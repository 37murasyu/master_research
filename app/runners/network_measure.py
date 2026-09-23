"""スマホから届いたランドマークで、三角測量から逆動力学までを回す。

なぜ既存の ``master_research_code.py`` を使わないか。スマホ経路では
**撮影と姿勢推定が不要**なので、あの 4,383 行の前半（カメラ制御・MediaPipe・
描画）が丸ごと要らない。フラグ分岐を足すと、映像が無い場合の描画経路まで
新設することになり、4,383 行に手を入れる羽目になる。

そこでオーケストレーションだけ新規に書き、**物理計算はすべて既存モジュールを
再利用する**。

再利用しているもの:
    push_up_model（座位プッシュアップのモデル。USB・オフライン経路と共有）
        estimate_gravity / joint_axes / push_up_torques / segment_from_storage
    utils.compute_local_torque / compute_joint_power / PushCycleDetector
    utils_dynamic.calculate_inertia_tensor
    link_vector_calculator_module.LinkVectorCalculator
    body_part_storage_module.BodyPartDataStorage

モデル（KNOWN_ISSUES §2-1）: 手を固定端に前腕 → 上腕の鎖を解き、体幹＋頭の荷重を肩に載せる。
重力は慣性テンソルを確定する初期フレームの体幹の向きから決める。かつては USB 経路と同じく
部位の並びが 1 つずれて ``wrist_R`` が右肘まわりのトルクになっており（§5-7）、下胴体に
体重 60 kg を丸ごと渡して 200〜300 N·m 出ていた（§5-8）。

既存 USB 経路と揃えてある点:
    - キーポイントの並び順（config.pose_keypoints の昇順）
    - リンク定義（part_calculations）と、関節ごとの局所軸（push_up_model.joint_axes）
    - 慣性テンソルの部位行と長さの取り方
    - サイクル検出の軸（既定 y）、閾値、mode='rise_to_rise'

**揃っていない点（重要）**:
    サイクルごとの量は仕事率 P = τ_y × (ω_リンク − ω_親)·y を積分した**仕事 [J]** で、既存の
    ``master_research_code.py`` が肘に対して使う特別な経路（``compute_cycle_energy_filtered``）は
    再現していない。したがって肘の値は USB 経路と**直接比較できない**。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2 as cv
import numpy as np

# これらはフレームが流れ始める前に払っておく。関数内 import にすると、
# 最初の数フレームの中で utils(0.12s) + utils_dynamic(0.70s) の読み込みが走り、
# その間 asyncio の受信ループが止まる（同期バッファの窓 2 秒の 1/3 を食う）。
from body_part_storage_module import BodyPartDataStorage
# 部位キーと重力の大きさは config.py が持っている。utils 経由で既に読み込まれているので
# 追加コストなしで再利用できる。
from config import G_SCALAR, INERTIA_LENGTH_FRAMES, SEGMENT_MASS_FRACTIONS
from config import g as _CONFIG_GRAVITY
from config import part_calculations
from config import part_keys as _PART_KEYS
from config import slot_of
from link_vector_calculator_module import LinkVectorCalculator
from push_up_model import (
    ARM_PARTS,
    arm_axes,
    estimate_gravity,
    hand_mass,
    push_up_joint_powers,
    push_up_torques,
    segment_from_storage,
    torso_load_mass,
    trunk_up_vectors,
)
from utils import PushCycleDetector, compute_local_torque
from utils_dynamic import calculate_inertia_tensor, compute_triangulate_transform_native

from energy_pipeline import angle_between

from app.gauge.thresholds import PartBand, part_bands
from app.hybrid.ekf import GRID_NS, EkfSettings, GridEkf
from app.hybrid.gravity import GravityChoice, choose_gravity
from app.hybrid.rep_detector import RepConfig, RepDetector
from app.tuning.ekf_profile import SCALE_REF_PAIR, body_scale_ratio
from app.hybrid.rep_work import MAX_STEP_S, RepAccumulator, WorkSample
from app.net.sync_buffer import PairedSample

# 歪み補正で扱う画像の外側の余白（幅・高さに対する比）。NetworkMeasurement._undistort を参照。
_UNDISTORT_MARGIN = 0.1


def _translate_image(P: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """画像座標を ``shift`` だけ平行移動したときの射影行列 ``S @ P``。

    像 x = P X を x' = x + shift に移すのは、同次座標で S = [[1, 0, sx], [0, 1, sy], [0, 0, 1]]
    を左から掛けることに等しい。座標と射影行列の両方に同じ S を掛ければ、三角測量の解 X は変わらない。
    """
    S = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])
    return S @ P

__all__ = ["NetworkMeasurement", "FrameResult", "MeasurementConfig", "ImplausibleBodyScale", "EXIT_IMPLAUSIBLE_SCALE"]

# 体格の検査で止めたときの終了コード（USB 経路の EXIT_IMPLAUSIBLE_SCALE と同じ。解像度の不一致と共用し、meta の error で見分ける）
EXIT_IMPLAUSIBLE_SCALE = 3


class ImplausibleBodyScale(ValueError):
    """先頭の窓の肩–肘の長さが人体の範囲（``ekf_profile.PLAUSIBLE_REF_LEN``）の外。

    座標の単位か校正が壊れている（校正の並進を m で保存すると 1/100 になる。2026-09-23 の実機は右上腕が 6.4 m）。
    そのまま逆動力学に入れるとトルクが桁違いになるので計測を止める。
    """

# 体幹から重力を決められないときの既定（三角測量の変換で z が上になる。カメラが水平という前提）
DEFAULT_GRAVITY = np.asarray(_CONFIG_GRAVITY, dtype=np.float64)


# リンク定義は config.part_calculations が正本（USB 経路と共通）。
# ここでは (start, end) のタプル形式に落として使う。
PART_LINKS: dict[str, tuple[int, int]] = {
    name: (spec["start"], spec["end"]) for name, spec in part_calculations.items()
}

# 部位キーは config.py が持っている（順序も一致）。
PART_KEYS = tuple(_PART_KEYS)

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class MeasurementConfig:
    """計測のパラメータ。既定値は既存 USB 経路に揃えてある。"""

    body_mass_kg: float = 60.0

    # 慣性テンソルを確定させるまでに必要なフレーム数（既存と同じ）
    # 慣性テンソルのリンク長を確定するまでに溜めるフレーム数。
    # 1 フレームの瞬時値だとその瞬間の三角測量誤差が全実行に固定される（再検算 R-6）。
    inertia_ready_frames: int = INERTIA_LENGTH_FRAMES
    dynamics_ready_frames: int = max(7, INERTIA_LENGTH_FRAMES)

    # サイクル検出の基準値を作るフレーム範囲（既存と同じ 5..14）
    baseline_first_frame: int = 5
    baseline_last_frame: int = 14
    # 基準値を確定させるのに最低限必要なサンプル数。欠測があっても
    # これだけ集まれば検出器を作る（1 フレームの欠測で無効化されないように）。
    baseline_min_samples: int = 3

    # サイクル検出に使う軸。既存の RT_CYCLE_AXIS（既定 'y'）に合わせる。
    cycle_axis: str = "y"
    cycle_threshold: float = 0.015
    cycle_velocity_epsilon: float = 0.01
    cycle_min_interval: int = 10
    cycle_mode: str = "rise_to_rise"
    cycle_negative_down: bool = True

    # 重力の決め方（push_up_model.estimate_gravity の mode）。慣性テンソルを確定する
    # 初期フレームの体幹の向きから決める。
    gravity_mode: str = "axis"
    # 盤の向きを吸着させる候補の軸（None なら 6 つ。app.hybrid.gravity.candidate_axes）、
    # 上位 2 つが近いときに優先する重力の向き（実行時の座標は z が上）と、その近さの幅（GRAVITY_AMBIG_DELTA）
    gravity_candidates: tuple[str, ...] | None = None
    gravity_preferred: str = "Z-"
    gravity_ambiguity: float = 0.08

    # 被験者の 1RM [kg]（部位 → 値、app.gauge.thresholds.load_one_rm）。None なら帯を出さない
    one_rm: Mapping[str, float | None] | None = None
    subject_id: str | None = None

    # 押し上げの回の区切り（関所を兼ねる、app.hybrid.rep_detector）
    rep: RepConfig = field(default_factory=RepConfig)

    # EKF（app.hybrid.ekf）。既定は有効・同梱の既定値の雑音。計測の子は EkfSettings.from_env() を渡す
    ekf: EkfSettings = field(default_factory=EkfSettings)

    # 保持するフレーム数の上限。長時間の計測でメモリを食い潰さないため。
    # 物理計算が実際に見るのは直近 2 フレームだけ（LinkVectorCalculator は
    # i と i-1、calculate_M_and_F は [-1] しか使わない）。
    history_limit: int = 600

    @property
    def cycle_axis_index(self) -> int:
        return _AXIS_INDEX.get(self.cycle_axis.strip().lower(), 1)


@dataclass
class FrameResult:
    """1 フレーム分の計算結果。"""

    t_ns: int
    # EKF の後の 3D 点（m、ランドマーク ID の昇順）。EKF が無効なら points_raw と同じ
    points_3d: np.ndarray
    local_torques: dict[str, np.ndarray] = field(default_factory=dict)
    cycle_detected: bool = False
    # サイクルごとの仕事 [J]。詳細はモジュール docstring の「揃っていない点」を参照。
    cycle_work_j: dict[str, float] = field(default_factory=dict)
    # 前の組からの実時間差 [s]。仕事はこの dt で積む（app.hybrid.rep_work）
    dt_s: float = 1.0 / 30.0
    # 関節ごとの仕事率 P = τ_y × ω_rel·y [W]（キーは local_torques と同じ）
    powers: dict[str, float] = field(default_factory=dict)
    # EKF の手前の 3D 点（三角測量の直後）。None なら points_3d と同じ（EKF を通していない記録）
    points_raw: np.ndarray | None = None
    # 同期バッファの格子の番号 round((t_ns − 最初の組の t_ns) / 33.3 ms)。抜けた組の分だけ飛ぶ
    grid_index: int = 0
    # EKF の速度 [m/s]（点 × 3）。EKF が無効なら None
    velocity: np.ndarray | None = None
    # このフレームで先頭の窓が閉じた（体格・重力・帯が決まった）
    window_closed: bool = False


class NetworkMeasurement:
    """ペアになったランドマークを順に流し込み、トルクと仕事を得る。"""

    # 物理計算が参照する過去フレーム数（LinkVectorCalculator が i-1 を見る）
    _REQUIRED_FRAMES = 2

    def __init__(
        self,
        projection_left: np.ndarray,
        projection_right: np.ndarray,
        pose_keypoints: Sequence[int],
        config: MeasurementConfig | None = None,
        lens: dict | None = None,
        *,
        tracker=None,
        board_up: np.ndarray | None = None,
    ):
        self.P0 = np.asarray(projection_left, dtype=np.float64)
        self.P1 = np.asarray(projection_right, dtype=np.float64)
        self.pose_keypoints = list(pose_keypoints)
        # 取り出し順はランドマーク ID の昇順で固定。毎フレーム並べ替えない。
        self._keypoints_in_id_order = sorted(self.pose_keypoints)
        self.config = config or MeasurementConfig()
        # 役割 → 内部パラメータ（K、歪み係数、画像寸法）。None なら歪み補正をしない。
        self.lens = lens
        # 三角測量に渡す射影行列。歪み補正をするときは平行移動を掛ける（_undistort）。
        self._triangulation_P = (self.P0, self.P1)
        self._shift: dict[str, np.ndarray] = {}
        if lens is not None:
            self._shift = {
                role: np.asarray(lens[role].size, dtype=np.float64) for role in ("cam0", "cam1")
            }
            self._triangulation_P = (
                _translate_image(self.P0, self._shift["cam0"]),
                _translate_image(self.P1, self._shift["cam1"]),
            )

        self._restart_dynamics()
        self.frame_index = 0
        self._prev_t_ns: int | None = None
        self._t0_ns: int | None = None
        self._prev_grid: int | None = None
        # 100 ms を超える抜けで速度の計算をやり直した回数
        self.dynamics_restarts = 0

        # EKF（app.hybrid.ekf）。無効なら None
        self.ekf = GridEkf(self.config.ekf, self.pose_keypoints) if self.config.ekf.enabled else None

        self._inertia: dict[str, np.ndarray] = {}
        self.gravity: np.ndarray | None = None
        self.gravity_choice: GravityChoice | None = None

        # 先頭の窓（肩と肘が有限の組を inertia_ready_frames 組）。閉じたら体格・重力・帯・回の区切りが決まる
        self._window_raw: list[np.ndarray] = []
        self._window_points: list[np.ndarray] = []
        self.window_closed = False
        self.window: dict = {}
        # 校正に盤を立てた向き（実行時の座標の単位ベクトル、app.hybrid.gravity.read_board_up）
        self.board_up = None if board_up is None else np.asarray(board_up, dtype=np.float64)
        self.up: np.ndarray | None = None
        self.baseline_height_m: float | None = None
        self.forearm_m: dict[str, float | None] = {}
        self.bands: dict[str, PartBand] = {}
        self.rep_detector: RepDetector | None = None
        # ゲージの状態（app.gauge.tracker.GaugeTracker）。None なら積まない
        self.tracker = tracker

        # サイクル検出
        self._baseline_sum = 0.0
        self._baseline_samples = 0
        self._detector = None
        self.cycle_work: dict[str, list[float]] = {k: [] for k in PART_KEYS}
        # 今の回の仕事。フレームごとの dt で積む（かつては確定したフレームの dt を全体に掛けていた）
        self.rep_work = RepAccumulator(PART_KEYS)

        self.results: list[FrameResult] = []

    # -- 入口 --------------------------------------------------------------
    def points_3d(self, pair: PairedSample) -> np.ndarray | None:
        """1 ペアの 3D 点（m、ランドマーク ID の昇順）。状態を持たないので、記録から三角測量し直すのにも使う
        （``app.hybrid.retriangulate``）。どちらかのロールが無ければ None。
        """
        keypoints0 = self._pixel_keypoints(pair, "cam0")
        keypoints1 = self._pixel_keypoints(pair, "cam1")
        if keypoints0 is None or keypoints1 is None:
            return None

        if self.lens is not None:
            keypoints0 = self._undistort(keypoints0, "cam0")
            keypoints1 = self._undistort(keypoints1, "cam1")
        return self._triangulate(keypoints0, keypoints1)

    def process(self, pair: PairedSample) -> FrameResult | None:
        """1 ペアを処理する。まだ計算できない段階では None を返す。"""
        raw = self.points_3d(pair)
        if raw is None:
            return None
        grid = self._grid_index(pair.t_ns)
        missing = 0 if self._prev_grid is None else max(0, grid - self._prev_grid - 1)
        self._prev_grid = grid

        dt = self._timestep(pair.t_ns)
        if self.frame_index > 0 and dt > MAX_STEP_S:
            # 長い抜けの後は、抜ける前のフレームとの差で速度・加速度を作らない（トルクが跳ねる）
            self._restart_dynamics()
            self.dynamics_restarts += 1
        if self.ekf is None:
            points, velocity = raw, None
        else:
            points, velocity = self.ekf.step(raw, missing)

        self._recent_points.append(points)
        if len(self._recent_points) > self._REQUIRED_FRAMES:
            del self._recent_points[0]

        self._update_links(dt)
        self._update_baseline(raw)

        result = FrameResult(t_ns=pair.t_ns, points_3d=points, dt_s=dt, points_raw=raw,
                             grid_index=grid, velocity=velocity)

        if not self.window_closed:
            self._collect_window(raw, points, result)

        if self.frame_index + 1 >= self.config.dynamics_ready_frames and self._inertia:
            dynamics = self._dynamics(points)
            if dynamics is not None:
                result.local_torques, result.powers, theta, tau_y = dynamics
                self.rep_work.add(WorkSample(dt=dt, powers=result.powers, theta=theta, tau_y=tau_y))
                self._accumulate_cycle(points, dt, result)

        self.frame_index += 1
        self._append_result(result)
        self._trim_storage()
        return result

    # -- 各段 --------------------------------------------------------------
    def _pixel_keypoints(self, pair: PairedSample, role: str) -> list[list[float]] | None:
        frame = pair.frames.get(role)
        if frame is None:
            return None
        # 既存 `_extract_keypoints_fast_single` と同じくランドマーク ID の昇順。
        # pose_keypoints の宣言順ではない（再検算 R-1。根拠は utils.extract_keypoints）。
        return [list(frame.pixel_xy(index)) for index in self._keypoints_in_id_order]

    def _undistort(self, keypoints, role: str) -> np.ndarray:
        """歪みを除いたピクセル座標に直し、三角測量用に平行移動して返す。

        - 画像の外側は余白（``_UNDISTORT_MARGIN``）までを補正する。MediaPipe は画面の
          少し外まで点を外挿して返す。それより遠い点は歪みの多項式の外挿が暴れるので NaN
        - 補正後の座標は、画像の内側の点でも端では負になる（樽型歪みは外へ押し出す）。
          OpenCV 側の三角測量は負の座標を「未検出」（USB 経路の -1）として捨てるので、
          画像寸法だけ平行移動して正に保つ。射影行列にも同じ移動を掛けてあるので
          （``_triangulation_P``）、三角測量の結果は変わらない
        """
        lens = self.lens[role]
        points = np.asarray(keypoints, dtype=np.float64)
        w, h = lens.size
        mx, my = w * _UNDISTORT_MARGIN, h * _UNDISTORT_MARGIN
        x, y = points[:, 0], points[:, 1]
        with np.errstate(invalid="ignore"):
            valid = (
                np.isfinite(points).all(axis=1)
                & (x >= -mx) & (x < w + mx) & (y >= -my) & (y < h + my)
            )
        result = np.full_like(points, np.nan)
        if valid.any():
            corrected = cv.undistortPoints(
                points[valid].reshape(-1, 1, 2), lens.K, lens.distortion, P=lens.K
            ).reshape(-1, 2)
            result[valid] = corrected + self._shift[role]
        return result

    def _triangulate(self, keypoints0, keypoints1) -> np.ndarray:
        """三角測量して既存と同じ座標系に変換する。

        既存 USB 経路（``master_research_code._triangulate_transform_batch``）が
        呼ぶのと同じ関数に委譲する。手書きすると、変換 (x,y,z)->(-x,-z,-y) と
        0.01 倍のスケール、欠測の扱いを二重に管理することになり、
        ネイティブ DLL の高速経路も使えない。
        """
        P0, P1 = self._triangulation_P
        return compute_triangulate_transform_native(
            P0, P1, keypoints0, keypoints1, scale=0.01
        )

    def _restart_dynamics(self) -> None:
        """リンクの速度の計算器・部位データ・直近の点を作り直す（最初と、100 ms を超える抜けの後）。"""
        self.storage = BodyPartDataStorage()
        self.calculators = {
            part: LinkVectorCalculator(
                spec["start"], spec["end"], spec.get("com_fraction", 0.5))
            for part, spec in part_calculations.items()
        }
        # 直近 2 フレームだけ持つ。物理計算はそれ以上遡らない。
        self._recent_points: list[np.ndarray] = []

    def _grid_index(self, t_ns: int) -> int:
        """同期バッファの格子の番号（最初の組を 0 とする）。"""
        if self._t0_ns is None:
            self._t0_ns = t_ns
        return round((t_ns - self._t0_ns) / GRID_NS)

    def ekf_provenance(self) -> dict:
        """EKF の出どころ（meta.json・サイドカー用）。"""
        return {"enabled": False} if self.ekf is None else self.ekf.provenance()

    def _timestep(self, t_ns: int) -> float:
        """前フレームとの実時間差。

        USB カメラの経路（master_research_code.py）は撮影時刻を持たず、ループのジッタを
        速度・加速度に持ち込まないよう間引き設定から dt を算出している
        （config.resolve_dynamics_dt）。こちらは**撮影時刻の差**を使える。
        無線のジッタがあっても、時刻で再標本化した後のグリッド間隔になる。
        """
        if self._prev_t_ns is None:
            self._prev_t_ns = t_ns
            return 1.0 / 30.0
        dt = (t_ns - self._prev_t_ns) / 1e9
        self._prev_t_ns = t_ns
        return dt if dt > 0 else 1.0 / 30.0

    def _update_links(self, dt: float) -> None:
        index = len(self._recent_points) - 1
        for part, calculator in self.calculators.items():
            result = calculator.calculate_link_vectors(self._recent_points, True, index, dt)
            if result[0] is None:
                continue
            r_vec, vel, omega, centroid, p1, acc, ang_acc = result
            self.storage.add_data(part, r_vec, vel, omega, centroid, p1, ang_acc, acc)

    def _cycle_value(self, points: np.ndarray) -> float:
        """サイクル検出に使うスカラー。既存は右手首相当の点の指定軸。"""
        return float(points[0][self.config.cycle_axis_index])

    def _update_baseline(self, points: np.ndarray) -> None:
        """サイクル検出の基準値（安定座位での値）を作る。

        欠測に強くしてある。既存はフレーム番号ちょうどで検出器を作るので、
        その 1 フレームが欠測だと以後一度も検出されない。ここでは
        実際に集まったサンプル数で平均し、範囲を過ぎた時点で確定させる。
        """
        config = self.config
        value = self._cycle_value(points)

        in_window = config.baseline_first_frame <= self.frame_index <= config.baseline_last_frame
        if in_window and math.isfinite(value):
            self._baseline_sum += value
            self._baseline_samples += 1

        if self._detector is not None:
            return
        if self.frame_index < config.baseline_last_frame:
            return
        if self._baseline_samples < config.baseline_min_samples:
            # まだ足りない。窓を過ぎても集まるまで待つ（全滅時は検出しない）。
            return

        baseline = self._baseline_sum / self._baseline_samples
        self._detector = PushCycleDetector(
            baseline,
            threshold=config.cycle_threshold,
            velocity_epsilon=config.cycle_velocity_epsilon,
            min_interval=config.cycle_min_interval,
            mode=config.cycle_mode,
            negative_down=config.cycle_negative_down,
        )

    def _collect_window(self, raw: np.ndarray, points: np.ndarray, result: FrameResult) -> None:
        """肩と肘が有限の組を先頭の窓に溜め、埋まったら閉じる。"""
        needed = [slot_of(name) for name in ("L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW")]
        if not np.all(np.isfinite(raw[needed])):
            return
        self._window_raw.append(raw)
        self._window_points.append(points)
        if len(self._window_raw) >= self.config.inertia_ready_frames:
            self._close_window()
            result.window_closed = True

    def _close_window(self) -> None:
        """先頭の窓で、体格の検査・EKF の掛け直し・慣性・重力・上向きと基準の高さ・前腕長・帯・回の区切りを決める。

        体格の検査は EKF の手前の値（``points_raw``）で、プロファイルの有無によらず行う（USB はプロファイル使用時だけ）。
        それ以外は EKF の後の値。
        """
        raw = np.stack(self._window_raw)
        points = np.stack(self._window_points)
        self._window_raw, self._window_points = [], []

        first, second = (self._keypoints_in_id_order.index(i) for i in SCALE_REF_PAIR)
        run_length = float(np.nanmedian(np.linalg.norm(raw[:, first] - raw[:, second], axis=1)))
        noise = None if self.ekf is None else self.ekf.noise
        scale_ref = noise.resolution.scale_ref if noise is not None and noise.origin == "profile" else None
        try:
            ratio = body_scale_ratio(scale_ref, run_length)
        except ValueError as error:
            raise ImplausibleBodyScale(str(error)) from error
        if scale_ref:
            self.ekf.set_scale(ratio)

        self._build_inertia(points)
        self.up = -self.gravity / np.linalg.norm(self.gravity)
        shoulders = 0.5 * (points[:, slot_of("L_SHOULDER")] + points[:, slot_of("R_SHOULDER")])
        self.baseline_height_m = float(np.nanmedian(shoulders @ self.up))

        def median_length(a: str, b: str) -> float | None:
            lengths = np.linalg.norm(points[:, slot_of(a)] - points[:, slot_of(b)], axis=1)
            lengths = lengths[np.isfinite(lengths)]
            return float(np.median(lengths)) if lengths.size else None

        self.forearm_m = {side: median_length(f"{side}_ELBOW", f"{side}_WRIST") for side in ("L", "R")}
        self.bands = part_bands(self.config.body_mass_kg, self.forearm_m, self.config.one_rm or {})
        if self.tracker is not None:
            self.tracker.set_bands(self.bands)
        if math.isfinite(self.baseline_height_m):
            self.rep_detector = RepDetector(self.baseline_height_m, self.config.rep)

        choice = self.gravity_choice
        self.window = {
            "ekf_scale_ratio": ratio if scale_ref else None,
            "ekf_run_length_m": run_length,
            "gravity": self.gravity.tolist(),
            "gravity_label": None if choice is None else choice.label,
            "gravity_source": None if choice is None else choice.source,
            "baseline_height_m": self.baseline_height_m,
            "forearm_len_m": dict(self.forearm_m),
        }
        self.window_closed = True

    def _build_inertia(self, samples: np.ndarray) -> None:
        """慣性テンソルと重力を確定させる。

        ``samples`` は (フレーム, 関節, 3) の点列。リンク長は**中央値**で決める。
        1 フレームの瞬時値だと三角測量の誤差がそのまま固定され、回帰式
        ``I = a*w + b*l + c`` は l に極端に敏感なので大きくずれる（再検算 R-6）。
        腕は左右で長さが違うので、テンソルも左右別に持つ（計画メモ E-1d）。

        重力は同じ初期フレームの体幹（腰中点 → 肩中点）の向きから決める（§1-5）。校正の meta に盤を立てた
        向き（``board_up``）があれば、最寄りの軸に吸着させて使う（``app.hybrid.gravity.choose_gravity``）。
        """
        mass = self.config.body_mass_kg

        def length(a: str, b: str) -> float:
            ia, ib = slot_of(a), slot_of(b)
            return float(np.nanmedian(np.linalg.norm(samples[:, ia] - samples[:, ib], axis=1)))

        self._inertia = {
            "upper_arm_R": calculate_inertia_tensor(3, mass, length("R_SHOULDER", "R_ELBOW")),
            "upper_arm_L": calculate_inertia_tensor(3, mass, length("L_SHOULDER", "L_ELBOW")),
            "forearm_R": calculate_inertia_tensor(4, mass, length("R_ELBOW", "R_WRIST")),
            "forearm_L": calculate_inertia_tensor(4, mass, length("L_ELBOW", "L_WRIST")),
        }
        ups = trunk_up_vectors(*(samples[:, slot_of(n)] for n in ("L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP")))
        config = self.config
        choice = choose_gravity(
            ups, self.board_up, magnitude=G_SCALAR, mode=config.gravity_mode,
            candidates=config.gravity_candidates, preferred_gravity=config.gravity_preferred,
            ambiguity=config.gravity_ambiguity)
        if choice.source == "default":
            # 腰が一度も取れなかった。受信ループを止めないよう、z が上（カメラが水平）の既定で続ける
            warnings.warn(f"初期フレームの体幹から重力を決められない（{choice.detail}）",
                          RuntimeWarning, stacklevel=2)
        self.gravity_choice = choice
        self.gravity = np.asarray(choice.vector, dtype=np.float64).copy()

    def _compute_local_torques(self, points: np.ndarray) -> dict[str, np.ndarray] | None:
        dynamics = self._dynamics(points)
        return None if dynamics is None else dynamics[0]

    def _dynamics(self, points: np.ndarray):
        """局所トルク・仕事率・肘角 θ・肘の τ_y。部位データが揃わなければ None。"""
        data = {name: self.storage.get_data(name) for name in PART_LINKS}
        needed = [part for parts in ARM_PARTS.values() for part in parts.values()] + ["both_shoulder"]
        if any(not data[name] for name in needed):
            return None

        mass = self.config.body_mass_kg
        load = torso_load_mass(mass)
        hand = hand_mass(mass)
        gravity = self.gravity
        up = -gravity

        local: dict[str, np.ndarray] = {}
        powers_by_key: dict[str, float] = {}
        theta: dict[str, float] = {}
        tau_y: dict[str, float] = {}
        for side, parts in ARM_PARTS.items():
            forearm = segment_from_storage(
                data[parts["forearm"]][-1], self._inertia[f"forearm_{side}"], mass * SEGMENT_MASS_FRACTIONS["forearm"])
            upper_arm = segment_from_storage(
                data[parts["upper_arm"]][-1], self._inertia[f"upper_arm_{side}"],
                mass * SEGMENT_MASS_FRACTIONS["upper_arm"])
            torques = push_up_torques(
                forearm, upper_arm,
                points[slot_of(f"{side}_WRIST")], points[slot_of(f"{side}_ELBOW")],
                points[slot_of(f"{side}_SHOULDER")], gravity, load, hand)
            axes = arm_axes(points, side)
            for joint, torque in torques.items():
                key = f"{joint}_{side}"
                link, parent = axes[joint]
                local[key] = compute_local_torque(torque, link, parent, up)
                self.storage.add_torque(key, local[key])
            powers = push_up_joint_powers(
                torques, axes, forearm, upper_arm, data["both_shoulder"][-1].get("omega"), up)
            for joint, power in powers.items():
                powers_by_key[f"{joint}_{side}"] = float(power)
            # 肘の濾波 E± の材料（USB と同じ θ = 肩→肘 と 肘→手首 のなす角、τ_y は肘の局所トルクの y）
            shoulder, elbow, wrist = (points[slot_of(f"{side}_{n}")] for n in ("SHOULDER", "ELBOW", "WRIST"))
            theta[f"elbow_{side}"] = angle_between(elbow - shoulder, wrist - elbow)
            tau_y[f"elbow_{side}"] = float(local[f"elbow_{side}"][1])
        return {key: local[key] for key in PART_KEYS}, powers_by_key, theta, tau_y

    def _accumulate_cycle(self, points: np.ndarray, dt: float, result: FrameResult) -> None:
        """サイクルを検出し、その区間の仕事を積む。"""
        if self._detector is None:
            return

        value = self._cycle_value(points)
        if not math.isfinite(value):
            return

        if self._detector.update(value, self.frame_index):
            self._close_rep(result)

    def _close_rep(self, result: FrameResult) -> None:
        """今の回を確定する。仕事はフレームごとの dt で積んだ値（``rep_work``）。"""
        result.cycle_detected = True
        for key, work in self.rep_work.reset().items():
            self.cycle_work[key].append(work.net)
            result.cycle_work_j[key] = work.net

    # -- メモリ管理 --------------------------------------------------------
    def _append_result(self, result: FrameResult) -> None:
        """結果を溜める。上限を超えたら古いものから捨てる。

        長時間の計測で溜め込むと、1 時間で 10 万件を超えてメモリを食い潰す。
        上流の SyncBuffer が時間窓で捨てているのと同じ理由。
        全件必要なら、呼び出し側が逐次書き出すこと。
        """
        self.results.append(result)
        limit = self.config.history_limit
        if limit > 0 and len(self.results) > limit:
            del self.results[: len(self.results) - limit]

    def _trim_storage(self) -> None:
        """BodyPartDataStorage を直近数フレームに切り詰める。

        既存の BodyPartDataStorage は部位ごとに毎フレーム辞書を積むだけで
        捨てる仕組みが無い。参照されるのは ``[-1]`` だけなので、
        少し残しておけば足りる。
        """
        keep = self._REQUIRED_FRAMES
        for part, entries in self.storage.storage.items():
            if len(entries) > keep:
                self.storage.storage[part] = entries[-keep:]
        for key, values in self.storage.torques.items():
            if len(values) > keep:
                self.storage.torques[key] = values[-keep:]

    # -- 参照 --------------------------------------------------------------
    @property
    def latest_cycle_work(self) -> dict[str, float]:
        """直近のサイクルの仕事 [J]。"""
        return {key: (values[-1] if values else 0.0) for key, values in self.cycle_work.items()}

    @property
    def cycle_count(self) -> int:
        return len(self.cycle_work[PART_KEYS[0]])
