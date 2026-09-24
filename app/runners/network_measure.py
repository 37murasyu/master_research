"""スマホから届いたランドマークで、三角測量から逆動力学までを回す。

なぜ既存の ``master_research_code.py`` を使わないか。スマホ経路では
**撮影と姿勢推定が不要**なので、あの 4,383 行の前半（カメラ制御・MediaPipe・
描画）が丸ごと要らない。フラグ分岐を足すと、映像が無い場合の描画経路まで
新設することになり、4,383 行に手を入れる羽目になる。

そこでオーケストレーションだけ新規に書き、**物理計算はすべて既存モジュールを
再利用する**。

再利用しているもの:
    push_up_model（座位プッシュアップのモデル。USB・オフライン経路と共有）
        joint_axes / push_up_torques / segment_from_storage（重力は app.hybrid.gravity 経由で estimate_gravity）
    utils.compute_local_torque / compute_joint_power
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

混成だけのもの（2026-09-24、USB 経路は触らない）:
    - 回の区切りと力学の関所は ``app.hybrid.rep_detector.RepDetector``（肩の中点の重力の上向きへの射影＝高さ）。
      USB と同じ ``PushCycleDetector``（左肩の y の往復）は、実行時の座標の y が奥行きなので、手を固定して体幹が
      上下するだけの押し上げで 1 回も閉じなかった。基準の高さは先頭の窓の中央値から始めて座面の高さを追う
    - 止めたときに開いたままの回は ``summary`` の ``unfinished_rep`` に残し、記録器が cycle_work に「未完」で書く
    - 先頭の窓で長さが決まらなかった部位（見えなかった肘・手首）は、窓が閉じた後に見えてから長さ・慣性・前腕長・
      帯・骨の長さの見張りの基準を決める（``late_segments``）。仕事を 1 フレームも積まなかった部位は、回の仕事を
      0.0 J ではなく NaN（記録は空欄）にする
    - トルクと仕事率は関所によらず毎フレーム計算して記録し、仕事とゲージには関所が開いている間だけ積む
      （座っている間の雑音の仕事を積まない。``MeasurementConfig.dyn_gate``）
    - 仕事はフレームごとの dt で積む（``app.hybrid.rep_work``）

**揃っていない点（重要）**:
    サイクルごとの量は仕事率 P = τ_y × (ω_リンク − ω_親)·y を積分した**仕事 [J]** で、既存の
    ``master_research_code.py`` が肘に対して使う特別な経路（``compute_cycle_energy_filtered``）は
    再現していない。したがって肘の値は USB 経路と**直接比較できない**。
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2 as cv
import numpy as np

# これらはフレームが流れ始める前に払っておく。関数内 import にすると、
# 最初の数フレームの中で utils(0.12s) + utils_dynamic(0.70s) の読み込みが走り、
# その間 asyncio の受信ループが止まる（約 0.8 秒、30 Hz で 25 フレームぶん受信が滞る）。
from body_part_storage_module import BodyPartDataStorage
# 部位キーと重力の大きさは config.py が持っている。utils 経由で既に読み込まれているので
# 追加コストなしで再利用できる。
from config import G_SCALAR, INERTIA_LENGTH_FRAMES, SEGMENT_MASS_FRACTIONS
from config import part_calculations
from config import part_keys as _PART_KEYS
from config import MP_LANDMARK, slot_of
from link_vector_calculator_module import LinkVectorCalculator
from push_up_model import (
    ARM_PARTS,
    arm_axes,
    hand_mass,
    push_up_joint_powers,
    push_up_torques,
    segment_from_storage,
    torso_load_mass,
    trunk_up_vectors,
)
from utils import compute_local_torque
from utils_dynamic import calculate_inertia_tensor, compute_triangulate_transform_native

from energy_pipeline import AdaptiveCutoff, EnergyFilterConfig, angle_between, compute_cycle_energy_filtered

from app.gauge.protocol import PART_NAMES
from app.gauge.thresholds import PartBand, part_bands
from app.hybrid.demo_gauge import DemoConfig, DemoGauge
from app.hybrid.ekf import EkfSettings, GridEkf
from app.hybrid.gravity import GravityChoice, choose_gravity
from app.hybrid.rep_detector import RepConfig, RepDetector, RepEvent
from app.tuning.ekf_profile import SCALE_REF_PAIR, body_scale_ratio
from app.hybrid.rep_work import PartWork, RepAccumulator, WorkSample
from app.net.sync_buffer import DEFAULT_GRID, GridSpec, PairedSample

# 歪み補正で扱う画像の外側の余白（幅・高さに対する比）。NetworkMeasurement._undistort を参照。
_UNDISTORT_MARGIN = 0.1


def _translate_image(P: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """画像座標を ``shift`` だけ平行移動したときの射影行列 ``S @ P``。

    像 x = P X を x' = x + shift に移すのは、同次座標で S = [[1, 0, sx], [0, 1, sy], [0, 0, 1]]
    を左から掛けることに等しい。座標と射影行列の両方に同じ S を掛ければ、三角測量の解 X は変わらない。
    """
    S = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])
    return S @ P


def _power_inputs_finite(joint: str, torque, axes, forearm, upper_arm, trunk_omega) -> bool:
    """関節の仕事率の入力（トルク・外側の部位の角速度・リンク・内側の部位の角速度）がすべて有限か。

    ``push_up_joint_powers`` は非有限が混じる関節の仕事率を 0 にする。見えない腕や慣性が決まらない腕の 0 を
    「0 W」として回に積むと、仕事は 0.0 J と普通の値に見える。ここで NaN に戻し、積まなかったフレームとして数える
    （``RepAccumulator`` は非有限の仕事率を積まない）。組は ``push_up_joint_powers`` の相対角速度と同じ。
    """
    omega, parent = {"wrist": (forearm.omega, None), "elbow": (upper_arm.omega, forearm.omega),
                     "shoulder": (upper_arm.omega, trunk_omega)}[joint]
    vectors = [torque, omega, axes[joint][0]] + ([parent] if parent is not None else [])
    return all(v is not None and np.all(np.isfinite(np.asarray(v, dtype=np.float64))) for v in vectors)


def _median_segment(frames: np.ndarray, ia: int, ib: int) -> float:
    """点列 (フレーム, 点, 3) の点 ``ia``–``ib`` の距離の中央値（有限のフレームだけ）。1 つも無ければ NaN。"""
    lengths = np.linalg.norm(frames[:, ia] - frames[:, ib], axis=1)
    lengths = lengths[np.isfinite(lengths)]
    return float(np.median(lengths)) if lengths.size else float("nan")


__all__ = ["NetworkMeasurement", "FrameResult", "MeasurementConfig", "ImplausibleBodyScale", "EXIT_IMPLAUSIBLE_SCALE"]

# 体格の検査で止めたときの終了コード（USB 経路の EXIT_IMPLAUSIBLE_SCALE と同じ。解像度の不一致と共用し、meta の error で見分ける）
EXIT_IMPLAUSIBLE_SCALE = 3


class ImplausibleBodyScale(ValueError):
    """先頭の窓の肩–肘の長さが人体の範囲（``ekf_profile.PLAUSIBLE_REF_LEN``）の外か、左右の肘が見えず測れない。

    範囲の外なら座標の単位か校正が壊れている（校正の並進を m で保存すると 1/100 になる。2026-09-23 の実機は右上腕が
    6.4 m）。そのまま逆動力学に入れるとトルクが桁違いになるので計測を止める。
    """


# 腕の長さの安全策で、長さがずれたフレームの後に積まないフレーム数を含めた幅（そのフレーム＋差分で速度・
# 加速度にそれを使う後の 2 フレーム）
_ARM_HISTORY = 3
# process の時間を中央値・95% の計算に残すフレーム数（30 Hz で 5 分）
_TIMING_WINDOW = 9000

# リンク定義は config.part_calculations が正本（USB 経路と共通）。
# ここでは (start, end) のタプル形式に落として使う。
PART_LINKS: dict[str, tuple[int, int]] = {
    name: (spec["start"], spec["end"]) for name, spec in part_calculations.items()
}

# 部位キーは config.py が持っている（順序も一致）。
PART_KEYS = tuple(_PART_KEYS)
# ゲージに出す部位（肩は出さない）。ゲージの行の書式が正本
GAUGE_PARTS = PART_NAMES

# 毎フレーム引き直さない位置索引と部位名
# 腕の力学に要る部位データ（どれかが無ければ _dynamics は None）
_DYN_NEEDED = tuple(part for parts in ARM_PARTS.values() for part in parts.values()) + ("both_shoulder",)
# 側 → (肩, 肘, 手首) の位置索引
_ARM_SLOTS = {side: tuple(slot_of(f"{side}_{n}") for n in ("SHOULDER", "ELBOW", "WRIST")) for side in ("L", "R")}
# 左右の肩の位置索引（高さ = 肩の中点）
_SHOULDER_SLOTS = (slot_of("L_SHOULDER"), slot_of("R_SHOULDER"))
# 慣性テンソルを長さから決める部位 → (utils_dynamic.calculate_inertia_tensor の係数の行, 始点, 終点)
_SEGMENTS = {
    "upper_arm_R": (3, "R_SHOULDER", "R_ELBOW"),
    "upper_arm_L": (3, "L_SHOULDER", "L_ELBOW"),
    "forearm_R": (4, "R_ELBOW", "R_WRIST"),
    "forearm_L": (4, "L_ELBOW", "L_WRIST"),
}


@dataclass
class MeasurementConfig:
    """計測のパラメータ。既定値は既存 USB 経路に揃えてある。"""

    body_mass_kg: float = 60.0

    # 慣性テンソルを確定させるまでに必要なフレーム数（既存と同じ）
    # 慣性テンソルのリンク長を確定するまでに溜めるフレーム数。
    # 1 フレームの瞬時値だとその瞬間の三角測量誤差が全実行に固定される（再検算 R-6）。
    inertia_ready_frames: int = INERTIA_LENGTH_FRAMES
    # 左右の肩と肘がそろう組が窓に満たないまま、肩が有限の組がこれだけたまったら、有限の値だけで窓を閉じる（約 5 秒）。
    # 片方の肘が画面の外に出続けると窓が永久に閉じず、トルクもゲージも出なかった（2026-09-24 のレビュー）
    window_timeout_frames: int = 150
    dynamics_ready_frames: int = max(7, INERTIA_LENGTH_FRAMES)

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
    # 力学の関所（HYBRID_DYN_GATE）。偽なら常に開いた扱い（回の区切りは RepDetector のまま）
    dyn_gate: bool = True
    # デモ（DEMO_MONO_GAUGE_ON=1）。None でなければ、ゲージの now をトルクではなく 3D の肩の上昇と肘角の変化で
    # 動かす（app.hybrid.demo_gauge）。回の区切り・トルク・記録は今までどおり
    demo: DemoConfig | None = None
    # OFFLINE_WRIST_CAPTURE: 終了時に前腕（肘→手首）と手首の局所 τ_y を npy に残す（USB と同じ形）
    offline_wrist_capture: bool = False
    # 肘の濾波 E± の前処理（energy_pipeline、USB の E_*）。計測の子は EnergyFilterConfig.from_env() を渡す
    energy_filter: EnergyFilterConfig = field(default_factory=EnergyFilterConfig)
    # 腕の長さの安全策: 先頭の窓の上腕長・前腕長（中央値）から、この比を超えてずれた腕の仕事率を回とゲージに
    # 積まない（トルクは記録する）。0 で無効。三角測量の誤りが続くと EKF でも吸収しきれず、2026-09-23 の実機の
    # 記録の再生で右腕の |τy| が最大 24 万 N·m になった
    arm_length_tolerance: float = 0.25

    # EKF（app.hybrid.ekf）。既定は有効・同梱の既定値の雑音。計測の子は EkfSettings.from_env() を渡す
    ekf: EkfSettings = field(default_factory=EkfSettings)

    # 同期バッファの格子（app.net.sync_buffer.GridSpec）。組み立てる側は SyncBuffer に同じものを渡す。
    # 格子の番号・EKF の dt・積む dt の上限・肘の濾波 E± の dt・記録の生 3D の格子はすべてこれを読む
    grid: GridSpec = DEFAULT_GRID

    # 保持するフレーム数の上限。長時間の計測でメモリを食い潰さないため。
    # 物理計算が実際に見るのは直近 2 フレームだけ（LinkVectorCalculator は
    # i と i-1、calculate_M_and_F は [-1] しか使わない）。
    history_limit: int = 600


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
    dt_s: float = DEFAULT_GRID.period_s
    # 関節ごとの仕事率 P = τ_y × ω_rel·y [W]（キーは local_torques と同じ）
    powers: dict[str, float] = field(default_factory=dict)
    # EKF の手前の 3D 点（三角測量の直後）。None なら points_3d と同じ（EKF を通していない記録）
    points_raw: np.ndarray | None = None
    # 同期バッファの格子の番号（最初の組を 0 とする）。組が持つ番号（PairedSample.grid_index）から引き、
    # 番号が無い組は round((t_ns − 最初の組の t_ns) / 格子の間隔)。抜けた組の分だけ飛ぶ
    grid_index: int = 0
    # 時刻の原点（grid_index が 0 の組の t_ns）。記録（app.hybrid.recorder）の t_s・生 3D の t もこれから測る。
    # None なら記録側が最初に書いた組の t_ns を使う（計測を通さずに作った結果）
    t0_ns: int | None = None
    # EKF の速度 [m/s]（点 × 3）。EKF が無効なら None
    velocity: np.ndarray | None = None
    # このフレームで先頭の窓が閉じた（体格・重力・帯が決まった）
    window_closed: bool = False
    # 関所が開いていた（このフレームの仕事を回とゲージに積んだ）。先読みで後から積んだフレームは偽のまま
    dyn_active: bool = False
    # 高さ = 肩の中点・上向き u [m]（窓が閉じる前・肩が無いフレームは NaN）
    height_m: float = float("nan")
    # このフレームの後の回の区切りの基準の高さ [m]（座面の高さを追う。窓が閉じる前は NaN）
    baseline_m: float = float("nan")
    # このフレームが属する回の番号（0 始まり＝それまでに確定した回の数）
    rep: int = 0
    # 腕の長さの安全策（L・R）。偽ならその腕の仕事率を回とゲージに積まなかった
    arm_ok: dict[str, bool] = field(default_factory=lambda: {"L": True, "R": True})
    # 回を確定したフレームだけ: 部位ごとの W+・W−（app.hybrid.rep_work.PartWork）と W_1RM [J]（帯が無い部位は None）
    cycle_parts: dict[str, PartWork] = field(default_factory=dict)
    cycle_w1rm: dict[str, float | None] = field(default_factory=dict)
    # このフレームの後のゲージの値（今の回の W+ [J]、部位 → 値）。tracker があればその値（デモなら置いた値）
    gauge_now: dict[str, float] = field(default_factory=dict)
    # 回を確定したフレームだけ: 肘の濾波 E±（部位 → {"e_pos","e_neg","fc","n_u"}）
    cycle_energy: dict[str, dict] = field(default_factory=dict)


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
        self.grid = self.config.grid
        # 時刻の原点（最初の組の t_ns）と、その組が持っていた格子の番号（番号が無い組なら None）
        self._t0_ns: int | None = None
        self._grid0: int | None = None
        self._prev_grid: int | None = None
        # 100 ms を超える抜けで速度の計算をやり直した回数
        self.dynamics_restarts = 0

        # process の時間 [s]（直近 _TIMING_WINDOW フレームと、全体の最大）。meta の timing に残す
        self._durations: deque[float] = deque(maxlen=_TIMING_WINDOW)
        self._duration_max = 0.0
        self._timed = 0

        self._demo = None if self.config.demo is None else DemoGauge(self.config.demo)
        # 肘の濾波 E± の適応カットオフ（E_FC_ADAPTIVE_ON=1 のときだけ動く）。毎フレーム左右の肘角の平均を渡す
        self._cutoff = AdaptiveCutoff(self.config.energy_filter, fps=self.grid.target_hz)

        # EKF（app.hybrid.ekf）。格子の間隔で回す。無効なら None
        self.ekf = (GridEkf(self.config.ekf, self.pose_keypoints, dt=self.grid.period_s)
                    if self.config.ekf.enabled else None)

        self._inertia: dict[str, np.ndarray] = {}
        self.gravity: np.ndarray | None = None
        self.gravity_choice: GravityChoice | None = None

        # 先頭の窓（肩と肘が有限の組を inertia_ready_frames 組）。閉じたら体格・重力・帯・回の区切りが決まる
        self._window_raw: list[np.ndarray] = []
        self._window_points: list[np.ndarray] = []
        # 肩が有限の組（窓が埋まらないときの予備）。直近 window_timeout_frames 組
        self._loose_raw: deque[np.ndarray] = deque(maxlen=max(1, self.config.window_timeout_frames))
        self._loose_points: deque[np.ndarray] = deque(maxlen=max(1, self.config.window_timeout_frames))
        self.window_closed = False
        self.window: dict = {}
        # 校正に盤を立てた向き（実行時の座標の単位ベクトル、app.hybrid.gravity.read_board_up）
        self.board_up = None if board_up is None else np.asarray(board_up, dtype=np.float64)
        self.up: np.ndarray | None = None
        self.baseline_height_m: float | None = None
        self.forearm_m: dict[str, float | None] = {}
        self.upper_arm_m: dict[str, float | None] = {}
        # 先頭の窓で長さが決まらなかった部位 → 窓が閉じた後に集めている長さ [m]（両端が見えたフレームだけ）
        self._late_lengths: dict[str, list[float]] = {}
        # 窓が閉じた後に長さを決めた部位 → 決めたフレーム
        self.late_segments: dict[str, int] = {}
        # 腕の長さの安全策: 最後に長さがずれてからのフレーム数と、積まなかったフレーム数
        self._arm_clean = {"L": _ARM_HISTORY, "R": _ARM_HISTORY}
        self.arm_guard_rejected = {"L": 0, "R": 0}
        self.bands: dict[str, PartBand] = {}
        self.rep_detector: RepDetector | None = None
        # ゲージの状態（app.gauge.tracker.GaugeTracker）。None なら積まない
        self.tracker = tracker

        # 確定した回。cycle_work は部位ごとの符号付きの仕事 W± [J]、cycles は回ごとの詳細（frame・t_ns・parts）
        self.cycle_work: dict[str, list[float]] = {k: [] for k in PART_KEYS}
        self.cycles: list[dict] = []
        # 押し上げでなかった回（最小の持ち上げに届かず捨てた）の数
        self.discarded_reps = 0
        # 今の回の仕事。フレームごとの dt で積む（かつては確定したフレームの dt を全体に掛けていた）。
        # 関所が開く前の輪の長さは回の区切りの設定（RepConfig.lookback_frames）に従う
        self.rep_work = RepAccumulator(PART_KEYS, max_step_s=self.grid.max_gap_s,
                                       lookahead=self.config.rep.lookback_frames)

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
        start = time.perf_counter()
        try:
            return self._process(pair)
        finally:
            elapsed = time.perf_counter() - start
            self._durations.append(elapsed)
            self._duration_max = max(self._duration_max, elapsed)
            self._timed += 1

    def timing(self) -> dict:
        """process の時間 [ms]（直近の中央値・95%・全体の最大）。受信スレッドの予算は 30 Hz で 33 ms。"""
        if not self._durations:
            return {"frames": 0, "median_ms": None, "p95_ms": None, "max_ms": None}
        recent = np.asarray(self._durations) * 1e3
        return {"frames": self._timed, "median_ms": float(np.median(recent)),
                "p95_ms": float(np.percentile(recent, 95)), "max_ms": self._duration_max * 1e3,
                "window": len(recent)}

    def _process(self, pair: PairedSample) -> FrameResult | None:
        raw = self.points_3d(pair)
        if raw is None:
            return None
        grid = self._grid_index(pair)
        missing = 0 if self._prev_grid is None else max(0, grid - self._prev_grid - 1)
        self._prev_grid = grid

        dt = self._timestep(pair.t_ns)
        if self.frame_index > 0 and dt > self.grid.max_gap_s:
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

        result = FrameResult(t_ns=pair.t_ns, points_3d=points, dt_s=dt, points_raw=raw,
                             grid_index=grid, t0_ns=self._t0_ns, velocity=velocity)

        if not self.window_closed:
            self._collect_window(raw, points, result)
        elif self._late_lengths:
            self._collect_late_segments(raw, points)

        if self.frame_index + 1 >= self.config.dynamics_ready_frames and self._inertia:
            dynamics = self._dynamics(points)
            sample = None
            result.arm_ok = self._arm_ok(points)
            if dynamics is not None:
                result.local_torques, result.powers, theta, tau_y = dynamics
                if self.config.energy_filter.fc_adaptive_on:   # 適応がオフなら fc は固定（step は何もしない）
                    angles = [v for v in theta.values() if math.isfinite(v)]
                    if angles:
                        self._cutoff.step(float(np.mean(angles)))
                sample = self._guarded_sample(
                    WorkSample(dt=dt, powers=result.powers, theta=theta, tau_y=tau_y), result.arm_ok)
            self._gate(points, velocity, dt, sample, result)

        if self._demo is not None and self.tracker is not None and self.window_closed:
            self.tracker.set_now(self._demo.update(points, self.up, self.bands))
        result.gauge_now = self._gauge_now()
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

    def _grid_index(self, pair: PairedSample) -> int:
        """同期バッファの格子の番号（最初の組を 0 とする）。

        組が番号を持っていればそれを使う（時刻から割り戻さない）。持っていない組（試験で直に作った組など）は
        最初の組からの時刻を格子の間隔で丸める。同期バッファの組の時刻は格子の上にあるので、どちらも同じ値になる。
        """
        if self._t0_ns is None:
            self._t0_ns = pair.t_ns
            self._grid0 = pair.grid_index
        if pair.grid_index is not None and self._grid0 is not None:
            return pair.grid_index - self._grid0
        return self.grid.index_of(pair.t_ns, self._t0_ns)

    def summary(self) -> dict:
        """計測を閉じるときに meta.json へ残す値（被験者の帯・重力・EKF・関所・腕の長さの安全策）。"""
        from config import OUTPUT_SCHEMA_VERSION

        choice = self.gravity_choice
        bands = self.bands
        return {
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "forearm_len_m": dict(self.forearm_m) or None,
            "upper_arm_len_m": dict(self.upper_arm_m) or None,
            "late_segments": dict(self.late_segments) or None,
            "w1rm_j": {part: band.w1rm for part, band in bands.items()} or None,
            "gauge_bands_j": {part: (None if band.band is None else list(band.band)) for part, band in bands.items()} or None,
            "gauge_band_reasons": {part: band.reason for part, band in bands.items() if band.reason} or None,
            "gravity": None if choice is None else {
                "source": choice.source, "label": choice.label, "vector": np.asarray(choice.vector).tolist(),
                "up_label": choice.up_label, "detail": choice.detail,
            },
            "ekf": self.ekf_provenance(),
            "dyn_gate": self.config.dyn_gate,
            "demo": self._demo is not None,
            "baseline_height_m": self.baseline_height_m,
            "rep_baseline": self._rep_baseline(),
            "reps": self.cycle_count,
            "unfinished_rep": self.unfinished_rep(),
            "discarded_reps": self.discarded_reps,
            "dynamics_restarts": self.dynamics_restarts,
            "arm_length_guard": {"tolerance": self.config.arm_length_tolerance,
                                 "rejected_frames": dict(self.arm_guard_rejected)},
            "timing": self.timing(),
        }

    def _rep_baseline(self) -> dict | None:
        """回の区切りの基準の高さ（先頭の窓の値・最後の値・置き換えた回数）。窓が閉じる前は None。"""
        detector = self.rep_detector
        if detector is None:
            return None
        return {"initial_m": detector.initial_baseline_m, "final_m": detector.baseline_m,
                "updates": detector.baseline_updates}

    def unfinished_rep(self) -> dict | None:
        """止めた時点で開いたままの回（関所が開いていて、まだ確定していない）。無ければ None。

        meta.json の ``unfinished_rep`` に残し、記録器（``app.hybrid.recorder.Recorder.close``）が cycle_work の末尾に
        「未完」（status=unfinished）の行を書く。frame・t_ns は最後に処理したフレーム。回の数（``reps``）には数えない。
        """
        detector = self.rep_detector
        if detector is None or not detector.is_open or self.frame_index == 0:
            return None
        parts = {}
        for key, work in self.rep_work.work().items():
            band = self.bands.get(key)
            counted = work.frames > 0   # 0 フレームの部位は「0 J」ではなく値なし（meta.json は NaN を書けない）
            parts[key] = {"work_j": work.net if counted else None, "work_pos_j": work.pos if counted else None,
                          "work_neg_j": work.neg if counted else None,
                          "w1rm_j": None if band is None else band.w1rm, "frames": work.frames}
        return {"frame": self.frame_index - 1, "t_ns": self._prev_t_ns, "open_s": detector.open_s,
                "max_lift_m": detector.max_lift_m, "parts": parts}

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
            return self.grid.period_s
        dt = (t_ns - self._prev_t_ns) / 1e9
        self._prev_t_ns = t_ns
        return dt if dt > 0 else self.grid.period_s

    def _update_links(self, dt: float) -> None:
        index = len(self._recent_points) - 1
        for part, calculator in self.calculators.items():
            result = calculator.calculate_link_vectors(self._recent_points, True, index, dt)
            if result[0] is None:
                continue
            r_vec, vel, omega, centroid, p1, acc, ang_acc = result
            self.storage.add_data(part, r_vec, vel, omega, centroid, p1, ang_acc, acc)

    def _collect_window(self, raw: np.ndarray, points: np.ndarray, result: FrameResult) -> None:
        """肩と肘が有限の組を先頭の窓に溜め、埋まったら閉じる。

        左右の肩が有限の組は予備にも溜め、窓が埋まらないまま ``window_timeout_frames`` 組たまったら、予備の組の
        有限の値だけで閉じる（見えない側の前腕長と帯は出さない）。
        """
        shoulders = [slot_of("L_SHOULDER"), slot_of("R_SHOULDER")]
        if not np.all(np.isfinite(raw[shoulders])):
            return
        self._loose_raw.append(raw)
        self._loose_points.append(points)
        needed = shoulders + [slot_of("L_ELBOW"), slot_of("R_ELBOW")]
        if np.all(np.isfinite(raw[needed])):
            self._window_raw.append(raw)
            self._window_points.append(points)
        timeout = self.config.window_timeout_frames
        if len(self._window_raw) >= self.config.inertia_ready_frames:
            self._close_window(self._window_raw, self._window_points, fallback=False)
        elif timeout > 0 and len(self._loose_raw) >= timeout:
            print("[計測] 先頭の窓: 左右の肩と肘がそろう組が 5 秒でたまらなかったので、見えている点だけで決めた"
                  "（見えない側の帯は出さない）。両方の画面に両腕が入っているか確かめること", file=sys.stderr)
            self._close_window(list(self._loose_raw), list(self._loose_points), fallback=True)
        else:
            return
        self._window_raw, self._window_points = [], []
        self._loose_raw.clear()
        self._loose_points.clear()
        result.window_closed = True

    def _close_window(self, raw_frames: list[np.ndarray], point_frames: list[np.ndarray], *,
                      fallback: bool = False) -> None:
        """先頭の窓で、体格の検査・EKF の掛け直し・慣性・重力・上向きと基準の高さ・前腕長・帯・回の区切りを決める。

        体格の検査は EKF の手前の値（``points_raw``）で、プロファイルの有無によらず行う（USB はプロファイル使用時だけ）。
        それ以外は EKF の後の値。
        """
        raw = np.stack(raw_frames)
        points = np.stack(point_frames)

        def ref_length(pair) -> float:
            return _median_segment(raw, *(self._keypoints_in_id_order.index(i) for i in pair))

        # 右の肩–肘（SCALE_REF_PAIR）が見えないときは左の肩–肘で確かめる（窓の予備で閉じたとき）
        run_length = ref_length(SCALE_REF_PAIR)
        if not math.isfinite(run_length):
            run_length = ref_length((MP_LANDMARK["L_SHOULDER"], MP_LANDMARK["L_ELBOW"]))
        if not math.isfinite(run_length):
            # 左右の肘が 1 フレームも見えなかった（予備で閉じた窓）。座標の単位の誤りとは別なので、案内も分ける
            raise ImplausibleBodyScale(
                "先頭の窓で左右の肘が一度も見えず、肩–肘の長さ（体格の検査）を測れない。"
                "両方のカメラの画面に両肘が入るように置き直すこと")
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
            length = _median_segment(points, slot_of(a), slot_of(b))
            return length if math.isfinite(length) else None

        self.forearm_m = {side: median_length(f"{side}_ELBOW", f"{side}_WRIST") for side in ("L", "R")}
        self.upper_arm_m = {side: median_length(f"{side}_SHOULDER", f"{side}_ELBOW") for side in ("L", "R")}
        # 窓で長さが決まらなかった部位は、窓が閉じた後に見えてから決める（_collect_late_segments）
        self._late_lengths = {name: [] for name in _SEGMENTS if self._segment_length(name) is None}
        self._set_bands()
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
            "fallback": fallback,
        }
        self.window_closed = True

    def _length_table(self, name: str) -> tuple[dict[str, float | None], str]:
        """部位（``_SEGMENTS`` の名前）の長さを持つ表（上腕長か前腕長）と側。"""
        kind, side = name.rsplit("_", 1)
        return (self.upper_arm_m if kind == "upper_arm" else self.forearm_m), side

    def _segment_length(self, name: str) -> float | None:
        """部位の決まった長さ [m]。決まっていなければ None。"""
        table, side = self._length_table(name)
        return table.get(side)

    def _set_bands(self) -> None:
        """実測の前腕長から帯を決め、ゲージにも渡す（窓が閉じたときと、前腕長を後から決めたとき）。"""
        self.bands = part_bands(self.config.body_mass_kg, self.forearm_m, self.config.one_rm or {})
        if self.tracker is not None:
            self.tracker.set_bands(self.bands)

    def _segment_inertia(self, name: str, length: float) -> np.ndarray:
        """部位の慣性テンソル。長さが有限の正でなければ NaN（慣性の回帰式に NaN の長さを渡さない）。"""
        if not (math.isfinite(length) and length > 0.0):
            return np.full((3, 3), np.nan)
        return calculate_inertia_tensor(_SEGMENTS[name][0], self.config.body_mass_kg, length)

    def _collect_late_segments(self, raw: np.ndarray, points: np.ndarray) -> None:
        """窓で長さが決まらなかった部位の長さを、両端が見えた（三角測量が有限の）フレームで集める。

        窓と同じ数（``inertia_ready_frames``）たまったら中央値で長さを決め、慣性・上腕長か前腕長・帯（前腕のとき。
        ゲージにも渡し直す）を埋める。骨の長さの見張り（``_arm_ok``）はこの長さを基準に検査を始める。
        """
        for name in list(self._late_lengths):
            _, start, end = _SEGMENTS[name]
            ends = [slot_of(start), slot_of(end)]
            if not (np.all(np.isfinite(raw[ends])) and np.all(np.isfinite(points[ends]))):
                continue
            lengths = self._late_lengths[name]
            lengths.append(float(np.linalg.norm(points[ends[0]] - points[ends[1]])))
            if len(lengths) < self.config.inertia_ready_frames:
                continue
            length = float(np.median(lengths))
            del self._late_lengths[name]
            self._inertia[name] = self._segment_inertia(name, length)
            table, side = self._length_table(name)
            table[side] = length
            if table is self.forearm_m:
                self._set_bands()
            self.late_segments[name] = self.frame_index
            print(f"[計測] 先頭の窓で決まらなかった {name} の長さを、見えてから決めた（{length:.3f} m、"
                  f"フレーム {self.frame_index}）", file=sys.stderr)

    def _build_inertia(self, samples: np.ndarray) -> None:
        """慣性テンソルと重力を確定させる。

        ``samples`` は (フレーム, 関節, 3) の点列。リンク長は**中央値**で決める。
        1 フレームの瞬時値だと三角測量の誤差がそのまま固定され、回帰式
        ``I = a*w + b*l + c`` は l に極端に敏感なので大きくずれる（再検算 R-6）。
        腕は左右で長さが違うので、テンソルも左右別に持つ（計画メモ E-1d）。

        重力は同じ初期フレームの体幹（腰中点 → 肩中点）の向きから決める（§1-5）。校正の meta に盤を立てた
        向き（``board_up``）があれば、最寄りの軸に吸着させて使う（``app.hybrid.gravity.choose_gravity``）。
        """
        # 窓で見えなかった部位の長さは NaN で、慣性も NaN（回帰式に渡さない）。見えてから決める（_collect_late_segments）
        self._inertia = {
            name: self._segment_inertia(name, _median_segment(samples, slot_of(start), slot_of(end)))
            for name, (_, start, end) in _SEGMENTS.items()
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

    def _dynamics(self, points: np.ndarray):
        """局所トルク・仕事率・肘角 θ・肘の τ_y。部位データが揃わなければ None。"""
        data = {name: self.storage.get_data(name) for name in PART_LINKS}
        if any(not data[name] for name in _DYN_NEEDED):
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
            shoulder, elbow, wrist = (points[i] for i in _ARM_SLOTS[side])
            forearm = segment_from_storage(
                data[parts["forearm"]][-1], self._inertia[f"forearm_{side}"], mass * SEGMENT_MASS_FRACTIONS["forearm"])
            upper_arm = segment_from_storage(
                data[parts["upper_arm"]][-1], self._inertia[f"upper_arm_{side}"],
                mass * SEGMENT_MASS_FRACTIONS["upper_arm"])
            torques = push_up_torques(forearm, upper_arm, wrist, elbow, shoulder, gravity, load, hand)
            axes = arm_axes(points, side)
            for joint, torque in torques.items():
                key = f"{joint}_{side}"
                link, parent = axes[joint]
                local[key] = compute_local_torque(torque, link, parent, up)
                self.storage.add_torque(key, local[key])
            trunk_omega = data["both_shoulder"][-1].get("omega")
            powers = push_up_joint_powers(torques, axes, forearm, upper_arm, trunk_omega, up)
            for joint, power in powers.items():
                finite = _power_inputs_finite(joint, torques[joint], axes, forearm, upper_arm, trunk_omega)
                powers_by_key[f"{joint}_{side}"] = float(power) if finite else float("nan")
            # 肘の濾波 E± の材料（USB と同じ θ = 肩→肘 と 肘→手首 のなす角、τ_y は肘の局所トルクの y）
            theta[f"elbow_{side}"] = angle_between(elbow - shoulder, wrist - elbow)
            tau_y[f"elbow_{side}"] = float(local[f"elbow_{side}"][1])
        return {key: local[key] for key in PART_KEYS}, powers_by_key, theta, tau_y

    def _arm_ok(self, points: np.ndarray) -> dict[str, bool]:
        """腕の長さの安全策。上腕長か前腕長が先頭の窓の中央値から許容の比を超えてずれた腕は偽。

        速度・加速度は直近のフレームとの差分なので、ずれたフレームの後の 2 フレームも偽にする。
        """
        tolerance = self.config.arm_length_tolerance
        if tolerance <= 0 or not self.window_closed:
            return {"L": True, "R": True}
        ok = {}
        for side, (shoulder, elbow, wrist) in _ARM_SLOTS.items():
            good = True
            for (a, b), reference in (((shoulder, elbow), self.upper_arm_m.get(side)),
                                      ((elbow, wrist), self.forearm_m.get(side))):
                if reference is None or not math.isfinite(reference) or reference <= 0:
                    continue   # 窓で長さを決められなかった腕は検査しない
                length = float(np.linalg.norm(points[a] - points[b]))
                if not (math.isfinite(length) and abs(length / reference - 1.0) <= tolerance):
                    good = False
            self._arm_clean[side] = self._arm_clean[side] + 1 if good else 0
            ok[side] = self._arm_clean[side] >= _ARM_HISTORY
        return ok

    def _guarded_sample(self, sample: WorkSample, arm_ok: dict[str, bool]) -> WorkSample:
        """長さがずれた腕の部位（手首・肘・肩）を仕事率・肘角・τ_y から外す。"""
        bad = [side for side, good in arm_ok.items() if not good]
        if not bad:
            return sample
        for side in bad:
            self.arm_guard_rejected[side] += 1

        def keep(key: str) -> bool:
            return not any(key.endswith(f"_{side}") for side in bad)

        return WorkSample(
            dt=sample.dt,
            powers={k: v for k, v in sample.powers.items() if keep(k)},
            theta={k: v for k, v in sample.theta.items() if keep(k)},
            tau_y={k: v for k, v in sample.tau_y.items() if keep(k)},
        )

    def _height(self, vectors: np.ndarray | None) -> float:
        """肩の中点の上向き成分（位置なら高さ [m]、速度なら上向きの速さ [m/s]）。"""
        if vectors is None or self.up is None:
            return float("nan")
        mid = 0.5 * (vectors[_SHOULDER_SLOTS[0]] + vectors[_SHOULDER_SLOTS[1]])
        return float(mid @ self.up)

    def _gate(self, points: np.ndarray, velocity: np.ndarray | None, dt: float,
              sample: WorkSample | None, result: FrameResult) -> None:
        """関所と回の区切り。開いている間だけ仕事とゲージに積み、閉じている間は先読みの輪に置く。"""
        result.rep = self.cycle_count
        detector = self.rep_detector
        gate = self.config.dyn_gate
        event = RepEvent.NONE
        result.height_m = self._height(points)
        if detector is not None:
            speed = self._height(velocity) if velocity is not None else None
            event = detector.update(result.height_m, speed, dt)
            result.baseline_m = detector.baseline_m
        if event is RepEvent.OPENED and gate:
            self.rep_work.release()
        is_open = (not gate) or (detector is not None and detector.is_open) \
            or event in (RepEvent.CLOSED, RepEvent.DISCARDED)
        result.dyn_active = is_open
        if sample is not None:
            if is_open:
                self.rep_work.add(sample)
            else:
                self.rep_work.hold(sample)
        self._sync_tracker()
        if event is RepEvent.CLOSED:
            self._close_rep(result)
        elif event is RepEvent.DISCARDED and gate:
            # 押し上げでなかった（持ち上げが 3 cm に届かない）。今の回の仕事を捨てる
            self.rep_work.reset()
            self.discarded_reps += 1
            if self.tracker is not None:
                self.tracker.discard_rep()

    def _elbow_energy(self) -> dict[str, dict]:
        """今の回の肘の濾波 E±（USB 経路と同じ ``compute_cycle_energy_filtered``、dt は格子の間隔（既定 1/30 s））。"""
        config = self.config.energy_filter
        fc = self._cutoff.fc if config.fc_adaptive_on else None
        work = self.rep_work.work()
        energy = {}
        for side in ("L", "R"):
            part = f"elbow_{side}"
            theta, tau = self.rep_work.series(part)
            e_pos, e_neg, info = compute_cycle_energy_filtered(theta, tau, self.grid.period_s, fc_override=fc,
                                                          config=config)
            frames = work[part].frames
            if frames == 0:   # 仕事を 1 フレームも積まなかった（肘のトルクが NaN）。0 J ではなく値なし
                e_pos = e_neg = float("nan")
            energy[part] = {"e_pos": e_pos, "e_neg": e_neg, "fc": info.get("fc"),
                            "n_u": int(info.get("n_u", 0)), "n_frames": frames}
        return energy

    def _gauge_now(self) -> dict[str, float]:
        """ゲージの今の値。tracker があればその値（デモなら置いた値）、無ければ今の回の W+。"""
        if self.tracker is not None:
            return self.tracker.values()
        work = self.rep_work.work()
        return {part: work[part].pos for part in GAUGE_PARTS}

    def _sync_tracker(self) -> None:
        """ゲージの now に今の回の W+ を置く。積むのは ``rep_work`` だけ（同じフレーム・同じ順で積むので値は同じ）。"""
        if self.tracker is None or self._demo is not None:
            return
        work = self.rep_work.work()
        self.tracker.set_now({part: work[part].pos for part in self.tracker.parts if part in work})

    def _close_rep(self, result: FrameResult) -> None:
        """今の回を確定する。仕事はフレームごとの dt で積んだ値（``rep_work``）。"""
        result.cycle_detected = True
        result.cycle_energy = self._elbow_energy()
        parts = self.rep_work.reset()
        for key, work in parts.items():
            # 有限の仕事率を 1 フレームも積まなかった部位は 0.0 J ではなく NaN（記録は空欄）
            net = work.net if work.frames else float("nan")
            self.cycle_work[key].append(net)
            result.cycle_work_j[key] = net
        result.cycle_parts = parts
        result.cycle_w1rm = {key: (self.bands[key].w1rm if key in self.bands else None) for key in parts}
        self.cycles.append({"frame": self.frame_index, "t_ns": result.t_ns, "parts": parts,
                            "energy": result.cycle_energy})
        if self.tracker is not None:
            self.tracker.close_rep()

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
