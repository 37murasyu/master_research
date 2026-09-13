"""スマホから届いたランドマークで、三角測量から逆動力学までを回す。

なぜ既存の ``master_research_code.py`` を使わないか。スマホ経路では
**撮影と姿勢推定が不要**なので、あの 4,383 行の前半（カメラ制御・MediaPipe・
描画）が丸ごと要らない。フラグ分岐を足すと、映像が無い場合の描画経路まで
新設することになり、4,383 行に手を入れる羽目になる。

そこでオーケストレーションだけ新規に書き、**物理計算はすべて既存モジュールを
再利用する**。既存の USB 経路は一切変更しないので回帰リスクがゼロ。

再利用しているもの:
    utils.compute_local_torque / PushCycleDetector
    utils_dynamic.calculate_inertia_tensor / calculate_M_and_F /
                  calculate_individual_torques
    link_vector_calculator_module.LinkVectorCalculator
    body_part_storage_module.BodyPartDataStorage

既存 USB 経路と揃えてある点:
    - キーポイントの並び順（config.pose_keypoints の宣言順）
    - リンク定義（part_calculations / links / parent_links）
    - 慣性テンソルの部位行と長さの取り方、r_g の重み、Imode / condition
    - サイクル検出の軸（既定 y）、閾値、mode='rise_to_rise'
    - ``compute_local_torque`` への parent_vec（肘面を基準に取る）

**揃っていない点（重要）**:
    サイクルごとの量は τ·ω を積分した**仕事 [J]** で、既存の
    ``master_research_code.py:3833-3843`` が肘・手首に対して使う特別な経路
    （``compute_cycle_energy_filtered`` と局所 y 成分の片側積算）は再現していない。
    したがって肘・手首の値は USB 経路と**直接比較できない**。肩・体幹に相当する
    「その他」分岐（P = τ·ω の積分）とは同じ定義。

注記: キーポイントの並び順が下流のインデックス演算と整合しているかには疑義がある
（code-review 指摘 #1 と同じ論点）。ここでは既存と同じ規約に揃えることを優先し、
規約自体は変更していない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# これらはフレームが流れ始める前に払っておく。関数内 import にすると、
# 最初の数フレームの中で utils(0.12s) + utils_dynamic(0.70s) の読み込みが走り、
# その間 asyncio の受信ループが止まる（同期バッファの窓 2 秒の 1/3 を食う）。
from body_part_storage_module import BodyPartDataStorage
# 部位キーと重力は config.py が持っている。utils 経由で既に読み込まれているので
# 追加コストなしで再利用できる。
from config import INERTIA_LENGTH_FRAMES
from config import g as GRAVITY
from config import part_calculations
from config import part_keys as _PART_KEYS
from config import slot_of
from link_vector_calculator_module import LinkVectorCalculator
from utils import PushCycleDetector, compute_local_torque
from utils_dynamic import (
    calculate_individual_torques,
    calculate_inertia_tensor,
    calculate_M_and_F,
    compute_triangulate_transform_native,
)

from app.net.sync_buffer import PairedSample

__all__ = ["NetworkMeasurement", "FrameResult", "MeasurementConfig"]


# リンク定義は config.part_calculations が正本（USB 経路と共通）。
# ここでは (start, end) のタプル形式に落として使う。
PART_LINKS: dict[str, tuple[int, int]] = {
    name: (spec["start"], spec["end"]) for name, spec in part_calculations.items()
}

# 局所トルクの基準リンク（既存 master_research_code.py の links 辞書と同じ）。
# 関節はランドマーク名で指す。位置索引を直書きすると pose_keypoints に点を足したときに
# 別の関節を指す（再検算 R-1 と同じ壊れ方）。
_TORQUE_LINK_IDS: dict[str, tuple[str, str]] = {
    "wrist_R": ("R_ELBOW", "R_WRIST"),        # 右手首 - 右肘
    "elbow_R": ("R_SHOULDER", "R_ELBOW"),     # 右肘 - 右肩
    "shoulder_R": ("L_SHOULDER", "R_SHOULDER"),   # 右肩 - 左肩
    "wrist_L": ("L_ELBOW", "L_WRIST"),        # 左手首 - 左肘
    "elbow_L": ("L_SHOULDER", "L_ELBOW"),     # 左肘 - 左肩
    "shoulder_L": ("R_SHOULDER", "L_SHOULDER"),   # 左肩 - 右肩
}


def _link_getter(origin: str, tip: str) -> Callable[[np.ndarray], np.ndarray]:
    a, b = slot_of(origin), slot_of(tip)
    return lambda p: p[b] - p[a]


TORQUE_LINKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    name: _link_getter(*ids) for name, ids in _TORQUE_LINK_IDS.items()
}

# 親リンク。局所座標の y 軸を親×z（肘面の法線）で取るために使う。
# 既存 master_research_code.py:3587-3594 と同じ対応。
PARENT_OF: dict[str, str | None] = {
    "wrist_R": "elbow_R",
    "elbow_R": "shoulder_R",
    "shoulder_R": None,
    "wrist_L": "elbow_L",
    "elbow_L": "shoulder_L",
    "shoulder_L": None,
}

# 仕事率 P = τ·ω を求めるときに、各関節へ対応させる部位の角速度。
# 既存 master_research_code.py:3790-3797 と同じ。
OMEGA_SOURCE: dict[str, str] = {
    "wrist_R": "forearm_R",
    "elbow_R": "upper_arm_R",
    "shoulder_R": "both_shoulder",
    "wrist_L": "forearm_L",
    "elbow_L": "up_arm_l",
    "shoulder_L": "both_shoulder",
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
    points_3d: np.ndarray
    local_torques: dict[str, np.ndarray] = field(default_factory=dict)
    cycle_detected: bool = False
    # サイクルごとの仕事 [J]。詳細はモジュール docstring の「揃っていない点」を参照。
    cycle_work_j: dict[str, float] = field(default_factory=dict)


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
    ):
        self.P0 = np.asarray(projection_left, dtype=np.float64)
        self.P1 = np.asarray(projection_right, dtype=np.float64)
        self.pose_keypoints = list(pose_keypoints)
        # 取り出し順はランドマーク ID の昇順で固定。毎フレーム並べ替えない。
        self._keypoints_in_id_order = sorted(self.pose_keypoints)
        self.config = config or MeasurementConfig()

        self.storage = BodyPartDataStorage()
        self.calculators = {
            part: LinkVectorCalculator(
                spec["start"], spec["end"], spec.get("com_fraction", 0.5))
            for part, spec in part_calculations.items()
        }

        # 直近 2 フレームだけ持つ。物理計算はそれ以上遡らない。
        self._recent_points: list[np.ndarray] = []
        self.frame_index = 0
        self._prev_t_ns: int | None = None

        self._inertia: dict[str, np.ndarray] = {}
        self._inertia_samples: list[np.ndarray] = []

        # サイクル検出
        self._baseline_sum = 0.0
        self._baseline_samples = 0
        self._detector = None
        self.cycle_work: dict[str, list[float]] = {k: [] for k in PART_KEYS}
        self._power_history: dict[str, list[float]] = {k: [] for k in PART_KEYS}

        self.results: list[FrameResult] = []

    # -- 入口 --------------------------------------------------------------
    def process(self, pair: PairedSample) -> FrameResult | None:
        """1 ペアを処理する。まだ計算できない段階では None を返す。"""
        keypoints0 = self._pixel_keypoints(pair, "cam0")
        keypoints1 = self._pixel_keypoints(pair, "cam1")
        if keypoints0 is None or keypoints1 is None:
            return None

        points = self._triangulate(keypoints0, keypoints1)
        self._recent_points.append(points)
        if len(self._recent_points) > self._REQUIRED_FRAMES:
            del self._recent_points[0]

        dt = self._timestep(pair.t_ns)
        self._update_links(dt)
        self._update_baseline(points)

        result = FrameResult(t_ns=pair.t_ns, points_3d=points)

        if not self._inertia:
            self._inertia_samples.append(points)
            if len(self._inertia_samples) >= self.config.inertia_ready_frames:
                self._build_inertia(np.stack(self._inertia_samples))
                self._inertia_samples = []

        if self.frame_index + 1 >= self.config.dynamics_ready_frames and self._inertia:
            torques = self._compute_local_torques(points)
            if torques is not None:
                result.local_torques = torques
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

    def _triangulate(self, keypoints0, keypoints1) -> np.ndarray:
        """三角測量して既存と同じ座標系に変換する。

        既存 USB 経路（``master_research_code._triangulate_transform_batch``）が
        呼ぶのと同じ関数に委譲する。手書きすると、変換 (x,y,z)->(-x,-z,-y) と
        0.01 倍のスケール、欠測の扱いを二重に管理することになり、
        ネイティブ DLL の高速経路も使えない。
        """
        return compute_triangulate_transform_native(
            self.P0, self.P1, keypoints0, keypoints1, scale=0.01
        )

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

    def _build_inertia(self, samples: np.ndarray) -> None:
        """慣性テンソルを確定させる。部位行と引数は既存と同じ。

        ``samples`` は (フレーム, 関節, 3) の点列。リンク長は**中央値**で決める。
        1 フレームの瞬時値だと三角測量の誤差がそのまま固定され、回帰式
        ``I = a*w + b*l + c`` は l に極端に敏感なので大きくずれる（再検算 R-6）。
        関節はランドマーク名で指す。位置索引を直書きすると pose_keypoints に
        点を足したときに別の関節を指す（再検算 R-1 と同じ壊れ方）。
        """
        mass = self.config.body_mass_kg

        def length(a: str, b: str) -> float:
            ia, ib = slot_of(a), slot_of(b)
            return float(np.nanmedian(np.linalg.norm(samples[:, ia] - samples[:, ib], axis=1)))

        # 胴体の半長 = |肩中点 − 腰中点| / 2
        shoulders = samples[:, slot_of("L_SHOULDER")] + samples[:, slot_of("R_SHOULDER")]
        hips = samples[:, slot_of("L_HIP")] + samples[:, slot_of("R_HIP")]
        half_body = 0.25 * float(np.nanmedian(np.linalg.norm(shoulders - hips, axis=1)))

        self._inertia = {
            "upper_arm": calculate_inertia_tensor(3, mass, length("L_SHOULDER", "L_ELBOW")),
            "forearm": calculate_inertia_tensor(4, mass, length("L_ELBOW", "L_WRIST")),
            "upper_body": calculate_inertia_tensor(1, mass, half_body),
            "lower_body": calculate_inertia_tensor(0, mass, half_body),
            "thigh": calculate_inertia_tensor(6, mass, length("R_HIP", "R_KNEE")),
        }

    def _compute_local_torques(self, points: np.ndarray) -> dict[str, np.ndarray] | None:
        data = {name: self.storage.get_data(name) for name in PART_LINKS}
        if any(not values for values in data.values()):
            return None

        mass = self.config.body_mass_kg
        # 部位質量。config.py の m1(上腕) m2(前腕) m4(太腿) と同じ係数。
        m_upper_arm = mass * 0.0227
        m_forearm = mass * 0.016
        m_thigh = mass * 0.11

        inertia = self._inertia

        def chain(arm: str, forearm: str, leg: str, condition: int):
            specs = [
                (inertia["upper_arm"], m_upper_arm, data[arm], {}),
                (inertia["forearm"], m_forearm, data[forearm], {}),
                (
                    inertia["upper_body"],
                    mass,
                    data["both_shoulder"],
                    {
                        "add_part_data": data["both_hip"],
                        "condition": condition,
                        "Imode": 3,
                        "Info_I3": points,
                    },
                ),
                (
                    inertia["lower_body"],
                    mass,
                    data["both_hip"],
                    {"add_part_data": data["both_shoulder"], "Imode": 4},
                ),
                (inertia["thigh"], m_thigh, data[leg], {}),
            ]
            rows = [
                calculate_M_and_F(tensor, mass_i, data_i, GRAVITY, **kwargs)
                for tensor, mass_i, data_i, kwargs in specs
            ]
            moments, forces, parts = (list(col) for col in zip(*rows))
            return moments, forces, parts

        try:
            Ms_r, Fs_r, parts_r = chain("upper_arm_R", "forearm_R", "upper_Leg_R", condition=1)
            Ms_l, Fs_l, parts_l = chain("up_arm_l", "forearm_L", "upper_Leg_L", condition=0)
        except (IndexError, KeyError, ValueError):
            return None

        def centroids(arm: str, forearm: str, leg: str) -> list[np.ndarray]:
            shoulder = data["both_shoulder"][-1]["centroid"]
            hip = data["both_hip"][-1]["centroid"]
            return [
                data[arm][-1]["centroid"],
                data[forearm][-1]["centroid"],
                (shoulder * 3 + hip) / 4,
                (shoulder + hip * 3) / 4,
                data[leg][-1]["centroid"],
            ]

        r_x = data["both_hip"][-1]["centroid"]
        tau_E = np.zeros(3)
        f_E = np.zeros(3)

        torques_r = calculate_individual_torques(
            Ms_r, Fs_r, np.array(centroids("upper_arm_R", "forearm_R", "upper_Leg_R")),
            tau_E, f_E, r_x, parts_r, self.storage,
        )
        torques_l = calculate_individual_torques(
            Ms_l, Fs_l, np.array(centroids("up_arm_l", "forearm_L", "upper_Leg_L")),
            tau_E, f_E, r_x, parts_l, self.storage,
        )

        global_torques = {
            "wrist_R": torques_r[0][0],
            "elbow_R": torques_r[1][0],
            "shoulder_R": torques_r[2][0],
            "wrist_L": torques_l[0][0],
            "elbow_L": torques_l[1][0],
            "shoulder_L": torques_l[2][0],
        }

        links = {key: builder(points) for key, builder in TORQUE_LINKS.items()}

        local = {}
        for key, global_torque in global_torques.items():
            parent_key = PARENT_OF[key]
            local[key] = compute_local_torque(
                global_torque,
                links[key],
                parent_vec=links[parent_key] if parent_key else None,
            )
            self.storage.add_torque(key, local[key])

        self._accumulate_power(global_torques, data)
        return local

    def _accumulate_power(
        self, global_torques: dict[str, np.ndarray], data: dict[str, list]
    ) -> None:
        """仕事率 P = τ·ω を溜める。既存 master_research_code.py:3789-3805 と同じ式。"""
        for key, part in OMEGA_SOURCE.items():
            torque = global_torques.get(key)
            entries = data.get(part)
            omega = entries[-1]["omega"] if entries else None
            if (
                torque is None
                or omega is None
                or not np.all(np.isfinite(torque))
                or not np.all(np.isfinite(omega))
            ):
                self._power_history[key].append(0.0)
            else:
                self._power_history[key].append(float(np.dot(torque, omega)))

    def _accumulate_cycle(self, points: np.ndarray, dt: float, result: FrameResult) -> None:
        """サイクルを検出し、その区間の仕事を積む。"""
        if self._detector is None:
            return

        value = self._cycle_value(points)
        if not math.isfinite(value):
            return

        if self._detector.update(value, self.frame_index):
            result.cycle_detected = True
            for key in PART_KEYS:
                series = self._power_history[key]
                if series:
                    work = float(np.sum(series) * dt)
                    self.cycle_work[key].append(work)
                    result.cycle_work_j[key] = work
                series.clear()

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
