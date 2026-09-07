"""スマホから届いたランドマークで、三角測量から逆動力学までを回す。

なぜ既存の ``master_research_code.py`` を使わないか。スマホ経路では
**撮影と姿勢推定が不要**なので、あの 4,383 行の前半（カメラ制御・MediaPipe・
描画）が丸ごと要らない。フラグ分岐を足すと、映像が無い場合の描画経路まで
新設することになり、4,383 行に手を入れる羽目になる。

そこでオーケストレーションだけ新規に書き、**物理計算はすべて既存モジュールを
再利用する**。既存の USB 経路は一切変更しないので回帰リスクがゼロ。

再利用しているもの:
    utils.DLT / compute_local_torque / PushCycleDetector
    utils_dynamic.calculate_inertia_tensor / calculate_M_and_F /
                  calculate_individual_torques
    link_vector_calculator_module.LinkVectorCalculator
    body_part_storage_module.BodyPartDataStorage

代償はオーケストレーションが 2 本になること。**規約は既存に忠実に合わせて**
あり（キーポイントの並び順、リンク定義、慣性テンソルの部位行、r_g の重み）、
同一入力で両経路の出力が一致することをテストで確かめる。

注記: キーポイントの並び順は既存実装（``_extract_keypoints_fast_single``）に
合わせて ``config.pose_keypoints`` の**宣言順**にしてある。この順序が下流の
インデックス演算と整合しているかは research 側の論点で、ここでは判断しない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from app.net.sync_buffer import PairedSample

__all__ = ["NetworkMeasurement", "FrameResult", "MeasurementConfig"]


# 既存 master_research_code.py と同じリンク定義
PART_LINKS: dict[str, tuple[int, int]] = {
    "upper_arm_R": (3, 1),
    "forearm_R": (5, 3),
    "both_shoulder": (0, 1),
    "both_hip": (6, 7),
    "up_arm_l": (2, 0),
    "forearm_L": (4, 2),
    "upper_Leg_R": (7, 9),
    "upper_Leg_L": (6, 8),
}

# 局所トルクに変換するときの基準リンク（既存の links 辞書と同じ）
TORQUE_LINKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "wrist_R": lambda p: p[4] - p[2],
    "elbow_R": lambda p: p[2] - p[0],
    "shoulder_R": lambda p: -(p[1] - p[0]),
    "wrist_L": lambda p: p[5] - p[3],
    "elbow_L": lambda p: p[3] - p[1],
    "shoulder_L": lambda p: (p[1] - p[0]),
}

PART_KEYS = ("wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L")


@dataclass
class MeasurementConfig:
    """計測のパラメータ。既定値は config.py に揃えてある。"""

    body_mass_kg: float = 60.0
    # 慣性テンソルを確定させるまでに必要なフレーム数（既存と同じ）
    inertia_ready_frames: int = 3
    dynamics_ready_frames: int = 7
    # サイクル検出の基準値を作るのに使うフレーム範囲（既存と同じ 5..14）
    baseline_first_frame: int = 5
    baseline_last_frame: int = 14


@dataclass
class FrameResult:
    """1 フレーム分の計算結果。"""

    t_ns: int
    points_3d: np.ndarray
    local_torques: dict[str, np.ndarray] = field(default_factory=dict)
    cycle_detected: bool = False
    impulses: dict[str, float] = field(default_factory=dict)


class NetworkMeasurement:
    """ペアになったランドマークを順に流し込み、トルクと力積を得る。"""

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
        self.config = config or MeasurementConfig()

        from body_part_storage_module import BodyPartDataStorage
        from link_vector_calculator_module import LinkVectorCalculator

        self.storage = BodyPartDataStorage()
        self.calculators = {
            part: LinkVectorCalculator(start, end) for part, (start, end) in PART_LINKS.items()
        }

        self.points_history: list[np.ndarray] = []
        self.frame_index = 0
        self._prev_t_ns: int | None = None

        # 慣性テンソル。既存と同じく 3 フレーム目で確定させる。
        self._inertia: dict[str, np.ndarray] = {}

        # サイクル検出
        self._baseline_z = 0.0
        self._detector = None
        self.impulse_records: dict[str, list[float]] = {k: [] for k in PART_KEYS}
        self._torque_history: dict[str, list[float]] = {k: [] for k in PART_KEYS}

        self.results: list[FrameResult] = []

    # -- 入口 --------------------------------------------------------------
    def process(self, pair: PairedSample) -> FrameResult | None:
        """1 ペアを処理する。まだ計算できない段階では None を返す。"""
        keypoints0 = self._pixel_keypoints(pair, "cam0")
        keypoints1 = self._pixel_keypoints(pair, "cam1")
        if keypoints0 is None or keypoints1 is None:
            return None

        points = self._triangulate(keypoints0, keypoints1)
        self.points_history.append(points)

        dt = self._timestep(pair.t_ns)
        self._update_links(dt)
        self._update_baseline(points)

        result = FrameResult(t_ns=pair.t_ns, points_3d=points)

        if len(self.points_history) == self.config.inertia_ready_frames:
            self._build_inertia(points)

        if len(self.points_history) >= self.config.dynamics_ready_frames and self._inertia:
            torques = self._compute_local_torques(points)
            if torques is not None:
                result.local_torques = torques
                self._accumulate_cycle(points, torques, dt, result)

        self.frame_index += 1
        self.results.append(result)
        return result

    # -- 各段 --------------------------------------------------------------
    def _pixel_keypoints(self, pair: PairedSample, role: str) -> list[list[float]] | None:
        frame = pair.frames.get(role)
        if frame is None:
            return None
        # 既存 `_extract_keypoints_fast_single` と同じく pose_keypoints の宣言順。
        return [
            list(frame.pixel_xy(index))
            for index in self.pose_keypoints
        ]

    def _triangulate(self, keypoints0, keypoints1) -> np.ndarray:
        """三角測量して既存と同じ座標系に変換する。

        変換 (x, y, z) -> (-x, -z, -y) と 0.01 倍のスケールは
        `_triangulate_transform_batch` に合わせてある。単位は m。
        """
        # pylint: disable=no-member
        import cv2 as cv

        pts0 = np.asarray(keypoints0, dtype=np.float64).T
        pts1 = np.asarray(keypoints1, dtype=np.float64).T
        homogeneous = cv.triangulatePoints(self.P0, self.P1, pts0, pts1)

        w = homogeneous[3, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = (homogeneous[:3, :] / w).T
        raw = np.where(np.isfinite(raw), raw, np.nan) * 0.01

        transformed = np.empty_like(raw)
        transformed[:, 0] = -raw[:, 0]
        transformed[:, 1] = -raw[:, 2]
        transformed[:, 2] = -raw[:, 1]
        return transformed

    def _timestep(self, t_ns: int) -> float:
        """前フレームとの実時間差。

        既存のリアルタイム経路は「PC の処理ループ速度」から dt を逆算していたが、
        こちらは**撮影時刻の差**を使える。無線のジッタがあっても、時刻で
        再標本化した後のグリッド間隔になるので等間隔が保たれる。
        """
        if self._prev_t_ns is None:
            self._prev_t_ns = t_ns
            return 1.0 / 30.0
        dt = (t_ns - self._prev_t_ns) / 1e9
        self._prev_t_ns = t_ns
        return dt if dt > 0 else 1.0 / 30.0

    def _update_links(self, dt: float) -> None:
        index = len(self.points_history) - 1
        for part, calculator in self.calculators.items():
            result = calculator.calculate_link_vectors(self.points_history, True, index, dt)
            if result[0] is None:
                continue
            r_vec, vel, omega, centroid, p1, acc, ang_acc = result
            self.storage.add_data(part, r_vec, vel, omega, centroid, p1, ang_acc, acc)

    def _update_baseline(self, points: np.ndarray) -> None:
        """サイクル検出の基準値（安定座位での z）を作る。既存と同じ 5..14 フレーム。"""
        from utils import PushCycleDetector

        value = float(points[0][2])
        if not math.isfinite(value):
            return  # NaN を混ぜると基準値が汚染され、以後一度も検出されなくなる

        span = self.config.baseline_last_frame - self.config.baseline_first_frame + 1
        if self.config.baseline_first_frame <= self.frame_index <= self.config.baseline_last_frame:
            self._baseline_z += value / span
        elif self.frame_index == self.config.baseline_last_frame + 1 and self._detector is None:
            self._detector = PushCycleDetector(self._baseline_z)

    def _build_inertia(self, points: np.ndarray) -> None:
        """慣性テンソルを確定させる。部位行と引数は既存と同じ。"""
        from utils_dynamic import calculate_inertia_tensor

        mass = self.config.body_mass_kg

        def length(a: int, b: int) -> float:
            return float(np.linalg.norm(points[a] - points[b]))

        half_body = 0.25 * float(
            np.linalg.norm(points[0] + points[1] - points[7] - points[6])
        )

        self._inertia = {
            "upper_arm": calculate_inertia_tensor(3, mass, length(0, 2)),
            "forearm": calculate_inertia_tensor(4, mass, length(2, 4)),
            "upper_body": calculate_inertia_tensor(1, mass, half_body),
            "lower_body": calculate_inertia_tensor(0, mass, half_body),
            "thigh": calculate_inertia_tensor(6, mass, length(9, 7)),
        }

    def _compute_local_torques(self, points: np.ndarray) -> dict[str, np.ndarray] | None:
        from utils import compute_local_torque
        from utils_dynamic import calculate_individual_torques, calculate_M_and_F

        data = {name: self.storage.get_data(name) for name in PART_LINKS}
        if any(not values for values in data.values()):
            return None

        mass = self.config.body_mass_kg
        # 部位質量。config.py の m1(上腕) m2(前腕) m4(太腿) と同じ係数。
        m_upper_arm = mass * 0.0227
        m_forearm = mass * 0.016
        m_thigh = mass * 0.11

        gravity = np.array([0.0, 0.0, -9.81])
        inertia = self._inertia

        def chain(arm: str, leg: str, condition: int):
            specs = [
                (inertia["upper_arm"], m_upper_arm, data[arm], {}),
                (inertia["forearm"], m_forearm, data[f"forearm_{'R' if condition else 'L'}"], {}),
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
            moments, forces, parts = [], [], []
            for tensor, segment_mass, part_data, kwargs in specs:
                M, F, name = calculate_M_and_F(tensor, segment_mass, part_data, gravity, **kwargs)
                moments.append(M)
                forces.append(F)
                parts.append(name)
            return moments, forces, parts

        try:
            Ms_r, Fs_r, parts_r = chain("upper_arm_R", "upper_Leg_R", condition=1)
            Ms_l, Fs_l, parts_l = chain("up_arm_l", "upper_Leg_L", condition=0)
        except (IndexError, KeyError, ValueError):
            return None

        def centroids(arm: str, leg: str) -> list[np.ndarray]:
            shoulder = data["both_shoulder"][-1]["centroid"]
            hip = data["both_hip"][-1]["centroid"]
            return [
                data[arm][-1]["centroid"],
                data[f"forearm_{'R' if arm.endswith('_R') else 'L'}"][-1]["centroid"],
                (shoulder * 3 + hip) / 4,
                (shoulder + hip * 3) / 4,
                data[leg][-1]["centroid"],
            ]

        r_x = data["both_hip"][-1]["centroid"]
        tau_E = np.zeros(3)
        f_E = np.zeros(3)

        torques_r = calculate_individual_torques(
            Ms_r, Fs_r, np.array(centroids("upper_arm_R", "upper_Leg_R")),
            tau_E, f_E, r_x, parts_r, self.storage,
        )
        torques_l = calculate_individual_torques(
            Ms_l, Fs_l, np.array(centroids("up_arm_l", "upper_Leg_L")),
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

        local = {}
        for key, global_torque in global_torques.items():
            link = TORQUE_LINKS[key](points)
            local[key] = compute_local_torque(global_torque, link)
            self.storage.add_torque(key, local[key])
        return local

    def _accumulate_cycle(
        self,
        points: np.ndarray,
        torques: dict[str, np.ndarray],
        dt: float,
        result: FrameResult,
    ) -> None:
        """サイクルを検出し、その区間の力積を積む。既存の計算と同じ。"""
        if self._detector is None:
            return

        z = float(points[0][2])
        if not math.isfinite(z):
            return

        if self._detector.update(z, self.frame_index):
            result.cycle_detected = True
            for key in PART_KEYS:
                series = np.asarray(self._torque_history[key], dtype=np.float64)
                if series.size:
                    positive = float(series[series > 0].sum() * dt)
                    negative = float(series[series < 0].sum() * dt)
                    impulse = max(abs(positive), abs(negative))
                    self.impulse_records[key].append(impulse)
                    result.impulses[key] = impulse
                self._torque_history[key].clear()

        for key in PART_KEYS:
            vector = torques.get(key)
            if vector is not None and math.isfinite(float(vector[2])):
                self._torque_history[key].append(float(vector[2]))

    # -- 参照 --------------------------------------------------------------
    @property
    def latest_impulses(self) -> dict[str, float]:
        return {
            key: (values[-1] if values else 0.0)
            for key, values in self.impulse_records.items()
        }

    @property
    def cycle_count(self) -> int:
        return len(self.impulse_records[PART_KEYS[0]])
