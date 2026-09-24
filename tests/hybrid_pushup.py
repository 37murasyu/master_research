"""テストの部品: 手を固定して体幹が持ち上がる座位プッシュアップの合成データ。

**なぜこの部品があるか。**

既存の合成動作（``test_network_measure._body_points``）は上肢を左右と奥行きに振るだけで、体幹は上下しない。
混成の経路の回の区切り（``app.hybrid.rep_detector``）と力学の関所は「肩の中点の重力の上向きへの射影（高さ）」で
押し上げを見るので、上下に動かない合成では 1 回も回が閉じない。実際の押し上げの形（手はアームレストに固定、
肘が伸びて体幹が 13 cm 上がる）で、回の数・関所・仕事の大きさを確かめる。

座標はカメラ座標（x 右、y 下、z 前方、**cm**）で作り、``runtime_m`` で実行時の座標（(−x, −z, −y) × 0.01 m、
z が上）に直す。点の並びはランドマーク ID の昇順（``config.pose_keypoints`` を並べ替えたもの）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import pose_keypoints

IDS = sorted(pose_keypoints)
SLOT = {lid: i for i, lid in enumerate(IDS)}

UPPER_ARM_CM = 28.0
FOREARM_CM = 25.0
DEPTH_CM = 250.0


@dataclass(frozen=True)
class PushUp:
    """押し上げの時間の形。静止 ``rest_s`` の後、``reps`` 回を ``period_s`` ごとに繰り返す。

    1 回は 上げ ``rise_s`` → 上で保持 ``hold_s`` → 下げ ``lower_s`` → 残りは座って休む。
    上げ下げは余弦で滑らかにつなぐ（速さが 0 から始まって 0 で終わる）。
    """

    reps: int = 3
    rest_s: float = 2.0
    period_s: float = 3.0
    rise_s: float = 1.0
    hold_s: float = 0.4
    lower_s: float = 1.0
    lift_cm: float = 13.0

    @property
    def duration_s(self) -> float:
        return self.rest_s + self.reps * self.period_s + 1.0

    def lift(self, t: float) -> float:
        """体幹の持ち上がり [cm]（0 が座った姿勢）。"""
        s = t - self.rest_s
        if s < 0 or s >= self.reps * self.period_s:
            return 0.0
        s = s % self.period_s
        if s < self.rise_s:
            return self.lift_cm * 0.5 * (1 - np.cos(np.pi * s / self.rise_s))
        s -= self.rise_s
        if s < self.hold_s:
            return self.lift_cm
        s -= self.hold_s
        if s < self.lower_s:
            return self.lift_cm * 0.5 * (1 + np.cos(np.pi * s / self.lower_s))
        return 0.0


def _elbow(shoulder: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """肩と手首から肘の位置（肘は後ろ＝カメラから遠い側へ曲がる）。"""
    axis = shoulder - wrist
    d = float(np.linalg.norm(axis))
    u = axis / d
    # 手首から肘までの、肩–手首の線に沿った長さと、線から離れる距離（余弦定理）
    along = (FOREARM_CM ** 2 - UPPER_ARM_CM ** 2 + d ** 2) / (2 * d)
    off = np.sqrt(max(FOREARM_CM ** 2 - along ** 2, 0.0))
    back = np.array([0.0, 0.0, 1.0])
    back = back - np.dot(back, u) * u
    back /= np.linalg.norm(back)
    return wrist + along * u + off * back


def body_cm(lift_cm: float, *, arm_offset_cm: float = 0.0) -> np.ndarray:
    """持ち上がり ``lift_cm`` のときの 16 点（カメラ座標、cm）。

    ``arm_offset_cm`` は手首と手を上下に平行移動する量。``lift_cm`` と同じにすると肩と手が一緒に上がり、
    腕の形（肘角）を保ったまま全体が動く（関節の仕事が 0 になる検算に使う）。
    """
    rise = -float(lift_cm)   # カメラの y は下向き
    shift = np.array([0.0, -float(arm_offset_cm), 0.0])
    by_id: dict[int, np.ndarray] = {}
    for side, sx, (sh, el, wr, pinky, index, thumb, hip, knee, ankle) in (
        ("L", -1.0, (11, 13, 15, 17, 19, 21, 23, 25, 27)),
        ("R", 1.0, (12, 14, 16, 18, 20, 22, 24, 26, 28)),
    ):
        wrist = np.array([sx * 21.0, 18.0, DEPTH_CM]) + shift
        # 肩は手首の真上の少し内側。座った姿勢で肩–手首は約 38 cm（肘の内角 約 91°、13 cm 上がると約 149°）
        shoulder = np.array([sx * 18.0, 18.0 - 38.0 + rise, DEPTH_CM])
        by_id[sh] = shoulder
        by_id[wr] = wrist
        by_id[el] = _elbow(shoulder, wrist)
        # 手はアームレストの上で前（カメラ側）を向く。前腕とほぼ直角（手首の軸を手のひらから作れる）
        by_id[pinky] = wrist + np.array([sx * 3.0, 1.0, -8.0])
        by_id[index] = wrist + np.array([sx * -1.0, 1.0, -9.0])
        by_id[thumb] = wrist + np.array([sx * -3.0, 0.0, -5.0])
        by_id[hip] = np.array([sx * 12.0, 22.0 + rise, DEPTH_CM + 5.0])
        by_id[knee] = np.array([sx * 12.0, 30.0, DEPTH_CM - 38.0])
        by_id[ankle] = np.array([sx * 12.0, 70.0, DEPTH_CM - 40.0])
    return np.array([by_id[lid] for lid in IDS], dtype=np.float64)


def runtime_m(points_cm: np.ndarray) -> np.ndarray:
    """カメラ座標 [cm] → 実行時の座標 (−x, −z, −y) [m]。三角測量（``compute_triangulate_transform_native``）と同じ。"""
    p = np.asarray(points_cm, dtype=np.float64)
    return np.stack([-p[..., 0], -p[..., 2], -p[..., 1]], axis=-1) * 0.01


def pushup_cm(t: float, motion: PushUp = PushUp(), *, arms_rigid: bool = False) -> np.ndarray:
    """時刻 t の 16 点（カメラ座標、cm）。``arms_rigid`` なら肘角を保ったまま腕ごと上下させる（関節の仕事は 0）。"""
    lift = motion.lift(t)
    return body_cm(lift, arm_offset_cm=lift if arms_rigid else 0.0)


def elbow_angle_deg(points_cm: np.ndarray, side: str = "R") -> float:
    sh, el, wr = ((12, 14, 16) if side == "R" else (11, 13, 15))
    a = points_cm[SLOT[sh]] - points_cm[SLOT[el]]
    b = points_cm[SLOT[wr]] - points_cm[SLOT[el]]
    return float(np.degrees(np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))))


def pushup_pairs(measurement, motion: PushUp = PushUp(), *, fps: float = 30.0, drop=(), arms_rigid=False,
                 noise_px: float = 0.0, seed: int = 0, seconds: float | None = None):
    """``measurement`` の射影行列で投影した組を時刻順に返す。``drop`` は組を作らない格子の番号の集まり。"""
    from test_network_measure import _pair_from_pixels, _project

    rng = np.random.default_rng(seed)
    total = motion.duration_s if seconds is None else seconds
    pairs = []
    for k in range(int(round(total * fps))):
        if k in drop:
            continue
        truth = pushup_cm(k / fps, motion, arms_rigid=arms_rigid)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)
        if noise_px:
            p0 = p0 + rng.normal(0.0, noise_px, p0.shape)
            p1 = p1 + rng.normal(0.0, noise_px, p1.shape)
        pairs.append(_pair_from_pixels(round(k * 1e9 / fps), p0, p1))
    return pairs


def run(measurement, motion: PushUp = PushUp(), **kw):
    """組を順に流し込み、結果の一覧を返す（``measurement.results`` の上限に左右されない）。"""
    results = []
    for pair in pushup_pairs(measurement, motion, **kw):
        result = measurement.process(pair)
        if result is not None:
            results.append(result)
    return results


def calibrated_pairs(motion: PushUp = PushUp(), *, drop=(), fps: float = 30.0, hide=(),
                     hide_until_s: float = float("inf")):
    """``test_hybrid_measure.calibration`` の校正（歪みつき、基線 35 cm）で撮った押し上げの組。

    ``hide`` のランドマーク ID は、``hide_until_s`` 秒まで Mac の画面のずっと外に置く（歪み補正で NaN になる）。
    """
    import cv2 as cv
    from test_hybrid_measure import geometry
    from test_network_measure import _pair_from_pixels

    intr, stereo = geometry()
    pairs = []
    for k in range(int(round(motion.duration_s * fps))):
        if k in drop:
            continue
        truth = pushup_cm(k / fps, motion)
        truth[:, 1] -= 5.0   # 両方の画像に収める
        a = cv.projectPoints(truth, np.zeros(3), np.zeros(3), intr.K, intr.distortion)[0].reshape(-1, 2)
        b = cv.projectPoints(truth, np.zeros(3), stereo.T, intr.K, intr.distortion)[0].reshape(-1, 2)
        if k / fps < hide_until_s:
            for landmark_id in hide:
                a[SLOT[landmark_id]] = (-5000.0, -5000.0)
        pairs.append(_pair_from_pixels(round(k * 1e9 / fps), a, b))
    return pairs
