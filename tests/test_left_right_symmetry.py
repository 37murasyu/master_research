"""逆動力学が左右対称に計算されることを固定する。

**なぜこのテストがあるか。**

左右の腕で仕事量が大きく食い違う（計画メモ E-1、``KNOWN_ISSUES.md`` §5-3）ので、
「コードが左右差を作っているのか、データ（姿勢）に左右差があるのか」を切り分けた。
方法は鏡映テスト: 骨格を左右反転した動きを流し、元の右と反転後の左が一致するかを見る。
コードが対称なら、データに依らず必ず一致する。

被験者 3（``Adjusted 3D Pose/3_0stereo_pose_scaled_with2d.csv``）での実測、
スマホ経路 ``NetworkMeasurement`` のトルクの鏡映誤差（相対、中央値）:

| 外したもの | 手首・肘 | 肩 |
|---|---|---|
| なし（修正前） | 1.3e-2 | 1.7e+0 |
| E-1b | 3e-3 | 1.7e+0 |
| E-1b・E-1d | 0 | 1.7e+0 |
| E-1b・E-1d・E-1f | 0 | 0 |

- E-1b ``calculate_M_and_F`` が右肩（``condition=1``）でだけ上胴体の ω・ω̇ を反転していた。
  上胴体は左右で共通の剛体なので、片側だけ反転する理由が無い
- E-1d 慣性テンソルのリンク長を左腕（と右脚）だけから取り、左右共通で使っていた
- E-1f 両チェーンとも、上胴体の関節位置に ``both_shoulder`` の始点（左肩）を使っていた。
  右肩のトルクが左肩まわりで計算されていた

3 つを外しても被験者 3 の実データの左右差はほぼ変わらない（手首 |τ| 中央値 209 vs 304 N·m）。
残る左右差はデータ由来。

計画メモの E-1a「``_local_y`` は鏡映で符号が反転するのが正常」は誤りだった。トルクも、
親リンクとの外積で作る y 軸も擬ベクトルなので、内積 ``τ·y`` は鏡映で**変わらない**。
左右で同じ符号条件で積算している現行の集計は正しい。下の ``TestFlexionComponentSign`` が
その性質を固定する（片側だけ軸を反転する「修正」を入れると落ちる）。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.net.sync_buffer import PairedSample
from app.runners.network_measure import MeasurementConfig, NetworkMeasurement
from config import pose_keypoints
from utils_dynamic import calculate_M_and_F

FPS = 30.0
FRAMES = 90

SLOTS = sorted(pose_keypoints)
MIRROR_ID = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 17: 18, 18: 17,
             19: 20, 20: 19, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27}


def _arm(shoulder, upper, forearm, flex, elbow_flex, abduction):
    """肩から肘・手首を置く。flex=0 で腕が真下、正で前（-y）へ上がる。"""
    elbow = shoulder + upper * np.array([abduction, -np.sin(flex), -np.cos(flex)])
    angle = flex + elbow_flex
    wrist = elbow + forearm * np.array([abduction, -np.sin(angle), -np.cos(angle)])
    return elbow, wrist


def _asymmetric_body(t: float) -> np.ndarray:
    """左右で長さも動きも違う上半身。リアルタイム経路の座標系（m、z が上）。

    非対称を 3 つとも表に出すための構成:
    - 上胴体が z 軸まわりに揺れる（角加速度がある）→ E-1b が効く
    - 腕と大腿の長さが左右で違う → E-1d が効く
    - 肩トルクを出す → E-1f が効く
    """
    phase = 2 * np.pi * 1.1 * t
    by_id = {
        11: np.array([-0.18, 0.0, 0.50]), 12: np.array([0.18, 0.0, 0.50]),
        23: np.array([-0.12, 0.0, 0.00]), 24: np.array([0.12, 0.0, 0.00]),
    }
    by_id[14], by_id[16] = _arm(by_id[12], 0.30, 0.26,
                                0.30 + 0.20 * np.sin(phase), 0.80 + 0.30 * np.sin(phase + 0.5), 0.10)
    by_id[13], by_id[15] = _arm(by_id[11], 0.27, 0.23,
                                0.25 + 0.15 * np.sin(phase + 0.3), 0.70 + 0.25 * np.sin(phase + 0.9), -0.10)
    by_id[26] = by_id[24] + np.array([0.0, -0.42, 0.0])
    by_id[25] = by_id[23] + np.array([0.0, -0.40, 0.0])
    by_id[28] = by_id[26] + np.array([0.0, 0.0, -0.40])
    by_id[27] = by_id[25] + np.array([0.0, 0.0, -0.40])
    for wrist, pinky, index, side in ((16, 18, 20, 1.0), (15, 17, 19, -1.0)):
        by_id[pinky] = by_id[wrist] + np.array([side * 0.03, -0.08, 0.0])
        by_id[index] = by_id[wrist] + np.array([-side * 0.01, -0.08, 0.02])

    yaw = 0.3 * np.sin(2 * np.pi * 0.7 * t)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([0.0, 0.05 * np.sin(phase), 1.0])
    return np.array([rotation @ by_id[pid] + shift for pid in SLOTS])


def _mirrored(points: np.ndarray) -> np.ndarray:
    """x を反転し、左右のランドマークを入れ替える（z を含む面での鏡映なので重力は変わらない）。"""
    flipped = points * np.array([-1.0, 1.0, 1.0])
    slot_of_id = {pid: slot for slot, pid in enumerate(SLOTS)}
    return np.array([flipped[slot_of_id[MIRROR_ID.get(pid, pid)]] for pid in SLOTS])


class _DirectPoints(NetworkMeasurement):
    """三角測量を飛ばして 3D 点を直接流す。

    確かめたいのは逆動力学の対称性だけ。三角測量の丸め（~1e-9 m）は加速度で dt² 分
    増幅されて ~1e-6 になり、ここで検出したい対称性の破れ（小さいもので 1e-3）との
    区別を鈍らせる。
    """

    def __init__(self, frames):
        projection = np.hstack([np.eye(3), np.zeros((3, 1))])
        super().__init__(projection, projection, pose_keypoints, MeasurementConfig(body_mass_kg=60.0))
        self._frames = iter(frames)

    def _pixel_keypoints(self, pair, role):
        return []

    def _triangulate(self, keypoints0, keypoints1):
        return next(self._frames)


def _local_torques(mirror: bool) -> list[dict[str, np.ndarray]]:
    frames = [_asymmetric_body(k / FPS) for k in range(FRAMES)]
    if mirror:
        frames = [_mirrored(p) for p in frames]
    measurement = _DirectPoints(frames)
    for k in range(FRAMES):
        measurement.process(PairedSample(t_ns=int(k / FPS * 1e9), frames={}))
    return [r.local_torques for r in measurement.results if r.local_torques]


@pytest.fixture(scope="module")
def torques():
    original, mirrored = _local_torques(False), _local_torques(True)
    assert len(original) == len(mirrored) > 30, "トルクが出たフレームが少なすぎて比較にならない"
    return original, mirrored


class TestUpperTorsoMoment:
    """E-1b 上胴体の回転の慣性力は、どちらの肩として計算しても同じ。"""

    def test_moment_does_not_depend_on_which_shoulder(self):
        # I = diag(2, 3, 4), ω = (1, 1, 0), ω̇ = (0, 1, 0) のとき
        #   I·ω̇ = (0, 3, 0)、ω × (I·ω) = (1, 1, 0) × (2, 3, 0) = (0, 0, 1)
        # なので M = (0, 3, 1)。ω と ω̇ を反転すると (0, -3, 1) になる。
        inertia = np.diag([2.0, 3.0, 4.0])
        part = [{"omega": np.array([1.0, 1.0, 0.0]), "dot_omega": np.array([0.0, 1.0, 0.0]),
                 "dot_dot_pg": np.zeros(3), "part_name": "both_shoulder"}]
        hip = [{"omega": np.zeros(3), "dot_omega": np.zeros(3),
                "dot_dot_pg": np.zeros(3), "part_name": "both_hip"}]
        points = np.zeros((6, 3))
        points[0], points[1] = [-0.18, 0.0, 0.5], [0.18, 0.0, 0.5]    # 左肩・右肩
        points[4], points[5] = [-0.25, -0.3, 0.1], [0.20, -0.3, 0.1]  # 左手首・右手首
        g = np.array([0.0, 0.0, -9.81])

        for condition, name in ((1, "右肩"), (0, "左肩")):
            moment, _, _ = calculate_M_and_F(
                inertia, 60.0, part, g, add_part_data=hip, condition=condition, Imode=3, Info_I3=points)
            np.testing.assert_allclose(moment, [0.0, 3.0, 1.0], atol=1e-12, err_msg=(
                f"{name}として計算した上胴体のモーメントが M = I·ω̇ + ω×(I·ω) = (0, 3, 1) と違う。"
                " 片側だけ ω・ω̇ を反転していないか確認すること（E-1b）"))


class TestMirroredMotion:
    """左右反転した動きなら、各関節のトルクの大きさは左右で入れ替わるだけ。"""

    @pytest.mark.parametrize("joint", ["wrist", "elbow", "shoulder"])
    def test_torque_magnitude_swaps_sides(self, torques, joint):
        original, mirrored = torques
        right = np.array([np.linalg.norm(frame[f"{joint}_R"]) for frame in original])
        left_of_mirror = np.array([np.linalg.norm(frame[f"{joint}_L"]) for frame in mirrored])
        np.testing.assert_allclose(left_of_mirror, right, rtol=1e-9, atol=1e-12, err_msg=(
            f"{joint}: 元の右と、左右反転した動きの左で |τ| が一致しない。"
            " コードが左右非対称に計算している（E-1b 片側だけの反転、E-1d 片腕の長さの流用、"
            "E-1f 片側の肩を関節位置に使う、のいずれか）"))


class TestFlexionComponentSign:
    """親リンクとの外積で軸を作る関節では、局所 y 成分は鏡映で符号を変えない。"""

    @pytest.mark.parametrize("joint", ["wrist", "elbow"])
    def test_local_y_keeps_its_sign(self, torques, joint):
        original, mirrored = torques
        right = np.array([frame[f"{joint}_R"][1] for frame in original])
        left_of_mirror = np.array([frame[f"{joint}_L"][1] for frame in mirrored])
        np.testing.assert_allclose(left_of_mirror, right, rtol=1e-9, atol=1e-12, err_msg=(
            f"{joint}: 左右反転した動きの左の local_y が、元の右と一致しない。"
            " τ も y 軸も擬ベクトルなので τ·y は鏡映で不変のはず。"
            "片側だけ局所軸や符号条件を反転していないか確認すること（計画メモ E-1a の導出は誤り）"))
