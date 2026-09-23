"""オフラインとスマホ経路が、同じ姿勢から同じ関節トルクを出すことを固定する。

**なぜこのテストがあるか。**

同じ力学モデルを経路ごとに別々に書いていたため、次のように食い違っていた。

- オフラインは重力を奥行き方向に置き、USB・スマホ経路は z を上にしていた（§1-5）
- USB・スマホ経路は部位の並びが 1 つずれ、``wrist_R`` が右肘まわりのトルクだった（§5-7）
- USB・スマホ経路は下胴体に体重を丸ごと渡し、オフラインは体幹＋頭を肩に載せていた（§5-8）
- 手首の局所軸が経路ごとに違った（§5-1）

いまは 3 経路とも ``push_up_model`` を呼ぶ。静止姿勢なら運動学の差分の取り方
（オフラインは中心差分、スマホ・USB は後退差分）にも依らないので、トルクは一致するはず。
USB 経路（``master_research_code.py``）は import できないので、同じ関数を呼んでいることを
AST で確かめる（``TestRealtimeWiring``）。
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import compute_torque_from_pose as ctp
from app.net.sync_buffer import PairedSample
from app.runners.network_measure import MeasurementConfig, NetworkMeasurement
from config import pose_keypoints

MAIN_SCRIPT = Path(__file__).resolve().parents[1] / "master_research_code.py"
SLOTS = sorted(pose_keypoints)
FRAMES = 45

# z が上の座標系（m）。左右で少し違う姿勢にして、左右の取り違えも見えるようにする。
BODY = {
    11: [-0.18, 0.00, 0.55], 12: [0.18, 0.00, 0.55],
    13: [-0.24, 0.03, 0.27], 14: [0.21, -0.02, 0.25],
    15: [-0.20, -0.08, 0.00], 16: [0.19, -0.11, 0.00],
    17: [-0.24, -0.01, -0.01], 18: [0.23, -0.03, -0.01],
    19: [-0.17, 0.00, -0.02], 20: [0.15, -0.02, -0.02],
    23: [-0.12, -0.05, 0.00], 24: [0.12, -0.05, 0.00],
    25: [-0.12, 0.35, 0.00], 26: [0.12, 0.35, 0.00],
    27: [-0.12, 0.35, -0.40], 28: [0.12, 0.35, -0.40],
}
JOINTS = ["wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L"]


class _DirectPoints(NetworkMeasurement):
    """三角測量を飛ばして 3D 点を直接流す。"""

    def __init__(self, frame: np.ndarray):
        projection = np.hstack([np.eye(3), np.zeros((3, 1))])
        super().__init__(projection, projection, pose_keypoints, MeasurementConfig(body_mass_kg=60.0))
        self._frame = frame

    def _pixel_keypoints(self, pair, role):
        return []

    def _triangulate(self, keypoints0, keypoints1):
        return self._frame.copy()


def _phone_local_torques() -> dict[str, np.ndarray]:
    frame = np.array([BODY[pid] for pid in SLOTS], dtype=float)
    measurement = _DirectPoints(frame)
    for k in range(FRAMES):
        measurement.process(PairedSample(t_ns=int(k / 30.0 * 1e9), frames={}))
    return measurement.results[-1].local_torques


def _offline_local_torques(tmp_path) -> dict[str, np.ndarray]:
    data = {"frame": np.arange(FRAMES)}
    for pid, p in sorted(BODY.items()):
        for axis, label in enumerate("xyz"):
            data[f"joint_{pid}_{label}"] = np.full(FRAMES, p[axis])
    csv = tmp_path / "pose.csv"
    pd.DataFrame(data).to_csv(csv, index=False)
    ctp.main(["--pose-csv", str(csv), "--out-dir", str(tmp_path / "out"), "--prefix", "p"])
    row = pd.read_csv(tmp_path / "out" / "p_torque.csv").iloc[-1]
    return {j: np.array([row[f"{j}_local_{a}"] for a in "xyz"]) for j in JOINTS}


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    return _phone_local_torques(), _offline_local_torques(tmp_path_factory.mktemp("offline"))


class TestSameTorqueFromTheSamePose:
    @pytest.mark.parametrize("joint", JOINTS)
    def test_phone_matches_offline(self, both, joint):
        phone, offline = both
        np.testing.assert_allclose(phone[joint], offline[joint], atol=1e-9, err_msg=(
            f"{joint}: 静止姿勢なのにスマホ経路とオフラインでトルクが違う。"
            " 重力・関節の基準点・局所軸・体幹荷重のどれかが経路で食い違っている"))

    def test_the_torso_load_dominates_the_wrist(self, both):
        """体幹荷重（片腕 17.34 kg）が入っていれば、手首トルクは 10 N·m 級になる。"""
        phone, _ = both
        assert 5.0 < np.linalg.norm(phone["wrist_R"]) < 40.0


class TestRealtimeWiring:
    """USB 経路（import できない）が同じ関数を呼んでいることを AST で確かめる。"""

    @pytest.fixture(scope="class")
    def called(self):
        tree = ast.parse(io.open(MAIN_SCRIPT, encoding="utf-8").read())
        return {
            node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            for node in ast.walk(tree) if isinstance(node, ast.Call)
        }

    @pytest.mark.parametrize("name", ["push_up_torques", "arm_axes", "segment_from_storage"])
    def test_uses_the_shared_model(self, called, name):
        assert name in called, f"master_research_code.py が {name} を呼んでいない"

    @pytest.mark.parametrize("name", ["calculate_individual_torques", "run_specs"])
    def test_no_longer_builds_its_own_chain(self, called, name):
        assert name not in called, (
            f"master_research_code.py がまだ {name} で独自の鎖を組んでいる（§5-7・§5-8 の再発）")
