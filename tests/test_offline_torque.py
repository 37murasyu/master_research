"""オフラインのトルク計算（``compute_torque_from_pose.py``）の約束を固定する。

**なぜこのテストがあるか。**

- §1-5 重力を全体座標の −z（奥行き）に置いていた。入力の ``Adjusted 3D Pose/*.csv`` は
  カメラ座標で y が鉛直下向き。被験者 3 の手首トルクは正しい向きで 1/3〜1/10 になる。
  いまは初期フレームの体幹の向きから重力を決める。同じ動きをどの座標系で与えても、
  トルクの大きさと局所成分は変わらないはず
- ``--torque-scale`` の既定が 0.01（N·cm → N·m 用）で、m 単位の入力ではトルクが 1/100 になっていた
- ``--wrist-base``（体幹荷重を載せる鎖、§2-1 で確定したモデル）が既定オフで、付け忘れると
  体幹荷重が入らなかった
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import compute_torque_from_pose as ctp

FPS = 30.0
G = 9.81
M_UPPER_ARM = 60.0 * 0.0227
M_FOREARM = 60.0 * 0.016
LOAD = 60.0 * (0.497 + 0.081) / 2

# z が上の座標系（x: 右、y: 前、z: 上）で組んだ上半身。
BODY_Z_UP = {
    11: [-0.18, 0.0, 0.55], 12: [0.18, 0.0, 0.55],     # 肩
    23: [-0.12, -0.05, 0.0], 24: [0.12, -0.05, 0.0],   # 腰（体幹は少し後ろに傾く）
    13: [-0.18, 0.0, 0.25], 14: [0.18, 0.0, 0.25],     # 肘
    15: [-0.18, -0.10, 0.0], 16: [0.18, -0.10, 0.0],   # 手首（肩より 0.10 m 後ろ）
}
HAND_Z_UP = {
    17: [-0.20, -0.02, -0.01], 18: [0.20, -0.02, -0.01],   # 小指（手は前へ）
    19: [-0.16, -0.02, -0.01], 20: [0.16, -0.02, -0.01],   # 人差し指
}

# z 上向き → カメラ座標（x: 右、y: 下、z: 奥行き）。行列式 +1 の回転。
TO_CAMERA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _frames(body: dict, n: int, motion: bool) -> dict[int, np.ndarray]:
    """関節 ID → (n, 3)。motion なら肩と肘を上下させる（手首は固定）。"""
    t = np.arange(n) / FPS
    lift = 0.04 * np.sin(2 * np.pi * 0.8 * t) if motion else np.zeros(n)
    out = {}
    for jid, p in body.items():
        series = np.tile(np.asarray(p, dtype=float), (n, 1))
        if jid in (11, 12, 13, 14):
            series[:, 2] += lift * (1.0 if jid in (11, 12) else 0.5)
        out[jid] = series
    return out


def _write_csv(path, joints: dict[int, np.ndarray], rotation=None, offset=None):
    n = len(next(iter(joints.values())))
    data = {"frame": np.arange(n)}
    for jid, series in sorted(joints.items()):
        pts = series if rotation is None else series @ rotation.T
        if offset is not None:
            pts = pts + offset
        for axis, label in enumerate("xyz"):
            data[f"joint_{jid}_{label}"] = pts[:, axis]
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _run(tmp_path, csv_path, *extra):
    out = tmp_path / "out"
    ctp.main(["--pose-csv", str(csv_path), "--out-dir", str(out), "--prefix", "p", *extra])
    torque = pd.read_csv(out / "p_torque.csv")
    meta = json.loads((out / "p_meta.json").read_text(encoding="utf-8"))
    return torque, meta


def _norms(df: pd.DataFrame, joint: str) -> np.ndarray:
    return np.linalg.norm(df[[f"{joint}_{a}" for a in "xyz"]].to_numpy(), axis=1)


class TestGravityFromTheTrunk:
    def test_camera_coordinates_put_gravity_along_plus_y(self, tmp_path):
        csv = _write_csv(tmp_path / "cam.csv", _frames(BODY_Z_UP, 60, False), TO_CAMERA, np.array([0, 0, 3.5]))
        _, meta = _run(tmp_path, csv)
        np.testing.assert_allclose(meta["gravity"]["vector"], [0.0, G, 0.0], atol=1e-9)

    @pytest.mark.parametrize("joint", ["wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L"])
    def test_torque_does_not_depend_on_the_coordinate_system(self, tmp_path, joint):
        joints = _frames(BODY_Z_UP, 90, True)
        z_up, _ = _run(tmp_path / "a", _write_csv(_mk(tmp_path / "a") / "zup.csv", joints))
        cam, _ = _run(tmp_path / "b", _write_csv(_mk(tmp_path / "b") / "cam.csv", joints, TO_CAMERA,
                                                  np.array([0, 0, 3.5])))
        np.testing.assert_allclose(_norms(cam, joint), _norms(z_up, joint), rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(cam[f"{joint}_local_y"], z_up[f"{joint}_local_y"], rtol=1e-6, atol=1e-9)


def _mk(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestDefaults:
    def test_torque_is_in_newton_metres_and_carries_the_torso(self, tmp_path):
        """静止姿勢の右手首 = 各部位と体幹荷重の重さ × 手首からの水平距離。"""
        csv = _write_csv(tmp_path / "s.csv", _frames(BODY_Z_UP, 60, False))
        torque, _ = _run(tmp_path, csv)
        # 肩と肘は手首から水平に 0.10 m 前なので、上腕の重心と体幹荷重も 0.10 m。
        # 前腕の重心は肘から手首へ 0.430 の位置 → 手首から 0.10 × 0.57 = 0.057 m
        expected = (0.10 * M_UPPER_ARM + 0.057 * M_FOREARM + 0.10 * LOAD) * G
        assert np.median(_norms(torque, "wrist_R")) == pytest.approx(expected, rel=1e-6)

    def test_meta_records_the_model(self, tmp_path):
        csv = _write_csv(tmp_path / "s.csv", _frames(BODY_Z_UP, 60, False))
        _, meta = _run(tmp_path, csv)
        assert meta["schema_version"] == 2
        assert meta["model"] == "wrist_base"
        assert meta["torso_load_mass_per_arm"] == pytest.approx(LOAD)
        assert meta["gravity"]["mode"] == "axis"

    def test_free_swing_chain_is_still_available(self, tmp_path):
        csv = _write_csv(tmp_path / "s.csv", _frames(BODY_Z_UP, 60, False))
        torque, meta = _run(tmp_path, csv, "--no-wrist-base")
        assert meta["model"] == "free_swing"
        assert np.all(_norms(torque, "wrist_R") == 0.0)


class TestIgnoredOptions:
    def test_a_dumbbell_with_the_push_up_model_is_warned(self, tmp_path):
        """ダンベルは腕を肩から吊る鎖（--no-wrist-base）でしか使わない。黙って捨てない。"""
        csv = _write_csv(tmp_path / "s.csv", _frames(BODY_Z_UP, 40, False))
        with pytest.warns(UserWarning, match="dumbbell"):
            _run(tmp_path, csv, "--dumbbell-mass-right", "5")


class TestElbowRecalculation:
    """``recalc_elbow_local_torque.py`` は肘の局所列を同じ軸で作り直すだけで、値を変えない。

    パイプライン（KNOWN_ISSUES §6-1）はトルク CSV の後にこれを通す。軸の作り方が違うと、
    肘の局所列だけ別の座標系になる（y は一致していたが z と x が前腕／上腕で入れ替わっていた）。
    """

    def test_recalculating_keeps_the_elbow_columns(self, tmp_path):
        from recalc_elbow_local_torque import recalc_pair

        # 肘を横に張り出し、腕を矢状面から外す。面内だとトルクが面の法線だけになり、
        # z と x の取り方の違いが値に出ない
        body = {**BODY_Z_UP, 13: [-0.26, 0.02, 0.25], 14: [0.26, 0.02, 0.25]}
        pose_csv = _write_csv(tmp_path / "m.csv", _frames(body, 90, True))
        torque, _ = _run(tmp_path, pose_csv)
        recalc_pair(pose_csv, tmp_path / "out" / "p_torque.csv", tmp_path / "recalc")
        again = pd.read_csv(tmp_path / "recalc" / "p_torque.csv")
        # トルク計算は平滑化後の姿勢で軸を作り、再計算は CSV の姿勢で作るので、1e-5 N·m 程度は違う。
        # 軸の取り方が違うと 10 N·m 級の差になる。
        for side in ("R", "L"):
            cols = [f"elbow_{side}_local_{a}" for a in "xyz"]
            np.testing.assert_allclose(again[cols].to_numpy(), torque[cols].to_numpy(), atol=1e-3)


class TestWristAxis:
    def test_hand_points_are_used_when_present(self, tmp_path):
        joints = _frames({**BODY_Z_UP, **HAND_Z_UP}, 60, False)
        _, meta = _run(tmp_path, _write_csv(tmp_path / "h.csv", joints))
        assert meta["wrist_axis"]["R"] == {"hand": 60, "elbow_plane": 0}

    def test_without_hand_points_the_elbow_plane_is_used(self, tmp_path):
        _, meta = _run(tmp_path, _write_csv(tmp_path / "n.csv", _frames(BODY_Z_UP, 60, False)))
        assert meta["wrist_axis"]["R"] == {"hand": 0, "elbow_plane": 60}
