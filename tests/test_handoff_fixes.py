"""引き継ぎで見つかった、解析ユーティリティの停止と部位名の誤記を防ぐ。"""

import json

import numpy as np
import pandas as pd
import pytest

import auto_detect_cycles_pose_ranges as cycles
import config
import estimate_joint_torque_from_pose as estimate


@pytest.mark.parametrize("start,end", [(100, 200), (-20, -1), (100, None)])
def test_range_without_frames_writes_unassigned_cycles(tmp_path, start, end):
    """範囲が収録区間から外れても、戻り値の個数不足でバッチ全体を止めないため。"""
    path = tmp_path / "pose.csv"
    original = pd.DataFrame({"frame": np.arange(8), "joint_12_y": np.linspace(0, 1, 8)})
    original.to_csv(path, index=False)
    cycles.process_file(path, start, end, None, None, False)
    result = pd.read_csv(tmp_path / "pose_with_cycles.csv")
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert (result.cycle_index == -1).all()
    assert not (tmp_path / "pose_cycles.png").exists()


@pytest.mark.parametrize("keep_local", [False, True])
@pytest.mark.parametrize("support", [False, True])
def test_legacy_estimator_writes_torque_and_metadata(tmp_path, keep_local, support):
    """旧 CLI の出力 API のずれを検出し、左右と関節の列対応も保証するため。"""
    points = {11: [-.2, .55, .12], 12: [.2, .55, .12],
              13: [-.2, .25, 0], 14: [.2, .25, 0],
              15: [-.2, 0, -.15], 16: [.2, 0, -.1]}
    data = {"frame": np.arange(10, 22)}
    for jid, point in points.items():
        for axis, value in zip("xyz", point):
            data[f"joint_{jid}_{axis}"] = np.full(12, value)
    path, out = tmp_path / "pose.csv", tmp_path / "torque.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    args = ["--pose-csv", str(path), "--out", str(out), "--skip-smoothing", "--include-gravity"]
    if keep_local:
        args.append("--keep-local")
    if support:
        args.append("--support-body-weight")
    assert estimate.main(args) == 0
    result = pd.read_csv(out)
    expected = {"frame", *estimate.global_columns()}
    if keep_local:
        expected.update(estimate.local_columns())
    assert set(result) == expected
    np.testing.assert_array_equal(result.frame, data["frame"])
    assert np.isfinite(result.to_numpy()).all()
    # 追加支持なしでは、肘トルクは前腕自重を支える逆向きのモーメント。
    if not support:
        for side, forearm_z in (("R", .1), ("L", .15)):
            expected_elbow = forearm_z * .430 * (60 * .016) * 9.81
            expected_shoulder = (.12 * .436 * (60 * .0227)
                                 + (.12 + forearm_z * .430) * (60 * .016)) * 9.81
            np.testing.assert_allclose(result[f"elbow_{side}_x"], expected_elbow)
            np.testing.assert_allclose(result[f"shoulder_{side}_x"], expected_shoulder)
    meta = json.loads(out.with_name("torque_meta.json").read_text())
    assert meta["frames"] == 12
    assert meta["gravity"] == [0.0, -9.81, 0.0]
    assert meta["shapes"]["tau_global_right"] == [12, 2, 3]


def test_effective_mass_names_the_torso_without_changing_values():
    """体幹係数を上肢と誤解して補正しないよう、名前と既存の分母の値を固定するため。"""
    assert config.EFFECTIVE_MASS_COEFFS["torso"] == pytest.approx(.276 + .19)
    assert "upper_limb" not in config.EFFECTIVE_MASS_COEFFS
    assert config.EFFECTIVE_MASS_BY_JOINT["wrist"] == pytest.approx(.615)
    assert config.EFFECTIVE_MASS_BY_JOINT["elbow"] == pytest.approx(.589)
