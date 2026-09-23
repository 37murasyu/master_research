"""ノイズ寄与の集計（``compute_cycle_noise_contrib.py``）がスコアと同じ仕事を分解することを固定する。

**なぜこのテストがあるか。**

このスクリプトはサイクル仕事を「信号 × 信号」「交差項」「ノイズ × ノイズ」に分ける。
分解する前の仕事はスコア（``compute_cycle_energy_elbow_wrist.py``）と同じでなければ意味が無い。
ところが §1-1・§1-2 と §5-2 をスコア側だけ直しており、こちらには残っていた（KNOWN_ISSUES §1-6）。

- 角速度に fps を掛けていた（30 倍）
- 角度（``arctan2``）を経由して微分し、トルクと別の軸の角速度を掛けていた
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compute_cycle_energy_elbow_wrist import RIGHT, _joint_powers
from compute_cycle_noise_contrib import side_noise_contrib

DT = 1.0 / 30.0
UPPER_ARM = 0.30
FOREARM = 0.25


def _frames(n: int, omega: float):
    """手首と肘を固定し、上腕だけが y 軸まわりに ω で回る（肘の相対角速度 = (0, ω, 0)）。"""
    theta = 0.3 + omega * DT * np.arange(n)
    wrist = np.zeros((n, 3))
    elbow = np.tile([0.0, 0.0, FOREARM], (n, 1))
    shoulder = elbow + UPPER_ARM * np.stack([np.sin(theta), np.zeros(n), np.cos(theta)], axis=1)
    pose = {}
    for name, series in (("wrist", wrist), ("elbow", elbow), ("shoulder", shoulder)):
        for axis, label in enumerate("xyz"):
            pose[f"joint_{RIGHT[name]}_{label}"] = series[:, axis]
    torque = {}
    for part, value in (("elbow", [0.0, 2.0, 0.0]), ("wrist", [0.0, 0.0, 0.0])):
        for axis, label in enumerate("xyz"):
            torque[f"{part}_R_{label}"] = np.full(n, value[axis])
    return pd.DataFrame(pose), pd.DataFrame(torque)


class TestSameWorkAsTheScore:
    def test_signed_work_equals_the_score_work(self):
        pose, torque = _frames(60, omega=1.5)
        cycles = np.ones(60, dtype=int)
        result = side_noise_contrib(pose, torque, "R", DT, fps=30.0, fc=3.0, cycle_idx=cycles)
        elbow_power, _ = _joint_powers(pose, torque, "R", DT)
        work = float(result["elbow_R"]["work_J_signed"].iloc[0])
        assert work == pytest.approx(float(np.sum(elbow_power) * DT), rel=1e-9)

    def test_angular_velocity_is_in_radians_per_second(self):
        # τ = 2 N·m、ω = 1.5 rad/s を 2 秒 → 6 J。fps を掛けると 180 J になる
        pose, torque = _frames(60, omega=1.5)
        result = side_noise_contrib(pose, torque, "R", DT, fps=30.0, fc=3.0, cycle_idx=np.ones(60, dtype=int))
        work = float(result["elbow_R"]["work_J_signed"].iloc[0])
        assert work == pytest.approx(6.0, rel=0.02), f"仕事 {work:.2f} J が 6 J から外れた（180 J なら fps の掛け戻し）"

    def test_components_add_up_to_the_total(self):
        pose, torque = _frames(60, omega=1.5)
        row = side_noise_contrib(pose, torque, "R", DT, fps=30.0, fc=3.0,
                                 cycle_idx=np.ones(60, dtype=int))["elbow_R"].iloc[0]
        assert row["work_sig"] + row["work_cross"] + row["work_noi"] == pytest.approx(row["work_J_signed"])


@pytest.mark.parametrize("excluded", [False, True])
def test_subject_four_is_only_excluded_when_requested(tmp_path, monkeypatch, excluded):
    """論文のノイズ評価は被験者4を含むため、コードに固定した除外で落とさない。"""
    import sys
    import compute_cycle_noise_contrib as noise

    pose_dir, torque_dir, out = (tmp_path / name for name in ("pose", "torque", "out"))
    pose_dir.mkdir()
    torque_dir.mkdir()
    pose, torque = _frames(60, omega=1.5)
    pose["frame"] = torque["frame"] = np.arange(60)
    pose["cycle_index"] = 1
    pose.to_csv(pose_dir / "4_0stereo_pose_lpf_with_cycles.csv", index=False)
    torque.to_csv(torque_dir / "4_0stereo_pose_torque_lpf.csv", index=False)
    args = ["noise", "--pose-dir", str(pose_dir), "--torque-dir", str(torque_dir),
            "--out-dir", str(out), "--pose-unit", "m"]
    if excluded:
        args += ["--exclude-subjects", "4"]
    monkeypatch.setattr(sys, "argv", args)
    assert noise.main() == 0
    files = sorted(out.glob("*.csv"))
    assert len(files) == (0 if excluded else 2)
    if not excluded:
        assert all(pd.read_csv(path).subject_id.eq(4).all() for path in files)
