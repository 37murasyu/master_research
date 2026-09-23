"""姿勢 CSV の低域通過フィルタで、カットオフを試技ごとに f0 から決めることを固定する（§6-1）。

**なぜこのテストがあるか。**

論文の `*_lpf.csv` は `tmp_filter_pose_torque.py` が 2 Hz 固定で作っていた。一方、論文の表 5 と
リアルタイム経路（`master_research_code.py` の E_FC_*）は「fc = 6 × f0（押し上げの基本周波数）を
2.1〜6.0 Hz に制限」で、被験者 8 は 5.63 Hz だった。2026-09-23 にオフラインもこれに揃えると決めた。

f0 はリアルタイム経路の `OnlineF0Estimator` と同じ計算（左右の肘角の平均 → Welch nperseg 256 →
0.3 Hz 以上のピーク）。30 fps なら周波数の刻みは 0.1172 Hz で、表 5 の 2.11 Hz・5.63 Hz は
その 3 倍・8 倍に 6 を掛けた値になる。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pose_lowpass as pl

FS = 30.0
REPO_ROOT = Path(__file__).resolve().parents[1]


def _arm_pose(n: int, f0: float, noise_hz: float | None = None, seed: int = 0) -> pd.DataFrame:
    """左右の肘が f0 で曲げ伸ばしする姿勢（joint_{id}_{axis} 列、m 単位）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    bend = 0.8 + 0.4 * np.sin(2 * np.pi * f0 * t)
    cols = {"frame": np.arange(n)}
    for side, (sh, el, wr) in {"L": (11, 13, 15), "R": (12, 14, 16)}.items():
        x = -0.18 if side == "L" else 0.18
        shoulder = np.tile([x, 0.0, 0.55], (n, 1))
        elbow = shoulder + np.array([0.0, 0.0, -0.30])
        wrist = elbow + 0.25 * np.stack([np.zeros(n), np.sin(bend), -np.cos(bend)], axis=1)
        if noise_hz is not None:
            wrist[:, 1] += 0.01 * np.sin(2 * np.pi * noise_hz * t)
        for jid, pts in ((sh, shoulder), (el, elbow), (wr, wrist)):
            for axis, label in enumerate("xyz"):
                cols[f"joint_{jid}_{label}"] = pts[:, axis] + rng.normal(0, 1e-4, n)
    return pd.DataFrame(cols)


class TestFundamental:
    @pytest.mark.parametrize("f0", [0.35, 0.47, 0.94])
    def test_the_bending_frequency_is_found(self, f0):
        est, snr = pl.estimate_f0(pl.elbow_angle(_arm_pose(1800, f0)), FS)
        assert est == pytest.approx(f0, abs=FS / 256)
        assert snr > pl.F0_SNR_DB


class TestCutoff:
    @pytest.mark.parametrize("f0, fc", [(3 * FS / 256, 2.109375), (8 * FS / 256, 5.625), (2.0, 6.0), (0.31, 2.1)])
    def test_six_times_f0_within_the_limits(self, f0, fc):
        assert pl.cutoff_hz(f0, snr_db=20.0) == pytest.approx(fc)

    def test_an_unclear_peak_falls_back(self):
        assert pl.cutoff_hz(0.5, snr_db=0.5) == pl.FALLBACK_FC_HZ

    def test_the_constants_match_the_realtime_path(self):
        """リアルタイム経路の E_FC_* の既定値と同じ。片方だけ変えると 2 経路の fc が食い違う。"""
        tree = ast.parse((REPO_ROOT / "master_research_code.py").read_text(encoding="utf-8"))
        defaults = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "getenv" and len(node.args) == 2:
                if isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant):
                    defaults[node.args[0].value] = node.args[1].value
        assert float(defaults["E_FC_K"]) == pl.FC_K
        assert float(defaults["E_FC_MIN"]) == pl.FC_MIN_HZ
        assert float(defaults["E_FC_MAX"]) == pl.FC_MAX_HZ
        assert float(defaults["E_F0_FMIN"]) == pl.F0_MIN_HZ
        assert float(defaults["E_F0_SNR_THRESHOLD"]) == pl.F0_SNR_DB


class TestLowpassFile:
    def test_the_cutoff_is_chosen_per_trial_and_recorded(self, tmp_path):
        src = tmp_path / "7_stereo_pose.csv"
        _arm_pose(1800, 0.47, noise_hz=10.0).to_csv(src, index=False)
        meta = pl.lowpass_pose_csv(src, tmp_path / "out")
        assert meta["f0_hz"] == pytest.approx(0.47, abs=FS / 256)
        assert meta["fc_hz"] == pytest.approx(6 * meta["f0_hz"])
        out = pd.read_csv(tmp_path / "out" / "7_stereo_pose_lpf.csv")
        assert list(out.columns) == list(pd.read_csv(src).columns)
        recorded = json.loads((tmp_path / "out" / "7_stereo_pose_lpf_meta.json").read_text(encoding="utf-8"))
        assert recorded["fc_hz"] == pytest.approx(meta["fc_hz"])

    def test_high_frequency_noise_is_removed(self, tmp_path):
        src = tmp_path / "s.csv"
        pose = _arm_pose(1800, 0.47, noise_hz=10.0)
        pose.to_csv(src, index=False)
        pl.lowpass_pose_csv(src, tmp_path / "out")
        out = pd.read_csv(tmp_path / "out" / "s_lpf.csv")
        clean = _arm_pose(1800, 0.47)
        residual = (out["joint_16_y"] - clean["joint_16_y"]).to_numpy()[100:-100]
        assert np.max(np.abs(residual)) < 0.002, "10 Hz の成分（振幅 1 cm）が残っている"

    def test_a_fixed_cutoff_can_be_given(self, tmp_path):
        src = tmp_path / "s.csv"
        _arm_pose(600, 0.47).to_csv(src, index=False)
        assert pl.lowpass_pose_csv(src, tmp_path / "out", fc=2.0)["fc_hz"] == 2.0
