"""肘のサイクルエネルギーの前処理（energy_pipeline）を本体と同じ計算に固定する。

**なぜこのテストがあるか。**

USB 経路（``master_research_code.py``）の肘の E± は、本体に直書きした ``compute_cycle_energy_filtered``
（LPF → 80 点に再標本化 → τ を分位で切る → dθ を制限 → ∫τdθ の正負）と、適応カットオフ
（``OnlineF0Estimator``・``_fc_scheduler``）で出している。``energy_pipeline.py`` は古い版のまま
（``fc_override`` と有限値の選別が無い）で、混成の経路から同じ値を出せなかった。

混成へ移植するため ``energy_pipeline`` を本体の今の版に入れ替えた。本体の定義がこの先直されたときに
食い違いに気づけるよう、本体を AST で読んで 6 つの定義だけを抜き出し（import するとカメラを開く）、
同じ入力で rtol 1e-12 で一致することを確かめる。

設定は ``EnergyFilterConfig.from_env`` で読む。``E_LPF_NATIVE_ON`` の既定は 0（Butterworth）にした。
本体の既定は 1 だが、GUI は 0 を渡す（有効だと 1 次指数フィルタになり既発表の数値と比べられない）。
"""

from __future__ import annotations

import ast
import collections
import io
import math
import os
import typing

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, welch

import energy_pipeline
from energy_pipeline import (
    AdaptiveCutoff,
    EnergyFilterConfig,
    OnlineF0Estimator,
    angle_between,
    compute_cycle_energy_filtered,
    fc_scheduler,
)
from utils_dynamic import compute_lpf_exp_fb_native

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_SCRIPT = os.path.join(REPO_ROOT, "master_research_code.py")
LEGACY_NAMES = (
    "OnlineF0Estimator",
    "_fc_scheduler",
    "_butter_lowpass_filtfilt",
    "_interp_uniform",
    "_winsorize",
    "compute_cycle_energy_filtered",
)


def _legacy(config: EnergyFilterConfig, *, scipy_ok: bool = True) -> dict:
    """本体の 6 定義を、``config`` の値を E_* に入れた名前空間で実行する（本体は import しない）。"""
    tree = ast.parse(io.open(MAIN_SCRIPT, encoding="utf-8").read())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in LEGACY_NAMES]
    assert sorted(n.name for n in nodes) == sorted(LEGACY_NAMES), "本体の定義が見つからない（名前が変わった？）"
    namespace = {
        "np": np, "math": math, "collections": collections, "Tuple": typing.Tuple,
        "butter": butter, "filtfilt": filtfilt, "welch": welch, "PchipInterpolator": PchipInterpolator,
        "_SCIPY_OK": scipy_ok, "compute_lpf_exp_fb_native": compute_lpf_exp_fb_native,
        "E_FC": config.fc, "E_LPF_ORDER": config.lpf_order, "E_RESAMPLE_N": config.resample_n,
        "E_MAX_DTH": config.max_dth, "E_WINSOR_PCTL_LOW": config.winsor_low,
        "E_WINSOR_PCTL_HIGH": config.winsor_high, "E_DEBUG": False,
        "E_LPF_NATIVE_ON": config.lpf_native_on, "E_FC_ADAPTIVE_ON": config.fc_adaptive_on,
        "E_FPS_MIN": config.fps_min, "E_FPS_MAX": config.fps_max,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, MAIN_SCRIPT, "exec"), namespace)  # noqa: S102
    return namespace


def _signals(n: int, *, nan_at=(), seed: int = 0, freq: float = 0.4, dt: float = 1 / 30):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    theta = 1.9 + 0.7 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(n)
    tau = 8.0 + 6.0 * np.cos(2 * np.pi * freq * t) + 0.8 * rng.standard_normal(n)
    for i in nan_at:
        if i < n:
            theta[i] = np.nan
            tau[(i + 3) % n] = np.inf
    return theta, tau


CONFIGS = {
    "butterworth": EnergyFilterConfig(),
    "native": EnergyFilterConfig(lpf_native_on=True),
    "adaptive": EnergyFilterConfig(fc_adaptive_on=True),
    "native_adaptive": EnergyFilterConfig(lpf_native_on=True, fc_adaptive_on=True, lpf_order=3, resample_n=60),
}


class TestSameAsTheMainScript:
    @pytest.mark.parametrize("name", list(CONFIGS))
    @pytest.mark.parametrize("n", [2, 3, 5, 12, 40, 150])
    @pytest.mark.parametrize("fc_override", [None, 3.0, 0.0])
    @pytest.mark.parametrize("nan_at", [(), (0, 7, 20, 21)])
    def test_cycle_energy(self, name, n, fc_override, nan_at):
        config = CONFIGS[name]
        legacy = _legacy(config)["compute_cycle_energy_filtered"]
        theta, tau = _signals(n, nan_at=nan_at, seed=n)
        e_pos, e_neg, info = compute_cycle_energy_filtered(theta, tau, 1 / 30, fc_override, config=config)
        l_pos, l_neg, l_info = legacy(theta, tau, 1 / 30, fc_override)
        assert e_pos == pytest.approx(l_pos, rel=1e-12, abs=0.0)
        assert e_neg == pytest.approx(l_neg, rel=1e-12, abs=0.0)
        assert {k: info[k] for k in l_info} == l_info

    @pytest.mark.parametrize("dt", [1 / 15, 1 / 60])
    def test_other_sampling_intervals(self, dt):
        config = CONFIGS["adaptive"]
        legacy = _legacy(config)["compute_cycle_energy_filtered"]
        theta, tau = _signals(90, dt=dt)
        new = compute_cycle_energy_filtered(theta, tau, dt, 2.5, config=config)
        old = legacy(theta, tau, dt, 2.5)
        assert new[:2] == pytest.approx(old[:2], rel=1e-12, abs=0.0)

    def test_without_scipy(self, monkeypatch):
        """SciPy が無い環境の代わりの計算（移動平均・線形補間）も同じ。"""
        config = CONFIGS["butterworth"]
        legacy = _legacy(config, scipy_ok=False)["compute_cycle_energy_filtered"]
        monkeypatch.setattr(energy_pipeline, "_SCIPY_OK", False)
        theta, tau = _signals(60)
        new = compute_cycle_energy_filtered(theta, tau, 1 / 30, config=config)
        old = legacy(theta, tau, 1 / 30)
        assert new[:2] == pytest.approx(old[:2], rel=1e-12, abs=0.0)

    def test_f0_estimator(self):
        config = EnergyFilterConfig()
        Legacy = _legacy(config)["OnlineF0Estimator"]
        new = OnlineF0Estimator(30.0, 4.0, 0.3, fps_min=config.fps_min, fps_max=config.fps_max)
        old = Legacy(30.0, 4.0, 0.3)
        theta, _ = _signals(400, nan_at=(50, 51, 300), freq=0.5)
        for i, th in enumerate(theta):
            if i in (150, 250):
                fps = 24.0 if i == 150 else 200.0  # 200 は上限で切られる
                new.set_fps(fps)
                old.set_fps(fps)
            new.step(th)
            old.step(th)
            if i % 17 == 0:
                assert new.estimate() == pytest.approx(old.estimate(), rel=1e-12, abs=1e-12)
        assert (new.fps, new.win_len, list(new.buffer)) == (old.fps, old.win_len, list(old.buffer))

    def test_fc_scheduler(self):
        legacy = _legacy(EnergyFilterConfig())["_fc_scheduler"]
        for f0, conf, prev in [(0.5, 10.0, 1.2), (0.5, 1.0, 1.2), (0.0, 10.0, 2.0), (2.0, 5.0, 3.0), (0.2, 4.0, 5.0)]:
            args = (f0, conf, prev, 2.1, 6.0, 6.0, 0.15, 3.0)
            assert fc_scheduler(*args) == pytest.approx(legacy(*args), rel=1e-12)

    @pytest.mark.parametrize("update_hz", [1.0, 4.0])
    def test_adaptive_cutoff_follows_the_main_loop(self, update_hz):
        """AdaptiveCutoff は本体の 3270〜3303 行（毎フレーム f0 に供給し、E_FC_UPDATE_HZ ごとに fc を更新）と同じ。"""
        config = EnergyFilterConfig(fc_adaptive_on=True, fc_update_hz=update_hz)
        ns = _legacy(config)
        theta, _ = _signals(600, freq=0.5, nan_at=(100,))
        fps_seq = [30.0] * 200 + [27.5] * 200 + [31.0] * 200

        estimator, fc, counter, expected = None, config.fc, 0, []
        for th, fps in zip(theta, fps_seq):
            f = float(np.clip(fps, config.fps_min, config.fps_max))
            if estimator is None:
                estimator = ns["OnlineF0Estimator"](fps=f, win_sec=config.f0_win_sec, fmin=config.f0_fmin)
            else:
                estimator.set_fps(f)
            interval = max(1, int(round(f / max(config.fc_update_hz, 1e-6))))
            estimator.step(th)
            counter += 1
            if counter >= interval:
                f0, conf = estimator.estimate()
                fc = ns["_fc_scheduler"](f0, conf, fc, config.fc_min, config.fc_max, config.fc_k,
                                         config.fc_ema_beta, config.f0_snr_threshold)
                counter = 0
            expected.append(fc)

        cutoff = AdaptiveCutoff(config, fps=30.0)
        got = [cutoff.step(th, fps=fps) for th, fps in zip(theta, fps_seq)]
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
        assert cutoff.fc == got[-1]
        assert got[-1] != config.fc, "0.5 Hz の周期で fc が動いていない"


class TestAdaptiveCutoff:
    def test_off_keeps_the_fixed_cutoff(self):
        cutoff = AdaptiveCutoff(EnergyFilterConfig(fc=1.5), fps=30.0)
        theta, _ = _signals(300, freq=0.5)
        assert {cutoff.step(th) for th in theta} == {1.5}

    def test_fc_moves_towards_k_times_f0(self):
        """0.5 Hz の押し上げなら fc は k·f0 = 3.0 Hz へ近づく（下限 2.1・上限 6.0 の内側）。"""
        config = EnergyFilterConfig(fc_adaptive_on=True)
        cutoff = AdaptiveCutoff(config, fps=30.0)
        theta, _ = _signals(30 * 40, freq=0.5)
        for th in theta:
            cutoff.step(th)
        assert cutoff.fc == pytest.approx(3.0, abs=0.3)
        assert cutoff.last_f0 == pytest.approx(0.5, abs=0.1)


class TestConfig:
    def test_defaults(self):
        c = EnergyFilterConfig.from_env({})
        assert c == EnergyFilterConfig()
        assert (c.fc, c.lpf_order, c.resample_n, c.max_dth, c.winsor_low, c.winsor_high) == (1.2, 2, 80, 0.25, 5.0, 95.0)
        assert c.lpf_native_on is False, "既定は Butterworth（GUI の既定と同じ。本体の既定 1 とは違う）"
        assert c.fc_adaptive_on is False
        assert (c.fc_min, c.fc_max, c.fc_k, c.f0_win_sec, c.fc_ema_beta, c.fc_update_hz) == (2.1, 6.0, 6.0, 4.0, 0.15, 1.0)
        assert (c.f0_fmin, c.f0_snr_threshold, c.fps_min, c.fps_max) == (0.3, 3.0, 5.0, 120.0)

    def test_values_come_from_the_environment(self):
        env = {
            "E_FC": "2.5", "E_LPF_ORDER": "3", "E_RESAMPLE_N": "60", "E_MAX_DTH": "0.3", "E_WLOW": "2",
            "E_WHIGH": "98", "E_LPF_NATIVE_ON": "1", "E_FC_ADAPTIVE_ON": "true", "E_FC_MIN": "1.5",
            "E_FC_MAX": "5", "E_FC_K": "4", "E_F0_WIN_SEC": "3", "E_FC_EMA_BETA": "0.2", "E_FC_UPDATE_HZ": "2",
            "E_F0_FMIN": "0.2", "E_F0_SNR_THRESHOLD": "4", "E_FPS_MIN": "10", "E_FPS_MAX": "60",
        }
        c = EnergyFilterConfig.from_env(env)
        assert (c.fc, c.lpf_order, c.resample_n, c.max_dth, c.winsor_low, c.winsor_high) == (2.5, 3, 60, 0.3, 2.0, 98.0)
        assert c.lpf_native_on is True and c.fc_adaptive_on is True
        assert (c.fc_min, c.fc_max, c.fc_k, c.f0_win_sec, c.fc_ema_beta, c.fc_update_hz) == (1.5, 5.0, 4.0, 3.0, 0.2, 2.0)
        assert (c.f0_fmin, c.f0_snr_threshold, c.fps_min, c.fps_max) == (0.2, 4.0, 10.0, 60.0)

    @pytest.mark.parametrize("raw, expected", [("0", False), ("off", False), (" YES ", True), ("", False), ("??", False)])
    def test_flags_are_read_like_env_flag(self, raw, expected):
        """真偽値は config.env_flag と同じ読み方（空・知らない値は既定）。"""
        assert EnergyFilterConfig.from_env({"E_LPF_NATIVE_ON": raw}).lpf_native_on is expected

    def test_a_broken_number_falls_back_to_the_default(self):
        with pytest.warns(RuntimeWarning):
            c = EnergyFilterConfig.from_env({"E_FC": "abc", "E_LPF_ORDER": ""})
        assert c.fc == 1.2 and c.lpf_order == 2

    def test_the_default_config_is_read_from_the_process_environment(self, monkeypatch):
        monkeypatch.setenv("E_RESAMPLE_N", "40")
        theta, tau = _signals(60)
        *_, info = compute_cycle_energy_filtered(theta, tau, 1 / 30)
        assert info["n_u"] == 40


class TestKeptApi:
    def test_angle_between_is_kept(self):
        """本体が ``from energy_pipeline import angle_between`` している。"""
        assert angle_between(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(math.pi / 2)
        assert angle_between(np.zeros(3), np.array([0, 1.0, 0])) == 0.0

    def test_info_reports_the_cutoff_used(self):
        theta, tau = _signals(60)
        *_, info = compute_cycle_energy_filtered(theta, tau, 1 / 30, 3.0, config=EnergyFilterConfig(fc_adaptive_on=True))
        assert info["fc"] == 3.0
        *_, info = compute_cycle_energy_filtered(theta, tau, 1 / 30, 3.0, config=EnergyFilterConfig())
        assert info["fc"] == 1.2  # 適応がオフなら上書きは効かない（本体と同じ）
