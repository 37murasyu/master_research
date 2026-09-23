"""混成の計測に EKF をつなぐ（``app.hybrid.ekf``、計画の T7）。

**なぜこのテストがあるか。**

2026-09-23 の実機の計測（Mac＋Pixel）は EKF も外れ値の除去も無く、右上腕が 4,734 m に飛んだ組がそのまま逆動力学に
入って手首のトルクが 100 万 N·m になった。USB 経路（``master_research_code.py``）と同じ ``LandmarkEKF``
（系列ごとの雑音、頑健な門）を混成にも入れる。混成に特有の点:

- 雑音は ``HYBRID_EKF_PROFILE`` のプロファイル、空なら同梱の既定値（builtin）。GUI が必ず渡す USB 向けの
  ``EKF_Q_ACC/EKF_R=1e-3`` は使わない（合成の押し上げで肘の W_pos が +52% になる。同梱値なら +10%）
- 同期バッファは 100 ms を超える穴で組を作らず、次の組の時刻が格子（33.3 ms）の n 倍跳ぶ。EKF は NaN の観測で
  dt=1/30 の予測を抜けた回数だけ進めてから更新する。0.5 s を超える抜けは作り直す
- 100 ms を超える抜けの後は、リンクの速度の計算器と部位データも作り直す（古い前フレームとの差で速度と
  トルクが跳ねないように）
- 記録と EKF の較正のため、EKF の手前（``points_raw``）と後（``points_3d``）を分けて持つ
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from app.hybrid.ekf import EkfSettings, GridEkf, hybrid_noise
from app.runners.network_measure import MeasurementConfig, NetworkMeasurement
from config import pose_keypoints
from hybrid_pushup import SLOT, PushUp, run
from test_network_measure import _body_points, _pair_from_pixels, _project, _stereo_projections

IDS = sorted(pose_keypoints)
DT = 1 / 30


def _measurement(ekf: EkfSettings | None = None, mass: float = 65.0) -> NetworkMeasurement:
    P0, P1 = _stereo_projections()
    config = MeasurementConfig(body_mass_kg=mass)
    if ekf is not None:
        config.ekf = ekf
    return NetworkMeasurement(P0, P1, pose_keypoints, config)


def _profile(tmp_path, *, r=1e-4, scale_len=0.28):
    """最小の較正プロファイル（``app.tuning.ekf_profile.read_profile`` が通る形）。"""
    entry = {"q_acc": 0.5, "r": r, "gate_std": 4.0, "n_eff": 500, "rho1": 0.1, "source": "fit"}
    profile = {
        "schema_version": 1, "frame": "runtime", "unit": "m", "dt": DT, "fps": 30.0,
        "bpf": {"low": 0.0, "high": 0.0, "order": 2},
        "scale_ref": {"pair": [12, 14], "median_len": scale_len},
        "series": {str(lid): {axis: dict(entry) for axis in "xyz"} for lid in IDS},
    }
    path = tmp_path / "ekf_profile_test.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


class TestSettings:
    def test_the_usb_scalars_are_not_used(self, monkeypatch):
        """GUI は USB 向けの EKF_Q_ACC/EKF_R=1e-3 と EKF_PROFILE を全件渡す。混成はどちらも読まない。"""
        monkeypatch.setenv("EKF_Q_ACC", "1e-3")
        monkeypatch.setenv("EKF_R", "1e-3")
        monkeypatch.setenv("EKF_PROFILE", "/nonexistent/usb_profile.json")
        monkeypatch.delenv("HYBRID_EKF_PROFILE", raising=False)
        settings = EkfSettings.from_env()
        noise = hybrid_noise(settings, IDS)
        assert settings.profile is None
        assert noise.origin == "builtin"
        np.testing.assert_allclose(noise.cfg.r, 2.59e-5)
        np.testing.assert_allclose(noise.cfg.q_acc, 0.122)

    def test_shared_switches_are_read(self, monkeypatch):
        monkeypatch.setenv("EKF_ENABLE", "0")
        monkeypatch.setenv("EKF_ROBUST_GATE", "false")
        monkeypatch.setenv("EKF_GATE_STD", "2.5")
        monkeypatch.setenv("EKF_MAX_GAP_S", "0.3")
        settings = EkfSettings.from_env()
        assert (settings.enabled, settings.robust_gate, settings.gate_std, settings.max_gap_s) == (False, False, 2.5, 0.3)

    def test_a_profile_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYBRID_EKF_PROFILE", str(_profile(tmp_path)))
        noise = hybrid_noise(EkfSettings.from_env(), IDS)
        assert noise.origin == "profile"
        np.testing.assert_allclose(noise.cfg.r, 1e-4)
        assert noise.provenance()["path"].endswith("ekf_profile_test.json")


class TestGrid:
    def test_missing_grid_steps_are_predicted_with_nan(self):
        """抜けた格子の数だけ、NaN の観測と dt=1/30 で予測してから更新する。"""
        ekf = GridEkf(EkfSettings(), IDS)
        calls = []
        inner = ekf._ekf.step

        def spy(meas, dt):
            calls.append((np.isnan(meas).all(), dt))
            return inner(meas, dt)

        ekf._ekf.step = spy
        points = np.ones((len(IDS), 3))
        ekf.step(points, missing=0)
        ekf.step(points, missing=2)
        assert calls == [(False, pytest.approx(DT)), (True, pytest.approx(DT)), (True, pytest.approx(DT)),
                         (False, pytest.approx(DT))]
        assert ekf.rebuilds == 0

    def test_a_gap_over_half_a_second_rebuilds(self):
        ekf = GridEkf(EkfSettings(), IDS)
        points = np.ones((len(IDS), 3))
        for _ in range(10):
            ekf.step(points, missing=0)
        jumped = points + 0.5
        pos, _ = ekf.step(jumped, missing=15)   # 0.53 s の抜け
        assert ekf.rebuilds == 1
        np.testing.assert_allclose(pos, jumped, err_msg="作り直した直後は観測そのもの")

    def test_the_scale_ratio_rescales_the_profile(self, tmp_path):
        ekf = GridEkf(EkfSettings(profile=str(_profile(tmp_path))), IDS)
        ekf.set_scale(2.0)
        np.testing.assert_allclose(ekf._ekf._r, 4e-4)
        assert ekf.scale_ratio == 2.0


class TestInTheMeasurement:
    def _steady(self, measurement, frames=40, jump_at=None):
        truth = _body_points(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)
        original = measurement.points_3d
        results = []
        for k in range(frames):
            if k == jump_at:
                def jumped(pair, _original=original):
                    points = _original(pair).copy()
                    points[SLOT[14], 0] += 3.0   # 右肘を 3 m 飛ばす（2026-09-23 の実機で起きた桁）
                    return points
                measurement.points_3d = jumped
            else:
                measurement.points_3d = original
            results.append(measurement.process(_pair_from_pixels(round(k * 1e9 / 30), p0, p1)))
        return results, truth

    def test_a_3m_jump_of_the_right_elbow_is_held_back(self):
        measurement = _measurement()
        results, truth = self._steady(measurement, jump_at=35)
        expected = truth[:, [0, 2, 1]] * -0.01
        jumped = results[35]
        assert abs(jumped.points_raw[SLOT[14], 0] - expected[SLOT[14], 0]) > 2.9, "記録には飛んだ値が残る"
        assert np.linalg.norm(jumped.points_3d[SLOT[14]] - expected[SLOT[14]]) < 0.05
        assert all(np.all(np.abs(v) < 200) for v in jumped.local_torques.values())

    def test_without_the_ekf_the_two_are_the_same(self):
        measurement = _measurement(EkfSettings(enabled=False))
        results, _ = self._steady(measurement, frames=5)
        assert measurement.ekf is None
        for result in results:
            np.testing.assert_array_equal(result.points_raw, result.points_3d)

    def test_the_provenance_is_kept(self):
        measurement = _measurement()
        assert measurement.ekf.noise.origin == "builtin"
        assert measurement.ekf_provenance()["enabled"] is True
        assert measurement.ekf_provenance()["origin"] == "builtin"
        assert _measurement(EkfSettings(enabled=False)).ekf_provenance() == {"enabled": False}

    def test_grid_index_and_missing_steps_follow_the_timestamps(self):
        measurement = _measurement()
        truth = _body_points(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)
        grid = [0, 1, 2, 5, 6]
        results = [measurement.process(_pair_from_pixels(1_000 + round(k * 1e9 / 30), p0, p1)) for k in grid]
        assert [r.grid_index for r in results] == grid


class TestAfterAGap:
    def test_torque_does_not_jump_right_after_a_long_gap(self):
        """150 ms を超える抜け（押し上げの途中）の直後に、古い前フレームとの差で速度とトルクが跳ねない。"""
        motion = PushUp(reps=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = run(_measurement(), motion)
            gap = set(range(75, 81))   # 押し上げの途中の 200 ms
            broken = run(_measurement(), motion, drop=gap)
        reference = max(abs(r.local_torques["elbow_R"][1]) for r in clean if r.local_torques)
        after = [r for r in broken if 81 <= r.grid_index <= 86 and r.local_torques]
        assert after, "抜けの後にトルクが出ていない"
        worst = max(abs(r.local_torques["elbow_R"][1]) for r in after)
        assert worst < 1.3 * reference, f"抜けの直後に {worst:.1f} N·m（抜けなしの最大 {reference:.1f}）"
        assert all(r.dt_s <= 0.1 for r in after[1:]), "抜けの直後のフレームだけが長い dt"


    def test_the_link_calculators_restart_after_a_gap_over_100ms(self):
        """100 ms を超える抜けの直後は、抜ける前のフレームとの差で速度を作らない（作り直す）。66 ms では作り直さない。"""
        measurement = _measurement()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = {r.grid_index: r for r in run(measurement, PushUp(reps=1), drop={40} | set(range(75, 81)))}
        assert measurement.dynamics_restarts == 1
        assert results[41].local_torques, "66 ms の抜けは作り直さない"
        assert not results[81].local_torques, "抜ける前のフレームとの差で速度を作っていない"
        assert results[82].local_torques and results[82].dt_s == pytest.approx(DT, rel=1e-6)


class TestWorkWithTheEkf:
    MOTION = PushUp(reps=2)

    def _w_pos(self, ekf: EkfSettings, noise_px: float) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = run(_measurement(ekf), self.MOTION, noise_px=noise_px, seed=3)
        return sum(max(r.powers.get("elbow_R", 0.0), 0.0) * r.dt_s for r in results
                   if r.t_ns / 1e9 >= self.MOTION.rest_s)

    def test_positive_elbow_work_stays_within_15_percent(self):
        """EKF の遅れや平滑で押し上げの仕事が大きく変わらない（同梱の既定値で +8%。GUI の 1e-3 だと +52%）。"""
        assert self._w_pos(EkfSettings(), 0.0) == pytest.approx(self._w_pos(EkfSettings(enabled=False), 0.0), rel=0.15)

    def test_with_noise_the_ekf_is_closer_to_the_truth(self):
        """W_pos = Σmax(P,0)·dt は雑音を整流して膨らむ（0.3 px で EKF なしは約 3 倍）。EKF はそれを抑える。"""
        truth = self._w_pos(EkfSettings(enabled=False), 0.0)
        with_ekf = self._w_pos(EkfSettings(), 0.3)
        without = self._w_pos(EkfSettings(enabled=False), 0.3)
        assert abs(with_ekf - truth) < 0.5 * abs(without - truth)


class TestDivergence:
    def test_a_fast_move_does_not_leave_the_filter_metres_away(self):
        """同梱の既定値で追える帯域は約 0.65 Hz。速い動き（1.2 Hz・奥行き 15 cm の揺れ）でも数 m ずれたままにならない。

        見張りが無いと、頑健な門が予測から外れるほど更新を弱め、この合成で右肘が 4 m ずれた。
        """
        measurement = _measurement()
        worst = 0.0
        for k in range(150):
            truth = _body_points(k / 30)
            result = measurement.process(_pair_from_pixels(
                round(k * 1e9 / 30), _project(measurement.P0, truth), _project(measurement.P1, truth)))
            expected = truth[:, [0, 2, 1]] * -0.01
            worst = max(worst, float(np.max(np.linalg.norm(result.points_3d - expected, axis=1))))
        assert worst < 0.5, f"{worst:.2f} m ずれた（見張りが無いと約 4 m）"
        assert measurement.ekf.resets > 0
