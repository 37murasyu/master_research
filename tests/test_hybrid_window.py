"""混成の計測の先頭の窓（肩と肘が有限の組を 30 組）で決めること（計画の T8）。

**なぜこのテストがあるか。**

2026-09-23 の実機の計測（Mac＋Pixel）は先頭 30 フレームの骨の長さが壊れており（右上腕 6.4 m）、それでも止まらずに
トルクを出し続けた。USB 経路は EKF のプロファイルを使うときだけ肩–肘の長さを検査していたが、混成はプロファイルの有無に
よらず検査し、人体の範囲（0.10〜0.60 m）の外なら終了コード 3 で止める（``meta.json`` の error に理由）。

同じ窓で、プロファイルの掛け直し（体格の比）、慣性、重力（校正に盤を立てた向きがあれば使う）、上向きと基準の高さ、
実測の前腕長 → ゲージの帯（論文 4.5.2 節の W_0.70・W_0.85）、回の区切り（``RepDetector``）を決める。
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from app.gauge.thresholds import part_bands
from app.gauge.tracker import GaugeTracker
from app.hybrid.ekf import EkfSettings
from app.hybrid.measurement import MeasurementSession
from app.net.protocol import LandmarkFrame
from app.runners.network_measure import ImplausibleBodyScale, MeasurementConfig, NetworkMeasurement
from config import pose_keypoints
from hybrid_pushup import SLOT, PushUp, body_cm, pushup_pairs, run, runtime_m
from test_hybrid_ekf import _profile
from test_network_measure import _pair_from_pixels, _project, _stereo_projections

ONE_RM = {"elbow_L": 20.0, "elbow_R": 22.0, "wrist_L": 8.0, "wrist_R": 9.0}


def _measurement(**kw) -> NetworkMeasurement:
    P0, P1 = _stereo_projections()
    config = MeasurementConfig(body_mass_kg=65.0, one_rm=kw.pop("one_rm", None), ekf=kw.pop("ekf", EkfSettings()))
    return NetworkMeasurement(P0, P1, pose_keypoints, config, **kw)


def _seated(measurement, frames=40, start=0):
    truth = body_cm(0.0)
    p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)
    results = []
    for k in range(start, start + frames):
        results.append(measurement.process(_pair_from_pixels(round(k * 1e9 / 30), p0, p1)))
    return results


class TestBodyScale:
    def test_a_hundredth_of_the_scale_stops_the_measurement(self):
        """校正の並進を m で入れた（1/100 の尺度）ときは、プロファイルが無くても止める。"""
        P0, _ = _stereo_projections()
        _, P1_metres = _stereo_projections(baseline_cm=0.5)
        measurement = NetworkMeasurement(P0, P1_metres, pose_keypoints, MeasurementConfig(body_mass_kg=65.0))
        truth = body_cm(0.0)
        p0, p1 = _project(P0, truth), _project(_stereo_projections()[1], truth)
        with pytest.raises(ImplausibleBodyScale, match="肩–肘"):
            for k in range(40):
                measurement.process(_pair_from_pixels(round(k * 1e9 / 30), p0, p1))
        assert measurement.frame_index == 29, "窓（30 組）が埋まった時点で止める"

    def test_the_session_exits_with_code_3(self, tmp_path):
        from test_hybrid_measure import geometry, projected
        from app.hybrid.calibration_io import load_calibration, save_calibration
        from app.hybrid.checkerboard import Board, Stereo

        intr, stereo = geometry()
        metres = Stereo(stereo.R, stereo.T / 100.0, 0, [], list(range(12)))   # 並進を m で保存してしまった校正
        cal = load_calibration(save_calibration(
            intr, intr, metres, Board(), root=tmp_path / "calibration",
            cameras=[{"kind": "mac", "device_id": "mac"}, {"kind": "pixel", "device_id": "pixel-1"}]))
        session = MeasurementSession(cal, root=tmp_path / "measure")
        session.on_landmarks(LandmarkFrame("cam1", 0, 1, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))
        _, a, b = projected()
        session.on_pairs([_pair_from_pixels(round(k * 1e9 / 30), a, b) for k in range(40)])
        assert session.exit_code == 3 and session.failed.is_set()
        assert "肩–肘" in session.error
        session.on_pairs([_pair_from_pixels(round(41 * 1e9 / 30), a, b)])   # 止めた後に届いた組は捨てる
        session.close()
        meta = json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))
        assert (meta["status"], meta["exit_code"]) == ("failed", 3)
        assert "肩–肘" in meta["error"]

    def test_frames_without_shoulders_do_not_fill_the_window(self):
        measurement = _measurement()
        truth = body_cm(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)
        blind = p0.copy()
        blind[[SLOT[11], SLOT[12]]] = np.nan
        for k in range(20):
            measurement.process(_pair_from_pixels(round(k * 1e9 / 30), blind, p1))
        assert measurement.window_closed is False
        _seated(measurement, frames=30, start=20)
        assert measurement.window_closed is True

    def test_a_profile_is_rescaled_by_the_body_ratio(self, tmp_path):
        measurement = _measurement(ekf=EkfSettings(profile=str(_profile(tmp_path, r=1e-4, scale_len=0.14))))
        _seated(measurement)
        ratio = 0.28 / 0.14   # 合成の上腕は 28 cm
        assert measurement.ekf.scale_ratio == pytest.approx(ratio, rel=1e-3)
        np.testing.assert_allclose(measurement.ekf._ekf._r, 1e-4 * ratio ** 2, rtol=2e-3)
        assert measurement.window["ekf_run_length_m"] == pytest.approx(0.28, abs=1e-3)


class TestWhatTheWindowDecides:
    def test_forearm_lengths_match_the_geometry(self):
        measurement = _measurement()
        _seated(measurement)
        assert measurement.forearm_m["R"] == pytest.approx(0.25, abs=1e-3)
        assert measurement.forearm_m["L"] == pytest.approx(0.25, abs=1e-3)

    def test_bands_follow_the_paper_thresholds(self):
        tracker = GaugeTracker()
        measurement = _measurement(one_rm=ONE_RM, tracker=tracker)
        _seated(measurement)
        expected = part_bands(65.0, measurement.forearm_m, ONE_RM)
        assert measurement.bands == expected
        frame = tracker.snapshot()
        for part, band in expected.items():
            assert frame.parts[part].band == band.band
            assert frame.parts[part].w1rm == pytest.approx(band.w1rm)

    def test_without_the_one_rm_there_is_no_band(self):
        tracker = GaugeTracker()
        measurement = _measurement(tracker=tracker)
        _seated(measurement)
        assert all(band.band is None and "1RM" in band.reason for band in measurement.bands.values())
        assert all(reading.band is None for reading in tracker.snapshot().parts.values())

    def test_up_and_the_seated_height(self):
        """実行時の座標は z が上。基準の高さは座った肩の中点の z。"""
        measurement = _measurement()
        _seated(measurement)
        np.testing.assert_allclose(measurement.up, [0.0, 0.0, 1.0], atol=1e-9)
        shoulders = runtime_m(body_cm(0.0))[[SLOT[11], SLOT[12]]].mean(axis=0)
        assert measurement.baseline_height_m == pytest.approx(shoulders[2], abs=1e-4)
        assert measurement.rep_detector is not None
        assert measurement.rep_detector.baseline_m == pytest.approx(measurement.baseline_height_m)

    def test_the_board_sets_the_gravity(self):
        """校正の meta に盤を立てた向きがあれば、最寄りの軸に吸着させて使う（符号は体幹）。"""
        measurement = _measurement(board_up=np.array([0.0, 0.1, -1.0]))   # 短辺の符号が逆に保存されていても
        _seated(measurement)
        assert measurement.gravity_choice.source == "checkerboard"
        assert measurement.gravity_choice.label == "Z-"
        np.testing.assert_allclose(measurement.gravity, [0.0, 0.0, -9.81], atol=1e-9)

    def test_the_session_reads_the_board_from_the_calibration(self, tmp_path):
        from test_hybrid_measure import calibration, projected

        cal = calibration(tmp_path)
        cal.meta["checkerboard_short_axis"] = {"vector_runtime": [0.0, 0.05, 1.0]}
        session = MeasurementSession(cal, root=tmp_path / "measure")
        session.on_landmarks(LandmarkFrame("cam1", 0, 1, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))
        _, a, b = projected()
        session.on_pairs([_pair_from_pixels(round(k * 1e9 / 30), a, b) for k in range(35)])
        assert session.exit_code == 0, session.error
        assert session.measurement.gravity_choice.source == "checkerboard"
        session.close()

    def test_no_dynamics_until_the_window_is_full(self):
        measurement = _measurement()
        results = _seated(measurement, frames=29)
        assert all(not r.local_torques for r in results)
        assert measurement.window_closed is False
