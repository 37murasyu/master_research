"""スマホ経路の計測オーケストレーションを検証する。

物理計算そのものは既存モジュールを再利用しているので、ここで見るのは
「正しく繋がっているか」。特に三角測量は、既知の 3D 点を投影して戻せるかで
確かめる。ここがずれると下流のトルクがすべて狂う。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.net.protocol import LANDMARK_COUNT, LandmarkFrame
from app.net.sync_buffer import InterpolatedFrame, PairedSample
from app.runners.network_measure import MeasurementConfig, NetworkMeasurement

WIDTH, HEIGHT = 1280, 720

# 既存 config.pose_keypoints と同じ 12 点
POSE_KEYPOINTS = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]


def _stereo_projections(baseline_m: float = 0.5, focal_px: float = 900.0):
    """左右に baseline だけ離した、平行なステレオ対の投影行列。"""
    K = np.array(
        [[focal_px, 0.0, WIDTH / 2], [0.0, focal_px, HEIGHT / 2], [0.0, 0.0, 1.0]]
    )
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([np.eye(3), np.array([[-baseline_m], [0.0], [0.0]])])
    return P0, P1


def _project(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """3D 点を画像座標へ落とす。"""
    homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    projected = (P @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def _pair_from_pixels(t_ns: int, pixels0: np.ndarray, pixels1: np.ndarray) -> PairedSample:
    """ピクセル座標から、受信層が出すのと同じ形のペアを組み立てる。"""

    def to_frame(role: str, pixels: np.ndarray) -> InterpolatedFrame:
        landmarks = [(0.0, 0.0, 0.0, 1.0)] * LANDMARK_COUNT
        landmarks = list(landmarks)
        for slot, landmark_id in enumerate(POSE_KEYPOINTS):
            x, y = pixels[slot]
            landmarks[landmark_id] = (x / WIDTH, y / HEIGHT, 0.0, 1.0)
        return InterpolatedFrame(role, t_ns, WIDTH, HEIGHT, landmarks)

    return PairedSample(
        t_ns=t_ns,
        frames={"cam0": to_frame("cam0", pixels0), "cam1": to_frame("cam1", pixels1)},
    )


def _body_points(t: float) -> np.ndarray:
    """12 関節ぶんの 3D 点。上肢だけ動かす。カメラ座標系（z が前方）。"""
    points = []
    for index in range(12):
        base_x = -0.3 + 0.06 * index
        base_y = -0.2 + 0.03 * index
        swing = 0.10 * np.sin(2 * np.pi * 1.2 * t) if index < 6 else 0.0
        points.append([base_x + swing, base_y, 2.5])
    return np.array(points, dtype=np.float64)


def _measurement() -> NetworkMeasurement:
    P0, P1 = _stereo_projections()
    return NetworkMeasurement(P0, P1, POSE_KEYPOINTS, MeasurementConfig(body_mass_kg=60.0))


class TestTriangulation:
    def test_recovers_known_3d_points(self):
        """既知の 3D 点を投影して戻したとき、元の点に一致すること。

        座標系の変換 (x, y, z) -> (-x, -z, -y) と 0.01 倍が既存実装と
        同じであることも、この逆変換で確かめている。
        """
        measurement = _measurement()
        truth = _body_points(0.0)

        pair = _pair_from_pixels(
            0,
            _project(measurement.P0, truth),
            _project(measurement.P1, truth),
        )
        result = measurement.process(pair)
        assert result is not None

        # 既存の変換を逆に辿って、元のカメラ座標系へ戻す
        recovered = result.points_3d / 0.01
        restored = np.empty_like(recovered)
        restored[:, 0] = -recovered[:, 0]
        restored[:, 1] = -recovered[:, 2]
        restored[:, 2] = -recovered[:, 1]

        np.testing.assert_allclose(restored, truth, atol=1e-6)

    def test_output_is_in_metres(self):
        """0.01 倍のスケールがかかり、単位が m になっていること。"""
        measurement = _measurement()
        truth = _body_points(0.0)
        pair = _pair_from_pixels(
            0, _project(measurement.P0, truth), _project(measurement.P1, truth)
        )
        points = measurement.process(pair).points_3d
        # 2.5m 先の被写体なので、変換後の該当成分は 0.025 のオーダー
        assert np.nanmax(np.abs(points)) < 1.0


class TestPipeline:
    def _run(self, frames: int = 40, fps: float = 30.0) -> NetworkMeasurement:
        measurement = _measurement()
        for index in range(frames):
            t = index / fps
            truth = _body_points(t)
            pair = _pair_from_pixels(
                int(t * 1e9),
                _project(measurement.P0, truth),
                _project(measurement.P1, truth),
            )
            measurement.process(pair)
        return measurement

    def test_torques_appear_once_enough_frames_accumulate(self):
        """既存と同じく、慣性テンソルと蓄積が揃うまでトルクは出ない。"""
        measurement = self._run(frames=40)
        with_torque = [r for r in measurement.results if r.local_torques]
        assert with_torque, "十分なフレーム数を流してもトルクが計算されていない"

    def test_no_torque_before_dynamics_are_ready(self):
        measurement = self._run(frames=5)
        assert all(not r.local_torques for r in measurement.results)

    def test_all_six_joints_are_produced(self):
        measurement = self._run(frames=40)
        torques = next(r.local_torques for r in measurement.results if r.local_torques)
        assert set(torques) == {
            "wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L",
        }

    def test_torque_values_are_finite(self):
        measurement = self._run(frames=40)
        for result in measurement.results:
            for name, vector in result.local_torques.items():
                assert np.all(np.isfinite(vector)), f"{name} に非有限値: {vector}"

    def test_timestep_comes_from_capture_timestamps(self):
        """dt は撮影時刻の差から取る。既存のリアルタイム経路は処理ループ速度から
        逆算していたが、こちらは時刻が刻まれているので実測値を使える。"""
        measurement = _measurement()
        truth = _body_points(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)

        measurement.process(_pair_from_pixels(0, p0, p1))
        measurement.process(_pair_from_pixels(50_000_000, p0, p1))  # 50ms 後
        assert measurement._timestep(100_000_000) == pytest.approx(0.05)


class TestRobustness:
    def test_missing_role_is_skipped(self):
        """片方のカメラしか無いペアは処理しない。"""
        measurement = _measurement()
        truth = _body_points(0.0)
        pixels = _project(measurement.P0, truth)
        pair = _pair_from_pixels(0, pixels, pixels)
        broken = PairedSample(t_ns=0, frames={"cam0": pair.frames["cam0"]})
        assert measurement.process(broken) is None

    def test_nan_does_not_poison_the_cycle_baseline(self):
        """NaN が基準値に混ざると、以後一度もサイクルを検出できなくなる。

        既存のリアルタイム経路が踏んでいた問題（code-review 指摘 #5）と
        同じ轍を踏まないこと。
        """
        measurement = _measurement()
        truth = _body_points(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)

        for index in range(20):
            if index == 7:
                # 三角測量が破綻する入力（両カメラで同一点）を混ぜる
                degenerate = np.full_like(p0, np.nan)
                measurement.process(_pair_from_pixels(index * 33_000_000, degenerate, degenerate))
            else:
                measurement.process(_pair_from_pixels(index * 33_000_000, p0, p1))

        assert np.isfinite(measurement._baseline_z), "基準値が NaN に汚染された"

    def test_reports_cycle_count_and_latest_impulses(self):
        measurement = _measurement()
        assert measurement.cycle_count == 0
        assert set(measurement.latest_impulses) == {
            "wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L",
        }
