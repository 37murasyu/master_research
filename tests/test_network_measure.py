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

# config から引く。値を写すと config 側の変更を検知できない。
# 抽出は ID 昇順で行われるので、点列の並びも昇順になる（再検算 R-1）。
from config import pose_keypoints as POSE_KEYPOINTS  # noqa: E402

POSE_KEYPOINTS_ORDERED = sorted(POSE_KEYPOINTS)


def _stereo_projections(baseline_cm: float = 50.0, focal_px: float = 900.0):
    """左右に baseline だけ離した、平行なステレオ対の投影行列。

    **並進は cm 単位**にしてある。既存パイプラインは三角測量の結果を 0.01 倍して
    m に直しており（``_triangulate_transform_batch``）、部位長も m で扱っている
    （KNOWN_ISSUES によれば前腕 0.19〜0.26 m）。したがって
    ``camera_parameters/*.dat`` の並進は cm 単位である。
    ここを m にすると、サイクル検出の閾値 0.015 が「1.5 m の移動」に相当する
    非現実的な設定になり、実データでは起こらない条件になってしまう。
    """
    K = np.array(
        [[focal_px, 0.0, WIDTH / 2], [0.0, focal_px, HEIGHT / 2], [0.0, 0.0, 1.0]]
    )
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([np.eye(3), np.array([[-baseline_cm], [0.0], [0.0]])])
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
        # 実装が sorted(pose_keypoints) の順で取り出すので、埋める側も昇順で揃える
        for slot, landmark_id in enumerate(POSE_KEYPOINTS_ORDERED):
            x, y = pixels[slot]
            landmarks[landmark_id] = (x / WIDTH, y / HEIGHT, 0.0, 1.0)
        return InterpolatedFrame(role, t_ns, WIDTH, HEIGHT, landmarks)

    return PairedSample(
        t_ns=t_ns,
        frames={"cam0": to_frame("cam0", pixels0), "cam1": to_frame("cam1", pixels1)},
    )


def _body_points(t: float) -> np.ndarray:
    """12 関節ぶんの 3D 点。上肢だけ動かす。カメラ座標系（z が前方）、**単位 cm**。

    被写体まで 250cm、上肢の振幅 10cm、押し出し周期 1.2Hz。
    奥行きも 15cm ほど前後させる（車椅子駆動では体幹が前後する）。

    索引はランドマーク ID の昇順（[0]左肩 [1]右肩 [2]左肘 [3]右肘 [4]左手首
    [5]右手首 [6]左腰 [7]右腰 [8]左膝 [9]右膝 [10]左足首 [11]右足首）。
    等間隔に並べるとリンク長が解剖学的にあり得ない値になり、慣性回帰式が
    適用範囲外に落ちてフォールバック警告が出る。実寸に近い配置にしてある。
    """
    phase = 2 * np.pi * 1.2 * t
    swing = 10.0 * np.sin(phase)     # 上肢の押し出し
    depth = 250.0 + 15.0 * np.sin(phase)
    # ランドマーク ID → 位置。pose_keypoints の構成が変わっても追随する。
    by_id = {
        11: [-18.0 + swing, -30.0, depth],   # 左肩
        12: [18.0 + swing, -30.0, depth],    # 右肩
        13: [-22.0 + swing, -6.0, depth],    # 左肘（上腕 約 24cm）
        14: [22.0 + swing, -6.0, depth],     # 右肘
        15: [-25.0 + swing, 15.0, depth],    # 左手首（前腕 約 21cm）
        16: [25.0 + swing, 15.0, depth],     # 右手首
        17: [-29.0 + swing, 22.0, depth],    # 左小指 MCP（手 約 8cm）
        18: [29.0 + swing, 22.0, depth],     # 右小指 MCP
        19: [-25.0 + swing, 23.0, depth + 4.0],   # 左人差指 MCP
        20: [25.0 + swing, 23.0, depth + 4.0],    # 右人差指 MCP
        21: [-23.0 + swing, 19.0, depth + 5.0],   # 左親指
        22: [23.0 + swing, 19.0, depth + 5.0],    # 右親指
        23: [-12.0, 20.0, depth],            # 左腰
        24: [12.0, 20.0, depth],             # 右腰
        25: [-12.0, 60.0, depth],            # 左膝（大腿 40cm）
        26: [12.0, 60.0, depth],             # 右膝
        27: [-12.0, 100.0, depth],           # 左足首（下腿 40cm）
        28: [12.0, 100.0, depth],            # 右足首
    }
    return np.array([by_id[pid] for pid in POSE_KEYPOINTS_ORDERED], dtype=np.float64)


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
        # 250cm 先の被写体。0.01 倍されるので m 単位で 2.5 前後になる
        assert 1.0 < np.nanmax(np.abs(points)) < 10.0


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

    def test_cycles_are_detected_for_realistic_motion(self):
        """現実的な大きさの往復運動でサイクルが検出されること。

        閾値（0.015）は既存パイプラインの単位系に合わせて調整されている。
        投影行列の並進を m 単位にすると、この閾値が「1.5 m の移動」に相当し、
        実データでは決して起こらない条件になる。cm 単位で組むこと。
        """
        measurement = self._run(frames=150)
        assert measurement.cycle_count > 0, (
            "1.2Hz・奥行き15cm の往復でサイクルが検出されない。"
            "投影行列の単位系（cm）を確認すること。"
        )

    def test_cycle_work_is_recorded_per_joint(self):
        measurement = self._run(frames=150)
        assert measurement.cycle_count > 0
        for key, values in measurement.cycle_work.items():
            assert len(values) == measurement.cycle_count, f"{key} の記録数が揃っていない"
            assert all(np.isfinite(v) for v in values), f"{key} に非有限値"

    def test_timestep_comes_from_capture_timestamps(self):
        """dt は撮影時刻の差から取る。既存のリアルタイム経路は処理ループ速度から
        逆算していたが、こちらは時刻が刻まれているので実測値を使える。"""
        measurement = _measurement()
        truth = _body_points(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)

        measurement.process(_pair_from_pixels(0, p0, p1))
        measurement.process(_pair_from_pixels(50_000_000, p0, p1))  # 50ms 後
        assert measurement._timestep(100_000_000) == pytest.approx(0.05)


def _body_points_rotating_right_arm(t: float) -> np.ndarray:
    """_body_points の右腕を、肘を曲げたまま肩まわりに一体で回す。

    肘角が変わらないので、肘での関節の仕事は 0 のはず。回転軸（カメラの z 軸）は
    上腕と前腕の両方に直交させてある。2 点リンクの角速度は軸に直交する成分しか
    取れないので、直交していないと両リンクの角速度の推定値が揃わない。
    """
    points = _body_points(t)
    slot = {pid: i for i, pid in enumerate(POSE_KEYPOINTS_ORDERED)}
    shoulder = points[slot[12]]
    angle = 1.2 * t   # 一方向に回し続ける（往復だとサイクル内で仕事が相殺して判定にならない）
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    elbow = np.array([4.0, 24.0, 0.0])            # 上腕 約 24cm（奥行きは肩と同じ）
    wrist = elbow + np.array([18.0, 12.0, 0.0])   # 前腕 約 22cm、肘は約 40° 曲げる
    points[slot[14]] = shoulder + rotation @ elbow
    points[slot[16]] = shoulder + rotation @ wrist
    return points


class TestJointPower:
    """サイクルごとの仕事は、関節の相対角速度から求める。"""

    def test_rigid_arm_rotation_does_no_work_at_the_elbow(self):
        """肘角を保ったまま腕全体が回るだけなら、wrist_R（前腕リンク）の仕事は 0。

        wrist_R の局所 y 軸は「上腕 × 前腕」の法線（肘の屈曲軸）なので、仕事率には
        前腕と上腕の相対角速度を使う。かつて前腕の絶対角速度との内積 τ·ω を使っており、
        腕を振るだけで仕事が出ていた（計画メモ A-4 (2)、H-B）。
        """
        measurement = _measurement()
        for index in range(150):
            t = index / 30.0
            truth = _body_points_rotating_right_arm(t)
            measurement.process(_pair_from_pixels(
                int(t * 1e9), _project(measurement.P0, truth), _project(measurement.P1, truth)))

        assert measurement.cycle_count > 0, "前提のサイクル検出が起きていない"
        work = np.abs(np.array(measurement.cycle_work["wrist_R"]))
        assert np.all(work < 0.05), (
            f"肘角が一定なのに wrist_R のサイクル仕事が {work} J 出た。"
            " 前腕の絶対角速度を使っていないか確認すること"
        )


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

        assert np.isfinite(measurement._baseline_sum), "基準値が NaN に汚染された"

    def test_missing_frame_at_detector_creation_does_not_disable_detection(self):
        """検出器が作られる瞬間のフレームが欠測でも、検出が死なないこと。

        既存実装は「フレーム番号ちょうど」で検出器を作るため、そのフレームが
        欠測だと以後一度もサイクルを検出できない。ここでは実際に集まった
        サンプル数で確定させるので、1 フレームの欠測では死なない。
        """
        measurement = _measurement()
        truth = _body_points(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)

        broken_frame = measurement.config.baseline_last_frame  # 検出器が作られる直前
        for index in range(20):
            if index == broken_frame:
                nan_pixels = np.full_like(p0, np.nan)
                measurement.process(_pair_from_pixels(index * 33_000_000, nan_pixels, nan_pixels))
            else:
                measurement.process(_pair_from_pixels(index * 33_000_000, p0, p1))

        assert measurement._detector is not None, (
            "1 フレームの欠測でサイクル検出器が作られなくなった"
        )

    def test_baseline_divides_by_the_samples_actually_collected(self):
        """欠測があっても平均が偏らないこと。

        固定の窓幅で割ると、欠測のぶんだけ基準値が 0 に寄る。
        PushCycleDetector は基準値との差を 0.015 m の閾値で見るので、
        ずれると検出のタイミングが狂う。
        """
        measurement = _measurement()
        truth = _body_points(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)

        skipped = {6, 8, 10}
        for index in range(16):
            if index in skipped:
                nan_pixels = np.full_like(p0, np.nan)
                measurement.process(_pair_from_pixels(index * 33_000_000, nan_pixels, nan_pixels))
            else:
                measurement.process(_pair_from_pixels(index * 33_000_000, p0, p1))

        window = (
            measurement.config.baseline_last_frame
            - measurement.config.baseline_first_frame
            + 1
        )
        assert measurement._baseline_samples == window - len(skipped), (
            "欠測フレームが基準値のサンプル数に数えられている"
        )

    def test_cycle_axis_follows_the_configuration(self):
        """既存の RT_CYCLE_AXIS と同じく、既定は y 軸であること。"""
        assert MeasurementConfig().cycle_axis == "y"
        assert MeasurementConfig().cycle_axis_index == 1
        assert MeasurementConfig(cycle_axis="z").cycle_axis_index == 2

    def test_history_is_bounded(self):
        """長時間の計測でメモリを食い潰さないこと。"""
        measurement = _measurement()
        measurement.config.history_limit = 20
        truth = _body_points(0.0)
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)

        for index in range(200):
            measurement.process(_pair_from_pixels(index * 33_000_000, p0, p1))

        assert len(measurement.results) <= 20
        assert len(measurement._recent_points) <= 2
        for entries in measurement.storage.storage.values():
            assert len(entries) <= 2, "部位データが際限なく溜まっている"

    def test_reports_cycle_count_and_latest_impulses(self):
        measurement = _measurement()
        assert measurement.cycle_count == 0
        assert set(measurement.latest_cycle_work) == {
            "wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L",
        }
