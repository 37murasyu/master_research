"""混成の計測の力学の関所と回の区切り（``RepDetector``、計画の T9）。

**なぜこのテストがあるか。**

混成の計測は USB 経路と同じ ``PushCycleDetector``（左肩の y 座標の往復）で回を区切っていたが、実行時の座標の y は
奥行きで、手を固定して体幹が上下するだけの押し上げでは 1 回も回が閉じなかった（合成データで確認）。回が閉じないと
ゲージが回ごとに 0 に戻らない。また座っている間も仕事を積んでおり、雑音 5 mm で肘の W_pos が 1 回 72〜160 J に
なって W_0.85（56.9 J）を超え、過負荷と誤表示した（論文 5.4 の過大評価の機序）。

そこで高さ h = 肩の中点・上向き u で押し上げを見る ``RepDetector`` を回の区切りと関所に使う:
トルクと仕事率は関所によらず毎フレーム計算して記録し、仕事とゲージには関所が開いている間だけ積む
（開く直前の 5 フレームは輪に置いて、開いたときに流し込む）。``HYBRID_DYN_GATE=0`` なら常に開いた扱い。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from app.gauge.tracker import GaugeTracker
from app.hybrid.ekf import EkfSettings
from app.runners.network_measure import MeasurementConfig, NetworkMeasurement
from config import pose_keypoints
from hybrid_pushup import PushUp, run
from test_network_measure import _stereo_projections

MOTION = PushUp(reps=3)


def _measurement(*, gate=True, ekf=True, tracker=None) -> NetworkMeasurement:
    P0, P1 = _stereo_projections()
    config = MeasurementConfig(body_mass_kg=65.0, dyn_gate=gate, ekf=EkfSettings(enabled=ekf))
    return NetworkMeasurement(P0, P1, pose_keypoints, config, tracker=tracker)


def _run(measurement, motion=MOTION, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run(measurement, motion, **kw)


class TestReps:
    def test_three_push_ups_are_three_reps(self):
        measurement = _measurement()
        results = _run(measurement)
        assert measurement.cycle_count == 3
        assert sum(r.cycle_detected for r in results) == 3
        closes = [r.t_ns / 1e9 for r in results if r.cycle_detected]
        for rep, t in enumerate(closes):
            start = MOTION.rest_s + rep * MOTION.period_s
            lowered = start + MOTION.rise_s + MOTION.hold_s + MOTION.lower_s
            assert lowered - 0.2 < t < start + MOTION.period_s, "下げ終わってから次の押し上げまでに閉じる"

    def test_sitting_records_torque_but_integrates_no_work(self):
        tracker = GaugeTracker()
        measurement = _measurement(tracker=tracker)
        results = _run(measurement, PushUp(reps=0), seconds=4.0)
        seated = [r for r in results if r.local_torques]
        assert seated, "座っている間もトルクは記録する"
        assert not any(r.dyn_active for r in results)
        assert all(w.pos == 0.0 and w.neg == 0.0 for w in measurement.rep_work.work().values())
        assert all(v == 0.0 for v in tracker.values().values())

    def test_the_elbow_work_of_one_rep_is_about_22_joules(self):
        """65 kg・13 cm の持ち上げで肘の W+ ≈ 22 J（計画の見積もり）、下げで W− ≈ −22 J。"""
        measurement = _measurement()
        _run(measurement)
        assert len(measurement.cycles) == 3
        for cycle in measurement.cycles:
            elbow = cycle["parts"]["elbow_R"]
            assert elbow.pos == pytest.approx(22.0, rel=0.2)
            assert elbow.neg == pytest.approx(-22.0, rel=0.2)

    def test_rigid_arms_moved_up_and_down_do_no_elbow_work(self):
        """肘角を保ったまま腕ごと上下しても、肘の仕事は ≈ 0（相対角速度で積む）。回は閉じる。"""
        measurement = _measurement()
        _run(measurement, arms_rigid=True)
        assert measurement.cycle_count == 3
        for cycle in measurement.cycles:
            assert abs(cycle["parts"]["elbow_R"].pos) < 0.5
            assert abs(cycle["parts"]["elbow_L"].neg) < 0.5

    def test_the_lookahead_catches_the_start_of_the_rise(self):
        """関所は高さ＋2 cm（この合成では上げ始めから 9 フレーム目）で開く。直前の 5 フレームを流し込むので、
        上げの仕事をほぼ取りこぼさない（落ちるのは動き出しの 3 フレーム、約 1%）。

        比べる範囲は上げ始めから回が閉じるフレームまで。着座の後は EKF の行き過ぎの揺り戻しで小さな正の仕事
        （約 1 J）が出るが、関所の外なので積まないのが正しい。
        """
        gated = _measurement()
        _run(gated, PushUp(reps=1))
        closed = gated.cycles[0]["t_ns"]
        measurement = _measurement(gate=False)
        results = _run(measurement, PushUp(reps=1))
        truth = sum(max(r.powers.get("elbow_R", 0.0), 0.0) * r.dt_s for r in results
                    if PushUp().rest_s * 1e9 <= r.t_ns <= closed)
        assert gated.cycles[0]["parts"]["elbow_R"].pos == pytest.approx(truth, rel=0.03)

    def test_without_the_gate_sitting_is_integrated_but_reps_still_close(self):
        measurement = _measurement(gate=False)
        results = _run(measurement)
        assert measurement.cycle_count == 3
        assert all(r.dyn_active for r in results if r.local_torques)

    def test_the_height_and_the_rep_index_are_reported(self):
        measurement = _measurement()
        results = _run(measurement, PushUp(reps=1))
        top = max(r.height_m for r in results if np.isfinite(r.height_m))
        assert top - measurement.baseline_height_m == pytest.approx(0.13, abs=0.005)
        assert [r.rep for r in results if r.cycle_detected] == [0]
        assert results[-1].rep == 1


class TestTracker:
    def test_the_gauge_rises_during_the_push_up(self):
        tracker = GaugeTracker()
        measurement = _measurement(tracker=tracker)
        motion = PushUp(reps=1)
        seen = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from hybrid_pushup import pushup_pairs
            for pair in pushup_pairs(measurement, motion):
                measurement.process(pair)
                seen.append((pair.t_ns / 1e9, tracker.values()["elbow_R"]))
        during = [v for t, v in seen if motion.rest_s + 0.3 < t < motion.rest_s + motion.rise_s]
        assert during and during[-1] > 5.0, "押し上げの途中でゲージが増えている"


class TestCloseRep:
    """回の確定（計画の T10）: ゲージは回ごとに 0 に戻り、直前の回の W_pos を prev に残す。記録には W±・W+・W−・スコア。"""

    ONE_RM = {"elbow_L": 20.0, "elbow_R": 22.0, "wrist_L": 8.0, "wrist_R": 9.0}

    def _closed_once(self):
        tracker = GaugeTracker()
        P0, P1 = _stereo_projections()
        config = MeasurementConfig(body_mass_kg=65.0, one_rm=self.ONE_RM)
        measurement = NetworkMeasurement(P0, P1, pose_keypoints, config, tracker=tracker)
        results = _run(measurement, PushUp(reps=1))
        return measurement, tracker, results

    def test_the_gauge_moves_now_to_prev(self):
        measurement, tracker, _ = self._closed_once()
        frame = tracker.snapshot()
        assert frame.rep == 1
        elbow = measurement.cycles[0]["parts"]["elbow_R"]
        assert frame.parts["elbow_R"].prev == pytest.approx(elbow.pos)
        assert frame.parts["elbow_R"].now == 0.0, "着座の後は関所が閉じていて積まない"
        assert frame.parts["wrist_L"].prev == pytest.approx(measurement.cycles[0]["parts"]["wrist_L"].pos)

    def test_the_closing_frame_carries_the_rep_values(self):
        measurement, _, results = self._closed_once()
        closing = next(r for r in results if r.cycle_detected)
        elbow = closing.cycle_parts["elbow_R"]
        assert closing.cycle_work_j["elbow_R"] == pytest.approx(elbow.pos + elbow.neg)
        assert closing.cycle_w1rm["elbow_R"] == pytest.approx(measurement.bands["elbow_R"].w1rm)
        assert closing.cycle_w1rm["shoulder_R"] is None, "肩には 1RM 相当の理論仕事が無い"


class TestArmLengthGuard:
    """腕の長さの安全策（2026-09-24 06:31 の追加）。

    2026-09-23 の実機の記録（置き方が悪い）の 20〜90 s を再生すると、EKF ありでも右腕の |τy| が 95% で 9,000 N·m、
    最大 24 万 N·m に跳ねた。三角測量の誤りが続けて起きると EKF では吸収しきれず、ゲージが偽の過負荷を出す。
    先頭の窓で決めた上腕長・前腕長から ±25% を超えてずれた腕は、そのフレーム（と、差分で速度・加速度にその
    フレームを使う後の 2 フレーム）の仕事率を回の仕事とゲージに積まない。トルクは今どおり記録する。
    """

    JUMP = range(75, 78)   # 上げの途中（関所が開いた後）の 3 フレーム

    def _run(self, jump: bool, tolerance: float = 0.25):
        from hybrid_pushup import SLOT, pushup_pairs

        tracker = GaugeTracker()
        P0, P1 = _stereo_projections()
        config = MeasurementConfig(body_mass_kg=65.0, ekf=EkfSettings(enabled=False),
                                   arm_length_tolerance=tolerance)
        measurement = NetworkMeasurement(P0, P1, pose_keypoints, config, tracker=tracker)
        original = measurement.points_3d

        def jumped(pair):
            points = original(pair).copy()
            points[SLOT[14], 0] += 3.0   # 右肘を 3 m 飛ばす
            return points

        results, peaks = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, pair in enumerate(pushup_pairs(measurement, PushUp(reps=1))):
                measurement.points_3d = jumped if (jump and k in self.JUMP) else original
                results.append(measurement.process(pair))
                peaks.append(tracker.values()["elbow_R"])
        return measurement, results, max(peaks)

    def test_a_jumping_arm_adds_no_work(self):
        clean, _, clean_peak = self._run(jump=False)
        guarded, results, peak = self._run(jump=True)
        right, right_clean = guarded.cycles[0]["parts"]["elbow_R"], clean.cycles[0]["parts"]["elbow_R"]
        assert right.pos <= right_clean.pos + 1e-9, "飛んだ腕の仕事が増えた"
        assert right.pos > 0.7 * right_clean.pos, "落とすのは飛んだ前後の数フレームだけ"
        assert peak <= clean_peak + 1e-9, "ゲージの now が飛びで増えた"
        left, left_clean = guarded.cycles[0]["parts"]["elbow_L"], clean.cycles[0]["parts"]["elbow_L"]
        assert left.pos == pytest.approx(left_clean.pos, rel=1e-9), "もう片方の腕は影響を受けない"
        flagged = [r.grid_index for r in results if not r.arm_ok["R"]]
        assert flagged == [75, 76, 77, 78, 79], "飛んだフレームと、差分にそれを使う後の 2 フレーム"
        assert all(r.arm_ok["L"] for r in results)
        assert all(results[k].local_torques for k in self.JUMP), "トルクは記録する"
        assert guarded.arm_guard_rejected == {"L": 0, "R": 5}

    def test_without_the_guard_the_jump_inflates_the_work(self):
        """安全策を切る（許容 0）と、同じ飛びで右肘の仕事が大きく振れる（このテストの前提の確認。この合成では
        W− が −25 J から −140 J 前後に振り切れる。飛びの向きしだいで W+ 側にも出る）。"""
        clean, _, _ = self._run(jump=False)
        unguarded, results, _ = self._run(jump=True, tolerance=0.0)
        work, work_clean = unguarded.cycles[0]["parts"]["elbow_R"], clean.cycles[0]["parts"]["elbow_R"]
        assert abs(work.pos - work_clean.pos) + abs(work.neg - work_clean.neg) > 2 * work_clean.pos
        assert all(r.arm_ok["R"] for r in results)
