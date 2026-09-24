"""先頭の窓が、片方の肘が見えないままでも一定時間で閉じることを確かめる（2026-09-24 のレビューの指摘）。

**なぜこのテストがあるか。**

先頭の窓は「左右の肩と左右の肘がすべて有限の組」を 30 組ためて、重力・前腕長・帯・回の区切りを決める。片方の肘が
一方のカメラで画面の外（余白 10% より外）に出続けると NaN になり、窓が永久に閉じず、トルクもゲージも最後まで出ない
（警告も無い）。本番の置き方の失敗で最も起きやすい形なので、肩が有限の組が ``window_timeout_frames``（既定 150 組＝約 5 秒）
たまっても窓が埋まらなければ、有限の値だけで窓を閉じる。見えない側の前腕長と帯は出さず、体格の検査は見える側の肩–肘で行う。
見えない側の長さは、窓が閉じた後に見えてから決める（``TestArmMeasuredAfterTheWindow``）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from app.gauge.tracker import GaugeTracker
from app.hybrid.ekf import EkfSettings
from app.runners.network_measure import ImplausibleBodyScale
from hybrid_pushup import SLOT, PushUp, body_cm, pushup_cm
from test_hybrid_window import ONE_RM, _measurement
from test_network_measure import _pair_from_pixels, _project


def _feed(measurement, frames, *, hide=()):
    truth = body_cm(0.0)
    p0 = _project(measurement.P0, truth).copy()
    p1 = _project(measurement.P1, truth)
    for landmark_id in hide:   # Mac の画面のずっと外（歪み補正で NaN になる）
        p0[SLOT[landmark_id]] = (-5000.0, -5000.0)
    for k in range(frames):
        measurement.process(_pair_from_pixels(round(k * 1e9 / 30), p0, p1))


def test_the_window_closes_when_one_elbow_stays_out_of_view():
    measurement = _measurement(one_rm=ONE_RM, tracker=GaugeTracker())
    _feed(measurement, 60, hide=(13,))          # 左肘（13）が見えない
    assert not measurement.window_closed, "既定の 150 組より前に閉じた"
    _feed(measurement, 200, hide=(13,))
    assert measurement.window_closed, "片方の肘が見えないと窓が永久に閉じない"
    assert measurement.window["fallback"] is True
    assert measurement.forearm_m["L"] is None and measurement.forearm_m["R"] is not None
    assert measurement.bands["elbow_R"].band is not None, "見えている右腕の帯が出ていない"
    assert measurement.bands["elbow_L"].band is None


def test_the_body_scale_check_uses_the_visible_side():
    """右肘が見えないときは、左の肩–肘で体格を確かめる（右が NaN だと必ず終了コード 3 になってしまう）。"""
    measurement = _measurement(one_rm=ONE_RM)
    _feed(measurement, 200, hide=(14,))         # 右肘（14）が見えない
    assert measurement.window_closed
    assert 0.1 < measurement.window["ekf_run_length_m"] < 0.6


def test_the_strict_window_is_used_when_everything_is_visible():
    measurement = _measurement(one_rm=ONE_RM)
    _feed(measurement, 40)
    assert measurement.window_closed
    assert measurement.window["fallback"] is False


def _pushups(measurement, motion, *, hide=(), hide_until_s=float("inf")):
    """押し上げを流す。``hide`` のランドマークは ``hide_until_s`` 秒まで Mac の画面のずっと外。"""
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(int(round(motion.duration_s * 30))):
            truth = pushup_cm(k / 30, motion)
            p0, p1 = _project(measurement.P0, truth).copy(), _project(measurement.P1, truth)
            if k / 30 < hide_until_s:
                for landmark_id in hide:
                    p0[SLOT[landmark_id]] = (-5000.0, -5000.0)
            results.append(measurement.process(_pair_from_pixels(round(k * 1e9 / 30), p0, p1)))
    return results


class TestArmMeasuredAfterTheWindow:
    """先頭の窓で長さが決まらなかった部位を、窓が閉じた後に見えてから決める（2026-09-24 のレビュー）。

    窓で決まらなかった腕は最後まで慣性が NaN で、トルクは NaN、仕事は 0.0 J と普通の値に見えた（左肘が先頭 6 s
    見えないと、見えるようになった後も左肘の W+ が 4 回とも 0.0。左手首が先頭 1.5 s 見えないと左手首の W+ が 0.0）。
    長さが決まっていない部位は有限のフレームで長さを集め続け、窓と同じ数たまったら慣性・前腕長・帯（ゲージにも
    渡し直す）・骨の長さの見張りの基準を埋める。
    """

    def test_an_elbow_seen_after_the_window_gets_its_arm(self):
        tracker = GaugeTracker()
        measurement = _measurement(one_rm=ONE_RM, tracker=tracker)
        results = _pushups(measurement, PushUp(reps=4, rest_s=8.0), hide=(13,), hide_until_s=6.0)
        assert measurement.window["fallback"] is True, "前提: 窓は左肘なしで閉じた"
        assert measurement.window["forearm_len_m"]["L"] is None
        assert measurement.upper_arm_m["L"] == pytest.approx(0.28, abs=0.01)
        assert measurement.forearm_m["L"] == pytest.approx(0.25, abs=0.01)
        assert np.isfinite(measurement._inertia["upper_arm_L"]).all()
        assert np.isfinite(measurement._inertia["forearm_L"]).all()
        late = [r for r in results if r.local_torques and r.t_ns / 1e9 > 7.5]
        assert late and all(np.isfinite(r.local_torques["elbow_L"]).all() for r in late)
        assert measurement.cycle_count == 4
        for cycle in measurement.cycles:
            left, right = cycle["parts"]["elbow_L"], cycle["parts"]["elbow_R"]
            assert left.pos == pytest.approx(right.pos, rel=0.05), "左右対称の押し上げで左肘だけ 0"
        band = measurement.bands["elbow_L"].band
        assert band is not None, "左腕の帯が出ていない"
        assert tracker.snapshot().parts["elbow_L"].band == pytest.approx(band), "ゲージに帯を渡し直していない"
        late_segments = measurement.summary()["late_segments"]
        assert set(late_segments) == {"upper_arm_L", "forearm_L"}
        assert all(6.0 * 30 <= frame <= 7.5 * 30 for frame in late_segments.values())

    def test_a_wrist_seen_after_the_window_gets_its_forearm(self):
        measurement = _measurement(one_rm=ONE_RM)
        _pushups(measurement, PushUp(reps=3, rest_s=4.0), hide=(15,), hide_until_s=1.5)
        assert measurement.window["fallback"] is False and measurement.window["forearm_len_m"]["L"] is None
        assert measurement.forearm_m["L"] == pytest.approx(0.25, abs=0.01)
        assert measurement.bands["wrist_L"].band is not None
        for cycle in measurement.cycles:
            left, right = cycle["parts"]["wrist_L"], cycle["parts"]["wrist_R"]
            assert right.pos > 0.5
            assert left.pos == pytest.approx(right.pos, rel=0.05), "左手首の W+ が 0"
        assert set(measurement.summary()["late_segments"]) == {"forearm_L"}

    def test_the_arm_guard_checks_the_late_arm(self):
        """骨の長さの見張りも、後から決めた長さを基準にする（決まらないうちは検査しない）。EKF を通すと 1 フレームの
        飛びは抑えられるので、見張りだけを見るために EKF を切る。"""
        measurement = _measurement(ekf=EkfSettings(enabled=False))
        _pushups(measurement, PushUp(reps=0, rest_s=8.0), hide=(13,), hide_until_s=6.0)
        truth = body_cm(0.0)
        truth[SLOT[13]] += (0.0, 20.0, 0.0)   # 左肘が 20 cm ずれた（三角測量の誤り）
        p0, p1 = _project(measurement.P0, truth), _project(measurement.P1, truth)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = measurement.process(_pair_from_pixels(round(8.0 * 1e9), p0, p1))
        assert result.arm_ok == {"L": False, "R": True}

    def test_a_segment_never_seen_is_left_without_inertia(self):
        """最後まで見えない部位の長さ・慣性は決めない（NaN の長さを慣性の回帰式に渡さない）。"""
        measurement = _measurement(one_rm=ONE_RM)
        _pushups(measurement, PushUp(reps=1, rest_s=6.0), hide=(13,))
        assert measurement.window_closed
        assert measurement.upper_arm_m["L"] is None and measurement.forearm_m["L"] is None
        assert np.isfinite(measurement._inertia["upper_arm_R"]).all()
        assert measurement.summary()["late_segments"] is None


def test_both_elbows_hidden_stops_with_an_elbow_message():
    """両肘が見えないまま窓が閉じると体格を確かめられず終了コード 3 で止める。案内は「座標の単位」ではなく肘。"""
    measurement = _measurement()
    with pytest.raises(ImplausibleBodyScale) as error:
        _pushups(measurement, PushUp(reps=0, rest_s=14.0), hide=(13, 14))
    assert "肘" in str(error.value) and "座標の単位" not in str(error.value)
