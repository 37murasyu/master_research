"""混成の押し上げの回の区切り（RepDetector、力学の関所）を固定する。

**なぜこのテストがあるか。**

混成の回の検出は USB の ``PushCycleDetector`` を使っていたが、上下だけの押し上げでは 0 回だった
（合成データで確認）。回が閉じないとゲージの now が回ごとに 0 に戻らず、prev も出ない。
また関所が無いと座っている間の雑音の仕事（論文 5.4 の過大評価の機序）が積まれ、雑音 5 mm で肘の
W_pos が 1 回 72〜160 J になって W_0.85 = 56.9 J を超え、過負荷と誤表示する。

そこで肩の中点の「重力の上向き」への射影（高さ, m）で開閉する状態機械にした:

- 開く: 高さが基準 + 2 cm を超える、または上向きの速さが 0.10 m/s を 2 フレーム続けて超える
- 閉じる: 開いてから 0.3 s 以上経ち、基準 + 1 cm 以内に 3 フレーム続いたとき。30 s を超えても閉じる
- 開いている間の最大の持ち上げが 3 cm 未満なら、回ではなかったとして捨てる（DISCARDED）
- 高さが NaN のフレームは状態を変えない
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.hybrid.rep_detector import RepConfig, RepDetector, RepEvent

DT = 1.0 / 30.0
BASE = 0.80  # 座った肩の中点の高さ [m]（値そのものに意味は無い）


def _rest(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds / DT)))


def _ramp(amplitude: float, seconds: float, up: bool = True) -> np.ndarray:
    """余弦の滑らかな上げ下げ（速さは両端で 0）。"""
    t = np.arange(1, int(round(seconds / DT)) + 1) * DT
    s = 0.5 * (1.0 - np.cos(np.pi * t / seconds))
    return amplitude * (s if up else 1.0 - s)


def _pushups(n: int, lift: float, up_s: float = 1.0, hold_s: float = 0.5, down_s: float = 1.0) -> np.ndarray:
    parts = [_rest(2.0)]
    for _ in range(n):
        parts += [_ramp(lift, up_s), np.full(int(hold_s / DT), lift), _ramp(lift, down_s, up=False), _rest(0.5)]
    parts.append(_rest(1.0))
    return np.concatenate(parts)


def _run(rise: np.ndarray, *, speed="diff", config=None, noise=0.0, seed=0):
    """高さの列を流して、フレームごとの事象を返す。speed="diff" は差分の速さ、None は検出器に任せる。"""
    rng = np.random.default_rng(seed)
    height = BASE + rise + noise * rng.standard_normal(len(rise))
    det = RepDetector(BASE, config or RepConfig())
    events = []
    prev = height[0]
    for h in height:
        v = (h - prev) / DT if speed == "diff" else None
        prev = h
        events.append(det.update(h, v, DT))
    return det, events, height


def _count(events, kind):
    return sum(1 for e in events if e is kind)


class TestPushUps:
    def test_three_13cm_lifts_are_three_reps(self):
        """計画の見積もりの押し上げ（肩 13 cm）を 3 回。回が 3 つ閉じる。"""
        det, events, _ = _run(_pushups(3, 0.13), noise=0.001)
        assert _count(events, RepEvent.OPENED) == 3
        assert _count(events, RepEvent.CLOSED) == 3
        assert _count(events, RepEvent.DISCARDED) == 0
        assert not det.is_open

    def test_the_detector_can_estimate_the_speed_itself(self):
        """速さを渡さなければ（None）、高さの差分を平滑して使う。回の数は変わらない。"""
        _, events, _ = _run(_pushups(3, 0.13), speed=None, noise=0.001)
        assert _count(events, RepEvent.CLOSED) == 3

    def test_a_fast_lift_opens_before_two_centimetres(self):
        """速い持ち上げは高さの条件（+2 cm）より前に速さで開く（押し上げの出だしの仕事を落とさない）。"""
        rise = np.concatenate([_rest(1.0), _ramp(0.13, 0.6), np.full(30, 0.13)])
        _, events, height = _run(rise)
        opened = events.index(RepEvent.OPENED)
        assert height[opened] - BASE < 0.02

    @pytest.mark.parametrize("speed", ["diff", None])
    @pytest.mark.parametrize("seconds", [0.6, 1.0])
    def test_the_lookback_covers_the_start(self, speed, seconds):
        """開いた時点から先読みの幅（5 フレーム）さかのぼれば、取りこぼす持ち上げは全体の 5% 未満。

        自前の速さ（平滑して約 1 フレーム遅れる）でも同じ。持ち上げの仕事の大半は位置エネルギーなので、
        取りこぼす高さの割合がそのまま取りこぼす仕事の目安になる。
        """
        rise = np.concatenate([_rest(1.0), _ramp(0.13, seconds), np.full(30, 0.13)])
        _, events, height = _run(rise, speed=speed)
        first_counted = events.index(RepEvent.OPENED) - RepConfig().lookback_frames
        assert height[first_counted] - BASE < 0.05 * 0.13

    def test_the_rep_closes_back_on_the_seat(self):
        """閉じるのは座面に戻ってから（基準 + 1 cm 以内が 3 フレーム）。持ち上げたままでは閉じない。"""
        rise = np.concatenate([_rest(1.0), _ramp(0.10, 0.8), np.full(90, 0.10), _ramp(0.10, 0.8, up=False), _rest(1.0)])
        _, events, height = _run(rise)
        closed = events.index(RepEvent.CLOSED)
        assert all(h - BASE <= 0.01 for h in height[closed - 2 : closed + 1])
        assert _count(events, RepEvent.CLOSED) == 1


class TestNotAPushUp:
    def test_sitting_with_3mm_noise_for_60_seconds_is_no_rep(self):
        """座ったまま σ 3 mm の雑音（生の差分の速さは σ≈0.13 m/s）でも回は 1 つも閉じない。"""
        _, events, _ = _run(_rest(60.0), noise=0.003)
        assert _count(events, RepEvent.CLOSED) == 0

    def test_sitting_with_the_detectors_own_speed_does_not_even_open(self):
        """検出器が平滑した速さなら、雑音で開きもしない（ゲージの now がちらつかない）。"""
        _, events, _ = _run(_rest(60.0), speed=None, noise=0.003)
        assert _count(events, RepEvent.OPENED) == 0

    def test_a_quick_15mm_bob_is_discarded(self):
        """速い 1.5 cm の揺れは速さで開くが、持ち上げが 3 cm に届かないので捨てる。"""
        rise = np.concatenate([_rest(1.0), _ramp(0.015, 0.12), _ramp(0.015, 0.12, up=False), _rest(1.0)])
        _, events, _ = _run(rise)
        assert _count(events, RepEvent.OPENED) == 1
        assert _count(events, RepEvent.DISCARDED) == 1
        assert _count(events, RepEvent.CLOSED) == 0

    def test_a_slow_15mm_sway_does_not_open(self):
        rise = np.concatenate([_rest(1.0), _ramp(0.015, 1.5), _ramp(0.015, 1.5, up=False), _rest(1.0)])
        _, events, _ = _run(rise)
        assert _count(events, RepEvent.OPENED) == 0


class TestRobustness:
    def test_nan_heights_do_not_change_the_state(self):
        det = RepDetector(BASE)
        assert det.update(BASE + 0.05, 0.0, DT) is RepEvent.OPENED
        for _ in range(100):
            assert det.update(math.nan, math.nan, DT) is RepEvent.NONE
        assert det.is_open
        # 閉じた状態でも同じ
        det2 = RepDetector(BASE)
        assert det2.update(math.nan, 5.0, DT) is RepEvent.NONE
        assert not det2.is_open

    def test_nan_speed_is_not_a_trigger(self):
        det = RepDetector(BASE)
        for _ in range(5):
            assert det.update(BASE, math.nan, DT) is RepEvent.NONE

    @pytest.mark.parametrize("dt", [5.0, 1e6, 0.0, -1.0, math.nan, math.inf])
    def test_odd_dt_does_not_crash(self, dt):
        """組の抜けで dt が大きく跳んでも落ちない。0・負・NaN は時間が進まないものとして扱う。"""
        det = RepDetector(BASE)
        det.update(BASE + 0.05, 0.0, DT)
        event = det.update(BASE + 0.05, 0.0, dt)
        assert event in (RepEvent.NONE, RepEvent.CLOSED)

    def test_a_long_gap_closes_an_open_rep(self):
        det = RepDetector(BASE)
        det.update(BASE + 0.05, 0.0, DT)
        assert det.update(BASE + 0.05, 0.0, 31.0) is RepEvent.CLOSED

    def test_an_open_rep_closes_after_30_seconds(self):
        """持ち上げたまま戻らなくても 30 s で閉じる（ゲージが永久に開いたままにならない）。"""
        det = RepDetector(BASE)
        events = [det.update(BASE + 0.06, 0.0, DT) for _ in range(int(35.0 / DT))]
        closed = events.index(RepEvent.CLOSED)
        assert closed * DT == pytest.approx(30.0, abs=0.1)

    def test_min_open_time(self):
        """開いてから 0.3 s は閉じない（押し上げの出だしで高さがまだ基準の近くにあるため）。"""
        det = RepDetector(BASE)
        det.update(BASE + 0.05, 0.0, DT)
        events = [det.update(BASE, 0.0, DT) for _ in range(20)]
        first = next(i for i, e in enumerate(events) if e is not RepEvent.NONE)
        assert (first + 1) * DT >= 0.3 - 1e-9
        assert events[first] is RepEvent.CLOSED  # 5 cm 持ち上げたので捨てない

    def test_the_config_defaults(self):
        c = RepConfig()
        assert (c.open_rise_m, c.open_speed_mps, c.close_band_m, c.close_frames) == (0.02, 0.10, 0.01, 3)
        assert (c.min_lift_m, c.min_open_s, c.max_open_s, c.lookback_frames) == (0.03, 0.3, 30.0, 5)
