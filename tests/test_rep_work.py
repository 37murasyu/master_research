"""回ごとの仕事を 1 フレームずつ Σ P_i·dt_i で積む（``app.hybrid.rep_work``）。

**なぜこのテストがあるか。**

混成の計測（``app.runners.network_measure``）は、サイクル確定のときに「サイクル全体の仕事率の和」に
**確定したフレームの dt だけ**を掛けていた（``network_measure.py:455``）。同期バッファは 100 ms を超える穴で
組を作らないので、組が抜けると次の組の dt が 2〜数倍になり、その dt で全フレームを積んで仕事が数倍に化けた。
仕事はフレームごとの dt で積み、100 ms を超える抜けはまたがない（その区間は積まない）。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.hybrid.rep_work import RepAccumulator, WorkSample
from app.net.sync_buffer import DEFAULT_GRID

PARTS = ("elbow_R", "wrist_R")


def _sample(p: float, dt: float = 1 / 30, **kw) -> WorkSample:
    return WorkSample(dt=dt, powers={part: p for part in PARTS}, **kw)


class TestSums:
    def test_each_frame_uses_its_own_dt(self):
        """Σ P_i·dt_i。dt が揺れても、最後の dt を全体に掛けない。"""
        acc = RepAccumulator(PARTS)
        dts = [1 / 30, 2 / 30, 1 / 30, 3 / 30]
        for dt in dts:
            assert acc.add(_sample(10.0, dt))
        assert acc.work()["elbow_R"].net == pytest.approx(10.0 * sum(dts))

    def test_positive_and_negative_are_split(self):
        """W+ = Σmax(P,0)·dt（論文 4.5.2 節のゲージの値）、W− = Σmin(P,0)·dt（負の値）、W± = W+ + W−。"""
        acc = RepAccumulator(PARTS)
        for p in (6.0, -3.0, 6.0, -3.0):
            acc.add(_sample(p, 0.05))
        work = acc.work()["wrist_R"]
        assert work.pos == pytest.approx(0.6)
        assert work.neg == pytest.approx(-0.3)
        assert work.net == pytest.approx(0.3)

    def test_a_step_over_100ms_is_not_integrated(self):
        """100 ms を超える dt（組の抜けの直後）はまたがない。その区間で腕がどう動いたかは分からない。"""
        acc = RepAccumulator(PARTS)
        assert acc.add(_sample(10.0, 1 / 30))
        assert not acc.add(_sample(10.0, DEFAULT_GRID.max_gap_s + 0.01))
        assert acc.work()["elbow_R"].net == pytest.approx(10.0 / 30)
        assert acc.skipped == 1

    def test_non_finite_power_adds_nothing(self):
        acc = RepAccumulator(PARTS)
        acc.add(WorkSample(dt=0.05, powers={"elbow_R": float("nan"), "wrist_R": 2.0}))
        assert acc.work()["elbow_R"].net == 0.0
        assert acc.work()["wrist_R"].net == pytest.approx(0.1)

    def test_each_part_counts_the_frames_it_integrated(self):
        """部位ごとに有限の仕事率を積んだフレーム数を持つ。0 なら「0 J」ではなく「積んでいない」（記録は空欄）。"""
        acc = RepAccumulator(PARTS)
        acc.add(WorkSample(dt=0.05, powers={"elbow_R": float("nan"), "wrist_R": 2.0}))
        acc.add(WorkSample(dt=0.05, powers={"wrist_R": 0.0}))
        acc.add(WorkSample(dt=0.5, powers={"elbow_R": 1.0, "wrist_R": 1.0}))   # dt が長すぎて積まない
        work = acc.work()
        assert (work["elbow_R"].frames, work["wrist_R"].frames) == (0, 2)
        assert acc.reset()["wrist_R"].frames == 2
        assert acc.work()["wrist_R"].frames == 0

    def test_reset_returns_the_rep_and_starts_over(self):
        acc = RepAccumulator(PARTS)
        acc.add(_sample(3.0, 0.1))
        done = acc.reset()
        assert done["elbow_R"].pos == pytest.approx(0.3)
        assert acc.work()["elbow_R"].pos == 0.0


class TestLookahead:
    def test_held_frames_join_the_rep_when_released(self):
        """関所が閉じている間の直近 5 フレームは輪に置き、開いたら流し込む（押し上げの立ち上がりを取りこぼさない）。"""
        acc = RepAccumulator(PARTS, lookahead=5)
        for p in range(1, 8):          # 7 フレーム置く。古い 2 つは輪から落ちる
            acc.hold(_sample(float(p), 0.1))
        assert acc.work()["elbow_R"].net == 0.0, "置いただけでは積まない"
        released = acc.release()
        assert len(released) == 5
        assert acc.work()["elbow_R"].net == pytest.approx(0.1 * (3 + 4 + 5 + 6 + 7))
        assert acc.release() == [], "2 回目は空"

    def test_dropped_hold_is_forgotten(self):
        acc = RepAccumulator(PARTS, lookahead=5)
        acc.hold(_sample(5.0))
        acc.drop_held()
        assert acc.release() == []


class TestSeries:
    def test_elbow_angle_and_torque_are_kept_for_the_filtered_energy(self):
        """肘の濾波 E±（``compute_cycle_energy_filtered``）の材料。積んだフレームだけ並べる。"""
        acc = RepAccumulator(PARTS)
        acc.add(_sample(1.0, theta={"elbow_R": 1.0}, tau_y={"elbow_R": -5.0}))
        acc.add(_sample(1.0, dt=0.5, theta={"elbow_R": 9.0}, tau_y={"elbow_R": 9.0}))   # 積まない
        acc.add(_sample(1.0, theta={"elbow_R": 1.1}, tau_y={"elbow_R": -6.0}))
        theta, tau = acc.series("elbow_R")
        np.testing.assert_allclose(theta, [1.0, 1.1])
        np.testing.assert_allclose(tau, [-5.0, -6.0])

    def test_series_are_bounded(self):
        """回が閉じないまま長く続いてもメモリを食い潰さない。"""
        acc = RepAccumulator(PARTS, max_series=10)
        for k in range(50):
            acc.add(_sample(1.0, theta={"elbow_R": float(k)}, tau_y={"elbow_R": 0.0}))
        theta, _ = acc.series("elbow_R")
        assert len(theta) == 10 and theta[-1] == 49.0
