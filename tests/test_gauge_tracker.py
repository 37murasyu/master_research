"""子プロセス側のゲージの状態（GaugeTracker）と行の書き出し（GaugeTicker）を固定する。

**なぜこのテストがあるか。**

ゲージの値は今の回の正の仕事 W_pos = Σmax(P, 0)·dt（論文の定義、設計書 §6.4）。受信スレッド
（PhoneLink の on_pairs）が計測の積んだ仕事を置き、メインスレッド（LiveSession.step の周回）が行を書くので、
状態はロック 1 つで守り、snapshot は中身の写しを返す（書いている途中の辞書を encode しない）。

行は ``sys.stdout.write`` の 1 回で書く。``print`` は本体と改行を別々に書くので、受信スレッドの
``print`` と 1 行が混ざり、GUI の decode が壊れた行として捨てる（UI 側の求め）。1 行は 512 バイト
（macOS の PIPE_BUF、QProcess の MergedChannels）未満でなければならない。

回の区切り: ``close_rep`` は prev に今の回を移して now を 0 に戻し、rep を 1 増やす。
``discard_rep``（1.5 cm の揺れなど、押し上げでなかった回）は now だけ 0 に戻し、prev と rep は変えない。
"""

from __future__ import annotations

import math
import threading

import pytest

from app.gauge import protocol
from app.gauge.protocol import GaugeFrame, PartReading
from app.gauge.thresholds import PartBand, part_bands
from app.gauge.tracker import GaugeTicker, GaugeTracker

PARTS = ("elbow_L", "elbow_R", "wrist_L", "wrist_R")


def _bands():
    return part_bands(
        65.0, {"L": 0.25, "R": 0.25}, {"elbow_L": 15.0, "elbow_R": 15.0, "wrist_L": 8.235, "wrist_R": 8.235}
    )


class TestSetNow:
    """now は計測（``rep_work`` の W+ = Σmax(P, 0)·dt）かデモが置く。積むのは tracker ではない。"""

    def test_set_now_places_the_values(self):
        t = GaugeTracker()
        t.set_now({"elbow_L": 3.5})
        assert t.values()["elbow_L"] == 3.5
        assert t.values()["elbow_R"] == 0.0

    @pytest.mark.parametrize("value", [math.nan, math.inf, None, "x"])
    def test_bad_values_are_ignored(self, value):
        """NaN・非有限の値は置かない（1 回の NaN で now が NaN にならない）。"""
        t = GaugeTracker()
        t.set_now({"wrist_R": 0.5})
        t.set_now({"wrist_R": value})
        assert t.values()["wrist_R"] == pytest.approx(0.5)

    def test_an_unknown_part_is_ignored(self):
        t = GaugeTracker()
        t.set_now({"shoulder_L": 100.0})
        assert set(t.values()) == set(PARTS)
        assert all(v == 0.0 for v in t.values().values())

    def test_a_subset_of_parts(self):
        t = GaugeTracker(parts=("elbow_L",))
        t.set_now({"elbow_R": 10.0})
        assert t.values() == {"elbow_L": 0.0}

    def test_set_now_overwrites_the_values_for_the_demo(self):
        t = GaugeTracker(source="demo")
        t.set_now({"elbow_L": 10.0})
        t.set_now({"elbow_L": 42.0, "wrist_L": 3.0, "unknown": 1.0})
        assert t.values() == {"elbow_L": 42.0, "elbow_R": 0.0, "wrist_L": 3.0, "wrist_R": 0.0}


class TestReps:
    def test_close_rep_moves_now_to_prev(self):
        t = GaugeTracker()
        t.set_now({"elbow_L": 20.0})
        t.close_rep()
        frame = t.snapshot()
        assert frame.rep == 1
        assert frame.parts["elbow_L"] == PartReading(now=0.0, prev=20.0, band=None, w1rm=None)
        t.set_now({"elbow_L": 5.0})
        t.close_rep()
        assert t.snapshot().parts["elbow_L"].prev == 5.0
        assert t.snapshot().rep == 2

    def test_discard_rep_keeps_prev_and_the_count(self):
        """押し上げでなかった回は数えない。直前の回の値（prev）も消さない。"""
        t = GaugeTracker()
        t.set_now({"elbow_L": 20.0})
        t.close_rep()
        t.set_now({"elbow_L": 7.0})
        t.discard_rep()
        frame = t.snapshot()
        assert frame.rep == 1
        assert frame.parts["elbow_L"].now == 0.0 and frame.parts["elbow_L"].prev == 20.0

    def test_prev_is_none_before_the_first_rep(self):
        """PartReading の約束: prev が None は「まだ 1 回も完了していない」。"""
        assert GaugeTracker().snapshot().parts["wrist_L"].prev is None


class TestSnapshot:
    def test_the_frame_has_all_parts_with_bands(self):
        t = GaugeTracker()
        t.set_bands(_bands())
        t.set_link("connected")
        frame = t.snapshot()
        assert isinstance(frame, GaugeFrame)
        assert frame.link == "connected" and frame.source == "measure" and frame.rep == 0
        assert tuple(frame.parts) == PARTS
        b = _bands()["elbow_L"]
        assert frame.parts["elbow_L"].band == b.band and frame.parts["elbow_L"].w1rm == b.w1rm

    def test_a_part_without_a_band(self):
        """帯を出せない部位（前腕長の範囲外・1RM が無い）は band も w1rm も None のまま出す。"""
        t = GaugeTracker()
        bands = dict(_bands())
        bands["wrist_R"] = PartBand(None, None, None, 0.25, "1RM が無い")
        t.set_bands(bands)
        reading = t.snapshot().parts["wrist_R"]
        assert reading.band is None and reading.w1rm is None

    def test_bands_not_given_are_cleared(self):
        t = GaugeTracker()
        t.set_bands(_bands())
        t.set_bands({"elbow_L": _bands()["elbow_L"]})
        frame = t.snapshot()
        assert frame.parts["elbow_L"].band is not None
        assert frame.parts["elbow_R"].band is None

    def test_the_snapshot_is_a_copy(self):
        t = GaugeTracker()
        frame = t.snapshot()
        t.set_now({"elbow_L": 10.0})
        assert frame.parts["elbow_L"].now == 0.0

    def test_link_accepts_a_bool_or_a_state(self):
        t = GaugeTracker()
        assert t.snapshot().link == "waiting"
        t.set_link(True)
        assert t.snapshot().link == "connected"
        t.set_link("waiting")
        assert t.snapshot().link == "waiting"
        with pytest.raises(ValueError):
            t.set_link("lost")

    @pytest.mark.parametrize("source", ["measure", "demo", "replay"])
    def test_sources(self, source):
        """再生の検証では source="replay" を出す（GUI が「再生」の印を出す）。"""
        assert GaugeTracker(source=source).snapshot().source == source

    def test_an_unknown_source_is_an_error(self):
        with pytest.raises(ValueError):
            GaugeTracker(source="usb")


class TestThreads:
    def test_updates_from_another_thread_are_not_lost(self):
        """受信スレッドが 1 万回置く間にメインスレッドが snapshot しても、合計が合う。"""
        t = GaugeTracker()
        n = 10_000

        def worker():
            for k in range(1, n + 1):
                t.set_now({"elbow_L": 0.03 * k})
                t.set_now({"wrist_R": 0.01 * k})

        th = threading.Thread(target=worker)
        th.start()
        seen = []
        while th.is_alive():
            frame = t.snapshot()
            seen.append(frame.parts["elbow_L"].now)
            protocol.encode(frame)
        th.join()
        assert t.values()["elbow_L"] == pytest.approx(n * 0.03)
        assert t.values()["wrist_R"] == pytest.approx(n * 0.01)
        assert seen == sorted(seen), "snapshot が途中で減った（ロックの外で書いている）"


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TestTicker:
    def test_each_tick_writes_exactly_one_line(self):
        """write は 1 行につき 1 回。print のように本体と改行を分けて書かない。"""
        t = GaugeTracker()
        t.set_bands(_bands())
        written: list[str] = []
        flushed: list[int] = []
        ticker = GaugeTicker(t, write=written.append, flush=lambda: flushed.append(1))
        assert ticker.tick() is True
        assert ticker.tick() is True  # 既定の period_s=0 は毎回出す（子のメインループごと）
        assert len(written) == 2 and len(flushed) == 2
        for line in written:
            assert line.startswith(protocol.PREFIX) and line.endswith("\n") and line.count("\n") == 1

    def test_the_period_limits_the_rate(self):
        clock = _Clock()
        written: list[str] = []
        ticker = GaugeTicker(GaugeTracker(), write=written.append, flush=lambda: None, period_s=0.1, clock=clock)
        assert ticker.tick() is True
        clock.now = 0.05
        assert ticker.tick() is False
        assert ticker.tick(force=True) is True  # 終わりに最後の状態を必ず出す
        clock.now = 0.2
        assert ticker.tick() is True
        assert len(written) == 3

    def test_a_newline_is_added_when_the_encoder_forgets_it(self):
        written: list[str] = []
        ticker = GaugeTicker(GaugeTracker(), encode=lambda f: "@@GAUGE {}", write=written.append, flush=lambda: None)
        ticker.tick()
        assert written == ["@@GAUGE {}\n"]

    def test_the_default_writes_to_the_current_stdout(self, capsys):
        """既定の出力先は呼んだ時点の sys.stdout（import 時に束縛しない。pytest の捕獲や差し替えに追随する）。"""
        GaugeTicker(GaugeTracker()).tick()
        out = capsys.readouterr().out
        assert out.startswith(protocol.PREFIX) and out.endswith("\n")

    def test_a_full_line_is_under_512_bytes(self):
        """4 部位・帯・W_1RM・大きな値と回数の入った最悪に近い行でも PIPE_BUF 未満。"""
        t = GaugeTracker(source="replay")
        t.set_bands(_bands())
        t.set_link("connected")
        for part in PARTS:
            t.set_now({part: 123456.789})
        for _ in range(9999):
            t.close_rep()
        for part in PARTS:
            t.set_now({part: 98765.4321})
        written: list[str] = []
        GaugeTicker(t, write=written.append, flush=lambda: None).tick()
        assert len(written[0].encode("utf-8")) < 512


class TestContract:
    def test_decode_keeps_the_four_parts_and_bands(self):
        """子が書いた行を GUI の decode が読んで、4 部位・帯・W_1RM・now・prev・rep を失わない。"""
        t = GaugeTracker()
        t.set_bands(_bands())
        t.set_link("connected")
        t.set_now({"elbow_L": 50.0})
        t.close_rep()
        t.set_now({"elbow_L": 12.34})
        frame = protocol.decode(protocol.encode(t.snapshot()))
        assert frame is not None
        assert tuple(frame.parts) == PARTS
        assert frame.rep == 1 and frame.link == "connected" and frame.source == "measure"
        assert frame.parts["elbow_L"].now == pytest.approx(12.3)
        assert frame.parts["elbow_L"].prev == pytest.approx(50.0)
        for part, b in _bands().items():
            lo, hi = frame.parts[part].band
            assert lo == pytest.approx(b.band[0], abs=0.05) and hi == pytest.approx(b.band[1], abs=0.05)
            assert frame.parts[part].w1rm == pytest.approx(b.w1rm, abs=0.05)
