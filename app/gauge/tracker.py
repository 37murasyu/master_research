"""子プロセス側のゲージの状態と、@@GAUGE の行の書き出し。

値は今の回の正の仕事 W_pos = Σmax(P, 0)·dt [J]（論文の定義、設計書 §6.4）。積むのは計測の
``app.hybrid.rep_work.RepAccumulator`` で、受信スレッド（PhoneLink の on_pairs）がその値を ``set_now`` で置き、
メインスレッド（子のメインループ）が ``GaugeTicker.tick`` で行を書く。
状態はロック 1 つで守り、``snapshot`` は写し（``protocol.GaugeFrame``）を返す。

行は ``sys.stdout.write`` の 1 回で書く。``print`` は本体と改行を別々に書くので、受信スレッドの
``print`` と 1 行が混ざりうる。丸め（小数 1 桁）・NaN の null・512 バイト未満は ``protocol.encode`` の約束。
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

from app.gauge import protocol
from app.gauge.protocol import GaugeFrame, PartReading, finite_or_none

if TYPE_CHECKING:  # 実行時は読まない（thresholds は重い import を持つ）
    from app.gauge.thresholds import PartBand


class GaugeTracker:
    """4 部位の今の回の W_pos・直前の回・帯・回数・接続の状態。すべてのメソッドはスレッド安全。"""

    def __init__(self, parts: Iterable[str] = protocol.PART_NAMES, *, source: str = "measure"):
        if source not in protocol.SOURCES:
            raise ValueError(f"source は {protocol.SOURCES} のどれか: {source!r}")
        self._lock = threading.Lock()
        self._parts = tuple(parts)
        self._source = source
        self._now: dict[str, float] = {p: 0.0 for p in self._parts}
        self._prev: dict[str, float | None] = {p: None for p in self._parts}
        self._bands: dict[str, PartBand | None] = {p: None for p in self._parts}
        self._rep = 0
        self._link = "waiting"

    @property
    def parts(self) -> tuple[str, ...]:
        return self._parts

    @property
    def rep(self) -> int:
        with self._lock:
            return self._rep

    def set_now(self, values: Mapping[str, float]) -> None:
        """今の回の値を置く（計測は ``rep_work`` の W+、デモは置く値）。対象外の部位・非有限の値は無視する。"""
        with self._lock:
            for part, value in values.items():
                v = finite_or_none(value)
                if part in self._now and v is not None:
                    self._now[part] = v

    def close_rep(self) -> None:
        """回を確定する: prev ← now、now ← 0、rep += 1。"""
        with self._lock:
            for part in self._parts:
                self._prev[part] = self._now[part]
                self._now[part] = 0.0
            self._rep += 1

    def discard_rep(self) -> None:
        """押し上げでなかった回を捨てる: now ← 0。prev と rep は変えない。"""
        with self._lock:
            for part in self._parts:
                self._now[part] = 0.0

    def set_bands(self, bands: Mapping[str, "PartBand | None"]) -> None:
        """部位ごとの帯（``thresholds.part_bands`` の戻り値）。渡されなかった部位は帯なしにする。"""
        with self._lock:
            self._bands = {p: bands.get(p) for p in self._parts}

    def set_link(self, state: str | bool) -> None:
        """Pixel との接続: "waiting" か "connected"（真偽値なら True が connected）。"""
        if isinstance(state, bool):
            state = "connected" if state else "waiting"
        if state not in protocol.LINKS:
            raise ValueError(f"link は {protocol.LINKS} のどれか: {state!r}")
        with self._lock:
            self._link = state

    def values(self) -> dict[str, float]:
        with self._lock:
            return dict(self._now)

    def snapshot(self) -> GaugeFrame:
        """今の状態の写し。丸めは ``protocol.encode`` がする。"""
        with self._lock:
            parts = {}
            for part in self._parts:
                band = self._bands[part]
                parts[part] = PartReading(
                    now=self._now[part],
                    prev=self._prev[part],
                    band=None if band is None else band.band,
                    w1rm=None if band is None else band.w1rm,
                )
            return GaugeFrame(link=self._link, rep=self._rep, source=self._source, parts=parts)


class GaugeTicker:
    """``tick`` ごとに @@GAUGE の行を 1 回の ``write`` で出す。

    既定の ``period_s=0`` は毎回出す（子のメインループ＝約 30 Hz ごと）。``write``・``flush`` を省くと、
    呼んだ時点の ``sys.stdout`` に書く（import 時に束縛すると、差し替えた標準出力に追随しない）。
    """

    def __init__(
        self,
        tracker: GaugeTracker,
        *,
        encode: Callable[[GaugeFrame], str] = protocol.encode,
        write: Callable[[str], object] | None = None,
        flush: Callable[[], object] | None = None,
        period_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._tracker = tracker
        self._encode = encode
        self._write = write
        self._flush = flush
        self._period_s = float(period_s)
        self._clock = clock
        self._last: float | None = None

    def tick(self, force: bool = False) -> bool:
        """前回から ``period_s`` 以上経っていれば（または ``force``）1 行を書いて True。"""
        now = self._clock()
        if not force and self._last is not None and now - self._last < self._period_s:
            return False
        line = self._encode(self._tracker.snapshot())
        if not line.endswith("\n"):
            line += "\n"
        (self._write or sys.stdout.write)(line)
        (self._flush or sys.stdout.flush)()
        self._last = now
        return True
