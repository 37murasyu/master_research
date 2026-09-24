"""混成の押し上げの回の区切り（力学の関所を兼ねる）。

高さは肩の中点の「重力の上向き」への射影 [m]。基準の初期値は先頭の窓の中央値（呼び出し側が渡す）で、以後は
座面の高さを追う（下の「基準」）。開いている間だけ仕事とゲージに積み、閉じている間はトルクを記録するだけにする
（座っている間の雑音の仕事を積まない。論文 5.4 の過大評価の機序）。

- 開く: 高さが基準 + ``open_rise_m`` を超える、または上向きの速さが ``open_speed_mps`` を
  ``open_speed_frames`` フレーム続けて超える
- 閉じる: 開いてから ``min_open_s`` 以上経ち、高さが基準 + ``close_band_m`` 以内に ``close_frames``
  フレーム続いたとき。基準より高く着座したとき（座り直し）は、高さが基準 + max(``open_rise_m``,
  ``settle_fraction`` × 最大の持ち上げ) 以内にいる間に速さが ``rest_speed_mps`` 以下のフレームが
  ``settle_frames`` たまったとき（着座した高さを基準にする）。``max_open_s`` を超えても閉じる
- 開いている間の最大の持ち上げが ``min_lift_m`` 未満なら CLOSED ではなく DISCARDED（回に数えない）
- 高さが NaN のフレームは開閉の状態を変えない

基準: 関所が閉じていて速さが ``rest_speed_mps`` 以下のフレームの高さを直近 ``baseline_window_s`` 秒ぶん持ち、
``baseline_min_frames`` 以上たまっていて、その中央値が今の基準から ``baseline_deadband_m`` を超えてずれたら基準を
中央値に置き換える。開いている間は動かさない（閉じる判定と最小の持ち上げは開く直前の基準に対して行う）。
かつては先頭の窓の中央値で固定しており、座り直して肩の中点が 1.5 cm 高くなると「基準 + 1 cm 以内」が満たされず
回が閉じなかった（30 s 開いたままで仕事を積み、過負荷と誤表示する）。先頭の窓で体を持ち上げていると基準が高く、
以後の押し上げがすべて DISCARDED になった（2026-09-24 のレビュー）。雑音（σ 数 mm）では基準は動かない。

``lookback_frames`` は呼び出し側（仕事の積算）が使う: 速さの条件は数フレーム遅れて開くので、開いた時点で
直前のこのフレーム数の仕事も今の回に入れる。

速さを ``None`` で渡すと、直近 ``SPEED_WINDOW_FRAMES`` フレームの高さの最小二乗の傾きを使う。生の差分は
雑音 σ 3 mm で σ≈0.13 m/s になり、座っているだけで 1 分に 20 回以上開く（すべて DISCARDED になるが、ゲージの
now がちらつく）。傾きなら開かない代わりに約 1 フレーム遅れて開くので、出だしは ``lookback_frames`` で拾う。
"""

from __future__ import annotations

import enum
import statistics
from collections import deque
from dataclasses import dataclass

from app.gauge.protocol import finite_or_none

__all__ = ["RepConfig", "RepDetector", "RepEvent"]

# 速さを自前で出すときの窓（σ 3 mm の雑音を 60 s 流して、乱数 10 通りとも開かない。EMA 0.5 は 10 通り中 2 回開いた）
SPEED_WINDOW_FRAMES = 5
_EPS_S = 1e-9


class RepEvent(enum.Enum):
    NONE = "none"
    OPENED = "opened"
    CLOSED = "closed"
    DISCARDED = "discarded"


@dataclass(frozen=True)
class RepConfig:
    open_rise_m: float = 0.02
    open_speed_mps: float = 0.10
    open_speed_frames: int = 2
    close_band_m: float = 0.01
    close_frames: int = 3
    min_lift_m: float = 0.03
    min_open_s: float = 0.3
    max_open_s: float = 30.0
    lookback_frames: int = 5
    # 基準の追従: 「速さがほぼ 0」の上限、中央値を取る直近の秒数、中央値に要るフレーム数、動かす最小のずれ
    rest_speed_mps: float = 0.05
    baseline_window_s: float = 2.0
    baseline_min_frames: int = 10
    baseline_deadband_m: float = 0.005
    # 基準より高く着座したときに閉じる: 静止のフレーム数と、最大の持ち上げに対する高さの比
    settle_frames: int = 6
    settle_fraction: float = 0.25


class RepDetector:
    """押し上げの回を開閉する状態機械。``update`` を 1 フレームごとに呼ぶ。"""

    def __init__(self, baseline_m: float, config: RepConfig | None = None):
        base = finite_or_none(baseline_m)
        if base is None:
            raise ValueError(f"基準の高さが有限でない: {baseline_m!r}")
        # 今の基準（座面の高さ）。先頭の窓の中央値から始めて、座っている間の高さを追う
        self.baseline_m = base
        self.initial_baseline_m = base
        # 基準を置き換えた回数
        self.baseline_updates = 0
        self.config = config or RepConfig()
        self._open = False
        self._elapsed = 0.0
        self._max_lift = 0.0
        self._in_band = 0
        # 開いている間に座面の近くで静止したフレームの高さ（基準より高く着座したときに閉じる）
        self._settled: list[float] = []
        self._fast = 0
        # 高さが有限のフレームの dt を足した時刻 [s]
        self._clock = 0.0
        # 速さを自前で出すときの (時刻, 高さ) の窓
        self._recent: deque[tuple[float, float]] = deque(maxlen=SPEED_WINDOW_FRAMES)
        # 関所が閉じていて静止したフレームの (時刻, 高さ)。直近 baseline_window_s 秒ぶん
        self._rest: deque[tuple[float, float]] = deque()

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def open_s(self) -> float:
        """開いてからの秒（閉じていれば 0）。"""
        return self._elapsed

    @property
    def max_lift_m(self) -> float:
        """開いている回の、開く直前の基準からの最大の持ち上げ [m]（閉じていれば 0）。"""
        return self._max_lift

    def _own_speed(self, height: float) -> float | None:
        """直近の窓の高さの最小二乗の傾き [m/s]。3 点に満たなければ None（速さの条件を使わない）。"""
        if self._recent and self._clock <= self._recent[-1][0]:
            self._recent.clear()  # 時間が進まなかった（dt が 0・不正）。傾きを作れないので窓を始め直す
        self._recent.append((self._clock, height))
        if len(self._recent) < 3:
            return None
        n = len(self._recent)
        t_mean = sum(t for t, _ in self._recent) / n
        h_mean = sum(h for _, h in self._recent) / n
        num = sum((t - t_mean) * (h - h_mean) for t, h in self._recent)
        den = sum((t - t_mean) ** 2 for t, _ in self._recent)
        return num / den if den > 0.0 else None

    def update(self, height_m: float, speed_mps: float | None, dt: float) -> RepEvent:
        """1 フレーム進める。``speed_mps`` は上向きの速さ（None なら高さから自前で出す）、``dt`` は前のフレームからの秒。"""
        height = finite_or_none(height_m)
        if height is None:
            # 開閉の状態は変えない。自前の速さの窓は、抜けをまたいで傾きを取らないように区切る
            self._recent.clear()
            return RepEvent.NONE
        step = finite_or_none(dt)
        step = step if step is not None and step > 0.0 else 0.0
        self._clock += step
        speed = self._own_speed(height) if speed_mps is None else finite_or_none(speed_mps)
        cfg = self.config
        still = speed is not None and abs(speed) <= cfg.rest_speed_mps
        rise = height - self.baseline_m

        if not self._open:
            if speed is not None and speed > cfg.open_speed_mps:
                self._fast += 1
            else:
                self._fast = 0
            if rise > cfg.open_rise_m or self._fast >= cfg.open_speed_frames:
                self._open = True
                self._elapsed = 0.0
                self._max_lift = max(rise, 0.0)
                self._in_band = 0
                self._settled.clear()
                self._fast = 0
                # 回の後は座り直しているかもしれない。基準の窓は着座し直してからのフレームで作り直す
                self._rest.clear()
                return RepEvent.OPENED
            if still:
                self._follow_seat(height)
            return RepEvent.NONE

        self._elapsed += step
        self._max_lift = max(self._max_lift, rise)
        self._in_band = self._in_band + 1 if rise <= cfg.close_band_m else 0
        # 座面の近く（開く高さ未満か、持ち上げの settle_fraction 以内）で静止したフレーム。近くを離れたら数え直す
        if rise <= max(cfg.open_rise_m, cfg.settle_fraction * self._max_lift):
            if still:
                self._settled.append(height)
        else:
            self._settled.clear()
        ready = self._elapsed + _EPS_S >= cfg.min_open_s
        if ready and self._in_band >= cfg.close_frames:
            return self._close()
        if ready and len(self._settled) >= cfg.settle_frames:
            # 基準より高く着座した。すぐ開き直さないよう、着座した高さを基準にする
            self._rebase(statistics.median(self._settled))
            return self._close()
        if self._elapsed + _EPS_S >= cfg.max_open_s:
            return self._close()
        return RepEvent.NONE

    def _rebase(self, seat: float) -> None:
        """座面の高さ ``seat`` が今の基準から不感帯を超えてずれていれば、基準を置き換える。"""
        if abs(seat - self.baseline_m) > self.config.baseline_deadband_m:
            self.baseline_m = seat
            self.baseline_updates += 1

    def _follow_seat(self, height: float) -> None:
        """関所が閉じていて静止したフレームの高さを窓に足し、中央値で基準を追う（``_rebase``）。"""
        cfg = self.config
        self._rest.append((self._clock, height))
        while self._rest and self._clock - self._rest[0][0] > cfg.baseline_window_s + _EPS_S:
            self._rest.popleft()
        if len(self._rest) < cfg.baseline_min_frames:
            return
        self._rebase(statistics.median(h for _, h in self._rest))

    def _close(self) -> RepEvent:
        lifted = self._max_lift >= self.config.min_lift_m
        self._open = False
        self._elapsed = 0.0
        self._max_lift = 0.0
        self._in_band = 0
        self._settled.clear()
        self._fast = 0
        return RepEvent.CLOSED if lifted else RepEvent.DISCARDED
