"""到着した姿勢ランドマークを、**時刻で**ペアにして返すバッファ。

なぜ必要か。三角測量（``utils.DLT``）は「2 台の同時刻の 2D 点」を前提にしている。
既存コードはこれを「同じループ回で読んだフレーム＝同時刻」という暗黙の仮定で
満たしていた（``master_research_code.py:2945-2946`` の逐次 grab）。USB なら
grab 間の差はミリ秒で済むが、無線では受信バッファの状態次第で数フレームずれる。

そこで各フレームに撮影時刻を刻んで送ってもらい（``protocol.LandmarkFrame``）、
受信側は到着順ではなく**時刻で組む**。1 秒程度のラグは許容できるので、
数百ミリ秒ぶんバッファしてから組めばよい。

さらに共通の等間隔グリッドへ**線形補間して再標本化**する。最近傍で組むより
整合が良い。ランドマークは単なる点列なので補間は自明かつ安価。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Sequence

from app.net.protocol import ROLES, LandmarkFrame, PixelCoordinates

__all__ = ["InterpolatedFrame", "PairedSample", "SyncBuffer"]

Landmarks = Sequence[tuple[float, float, float, float]]


def _time_of(frame: LandmarkFrame) -> int:
    """bisect の key。フレーム自身が持つ撮影時刻で並べる。"""
    return frame.t_capture_ns


@dataclass(frozen=True)
class InterpolatedFrame(PixelCoordinates):
    """グリッド時刻における 1 台ぶんのランドマーク。

    ピクセル換算の規約は ``PixelCoordinates`` が持つ。送信側と受信側で
    別々に実装すると、片方だけ直したときに黙ってずれる。
    """

    role: str
    t_ns: int
    width: int
    height: int
    landmarks: Landmarks


@dataclass(frozen=True)
class PairedSample:
    """同一時刻に揃った全ロールぶんのランドマーク。ここから三角測量へ渡す。"""

    t_ns: int
    frames: dict[str, InterpolatedFrame]


@dataclass
class _Stats:
    emitted: int = 0
    dropped_gap: int = 0
    dropped_late: int = 0
    rejected: int = 0
    # 直近 1 サンプルの位相差。値は素直だがジッタでよく揺れる。
    last_role_skew_ms: float = 0.0
    # 指数移動平均。UI に出すのも、同期品質を判定するのもこちらを使う。
    # 瞬時値はジッタの分散をそのまま拾うため、単発の値で良否を決めてはいけない。
    mean_role_skew_ms: float = 0.0
    max_role_skew_ms: float = 0.0
    _skew_samples: int = 0

    # 平滑化の強さ。0.1 だとおよそ直近 20 サンプルぶんを見る。
    SKEW_ALPHA = 0.1

    def observe_skew(self, skew_ms: float) -> None:
        self.last_role_skew_ms = skew_ms
        self.max_role_skew_ms = max(self.max_role_skew_ms, skew_ms)
        if self._skew_samples == 0:
            self.mean_role_skew_ms = skew_ms
        else:
            self.mean_role_skew_ms += self.SKEW_ALPHA * (skew_ms - self.mean_role_skew_ms)
        self._skew_samples += 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "emitted": self.emitted,
            "dropped_gap": self.dropped_gap,
            "dropped_late": self.dropped_late,
            "rejected": self.rejected,
            "last_role_skew_ms": self.last_role_skew_ms,
            "mean_role_skew_ms": self.mean_role_skew_ms,
            "max_role_skew_ms": self.max_role_skew_ms,
        }


class SyncBuffer:
    """ロールごとにフレームを溜め、揃った時刻から順に取り出す。

    Parameters
    ----------
    roles:
        揃うべきロール。既定は ``("cam0", "cam1")``。
    target_hz:
        再標本化するグリッドの周波数。既存パイプラインは 30 fps 前提
        （``config.fps = 30``）なのでそれに合わせるのが既定。
    window_sec:
        保持する時間窓。長いほどジッタに強いが、その分だけ表示が遅れる。
    max_gap_ms:
        補間を許す最大の欠測幅。これを超える穴は補間せず捨てる。
        長い穴を線形補間で埋めると、実際には動いていた手を
        「まっすぐ動いた」ことにしてしまうため。
    """

    def __init__(
        self,
        roles: Sequence[str] = ROLES,
        target_hz: float = 30.0,
        window_sec: float = 2.0,
        max_gap_ms: float = 100.0,
    ):
        if target_hz <= 0:
            raise ValueError("target_hz は正の値である必要があります")

        self.roles = tuple(roles)
        self.period_ns = round(1_000_000_000 / target_hz)
        self.window_ns = round(window_sec * 1_000_000_000)
        self.max_gap_ns = round(max_gap_ms * 1_000_000)

        # ロールごとに時刻昇順で保持する。到着順は当てにしない。
        # 時刻はフレーム自身が持っているので別のリストにはしない
        # （二重管理すると「2 本の index が揃っている」という不変条件を
        #  push / evict のたびに維持する責任が生まれる）。
        self._frames: dict[str, list[LandmarkFrame]] = {r: [] for r in self.roles}

        self._next_grid_ns: int | None = None
        self._stats = _Stats()

    # -- 入力 --------------------------------------------------------------
    def push(self, frame: LandmarkFrame) -> None:
        """フレームを受け取る。未知のロールや遅すぎるものは捨てる。"""
        frames = self._frames.get(frame.role)
        if frames is None:
            self._stats.rejected += 1
            return

        index = bisect.bisect_left(frames, frame.t_capture_ns, key=_time_of)

        if index < len(frames) and frames[index].t_capture_ns == frame.t_capture_ns:
            return  # 同時刻の重複。再送などで起こりうる

        # 既に処理を終えた時刻より古いフレームは使い道がない
        if self._next_grid_ns is not None and frame.t_capture_ns < self._next_grid_ns - self.max_gap_ns:
            self._stats.dropped_late += 1
            return

        frames.insert(index, frame)

    # -- 出力 --------------------------------------------------------------
    def drain(self) -> list[PairedSample]:
        """今の時点で組めるペアをすべて返す。"""
        if not self._ensure_grid_origin():
            return []

        pairs: list[PairedSample] = []
        while True:
            assert self._next_grid_ns is not None
            t = self._next_grid_ns

            status = self._can_resolve(t)
            if status == "wait":
                break  # まだデータが足りない。次の push を待つ

            if status == "ok":
                sample = self._resolve(t)
                if sample is not None:
                    pairs.append(sample)
                    self._stats.emitted += 1
                    self._update_skew(t)
            else:  # "skip"
                self._stats.dropped_gap += 1

            self._next_grid_ns = t + self.period_ns

        self._evict()
        return pairs

    # -- 内部 --------------------------------------------------------------
    def _ensure_grid_origin(self) -> bool:
        """全ロールにデータが揃った時点でグリッドの原点を決める。

        原点は「各ロールの最初の時刻のうち最も遅いもの」。それより前は
        片方しかデータが無く、どうやってもペアにならないため。
        """
        if self._next_grid_ns is not None:
            return True
        if any(not frames for frames in self._frames.values()):
            return False
        self._next_grid_ns = max(f[0].t_capture_ns for f in self._frames.values())
        return True

    def _can_resolve(self, t: int) -> str:
        """グリッド時刻 t を解決できるか。"ok" / "skip" / "wait" を返す。"""
        for role in self.roles:
            frames = self._frames[role]
            if not frames:
                return "wait"
            if t > frames[-1].t_capture_ns:
                return "wait"  # 将来のデータで解決できるかもしれない
            if t < frames[0].t_capture_ns:
                return "skip"  # もう手に入らない

            index = bisect.bisect_left(frames, t, key=_time_of)
            if index < len(frames) and frames[index].t_capture_ns == t:
                continue  # ちょうどサンプルがある
            gap = frames[index].t_capture_ns - frames[index - 1].t_capture_ns
            if gap > self.max_gap_ns:
                return "skip"  # 欠測が長すぎる。補間で埋めない
        return "ok"

    def _resolve(self, t: int) -> PairedSample | None:
        frames: dict[str, InterpolatedFrame] = {}
        for role in self.roles:
            interpolated = self._interpolate(role, t)
            if interpolated is None:
                return None
            frames[role] = interpolated
        return PairedSample(t_ns=t, frames=frames)

    def _interpolate(self, role: str, t: int) -> InterpolatedFrame | None:
        buffered = self._frames[role]

        index = bisect.bisect_left(buffered, t, key=_time_of)
        if index < len(buffered) and buffered[index].t_capture_ns == t:
            exact = buffered[index]
            return InterpolatedFrame(
                role=role,
                t_ns=t,
                width=exact.width,
                height=exact.height,
                landmarks=list(exact.landmarks),
            )

        if index == 0 or index >= len(buffered):
            return None

        before, after = buffered[index - 1], buffered[index]
        span = after.t_capture_ns - before.t_capture_ns
        if span <= 0:
            return None
        ratio = (t - before.t_capture_ns) / span

        landmarks = [
            (
                b[0] + (a[0] - b[0]) * ratio,
                b[1] + (a[1] - b[1]) * ratio,
                b[2] + (a[2] - b[2]) * ratio,
                # visibility は補間せず、慎重な側（低い方）を採る。
                # 片方が未検出なら、その区間の点は信用しないほうがよい。
                min(b[3], a[3]),
            )
            for b, a in zip(before.landmarks, after.landmarks)
        ]

        return InterpolatedFrame(
            role=role,
            t_ns=t,
            width=before.width,
            height=before.height,
            landmarks=landmarks,
        )

    def _update_skew(self, t: int) -> None:
        """2 台の位相差を記録する。UI で同期品質を見せるのに使う。"""
        nearest: list[int] = []
        for role in self.roles:
            frames = self._frames[role]
            if not frames:
                return
            index = bisect.bisect_left(frames, t, key=_time_of)
            candidates = [
                frames[i].t_capture_ns for i in (index - 1, index) if 0 <= i < len(frames)
            ]
            if not candidates:
                return
            nearest.append(min(candidates, key=lambda x: abs(x - t)))
        self._stats.observe_skew((max(nearest) - min(nearest)) / 1_000_000)

    def _evict(self) -> None:
        """時間窓より古いフレームを捨てる。長時間の計測でメモリを食わないため。"""
        newest = max(
            (f[-1].t_capture_ns for f in self._frames.values() if f), default=None
        )
        if newest is None:
            return
        cutoff = newest - self.window_ns

        for frames in self._frames.values():
            keep_from = bisect.bisect_left(frames, cutoff, key=_time_of)
            # 補間には「t の直前のサンプル」が要るので、必ず 2 個は残す
            keep_from = min(keep_from, max(0, len(frames) - 2))
            if keep_from > 0:
                del frames[:keep_from]

    # -- 観測 --------------------------------------------------------------
    def buffered_count(self, role: str) -> int:
        return len(self._frames.get(role, ()))

    @property
    def stats(self) -> dict[str, float | int]:
        return self._stats.as_dict()
