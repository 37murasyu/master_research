"""スマホのふりをして PC に接続する検証用クライアント。

Android アプリが無い段階で、PC 側（受信層・同期バッファ・三角測量・ゲージ）を
最後まで作り切って検証するために使う。実機と同じプロトコルを話すので、
これが通れば Android 側は同じ電文を出すだけでよい。

ジッタとパケットロスを注入できる。Wi-Fi の劣化を再現して、
同期バッファがどこまで耐えるかを実測するのが目的。

使い方::

    # 2 台ぶんを同時に流す
    python -m app.net.mock_sender --url ws://127.0.0.1:8765 --duration 10

    # 片方だけ、ジッタとロスを入れて
    python -m app.net.mock_sender --role cam0 --jitter-ms 50 --loss 0.05
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import time

import websockets

from app.net import protocol as p

__all__ = ["MockPhone", "synthetic_pose"]

# 端末の単調時計を模す。実機の SystemClock.elapsedRealtimeNanos() に相当し、
# PC の時計とは無関係な原点を持つ（そこを同期で埋める）。
_PHONE_CLOCK_SKEW_NS = 987_654_321_000


def _phone_clock() -> int:
    return time.monotonic_ns() + _PHONE_CLOCK_SKEW_NS


def synthetic_pose(t_sec: float, role: str) -> list[tuple[float, float, float, float]]:
    """車椅子駆動を模した往復運動のランドマーク列を作る。

    実際の押し出し周期はおよそ 1〜1.5 Hz。手首まわりが大きく動き、
    体幹はほぼ止まっている、という現実に近い形にしてある。
    2 台で少しだけ視差が付くよう、role で位相と横位置をずらす。
    """
    cycle_hz = 1.2
    phase = 2 * math.pi * cycle_hz * t_sec

    # 2 台の視差。押し出し周期に合わせて変化させる。
    # 固定値にすると三角測量の結果が「奥行き一定」になり、
    # 奥行き方向のサイクル検出が原理的に働かなくなる。
    # 実際の計測では体が前後に動くので、視差も周期的に変わる。
    base_parallax = 0.045
    parallax = (base_parallax + 0.015 * math.sin(phase)) if role == "cam1" else 0.0

    points: list[tuple[float, float, float, float]] = []
    for index in range(p.LANDMARK_COUNT):
        # 上肢（11-16）だけ大きく動かし、それ以外はわずかに揺らす
        is_arm = 11 <= index <= 16
        amplitude = 0.12 if is_arm else 0.01
        x = 0.5 + parallax + amplitude * math.sin(phase + index * 0.15)
        y = 0.5 + amplitude * math.cos(phase + index * 0.15) * 0.6
        z = 0.1 * math.sin(phase)
        points.append((x, y, z, 1.0))
    return points


class MockPhone:
    """1 台ぶんの模擬端末。"""

    def __init__(
        self,
        url: str,
        role: str,
        fps: float = 30.0,
        jitter_ms: float = 0.0,
        loss: float = 0.0,
        device: str = "MockPhone",
        session: str = "mock",
        seed: int | None = None,
    ):
        self.url = url
        self.role = role
        self.fps = fps
        self.jitter_ms = jitter_ms
        self.loss = loss
        self.device = device
        self.session = session
        self._random = random.Random(seed)

        self.offset_ns = 0
        self.rtt_ns = 0
        self.sent = 0
        self.dropped = 0

    async def run(self, duration_sec: float) -> None:
        async with websockets.connect(self.url) as ws:
            await ws.send(p.encode(p.Hello(self.role, self.device, self.session)))
            await self._synchronize(ws)
            await self._stream(ws, duration_sec)

    # -- 時刻同期 ----------------------------------------------------------
    async def _synchronize(self, ws, samples: int = 20) -> None:
        """往復測定を繰り返し、RTT 最小のサンプルを採る。

        平均を取らないのは、一時的な輻輳による外れ値に引きずられるため。
        往復が速かった回ほど「往路と復路が等しい」という仮定が成り立ちやすい。
        """
        measurements = []
        for _ in range(samples):
            t1 = _phone_clock()
            await ws.send(p.encode(p.SyncRequest(t1=t1)))
            reply = p.decode(await ws.recv())
            t4 = _phone_clock()
            if isinstance(reply, p.SyncResponse):
                measurements.append(p.compute_clock_offset(reply.t1, reply.t2, reply.t3, t4))
            await asyncio.sleep(0.005)

        best = p.best_offset(measurements)
        if best is not None:
            self.offset_ns = best.offset_ns
            self.rtt_ns = best.rtt_ns

    # -- 送信 --------------------------------------------------------------
    async def _stream(self, ws, duration_sec: float) -> None:
        period = 1.0 / self.fps
        started = time.monotonic()
        seq = 0

        while time.monotonic() - started < duration_sec:
            elapsed = time.monotonic() - started

            if self._random.random() < self.loss:
                self.dropped += 1  # パケットロスを模す
            else:
                # 撮影時刻は「端末時計 + 同期で求めたずれ」＝ PC 時計の値。
                # 到着揺らぎとは別物で、ここがずれないことが同期の肝。
                t_capture = _phone_clock() + self.offset_ns
                frame = p.LandmarkFrame(
                    role=self.role,
                    seq=seq,
                    t_capture_ns=t_capture,
                    width=1280,
                    height=720,
                    landmarks=synthetic_pose(elapsed, self.role),
                )
                await ws.send(p.encode(frame))
                self.sent += 1

            seq += 1
            wait = period
            if self.jitter_ms:
                wait += self._random.gauss(0, self.jitter_ms / 1000.0)
            await asyncio.sleep(max(0.0, wait))


async def _main_async(args: argparse.Namespace) -> int:
    roles = [args.role] if args.role else ["cam0", "cam1"]
    phones = [
        MockPhone(
            url=args.url,
            role=role,
            fps=args.fps,
            jitter_ms=args.jitter_ms,
            loss=args.loss,
            seed=index,
        )
        for index, role in enumerate(roles)
    ]

    await asyncio.gather(*(phone.run(args.duration) for phone in phones))

    for phone in phones:
        print(
            f"  {phone.role}: 送信 {phone.sent} / 破棄 {phone.dropped} / "
            f"時計ずれ {phone.offset_ns / 1e6:.2f} ms / RTT {phone.rtt_ns / 1e6:.2f} ms"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="接続先")
    parser.add_argument("--role", choices=p.ROLES, help="省略すると 2 台ぶん同時に流す")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=10.0, help="送信する秒数")
    parser.add_argument("--jitter-ms", type=float, default=0.0, help="送信間隔の揺らぎ")
    parser.add_argument("--loss", type=float, default=0.0, help="パケットロス率 (0.0-1.0)")
    args = parser.parse_args()

    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
