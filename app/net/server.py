"""スマホから姿勢ランドマークを受け取る WebSocket サーバ。

役割は 3 つだけに絞ってある。

1. **時刻サーバになる** — 端末からの ``sync_req`` に受信時刻と送信時刻を付けて返す。
   端末はそこから自分の時計と PC の時計のずれを求める（NTP と同じ原理）。
   両端末が同一の PC 時計に揃うので、結果として端末どうしも揃う。
2. **ランドマークを同期バッファへ流す**。
3. **揃ったペアをコールバックで渡す**。

トランスポート（WebSocket）と処理（``SessionHandler``）を分けてある。
処理側はソケットを開かずに検証できる。

時計には ``time.monotonic_ns()`` を使う。壁時計（``time.time_ns()``）は
NTP 補正や夏時間で飛ぶことがあり、計測中に時間が巻き戻ると
同期バッファのペアリングが壊れるため。
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import signal
import socket
import time
from typing import Callable, Iterable

import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from app.net import protocol as p
from app.net.sync_buffer import PairedSample, SyncBuffer

__all__ = ["SessionHandler", "LandmarkServer", "DEFAULT_PORT", "local_ip", "port_conflict"]

DEFAULT_PORT = 8765


def local_ip() -> str:
    """端末から到達できる LAN 上の自分の IP を返す。

    ``0.0.0.0`` は待ち受けには使えても、QR で端末に配る接続先にはならない。
    外部宛の UDP ソケットを「繋ぐ」だけで実際には送信せず、
    OS が選んだ送信元アドレスを読み取る（経路表を引く定番の方法）。
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))  # パケットは飛ばない
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def port_conflict(port: int, host: str = "127.0.0.1") -> str | None:
    """``host:port`` を既に誰かが握っていれば、その旨の説明を返す。

    ``0.0.0.0`` への bind は、別プロセスが ``127.0.0.1`` の同じポートを
    握っていても成功してしまう。その状態では「サーバは起動したように見えるのに
    端末は別のサービスに繋がる」という、最も気づきにくい壊れ方をする。
    起動前に明示的に確かめる。

    ``port=0``（空きポート自動割り当て）は衝突しようがないので常に None。
    """
    if port == 0:
        return None

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.3)
    try:
        probe.connect((host, port))
        return f"{host}:{port} に既に応答するプロセスがあります"
    except OSError:
        return None
    finally:
        probe.close()


Clock = Callable[[], int]


class SessionHandler:
    """1 接続ぶんのメッセージ処理。トランスポートに依存しない。"""

    def __init__(self, buffer: SyncBuffer, clock: Clock = time.monotonic_ns):
        self._buffer = buffer
        self._clock = clock
        self.role: str | None = None
        self.device: str | None = None
        self.frames_received = 0
        self.errors = 0

    def handle(self, raw: str | bytes) -> str | None:
        """受信メッセージを処理し、返信が必要なら文字列で返す。

        **壊れたメッセージで例外を投げない**。無線ではパケットの破損や
        version 違いが日常的に起きるので、1 通の不正で計測を止めない。
        """
        try:
            message = p.decode(raw)
        except p.ProtocolError:
            self.errors += 1
            return None

        if isinstance(message, p.SyncRequest):
            t2 = self._clock()
            t3 = self._clock()
            return p.encode(p.SyncResponse(t1=message.t1, t2=t2, t3=t3))

        if isinstance(message, p.LandmarkFrame):
            self.role = message.role
            self.frames_received += 1
            self._buffer.push(message)
            return None  # 毎フレーム返信すると無駄な往復が増える

        if isinstance(message, p.Hello):
            self.role = message.role
            self.device = message.device
            return None

        return None


class LandmarkServer:
    """WebSocket で待ち受け、揃ったペアをコールバックへ渡す。"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        buffer: SyncBuffer | None = None,
        on_pairs: Callable[[Iterable[PairedSample]], None] | None = None,
        clock: Clock = time.monotonic_ns,
        session: str | None = None,
    ):
        self.host = host
        self._requested_port = port
        self.buffer = buffer if buffer is not None else SyncBuffer()
        self._on_pairs = on_pairs
        self._clock = clock
        self.session = session or secrets.token_hex(4)

        self._server: Server | None = None
        self._handlers: dict[int, SessionHandler] = {}

    # -- 起動・停止 --------------------------------------------------------
    async def start(self) -> None:
        conflict = port_conflict(self._requested_port)
        if conflict is not None:
            # ここを黙って通すと最悪の壊れ方をする。0.0.0.0 への bind は成功する一方、
            # 127.0.0.1 を既に別プロセスが握っていると、より具体的な束縛が優先され、
            # 「サーバは起動したように見えるのに端末は別のサービスに繋がる」状態になる。
            raise OSError(
                f"ポート {self._requested_port} は既に使われています（{conflict}）。\n"
                f"  --port で別の番号を指定するか、そのプロセスを止めてください。"
            )
        self._server = await serve(self._on_connection, self.host, self._requested_port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def port(self) -> int:
        """実際に待ち受けているポート。port=0 で起動した場合はここで確認する。"""
        if self._server is None:
            return self._requested_port
        for sock in self._server.sockets:
            return sock.getsockname()[1]
        return self._requested_port

    # -- 接続の受け口 ------------------------------------------------------
    async def _on_connection(self, connection: ServerConnection) -> None:
        handler = SessionHandler(self.buffer, clock=self._clock)
        self._handlers[id(connection)] = handler
        try:
            async for raw in connection:
                reply = handler.handle(raw)
                if reply is not None:
                    await connection.send(reply)
                self._flush_pairs()
        except websockets.exceptions.ConnectionClosed:
            pass  # 端末が離脱しただけ。計測は続行する
        finally:
            self._handlers.pop(id(connection), None)

    def _flush_pairs(self) -> None:
        pairs = self.buffer.drain()
        if pairs and self._on_pairs is not None:
            self._on_pairs(pairs)

    # -- 接続情報 ----------------------------------------------------------
    def connect_url(self, role: str, host: str | None = None) -> str:
        """端末に QR で渡す接続先。

        host を省略すると設定された待ち受けアドレスを使うが、``0.0.0.0`` は
        端末から到達できないので、実際に配るときは LAN の IP を渡すこと。
        """
        target = host or self.host
        return f"ws://{target}:{self.port}/?session={self.session}&role={role}"

    @property
    def connected_roles(self) -> list[str]:
        return sorted({h.role for h in self._handlers.values() if h.role})

    @property
    def stats(self) -> dict[str, object]:
        return {
            "clients": len(self._handlers),
            "roles": self.connected_roles,
            "frames_received": sum(h.frames_received for h in self._handlers.values()),
            "protocol_errors": sum(h.errors for h in self._handlers.values()),
            **self.buffer.stats,
        }


def _print_qr(url: str) -> None:
    """接続先をターミナルに QR で描く。

    端末側は QR で接続先と役割を受け取る。手で URL を打たせると、
    2 台とも同じ役割にしてしまう事故が起きやすい。

    segno が無くても起動は妨げない（URL を手入力すれば繋がる）。
    """
    try:
        import segno
    except ImportError:
        print("  （segno が無いため QR を省略します: pip install segno）")
        return

    try:
        segno.make(url, error="m").terminal(compact=True)
    except Exception as exc:  # pragma: no cover - 端末依存
        print(f"  （QR を描けませんでした: {exc}）")


# ---------------------------------------------------------------------------
# 単体起動（動作確認・Phase 3 の GUI から切り離してのデバッグ用）
# ---------------------------------------------------------------------------
async def _main_async(args: argparse.Namespace) -> int:
    received: list[PairedSample] = []
    server = LandmarkServer(
        host=args.host,
        port=args.port,
        buffer=SyncBuffer(
            target_hz=args.hz, window_sec=args.window, max_gap_ms=args.max_gap_ms
        ),
        on_pairs=received.extend,
    )
    try:
        await server.start()
    except OSError as exc:
        # ポート衝突など。利用者に見せるのはトレースバックではなく対処法。
        print(f"起動できませんでした:\n{exc}")
        return 1

    advertise = local_ip() if args.host in ("0.0.0.0", "") else args.host
    print(f"受信サーバを起動しました  session={server.session}\n")
    for role in p.ROLES:
        url = server.connect_url(role, host=advertise)
        print(f"■ {role}")
        print(f"  {url}")
        if not args.no_qr:
            _print_qr(url)
        print()
    print("端末のアプリで QR を読み取ってください。cam0 と cam1 で別々の QR です。")
    print("Ctrl-C で終了します。\n")

    # SIGINT / SIGTERM をイベントに変換する。
    # asyncio のループ内で待っている間、KeyboardInterrupt は確実には届かない
    # （バックグラウンド実行だと特に）。ハンドラを明示登録して終了を待ち合わせる。
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        try:
            loop.add_signal_handler(signum, stopping.set)
        except NotImplementedError:  # Windows では未対応
            pass

    try:
        while not stopping.is_set():
            try:
                await asyncio.wait_for(stopping.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # 1 秒ごとに統計を出すためのタイムアウト
            stats = server.stats
            print(
                f"  接続 {stats['clients']}台 {stats['roles']}  "
                f"受信 {stats['frames_received']}  ペア {stats['emitted']}  "
                f"欠測破棄 {stats['dropped_gap']}  "
                f"位相差 {stats['mean_role_skew_ms']:.1f}ms(最大{stats['max_role_skew_ms']:.1f})  "
                f"不正 {stats['protocol_errors']}"
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await server.stop()
        print(f"\n終了しました。取り出したペア: {len(received)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="スマホからランドマークを受け取るサーバ")
    parser.add_argument("--host", default="0.0.0.0", help="待ち受けアドレス")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hz", type=float, default=30.0, help="再標本化するグリッド周波数")
    parser.add_argument("--window", type=float, default=2.0, help="バッファの時間窓（秒）")
    parser.add_argument("--max-gap-ms", type=float, default=100.0, help="補間を許す最大欠測幅")
    parser.add_argument("--no-qr", action="store_true", help="QR コードを表示しない")
    args = parser.parse_args()

    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
