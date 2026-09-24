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
from typing import Callable, Iterable, Sequence

import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from app.net import protocol as p
from app.net.sync_buffer import DEFAULT_GRID, DEFAULT_WINDOW_SEC, GridSpec, PairedSample, SyncBuffer

__all__ = [
    "SessionHandler",
    "LandmarkServer",
    "DEFAULT_PORT",
    "CLOSE_TAKEN_OVER",
    "check_injectable",
    "close_reason",
    "local_ip",
    "port_conflict",
]

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

# 同じ役割の新しい接続に席を譲って閉じるときのコード。4000〜4999 はアプリが自由に使える範囲。
CLOSE_TAKEN_OVER = 4000


def close_reason(text: str, limit: int = 123) -> str:
    """WebSocket の close の理由に収まるよう、UTF-8 で ``limit`` バイト以内に切る。

    上限は文字数ではなくバイト数。日本語は 1 文字 3 バイトなので、文字数で切ると
    41 文字を超えたところで上限を越える。文字の途中では切らない。
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def check_injectable(
    frame: p.LandmarkFrame, remote_roles: Sequence[str], buffer_roles: Sequence[str]
) -> None:
    """PC 側で作ったフレームを注入してよいか確かめる。だめなら ValueError。

    注入は電文の復号（``p.decode``）を通らないので、同じ条件をここで課す。
    取り違えはプログラムの誤りなので、壊れた電文とは違い例外にする。
    """
    if frame.role in remote_roles or frame.role not in buffer_roles:
        raise ValueError(
            f"{frame.role} は注入できません（端末の役割: {tuple(remote_roles)}、"
            f"バッファの役割: {tuple(buffer_roles)}）"
        )
    if len(frame.landmarks) != p.LANDMARK_COUNT or any(len(pt) != 4 for pt in frame.landmarks):
        raise ValueError(
            f"ランドマークは {p.LANDMARK_COUNT} 点の (x, y, z, visibility) である必要があります"
        )


class SessionHandler:
    """1 接続ぶんのメッセージ処理。トランスポートに依存しない。"""

    def __init__(
        self,
        buffer: SyncBuffer,
        clock: Clock = time.monotonic_ns,
        on_calibration_frame: Callable[[p.CalibrationFrame], None] | None = None,
        on_hello: Callable[[p.Hello], str | None] | None = None,
        remote_roles: Sequence[str] = p.ROLES,
        on_landmarks: Callable[[p.LandmarkFrame], None] | None = None,
        accept_frame: Callable[[p.LandmarkFrame], bool] | None = None,
        require_hello: bool = False,
    ):
        self._buffer = buffer
        self._clock = clock
        self._on_calibration_frame = on_calibration_frame
        # 名乗りを受け入れるかの判定。断る理由を返してもらう（None なら受け入れ）。
        self._on_hello = on_hello
        # 端末に許す役割。混成構成では cam0 を PC のカメラが受け持つので cam1 だけになる。
        self._remote_roles = tuple(remote_roles)
        self._on_landmarks = on_landmarks
        self._accept_frame = accept_frame
        self._require_hello = require_hello
        self.role: str | None = None
        self.device: str | None = None
        self.device_id: str | None = None
        self.hello: p.Hello | None = None
        # 断った理由。接続を閉じる側が、利用者に見せる文言として使う。
        self.rejection: str | None = None
        # 同じ役割の新しい接続に席を譲った理由。以後この接続の電文は使わない。
        self.retired: str | None = None
        self.frames_received = 0
        self.errors = 0
        # 役割の食い違いで捨てた電文の数。
        self.rejected = 0

    def retire(self, reason: str) -> None:
        """席を譲る。閉じ終わるまでに届いた電文も使わない。"""
        self.retired = reason
        self.role = None
        self.hello = None

    def _accepts(self, role: str) -> bool:
        """この接続から来た ``role`` の電文を使ってよいか。"""
        if self._require_hello and self.hello is None:
            return False
        if role not in self._remote_roles:
            return False
        # 名乗った役割と違う点は使わない。混ざると補間が 2 台の間を行き来する。
        return self.hello is None or role == self.hello.role

    def handle(self, raw: str | bytes) -> str | None:
        """受信メッセージを処理し、返信が必要なら文字列で返す。

        **壊れたメッセージで例外を投げない**。無線ではパケットの破損や
        version 違いが日常的に起きるので、1 通の不正で計測を止めない。
        """
        if self.rejection is not None or self.retired is not None:
            return None  # 閉じる途中の接続

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
            if not self._accepts(message.role):
                self.rejected += 1
                return None
            self.role = message.role
            self.frames_received += 1
            if self._accept_frame is not None and not self._accept_frame(message):
                return None
            self._buffer.push(message)
            if self._on_landmarks is not None:
                self._on_landmarks(message)
            return None  # 毎フレーム返信すると無駄な往復が増える

        if isinstance(message, p.Hello):
            if message.role not in self._remote_roles:
                self.rejection = (
                    f"{message.role} は PC のカメラが受け持っています。"
                    f"{'・'.join(self._remote_roles)} の QR を読んでください"
                )
                return None
            reason = self._on_hello(message) if self._on_hello is not None else None
            if reason is not None:
                # 役割として数えない。数えると「2 台つながった」と見えてしまう。
                self.rejection = reason
                return None
            self.role = message.role
            self.device = message.device
            self.device_id = message.device_id
            self.hello = message
            return None

        if isinstance(message, p.CalibrationFrame):
            if not self._accepts(message.role):
                self.rejected += 1
                return None
            self.role = self.role or message.role
            if self._on_calibration_frame is not None:
                self._on_calibration_frame(message)
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
        on_calibration_frame: Callable[[p.CalibrationFrame], None] | None = None,
        on_hello: Callable[[p.Hello], str | None] | None = None,
        remote_roles: Sequence[str] = p.ROLES,
        on_landmarks: Callable[[p.LandmarkFrame], None] | None = None,
        accept_frame: Callable[[p.LandmarkFrame], bool] | None = None,
        require_hello: bool = False,
        close_timeout: float = 10.0,
    ):
        self.host = host
        self._requested_port = port
        # 閉じるときに相手の応答を待つ上限（秒）。websockets の既定は 10 秒。
        # 応答しない端末が 1 台いると、stop() がこの時間だけ戻らない。
        self._close_timeout = close_timeout
        self.buffer = buffer if buffer is not None else SyncBuffer()
        self._on_pairs = on_pairs
        self._clock = clock
        self.session = session or secrets.token_hex(4)
        self._on_calibration_frame = on_calibration_frame
        self._on_hello = on_hello

        unknown = set(remote_roles) - set(self.buffer.roles)
        if unknown:
            raise ValueError(f"バッファに無い役割は端末に許せません: {sorted(unknown)}")
        # 端末（Wi-Fi）が受け持つ役割。残りの役割は PC 自身が inject で入れる。
        self.remote_roles = tuple(remote_roles)
        # 受信した点も注入した点も、組になる前にここへ流す（生 2D の記録と表示用）。
        self._on_landmarks = on_landmarks
        self._accept_frame = accept_frame
        self._require_hello = require_hello
        self._injected = 0
        self._finished_counts = dict(frames_received=0, errors=0, rejected=0)

        self._server: Server | None = None
        self._handlers: dict[int, SessionHandler] = {}
        # 撮影指示を送るため、接続そのものも持つ。
        self._connections: dict[int, ServerConnection] = {}
        # 後勝ちで閉じている途中の接続（_take_over）。
        self._closing: set[asyncio.Task] = set()

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
        # 既定の上限は 1MiB。校正用の JPEG（720p を base64 にしたもの）が
        # 超えることがあり、超えると接続ごと切れる。プロトコル側の上限
        # （p.MAX_CALIBRATION_BYTES）で弾き、切断では終わらせない。
        self._server = await serve(
            self._on_connection,
            self.host,
            self._requested_port,
            max_size=p.MAX_CALIBRATION_BYTES * 2,
            close_timeout=self._close_timeout,
        )

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
        sockets = self._server.sockets
        if not sockets:
            return self._requested_port
        return sockets[0].getsockname()[1]

    # -- 接続の受け口 ------------------------------------------------------
    async def _on_connection(self, connection: ServerConnection) -> None:
        handler = SessionHandler(
            self.buffer,
            clock=self._clock,
            on_calibration_frame=self._on_calibration_frame,
            on_hello=self._on_hello,
            remote_roles=self.remote_roles,
            on_landmarks=self._on_landmarks,
            accept_frame=self._accept_frame,
            require_hello=self._require_hello,
        )
        key = id(connection)
        self._handlers[key] = handler
        self._connections[key] = connection
        try:
            async for raw in connection:
                had_hello = handler.hello is not None
                reply = handler.handle(raw)
                if reply is not None:
                    await connection.send(reply)
                if handler.rejection is not None:
                    # 受け入れられない端末は、理由を伝えて閉じる。つないだ人が
                    # 「QR を読み直す」「正しい端末を使う」と判断できるように。
                    await connection.close(code=1008, reason=close_reason(handler.rejection))
                    break
                if not had_hello and handler.hello is not None:
                    self._take_over(key, handler.hello.role)
                self._flush_pairs()
        except websockets.exceptions.ConnectionClosed:
            pass  # 端末が離脱しただけ。計測は続行する
        finally:
            for name in self._finished_counts:
                self._finished_counts[name] += getattr(handler, name)
            self._handlers.pop(key, None)
            self._connections.pop(key, None)

    def _take_over(self, newcomer: int, role: str) -> None:
        """同じ役割の古い接続を閉じ、新しい方に役割を渡す（後勝ち）。

        アプリを入れ直したり QR を読み直したりすると、古い接続は相手が消えたことに
        気づくまで（数十秒）残る。先勝ちにすると、その間は読み直した端末が使えない。

        閉じ終わるのは**待たない**。相手が応答しないと close は close_timeout（既定 10 秒）
        まで戻らず、その間、新しい端末の時刻同期が止まる。役割は ``retire`` で
        すぐ外れるので、閉じ終わる前に届いた電文も使われない。
        """
        for key, other in list(self._handlers.items()):
            if key == newcomer or other.role != role:
                continue
            reason = f"同じ役割（{role}）で別の接続が来たため切断しました"
            other.retire(reason)
            connection = self._connections.get(key)
            if connection is not None:
                task = asyncio.create_task(
                    connection.close(code=CLOSE_TAKEN_OVER, reason=close_reason(reason))
                )
                # 参照を持たないタスクは途中で回収されうる
                self._closing.add(task)
                task.add_done_callback(self._closing.discard)

    def inject(self, frame: p.LandmarkFrame) -> None:
        """PC 自身のカメラで作ったフレームを、端末の点と同じバッファへ入れる。

        **ループのスレッドから呼ぶこと**（同期バッファにロックが無い）。別スレッドからは
        ``loop.call_soon_threadsafe(server.inject, frame)`` で渡す。

        注入は電文の検証を通らないので、ここで同じ条件を確かめる（``check_injectable``）。
        """
        check_injectable(frame, self.remote_roles, self.buffer.roles)
        if self._accept_frame is not None and not self._accept_frame(frame):
            return
        self.buffer.push(frame)
        self._injected += 1
        if self._on_landmarks is not None:
            self._on_landmarks(frame)
        # 端末からの受信を待たずに掃き出す。待つと、端末が止まった瞬間にペアも止まる。
        self._flush_pairs()

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
    def devices(self) -> dict[str, p.Hello]:
        """役割 → 名乗り。どの端末がどちらで繋がったかの照合に使う。"""
        return {h.role: h.hello for h in self._handlers.values() if h.role and h.hello}

    async def request_capture(
        self,
        capture_id: int,
        at_ns: int | None = None,
        max_width: int | None = None,
        quality: int | None = None,
    ) -> int:
        """繋がっている端末に、校正用の撮影を指示する。送った台数を返す。

        両端末へ**同じ目標時刻**を渡す。端末は PC 時計に同期しているので、
        ネットワークの遅延差があっても、ほぼ同じ瞬間のフレームが揃う。
        ``max_width`` と ``quality`` はライブ表示用（``p.CaptureRequest``）。
        """
        message = p.encode(
            p.CaptureRequest(id=capture_id, at_ns=at_ns, max_width=max_width, quality=quality)
        )
        sent = 0
        for key, connection in list(self._connections.items()):
            handler = self._handlers.get(key)
            if handler is None or not handler.role:
                continue  # まだ名乗っていない接続には送らない
            try:
                await connection.send(message)
                sent += 1
            except websockets.exceptions.ConnectionClosed:
                continue  # 離脱した端末。撮影指示は次の周回で届く
        return sent

    @property
    def stats(self) -> dict[str, object]:
        return {
            "clients": len(self._handlers),
            "roles": self.connected_roles,
            "frames_received": self._finished_counts["frames_received"] + sum(h.frames_received for h in self._handlers.values()),
            "frames_injected": self._injected,
            "protocol_errors": self._finished_counts["errors"] + sum(h.errors for h in self._handlers.values()),
            "role_mismatches": self._finished_counts["rejected"] + sum(h.rejected for h in self._handlers.values()),
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
class _PairCounter:
    """受け取ったペアの数だけ数える。

    ``received.extend`` を渡すとリスト全体が実行中ずっと生き続ける。
    このデバッグ用サーバは Ctrl-C まで流しっぱなしで使うので、
    30fps で 10 分回すと 18,000 サンプル（100MB 超）が溜まる。
    最後に件数を出すだけなら int 1 個で足りる。
    """

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, pairs) -> None:
        self.count += len(pairs)


async def _main_async(args: argparse.Namespace) -> int:
    received = _PairCounter()
    server = LandmarkServer(
        host=args.host,
        port=args.port,
        buffer=SyncBuffer(window_sec=args.window, grid=GridSpec(args.hz, args.max_gap_ms)),
        on_pairs=received,
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
        print(f"\n終了しました。取り出したペア: {received.count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="スマホからランドマークを受け取るサーバ")
    parser.add_argument("--host", default="0.0.0.0", help="待ち受けアドレス")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hz", type=float, default=DEFAULT_GRID.target_hz, help="再標本化するグリッド周波数")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_SEC, help="バッファの保持時間（秒）")
    parser.add_argument("--max-gap-ms", type=float, default=DEFAULT_GRID.max_gap_ms, help="補間を許す最大欠測幅")
    parser.add_argument("--no-qr", action="store_true", help="QR コードを表示しない")
    args = parser.parse_args()

    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
