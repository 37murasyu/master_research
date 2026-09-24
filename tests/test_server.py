"""受信サーバを検証する。

トランスポート（WebSocket）とメッセージ処理を分けてある。
``SessionHandler`` は 1 接続ぶんの処理を担う純粋なクラスで、
ソケットを開かずに検証できる。``LandmarkServer`` は実際に通信する層で、
こちらは統合テストで確かめる。
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from app.net import protocol as p
from app.net.server import MAX_FUTURE_NS, LandmarkServer, SessionHandler
from app.net.sync_buffer import GridSpec, SyncBuffer


class FakeClock:
    """テスト用の時計。呼ぶたびに一定量進む。"""

    def __init__(self, start: int = 1_000_000_000, step: int = 1_000_000):
        self.now = start
        self.step = step

    def __call__(self) -> int:
        value = self.now
        self.now += self.step
        return value


def _landmarks_message(role: str, seq: int, t_ns: int) -> str:
    return p.encode(
        p.LandmarkFrame(
            role=role,
            seq=seq,
            t_capture_ns=t_ns,
            width=1280,
            height=720,
            landmarks=[(0.5, 0.5, 0.0, 1.0)] * p.LANDMARK_COUNT,
        )
    )


class TestSessionHandler:
    def test_sync_request_gets_response_with_both_timestamps(self):
        handler = SessionHandler(SyncBuffer(), clock=FakeClock())
        reply = handler.handle(p.encode(p.SyncRequest(t1=12345)))
        assert reply is not None

        response = p.decode(reply)
        assert isinstance(response, p.SyncResponse)
        assert response.t1 == 12345, "端末の t1 をそのまま返すこと"
        assert response.t3 >= response.t2, "受信時刻より送信時刻が後であること"

    def test_landmarks_are_forwarded_to_the_buffer(self):
        buffer = SyncBuffer(grid=GridSpec(target_hz=10.0))
        handler = SessionHandler(buffer, clock=FakeClock())
        handler.handle(_landmarks_message("cam0", 0, 0))
        assert buffer.buffered_count("cam0") == 1

    def test_landmarks_get_no_reply(self):
        """毎フレーム返信すると無駄な往復が増える。応答は同期のときだけ。"""
        handler = SessionHandler(SyncBuffer(), clock=FakeClock())
        assert handler.handle(_landmarks_message("cam0", 0, 0)) is None

    def test_hello_records_the_role(self):
        handler = SessionHandler(SyncBuffer(), clock=FakeClock())
        handler.handle(p.encode(p.Hello(role="cam1", device="Pixel 8", session="s1")))
        assert handler.role == "cam1"
        assert handler.device == "Pixel 8"

    def test_malformed_message_does_not_raise(self):
        """壊れたメッセージ 1 通で計測を止めない。無線では日常的に起きる。"""
        handler = SessionHandler(SyncBuffer(), clock=FakeClock())
        assert handler.handle("これはJSONではない") is None
        assert handler.errors == 1

    @pytest.mark.parametrize(
        "raw",
        [
            json.dumps({"type": "landmarks", "role": "cam0", "seq": 0, "t_capture_ns": 1, "w": 1280, "h": 720,
                        "lm": [[10**400, 0.5, 0, 1]] * p.LANDMARK_COUNT}),
            '{"type":"landmarks","lm":' + "[" * 100_000 + "]" * 100_000 + "}",
        ],
        ids=["huge-int-coordinate", "deep-nesting"],
    )
    def test_hostile_message_is_counted_not_raised(self, raw):
        """以前は OverflowError・RecursionError が外へ出て、受信サーバが接続ごと 1011 で切っていた。"""
        handler = SessionHandler(SyncBuffer(), clock=FakeClock())
        assert handler.handle(raw) is None
        assert handler.errors == 1

    def test_keeps_working_after_a_malformed_message(self):
        buffer = SyncBuffer(grid=GridSpec(target_hz=10.0))
        handler = SessionHandler(buffer, clock=FakeClock())
        handler.handle("{壊れている")
        handler.handle(_landmarks_message("cam0", 0, 0))
        assert buffer.buffered_count("cam0") == 1


S = 1_000_000_000  # 1 秒 = 10^9 ナノ秒


class TestCaptureTimeCheck:
    """撮影時刻が受信時の PC の時計から外れた点は、同期バッファへ入れずに捨てて数える。

    未来の撮影時刻の点が 1 枚でも入ると、同期バッファはそれを最新として相手の点を捨て、格子もその時刻までの
    穴を「補間できない」と飛ばすので、以後の組が全滅していた（60 s 先の 1 枚で、以後の 15 s が 0 組）。
    保持時間より古い点は、相手の点がもう捨てられているので組にならない。
    """

    def _handler(self, now: int, **kwargs):
        buffer = SyncBuffer()
        handler = SessionHandler(buffer, clock=FakeClock(start=now, step=0), **kwargs)
        return handler, buffer

    def test_future_capture_time_is_dropped_and_counted(self):
        handler, buffer = self._handler(100 * S)
        handler.handle(_landmarks_message("cam1", 0, 160 * S))
        assert buffer.buffered_count("cam1") == 0
        assert handler.time_rejected == 1

    def test_clock_sync_error_within_the_margin_is_kept(self):
        """端末の時刻合わせの誤差（数十 ms）で PC の時計より少し先になった点は使う。"""
        handler, buffer = self._handler(100 * S)
        handler.handle(_landmarks_message("cam1", 0, 100 * S + MAX_FUTURE_NS))
        assert buffer.buffered_count("cam1") == 1
        assert handler.time_rejected == 0

    def test_capture_time_older_than_the_window_is_dropped(self):
        handler, buffer = self._handler(100 * S)
        handler.handle(_landmarks_message("cam1", 0, 100 * S - buffer.window_ns - 1))
        handler.handle(_landmarks_message("cam1", 1, 100 * S - buffer.window_ns))
        assert buffer.buffered_count("cam1") == 1
        assert handler.time_rejected == 1

    def test_dropped_points_are_not_passed_on(self):
        """生 2D の記録や表示（on_landmarks）にも渡さない。未来の点が並びを壊す。"""
        seen: list = []
        handler, _buffer = self._handler(100 * S, on_landmarks=seen.append)
        handler.handle(_landmarks_message("cam1", 0, 160 * S))
        handler.handle(_landmarks_message("cam1", 1, 100 * S - 150_000_000))
        assert [frame.seq for frame in seen] == [1]

    @pytest.mark.parametrize("offset_s", [3.0, 60.0])
    def test_one_future_point_does_not_stop_the_pairs(self, offset_s):
        """PC のカメラ（注入）は撮影直後に、Pixel は 150 ms 遅れで届く。5 s の時点で 1 枚だけ未来の点が来る。"""
        clock = FakeClock(start=0, step=0)
        pairs: list = []
        server = LandmarkServer(buffer=SyncBuffer(), clock=clock, remote_roles=("cam1",), on_pairs=pairs.extend)
        handler = SessionHandler(server.buffer, clock=clock, remote_roles=("cam1",))
        t0, period = 1000 * S, 33_333_333
        events = []
        for i in range(20 * 30):
            t = t0 + i * period
            events.append((t + 5_000_000, _local_frame(t)))
            events.append((t + 150_000_000, _landmarks_message("cam1", i, t + 7_000_000)))
        bad_at = t0 + 5 * S + 150_000_000 + 1
        events.append((bad_at, _landmarks_message("cam1", 99_999, t0 + 5 * S + round(offset_s * S))))
        events.sort(key=lambda e: e[0])

        before = 0
        for arrival, item in events:
            clock.now = arrival
            if arrival <= bad_at:
                before = len(pairs)
            if isinstance(item, p.LandmarkFrame):
                server.inject(item)
            else:
                handler.handle(item)

        assert handler.time_rejected == 1
        assert len(pairs) - before >= 15 * 30 - 5, "未来の点の後の組が失われている"


class TestServerIntegration:
    def test_two_clients_produce_paired_samples(self):
        """2 台ぶん繋いで、時刻の揃ったペアが取り出せること。"""

        async def scenario():
            pairs: list = []
            server = LandmarkServer(
                host="127.0.0.1",
                port=0,  # 空きポートを自動で取る
                buffer=SyncBuffer(grid=GridSpec(target_hz=10.0, max_gap_ms=250.0)),
                on_pairs=pairs.extend,
            )
            await server.start()
            try:
                import websockets

                url = f"ws://127.0.0.1:{server.port}"
                async with websockets.connect(url) as cam0, websockets.connect(url) as cam1:
                    await cam0.send(p.encode(p.Hello("cam0", "test", "s")))
                    await cam1.send(p.encode(p.Hello("cam1", "test", "s")))

                    base = time.monotonic_ns()  # 撮影時刻は PC の時計（端末は時刻合わせ済み）
                    for i in range(6):
                        t = base + i * 100_000_000  # 100ms 刻み
                        await cam0.send(_landmarks_message("cam0", i, t))
                        await cam1.send(_landmarks_message("cam1", i, t))

                    # 受信とバッファ処理が回るのを待つ
                    for _ in range(50):
                        if pairs:
                            break
                        await asyncio.sleep(0.02)
            finally:
                await server.stop()
            return pairs

        pairs = asyncio.run(asyncio.wait_for(scenario(), timeout=15))
        assert pairs, "ペアが 1 つも出ていない"
        assert all(set(pair.frames) == {"cam0", "cam1"} for pair in pairs)

    def test_a_hostile_message_keeps_the_connection(self):
        """壊れた電文 1 通で接続を切らず、protocol_errors に数える（以前は 1011 で切れ、数えなかった）。"""

        async def scenario():
            server = LandmarkServer(host="127.0.0.1", port=0, buffer=SyncBuffer())
            await server.start()
            try:
                import websockets

                async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.send(p.encode(p.Hello("cam1", "Pixel 7a", "s", "id-1")))
                    await ws.send(json.dumps({
                        "type": "landmarks", "role": "cam1", "seq": 0, "t_capture_ns": time.monotonic_ns(),
                        "w": 1280, "h": 720, "lm": [[10**400, 0.5, 0, 1]] * p.LANDMARK_COUNT,
                    }))
                    await ws.send(p.encode(p.SyncRequest(t1=1)))
                    reply = p.decode(await asyncio.wait_for(ws.recv(), timeout=5))
                    return reply, server.stats
            finally:
                await server.stop()

        reply, stats = asyncio.run(asyncio.wait_for(scenario(), timeout=15))
        assert isinstance(reply, p.SyncResponse), "壊れた電文の後も同じ接続で時刻同期に応える"
        assert stats["protocol_errors"] == 1
        assert stats["clients"] == 1

    def test_stats_count_points_dropped_by_capture_time(self):
        """撮影時刻で捨てた点は統計（time_rejected）に出す。接続が切れた後も数え続ける。"""

        async def scenario():
            server = LandmarkServer(host="127.0.0.1", port=0, buffer=SyncBuffer())
            await server.start()
            try:
                import websockets

                async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.send(p.encode(p.Hello("cam1", "Pixel 7a", "s", "id-1")))
                    await ws.send(_landmarks_message("cam1", 0, time.monotonic_ns() + 60 * S))
                    await ws.send(p.encode(p.SyncRequest(t1=1)))
                    await asyncio.wait_for(ws.recv(), timeout=5)  # ここまでの電文は処理済み
                    connected = server.stats
                for _ in range(50):
                    if server.stats["clients"] == 0:
                        break
                    await asyncio.sleep(0.02)
                return connected, server.stats
            finally:
                await server.stop()

        connected, finished = asyncio.run(asyncio.wait_for(scenario(), timeout=15))
        assert connected["time_rejected"] == 1
        assert connected["frames_received"] == 1
        assert finished["clients"] == 0 and finished["time_rejected"] == 1

    def test_time_sync_round_trip_over_the_wire(self):
        async def scenario():
            server = LandmarkServer(host="127.0.0.1", port=0, buffer=SyncBuffer())
            await server.start()
            try:
                import websockets

                async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.send(p.encode(p.SyncRequest(t1=1000)))
                    reply = await asyncio.wait_for(ws.recv(), timeout=5)
                return p.decode(reply)
            finally:
                await server.stop()

        response = asyncio.run(asyncio.wait_for(scenario(), timeout=15))
        assert isinstance(response, p.SyncResponse)
        assert response.t1 == 1000

    def test_reports_a_connect_url_for_the_qr_code(self):
        """PC が QR で配る接続先。role と session を含むこと。"""
        server = LandmarkServer(host="127.0.0.1", port=8765, buffer=SyncBuffer())
        url = server.connect_url("cam0")
        assert url.startswith("ws://")
        assert "role=cam0" in url
        assert "session=" in url


class TestPortConflict:
    def test_port_zero_never_conflicts(self):
        from app.net.server import port_conflict

        assert port_conflict(0) is None

    def test_free_port_reports_no_conflict(self):
        import socket

        from app.net.server import port_conflict

        # 一度 bind して即座に閉じ、確実に空いている番号を得る
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
        probe.close()

        assert port_conflict(free_port) is None

    def test_occupied_port_is_detected(self):
        """0.0.0.0 への bind は成功してしまうので、事前に気づけることが重要。"""
        import socket

        from app.net.server import port_conflict

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            message = port_conflict(port)
            assert message is not None
            assert str(port) in message
        finally:
            listener.close()

    def test_start_refuses_to_run_on_an_occupied_port(self):
        import asyncio
        import socket

        from app.net.server import LandmarkServer

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            server = LandmarkServer(host="127.0.0.1", port=port, buffer=SyncBuffer())
            with pytest.raises(OSError) as exc:
                asyncio.run(server.start())
            assert str(port) in str(exc.value)
        finally:
            listener.close()


class TestCalibrationCapture:
    """校正用の撮影指示と、返ってきた画像の受け渡し。

    撮影は PC 側から指示する。端末に撮影ボタンを置くと、ボードを持つ人と
    端末を操作する人の 2 人が要る上、2 台の撮影時刻も揃わない。
    """

    def test_capture_request_reaches_the_phones(self):
        async def scenario():
            server = LandmarkServer(host="127.0.0.1", port=0, buffer=SyncBuffer())
            await server.start()
            try:
                import websockets

                url = f"ws://127.0.0.1:{server.port}"
                async with websockets.connect(url) as cam0, websockets.connect(url) as cam1:
                    await cam0.send(p.encode(p.Hello("cam0", "Pixel 7a", "s", "id-0")))
                    await cam1.send(p.encode(p.Hello("cam1", "Pixel 7a", "s", "id-1")))
                    for _ in range(50):
                        if len(server.connected_roles) == 2:
                            break
                        await asyncio.sleep(0.02)

                    await server.request_capture(7, at_ns=1234)
                    return [
                        p.decode(await asyncio.wait_for(cam0.recv(), timeout=5)),
                        p.decode(await asyncio.wait_for(cam1.recv(), timeout=5)),
                    ]
            finally:
                await server.stop()

        received = asyncio.run(asyncio.wait_for(scenario(), timeout=20))
        assert all(isinstance(m, p.CaptureRequest) for m in received)
        assert {m.id for m in received} == {7}
        assert {m.at_ns for m in received} == {1234}, "両端末に同じ目標時刻を渡すこと"

    def test_calibration_frames_reach_the_callback(self):
        frames: list = []
        handler = SessionHandler(SyncBuffer(), on_calibration_frame=frames.append)

        handler.handle(p.encode(p.CalibrationFrame(
            role="cam0", id=3, t_capture_ns=1, width=1280, height=720, jpeg=b"\xff\xd8\xff\xd9",
        )))

        assert len(frames) == 1
        assert frames[0].role == "cam0"

    def test_devices_are_reported_per_role(self):
        """どの端末がどちらの役割で繋がったか。校正結果との照合に使う。"""

        async def scenario():
            server = LandmarkServer(host="127.0.0.1", port=0, buffer=SyncBuffer())
            await server.start()
            try:
                import websockets

                async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.send(p.encode(p.Hello("cam0", "Pixel 7a", "s", "id-0")))
                    for _ in range(50):
                        if server.devices:
                            break
                        await asyncio.sleep(0.02)
                    return dict(server.devices)
            finally:
                await server.stop()

        devices = asyncio.run(asyncio.wait_for(scenario(), timeout=20))
        assert devices["cam0"].device_id == "id-0"

    def test_a_large_calibration_frame_fits_through(self):
        """720p の JPEG は base64 にすると 1MiB を超えることがある。

        websockets の既定の上限は 1MiB で、超えると接続ごと切れる。
        校正の途中で端末が落ちる形になり、原因が分かりにくい。
        """

        async def scenario():
            frames: list = []
            server = LandmarkServer(
                host="127.0.0.1", port=0, buffer=SyncBuffer(),
                on_calibration_frame=frames.append,
            )
            await server.start()
            try:
                import websockets

                big = b"\xff\xd8" + b"\x00" * 1_500_000
                message = p.encode(p.CalibrationFrame(
                    role="cam0", id=1, t_capture_ns=1, width=1280, height=720, jpeg=big,
                ))
                async with websockets.connect(
                    f"ws://127.0.0.1:{server.port}", max_size=None
                ) as ws:
                    await ws.send(message)
                    for _ in range(100):
                        if frames:
                            break
                        await asyncio.sleep(0.02)
                return frames
            finally:
                await server.stop()

        frames = asyncio.run(asyncio.wait_for(scenario(), timeout=30))
        assert len(frames) == 1, "大きな校正フレームが届いていない"

class TestHelloScreening:
    """名乗りの段階で受け入れるかを決める。

    設置ファイルと違う端末をそのまま受け入れると、別の機種の内部パラメータで
    三角測量することになる。値は出るが正しくない、という最悪の壊れ方をする。
    """

    def test_rejected_hello_is_not_counted_as_connected(self):
        handler = SessionHandler(SyncBuffer(), on_hello=lambda hello: "知らない端末です")

        reply = handler.handle(p.encode(p.Hello("cam0", "別の機種", "s", "id-x")))

        assert handler.rejection == "知らない端末です"
        assert handler.role is None, "拒否した接続を役割として数えない"
        assert reply is None

    def test_accepted_hello_records_the_role(self):
        handler = SessionHandler(SyncBuffer(), on_hello=lambda hello: None)

        handler.handle(p.encode(p.Hello("cam0", "Pixel 7a", "s", "id-0")))

        assert handler.role == "cam0"
        assert handler.rejection is None


class TestCloseReason:
    """close の理由は UTF-8 で 123 バイトまで。超えると相手に 1011 が届き、理由が消える。"""

    def test_long_japanese_reason_fits_in_123_bytes(self):
        from app.net.server import close_reason

        reason = close_reason("あ" * 60)
        assert len(reason.encode("utf-8")) <= 123
        assert reason == "あ" * 41, "文字の途中で切らないこと"

    def test_short_reason_is_unchanged(self):
        from app.net.server import close_reason

        assert close_reason("QR を読み直してください") == "QR を読み直してください"


async def _silent_client(port: int, hello: p.Hello):
    """名乗った後は何も読まず、close にも応えない端末（電源が落ちた端末の代わり）。

    websockets のクライアントは裏で close に応えてしまうので、生のソケットで
    ハンドシェイクと 1 通だけを送る。
    """
    import base64
    import os

    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    key = base64.b64encode(os.urandom(16)).decode()
    writer.write(
        (
            f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ).encode()
    )
    await reader.readuntil(b"\r\n\r\n")

    payload = p.encode(hello).encode()
    assert len(payload) < 126
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    writer.write(bytes([0x81, 0x80 | len(payload)]) + mask + masked)
    await writer.drain()
    return reader, writer


async def _closed_code(ws) -> int | None:
    """相手に閉じられるまで読み捨て、閉じたときのコードを返す。"""
    import websockets

    try:
        while True:
            await asyncio.wait_for(ws.recv(), timeout=5)
    except websockets.exceptions.ConnectionClosed as exc:
        return exc.rcvd.code if exc.rcvd is not None else None


class TestRemoteRoles:
    """混成構成では cam0 を PC のカメラが受け持つ。端末には cam1 だけを許す。

    端末が誤って cam0 を名乗ると、PC のカメラと同じ役割の点が 1 つのバッファに
    混ざり、補間が 2 台のカメラの間を行き来する。値は出るが意味が無い。
    """

    def test_hello_for_a_local_role_is_rejected(self):
        handler = SessionHandler(SyncBuffer(), remote_roles=("cam1",))

        handler.handle(p.encode(p.Hello("cam0", "Pixel 7a", "s", "id-0")))

        assert handler.role is None
        assert handler.rejection is not None and "cam0" in handler.rejection

    def test_frames_for_a_local_role_are_dropped(self):
        buffer = SyncBuffer(grid=GridSpec(target_hz=10.0))
        handler = SessionHandler(buffer, remote_roles=("cam1",))

        handler.handle(_landmarks_message("cam0", 0, 0))

        assert buffer.buffered_count("cam0") == 0
        assert handler.rejected == 1

    def test_frames_must_match_the_announced_role(self):
        """cam1 と名乗った接続から cam0 の点が来たら使わない。"""
        buffer = SyncBuffer(grid=GridSpec(target_hz=10.0))
        handler = SessionHandler(buffer)
        handler.handle(p.encode(p.Hello("cam1", "Pixel 7a", "s", "id-1")))

        handler.handle(_landmarks_message("cam0", 0, 0))

        assert buffer.buffered_count("cam0") == 0
        assert handler.rejected == 1

    def test_local_role_is_closed_with_policy_violation(self):
        """端末側で理由を表示できるよう、1008 と理由を付けて閉じる。"""

        async def scenario():
            server = LandmarkServer(
                host="127.0.0.1", port=0, buffer=SyncBuffer(), remote_roles=("cam1",)
            )
            await server.start()
            try:
                import websockets

                async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.send(p.encode(p.Hello("cam0", "Pixel 7a", "s", "id-0")))
                    return await _closed_code(ws)
            finally:
                await server.stop()

        assert asyncio.run(asyncio.wait_for(scenario(), timeout=15)) == 1008

    def test_newer_connection_takes_over_the_role(self):
        """アプリを入れ直して QR を読み直したとき、古い接続の切断を待たずに使える。"""

        async def scenario():
            server = LandmarkServer(host="127.0.0.1", port=0, buffer=SyncBuffer())
            await server.start()
            try:
                import websockets

                url = f"ws://127.0.0.1:{server.port}"
                async with websockets.connect(url) as old, websockets.connect(url) as new:
                    await old.send(p.encode(p.Hello("cam1", "Pixel 7a", "s", "id-old")))
                    for _ in range(50):
                        if server.devices:
                            break
                        await asyncio.sleep(0.02)
                    await new.send(p.encode(p.Hello("cam1", "Pixel 7a", "s", "id-new")))
                    old_code = await _closed_code(old)
                    for _ in range(50):
                        if len(server.stats["roles"]) == 1 and server.stats["clients"] == 1:
                            break
                        await asyncio.sleep(0.02)
                    return old_code, server.devices["cam1"].device_id
            finally:
                await server.stop()

        old_code, device_id = asyncio.run(asyncio.wait_for(scenario(), timeout=15))
        assert old_code == 4000
        assert device_id == "id-new"

    def test_takeover_does_not_wait_for_an_unresponsive_old_phone(self):
        """後勝ちが要るのは、古い接続の相手がもう応答しないとき（アプリの再起動など）。

        古い接続を閉じ終わるのを待つと、websockets の close_timeout（10 秒）の間、
        新しい端末の時刻同期が止まる。
        """

        async def scenario():
            server = LandmarkServer(host="127.0.0.1", port=0, buffer=SyncBuffer())
            await server.start()
            silent = None
            try:
                import websockets

                silent = await _silent_client(server.port, p.Hello("cam1", "Pixel 7a", "s", "old"))
                for _ in range(50):
                    if server.devices:
                        break
                    await asyncio.sleep(0.02)

                async with websockets.connect(f"ws://127.0.0.1:{server.port}") as new:
                    await new.send(p.encode(p.Hello("cam1", "Pixel 7a", "s", "new")))
                    started = asyncio.get_running_loop().time()
                    await new.send(p.encode(p.SyncRequest(t1=1)))
                    await asyncio.wait_for(new.recv(), timeout=8)
                    return asyncio.get_running_loop().time() - started
            finally:
                if silent is not None:
                    silent[1].close()
                await server.stop()

        elapsed = asyncio.run(asyncio.wait_for(scenario(), timeout=30))
        assert elapsed < 1.0, f"時刻同期の応答に {elapsed:.1f} 秒かかった"


def _local_frame(t_ns: int, role: str = "cam0") -> p.LandmarkFrame:
    return p.LandmarkFrame(
        role=role,
        seq=0,
        t_capture_ns=t_ns,
        width=1280,
        height=720,
        landmarks=[(0.5, 0.5, 0.0, 1.0)] * p.LANDMARK_COUNT,
    )


class TestInject:
    """PC 自身のカメラの点を、端末の点と同じバッファへ入れる口。

    ループのスレッドから呼ぶ前提（同期バッファにロックが無いため）。
    """

    def test_injected_frames_pair_with_remote_frames(self):
        async def scenario():
            pairs: list = []
            seen: list = []
            server = LandmarkServer(
                host="127.0.0.1",
                port=0,
                buffer=SyncBuffer(grid=GridSpec(target_hz=10.0, max_gap_ms=250.0)),
                on_pairs=pairs.extend,
                on_landmarks=seen.append,
                remote_roles=("cam1",),
            )
            await server.start()
            try:
                import websockets

                async with websockets.connect(f"ws://127.0.0.1:{server.port}") as cam1:
                    await cam1.send(p.encode(p.Hello("cam1", "Pixel 7a", "s", "id-1")))
                    base = time.monotonic_ns()
                    for i in range(6):
                        await cam1.send(_landmarks_message("cam1", i, base + i * 100_000_000))
                    for _ in range(50):
                        if server.stats["frames_received"] == 6:
                            break
                        await asyncio.sleep(0.02)
                    # 端末からはもう何も来ない。注入だけでペアが出ること
                    for i in range(6):
                        server.inject(_local_frame(base + i * 100_000_000))
                    return pairs, seen
            finally:
                await server.stop()

        pairs, seen = asyncio.run(asyncio.wait_for(scenario(), timeout=15))
        assert pairs, "注入したフレームと端末のフレームが組めていない"
        assert all(set(pair.frames) == {"cam0", "cam1"} for pair in pairs)
        assert {frame.role for frame in seen} == {"cam0", "cam1"}, "受信も注入も横から見えること"

    def test_inject_refuses_a_remote_role(self):
        """同じ役割を端末と PC の両方から入れると、補間が 2 台の間を行き来する。"""
        server = LandmarkServer(buffer=SyncBuffer(), remote_roles=("cam1",))
        with pytest.raises(ValueError):
            server.inject(_local_frame(0, role="cam1"))

    def test_inject_refuses_a_malformed_frame(self):
        """注入は電文の検証を通らない。点数が違えば三角測量の索引がずれる。"""
        server = LandmarkServer(buffer=SyncBuffer(), remote_roles=("cam1",))
        frame = p.LandmarkFrame(
            role="cam0", seq=0, t_capture_ns=0, width=1280, height=720,
            landmarks=[(0.5, 0.5, 0.0, 1.0)] * 12,
        )
        with pytest.raises(ValueError):
            server.inject(frame)

    def test_capture_request_carries_size_and_quality(self):
        async def scenario():
            server = LandmarkServer(host="127.0.0.1", port=0, buffer=SyncBuffer())
            await server.start()
            try:
                import websockets

                async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                    await ws.send(p.encode(p.Hello("cam1", "Pixel 7a", "s", "id-1")))
                    for _ in range(50):
                        if server.devices:
                            break
                        await asyncio.sleep(0.02)
                    await server.request_capture(5, max_width=640, quality=70)
                    return p.decode(await asyncio.wait_for(ws.recv(), timeout=5))
            finally:
                await server.stop()

        request = asyncio.run(asyncio.wait_for(scenario(), timeout=15))
        assert request == p.CaptureRequest(id=5, max_width=640, quality=70)
