"""受信サーバを検証する。

トランスポート（WebSocket）とメッセージ処理を分けてある。
``SessionHandler`` は 1 接続ぶんの処理を担う純粋なクラスで、
ソケットを開かずに検証できる。``LandmarkServer`` は実際に通信する層で、
こちらは統合テストで確かめる。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.net import protocol as p
from app.net.server import LandmarkServer, SessionHandler
from app.net.sync_buffer import SyncBuffer


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
        buffer = SyncBuffer(target_hz=10.0)
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

    def test_keeps_working_after_a_malformed_message(self):
        buffer = SyncBuffer(target_hz=10.0)
        handler = SessionHandler(buffer, clock=FakeClock())
        handler.handle("{壊れている")
        handler.handle(_landmarks_message("cam0", 0, 0))
        assert buffer.buffered_count("cam0") == 1


class TestServerIntegration:
    def test_two_clients_produce_paired_samples(self):
        """2 台ぶん繋いで、時刻の揃ったペアが取り出せること。"""

        async def scenario():
            pairs: list = []
            server = LandmarkServer(
                host="127.0.0.1",
                port=0,  # 空きポートを自動で取る
                buffer=SyncBuffer(target_hz=10.0, max_gap_ms=250.0),
                on_pairs=pairs.extend,
            )
            await server.start()
            try:
                import websockets

                url = f"ws://127.0.0.1:{server.port}"
                async with websockets.connect(url) as cam0, websockets.connect(url) as cam1:
                    await cam0.send(p.encode(p.Hello("cam0", "test", "s")))
                    await cam1.send(p.encode(p.Hello("cam1", "test", "s")))

                    for i in range(6):
                        t = i * 100_000_000  # 100ms 刻み
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
