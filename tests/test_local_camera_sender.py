"""スマホ1台とPCカメラ1台を、同じランドマーク同期経路に流せることを確認する。"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from app.net import protocol as p


def test_camera_frame_uses_capture_time_and_full_image_dimensions():
    """推論後の時刻や縮小画像サイズを送ると、同期と三角測量を誤るため。"""
    from app.net.local_camera_sender import CameraLandmarkReader

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    source = SimpleNamespace(read=lambda: (True, image))
    def process(rgb):
        assert rgb[0, 0].tolist() == [255, 0, 0]
        return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=[
            SimpleNamespace(x=.25, y=.75, z=0., visibility=1.) for _ in range(33)]))
    reader = CameraLandmarkReader(source, SimpleNamespace(process=process), "cam1", clock=lambda: 123)
    frame = reader.read()
    assert frame.t_capture_ns == 123
    assert frame.role == "cam1"
    assert (frame.width, frame.height) == (320, 240)
    assert frame.pixel_xy(0) == (80., 180.)
    assert p.decode(p.encode(frame)) == frame


def test_missing_pose_is_skipped_and_capture_failure_is_reported():
    """未検出を偽の零座標として流さず、カメラ切断で無限ループしないため。"""
    from app.net.local_camera_sender import CameraLandmarkReader

    source = SimpleNamespace(read=lambda: (True, np.zeros((8, 8, 3), dtype=np.uint8)))
    reader = CameraLandmarkReader(source, SimpleNamespace(process=lambda _: SimpleNamespace(pose_landmarks=None)))
    assert reader.read() is None
    source.read = lambda: (False, None)
    with pytest.raises(RuntimeError, match="カメラ"):
        reader.read()


def test_phone_and_local_camera_produce_pairs_without_blocking_server():
    """PC推論中もスマホを受信でき、同じPC時計で2視点のペアを作れることを保証する。"""
    from app.net.local_camera_sender import send_camera
    from app.net.mock_sender import MockPhone, synthetic_pose
    from app.net.server import LandmarkServer
    import time

    class Reader:
        def __init__(self):
            self.count = 0
        def read(self):
            # 同期処理がイベントループを塞ぐ実装だと、サーバの受信が遅延する。
            captured = time.monotonic_ns()
            time.sleep(.01)
            self.count += 1
            return p.LandmarkFrame("cam1", self.count, captured, 1280, 720, synthetic_pose(0, "cam1"))

    async def scenario():
        pairs = []
        server = LandmarkServer(host="127.0.0.1", port=0, on_pairs=pairs.extend)
        await server.start()
        try:
            url = f"ws://127.0.0.1:{server.port}"
            reader = Reader()
            sent, _ = await asyncio.gather(
                send_camera(url, reader, role="cam1", duration_sec=1.0),
                MockPhone(url, "cam0").run(1.0),
            )
            return pairs, sent
        finally:
            await server.stop()
    pairs, sent = asyncio.run(asyncio.wait_for(scenario(), 10))
    assert sent > 5
    assert len(pairs) > 5
    assert all(set(pair.frames) == {"cam0", "cam1"} for pair in pairs)


@pytest.mark.parametrize("fails", [False, True])
def test_sender_closes_camera_and_estimator_on_exit(monkeypatch, fails):
    """送信終了・接続失敗のどちらでも、同じスレッドでカメラを解放するため。"""
    from app.net import local_camera_sender as sender
    import threading

    events = []
    def opened(*_):
        events.append(("open", threading.get_ident()))
        return SimpleNamespace(close=lambda: events.append(("close", threading.get_ident())))
    async def send(*_, **__):
        if fails:
            raise OSError("接続失敗")
        return 1
    monkeypatch.setattr(sender, "open_reader", opened)
    monkeypatch.setattr(sender, "send_camera", send)
    args = SimpleNamespace(camera=0, role="cam1", width=320, height=240, url="ws://localhost:8765",
                           session="test", fps=30., duration=.1)
    if fails:
        with pytest.raises(OSError, match="接続失敗"):
            asyncio.run(sender._main_async(args))
    else:
        assert asyncio.run(sender._main_async(args)) == 0
    assert [name for name, _ in events] == ["open", "close"]
    assert events[0][1] == events[1][1]


def test_sender_rejects_other_pc_clock():
    """時刻補正しないPCカメラを別PCのサーバにつないで同期を壊さないため。"""
    from app.net.local_camera_sender import send_camera

    with pytest.raises(ValueError, match="localhost"):
        asyncio.run(send_camera("ws://192.0.2.1:8765", None))
