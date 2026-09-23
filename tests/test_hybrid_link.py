"""混成構成の受信側（PC のカメラ＝cam0、Wi-Fi の端末＝cam1）を検証する。

``PhoneLink`` は asyncio のサーバを背景スレッドで回し、メインスレッド（カメラと表示）と
受け渡す。同期バッファにロックが無いので、PC 側の点はループのスレッドへ渡して入れる。
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.hybrid.link import CALIBRATION, OFF, PREVIEW, CaptureMode, CaptureScheduler, PhoneLink
from app.net import protocol as p
from app.net.mock_sender import MockPhone, synthetic_capture, synthetic_pose


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class TestCaptureScheduler:
    """撮影要求の出し方。端末の JPEG 化が追いつかないまま要求を溜めない。"""

    def test_off_issues_nothing(self):
        scheduler = CaptureScheduler(OFF, clock=FakeClock())
        assert scheduler.next_request() is None

    def test_only_one_request_is_outstanding(self):
        scheduler = CaptureScheduler(CaptureMode(hz=100.0), clock=FakeClock())
        first = scheduler.next_request()
        assert first is not None
        assert scheduler.next_request() is None, "応答が来る前に次を出さない"

    def test_next_request_waits_for_the_interval(self):
        clock = FakeClock()
        scheduler = CaptureScheduler(CaptureMode(hz=4.0), clock=clock)
        first = scheduler.next_request()
        assert scheduler.complete(first.id)

        clock.now += 0.1
        assert scheduler.next_request() is None, "4 Hz なら 0.25 秒は空ける"
        clock.now += 0.2
        second = scheduler.next_request()
        assert second is not None and second.id != first.id

    def test_gives_up_on_a_lost_request(self):
        """電文が消えたり端末が撮影に対応していなかったりしても止まらない。"""
        clock = FakeClock()
        scheduler = CaptureScheduler(CaptureMode(hz=4.0), timeout_s=1.0, clock=clock)
        scheduler.next_request()

        clock.now += 0.9
        assert scheduler.next_request() is None
        clock.now += 0.2
        assert scheduler.next_request() is not None
        assert scheduler.timeouts == 1

    def test_unknown_reply_is_not_counted(self):
        scheduler = CaptureScheduler(CaptureMode(hz=4.0), clock=FakeClock())
        scheduler.next_request()
        assert not scheduler.complete(999)

    def test_mode_sets_size_and_quality(self):
        scheduler = CaptureScheduler(PREVIEW, clock=FakeClock())
        request = scheduler.next_request()
        assert (request.max_width, request.quality) == (PREVIEW.max_width, PREVIEW.quality)

        scheduler.complete(request.id)
        scheduler.set_mode(CALIBRATION)
        scheduler._clock.now += 10
        request = scheduler.next_request()
        assert request.max_width is None, "校正は全解像度"
        assert request.quality == CALIBRATION.quality


def _local_frame(t_ns: int) -> p.LandmarkFrame:
    return p.LandmarkFrame(
        role="cam0",
        seq=0,
        t_capture_ns=t_ns,
        width=1280,
        height=720,
        landmarks=synthetic_pose(t_ns / 1e9, "cam0"),
    )


def _run_phone(phone: MockPhone, duration: float) -> threading.Thread:
    """模擬端末を別スレッドで流す（PhoneLink のループとは別のループ）。"""
    errors: list[BaseException] = []

    def target() -> None:
        try:
            asyncio.run(phone.run(duration))
        except BaseException as exc:  # pragma: no cover - 失敗時にテストへ伝える
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.errors = errors  # type: ignore[attr-defined]
    thread.start()
    return thread


def _inject_for(link: PhoneLink, seconds: float, fps: float = 30.0) -> int:
    """PC のカメラの代わりに、メインスレッドから cam0 を流す。"""
    sent = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        link.inject(_local_frame(time.monotonic_ns()))
        sent += 1
        time.sleep(1.0 / fps)
    return sent


def _wait_until(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


class TestPhoneLink:
    def test_local_and_remote_frames_become_pairs(self):
        pairs: list = []
        link = PhoneLink(host="127.0.0.1", port=0, on_pairs=pairs.extend)
        link.start()
        try:
            phone = MockPhone(link.url, "cam1", session=link.session, device_id="pixel-1")
            thread = _run_phone(phone, duration=2.5)
            assert _wait_until(lambda: link.status().devices), "端末が名乗れていない"
            running = link.status()
            _inject_for(link, seconds=2.5)
            thread.join(timeout=10)
            assert not thread.errors, thread.errors
        finally:
            link.stop()

        assert pairs, "ペアが 1 つも出ていない"
        assert all(set(pair.frames) == {"cam0", "cam1"} for pair in pairs)
        # 端末は同期に少し時間を使うので、端末の送信数を基準にする
        assert len(pairs) >= phone.sent * 0.7, f"ペアが少ない: {len(pairs)} / 端末 {phone.sent}"
        assert running.devices["cam1"].device_id == "pixel-1"
        assert not link.status().devices, "停止後は端末を数えない"

    def test_preview_images_arrive_at_the_requested_size(self):
        import cv2 as cv
        import numpy as np

        link = PhoneLink(host="127.0.0.1", port=0, capture_mode=PREVIEW)
        link.start()
        try:
            phone = MockPhone(
                link.url, "cam1", session=link.session, capture_fn=synthetic_capture
            )
            thread = _run_phone(phone, duration=3.0)
            captured = []
            assert _wait_until(lambda: captured.append(link.take_capture()) or captured[-1], 5.0)
            thread.join(timeout=10)
        finally:
            link.stop()

        frame = captured[-1]
        image = cv.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv.IMREAD_COLOR)
        assert image.shape[:2] == (360, 640)
        assert (frame.width, frame.height) == (640, 360)
        # 停止後はもう届かない。取り出した画像を 2 度返さないこと
        link.take_capture()
        assert link.take_capture() is None, "同じ画像を 2 度返さない"

    def test_nearest_remote_landmarks_follow_the_image_time(self):
        """Pixel の画像に重ねる骨格は、画像と同じ頃の点を使う。"""
        link = PhoneLink(host="127.0.0.1", port=0)
        link.start()
        try:
            phone = MockPhone(link.url, "cam1", session=link.session)
            thread = _run_phone(phone, duration=1.5)
            thread.join(timeout=10)
            assert _wait_until(lambda: link.status().frames_received > 10)
            now = time.monotonic_ns()
            nearest = link.nearest_remote(now, tolerance_ns=5_000_000_000)
            far = link.nearest_remote(now + 60_000_000_000, tolerance_ns=20_000_000)
        finally:
            link.stop()

        assert nearest is not None and nearest.role == "cam1"
        assert far is None, "許容幅を超えて離れた点は重ねない"

    def test_stale_qr_is_rejected(self):
        """前回起動したときの QR で繋いだ端末は断る。"""
        import websockets

        link = PhoneLink(host="127.0.0.1", port=0)
        link.start()

        async def connect_with_old_session() -> int | None:
            async with websockets.connect(link.url) as ws:
                await ws.send(p.encode(p.Hello("cam1", "Pixel 7a", "old-session", "id")))
                try:
                    while True:
                        await asyncio.wait_for(ws.recv(), timeout=5)
                except websockets.exceptions.ConnectionClosed as exc:
                    return exc.rcvd.code if exc.rcvd is not None else None

        try:
            code = asyncio.run(connect_with_old_session())
        finally:
            link.stop()
        assert code == 1008

    def test_inject_checks_the_frame_in_the_caller_thread(self):
        """ループの中で例外になると呼び出し側に届かない。渡す前に確かめる。"""
        link = PhoneLink(host="127.0.0.1", port=0)
        link.start()
        try:
            bad = p.LandmarkFrame(
                role="cam1", seq=0, t_capture_ns=0, width=1280, height=720,
                landmarks=synthetic_pose(0.0, "cam1"),
            )
            with pytest.raises(ValueError):
                link.inject(bad)
        finally:
            link.stop()

    def test_callback_errors_are_counted_and_do_not_stop_the_link(self):
        def broken(pairs) -> None:
            raise RuntimeError("計算側の不具合")

        link = PhoneLink(host="127.0.0.1", port=0, on_pairs=broken)
        link.start()
        try:
            phone = MockPhone(link.url, "cam1", session=link.session)
            thread = _run_phone(phone, duration=1.5)
            assert _wait_until(lambda: link.status().devices)
            _inject_for(link, seconds=1.5)
            thread.join(timeout=10)
            status = link.status()
        finally:
            link.stop()

        assert status.callback_errors > 0
        assert status.frames_received > 10, "例外の後も受信を続けていること"

    def test_occupied_port_fails_at_start(self):
        import socket

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            link = PhoneLink(host="127.0.0.1", port=port)
            with pytest.raises(OSError):
                link.start()
        finally:
            listener.close()
