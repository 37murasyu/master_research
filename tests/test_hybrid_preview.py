import time
from types import SimpleNamespace
import cv2 as cv
import numpy as np
import pytest
from app.hybrid.mac_camera import MacCamera
from app.hybrid.pose_detector import PoseDetector
from app.hybrid.display import compose
from app.hybrid.live import LiveSession
from app.hybrid.link import PhoneLink, PREVIEW
from app.net.mock_sender import MockPhone, synthetic_capture
from app.core.stop_request import StopRequest
from app.runners.hybrid_preview import run_session
from hybrid_fakes import FakeCamera, FakeDetector
from test_hybrid_link import _run_phone


def test_corrupt_capture_is_skipped_and_not_returned():
    """壊れた JPEG は捨て、呼び出し側へ返さない。

    返すと、校正ランナーが同じ JPEG を復号し直して例外で落ち、集めた盤が消えていた。
    """
    from app.hybrid.link import LinkStatus
    from app.net.protocol import CalibrationFrame

    corrupt = CalibrationFrame("cam1", 1, 1, 640, 360, b"\xff\xd8" + b"\x00" * 16)
    link = SimpleNamespace(
        inject=lambda frame: None,
        take_capture=lambda: corrupt,
        nearest_remote=lambda t, tolerance: None,
        status=lambda: LinkStatus(),
        url="ws://127.0.0.1:1/?session=s&role=cam1",
    )
    session = LiveSession(FakeCamera(), FakeDetector(), link, output=lambda image: None)

    assert session.step() is None
    assert session.remote_image is None
    assert session.bad_captures == 1


def test_qr_is_readable():
    url = "ws://192.168.1.10:8765/?session=1234567890abcdef&role=cam1"
    panel = compose(None, None, url=url)
    decoded, _, _ = cv.QRCodeDetector().detectAndDecode(panel[:, 640:])
    assert decoded == url


def test_camera_checks_resolution_and_timestamp_order():
    events = []

    class Capture:
        def set(self, *args):
            pass

        def read(self):
            return True, np.zeros((720, 1280, 3), np.uint8)

        def grab(self):
            events.append("grab")
            return True

        def retrieve(self):
            events.append("retrieve")
            return self.read()

        def release(self):
            events.append("release")

    def clock():
        events.append("clock")
        return 123

    camera = MacCamera(opener=lambda _: (None, Capture()), clock=clock)
    assert camera.read()[0] == 123
    assert events == ["grab", "clock", "retrieve"]
    camera.close()

    class Wrong(Capture):
        def read(self):
            return True, np.zeros((480, 640, 3), np.uint8)

    with pytest.raises(ValueError, match="1280"):
        MacCamera(opener=lambda _: (None, Wrong()))
    assert events[-1] == "release"


def test_video_timestamps_and_missing_visibility():
    stamps = []
    landmark = SimpleNamespace(x=0.2, y=0.3, z=0, visibility=None)

    class Detector:
        def detect_for_video(self, image, stamp):
            stamps.append(stamp)
            return SimpleNamespace(pose_landmarks=[[landmark] * 33])

        def close(self):
            pass

    detector = PoseDetector(detector=Detector(), image_factory=lambda x: x)
    for t in [2000000, 2000000, 1000000]:
        points = detector.detect(np.zeros((10, 10, 3), np.uint8), t)
    assert stamps == [2, 3, 4]
    assert points[0][3] == 1.0


def test_live_session_with_phone():
    images = []
    link = PhoneLink(host="127.0.0.1", port=0, capture_mode=PREVIEW)
    link.start()
    phone = MockPhone(
        link.url, "cam1", session=link.session, capture_fn=synthetic_capture
    )
    thread = _run_phone(phone, 1.5)
    session = LiveSession(FakeCamera(), FakeDetector(), link, output=images.append)
    try:
        until = time.monotonic() + 1.7
        while time.monotonic() < until:
            session.step()
            time.sleep(0.03)
        assert session.remote_image is not None
        assert link.status().pairs > 0
        assert images[-1].shape == (440, 1280, 3)
    finally:
        thread.join(5)
        link.stop()
    assert not thread.errors


def test_stop_file_exits_without_camera_read(tmp_path):
    stop = tmp_path / "stop"
    stop.touch()
    session = SimpleNamespace(step=lambda: pytest.fail("must stop before next frame"))
    run_session(session, StopRequest(stop), poll_key=lambda: -1)
