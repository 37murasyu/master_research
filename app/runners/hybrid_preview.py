"""CLI: python -m app.runners.hybrid_preview --camera 0."""

import argparse
from contextlib import ExitStack
import sys
import cv2 as cv
from app.core.stop_request import StopRequest
from app.hybrid.mac_camera import MacCamera
from app.hybrid.pose_detector import PoseDetector
from app.hybrid.link import PhoneLink, PREVIEW
from app.hybrid.live import LiveSession

WINDOW = "Mac + Pixel"


def poll_window():
    key = cv.waitKey(1) & 0xFF
    try:
        if cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1:
            return ord("q")
    except cv.error:
        return ord("q")
    return key


def run_session(session, stop, *, poll_key=poll_window):
    try:
        while not stop.requested():
            session.step()
            if poll_key() in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Mac + Pixel ライブ表示")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    stop = StopRequest.from_environment()
    stop.install_signal_handlers()
    with ExitStack() as stack:
        camera = MacCamera(args.camera)
        stack.callback(camera.close)
        detector = PoseDetector()
        stack.callback(detector.close)
        link = PhoneLink(port=args.port, capture_mode=PREVIEW)
        link.start()
        stack.callback(link.stop)
        stack.callback(cv.destroyAllWindows)
        print(f"Pixel cam1: {link.url}")
        run_session(LiveSession(camera, detector, link), stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
