"""CLI: python -m app.runners.hybrid_preview --camera 0."""

import argparse
from contextlib import ExitStack
import sys
import cv2 as cv
from app.core.stop_request import StopRequest
from app.hybrid.mac_camera import MacCamera, default_camera_index
from app.hybrid.pose_detector import PoseDetector
from app.hybrid.session import stable_session
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
    parser.add_argument("--camera", type=int, default=None, help="Mac のカメラ番号（既定は CAM0）")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--advertise-host", help="QR に載せる PC のアドレス。既定は自動判定（VPN 接続中は誤ることがある）")
    parser.add_argument("--new-session", action="store_true", help="session を作り直す（前に QR を読んだ端末は自動でつながらなくなる）")
    args = parser.parse_args(argv)
    if args.camera is None:
        args.camera = default_camera_index()
    stop = StopRequest.from_environment()
    stop.install_signal_handlers()
    with ExitStack() as stack:
        camera = MacCamera(args.camera)
        stack.callback(camera.close)
        detector = PoseDetector()
        stack.callback(detector.close)
        link = PhoneLink(
            port=args.port, advertise_host=args.advertise_host,
            session=stable_session(renew=args.new_session), capture_mode=PREVIEW
        )
        link.start()
        stack.callback(link.stop)
        stack.callback(cv.destroyAllWindows)
        print(f"Pixel cam1: {link.url}")
        run_session(LiveSession(camera, detector, link), stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
