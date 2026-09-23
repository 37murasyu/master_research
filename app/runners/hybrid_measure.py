"""Calibrated Mac + Pixel measurement CLI."""

import argparse
from contextlib import ExitStack
import os
import sys
import cv2 as cv
from app.core.stop_request import StopRequest
from app.hybrid.calibration_io import load_calibration
from app.hybrid.session import stable_session
from app.hybrid.link import PhoneLink, CaptureMode
from app.hybrid.live import LiveSession
from app.hybrid.mac_camera import MacCamera, default_camera_index
from app.hybrid.measurement import MeasurementSession
from app.hybrid.pose_detector import PoseDetector
from app.runners.hybrid_preview import poll_window
from app.runners.network_measure import MeasurementConfig


def _default_body_mass(fallback: float = 60.0) -> float:
    """設定 BODY_MASS_KG を読む。数でなければ既定値（起動の段階で落とさない）。"""
    raw = os.environ.get("BODY_MASS_KG")
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        print(f"BODY_MASS_KG={raw!r} は数ではないため {fallback} kg を使います", file=sys.stderr)
        return fallback


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Mac + Pixel 計測")
    parser.add_argument("--calibration", default="latest")
    parser.add_argument("--camera", type=int, default=None, help="Mac のカメラ番号（既定は CAM0）")
    parser.add_argument("--body-mass", type=float, default=None, help="体重 kg（既定は BODY_MASS_KG）")
    parser.add_argument("--gravity-mode", choices=("axis", "trunk"), default="axis")
    parser.add_argument("--preview-hz", type=float, default=2.0)
    parser.add_argument("--cam0-offset-ms", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--advertise-host", help="QR に載せる PC のアドレス。既定は自動判定（VPN 接続中は誤ることがある）")
    parser.add_argument("--new-session", action="store_true", help="session を作り直す（前に QR を読んだ端末は自動でつながらなくなる）")
    args = parser.parse_args(argv)
    if args.camera is None:
        args.camera = default_camera_index()
    if args.body_mass is None:
        args.body_mass = _default_body_mass()
    if args.body_mass <= 0 or args.preview_hz < 0:
        parser.error("体重は正の値、表示 Hz は0以上にしてください")
    stop = StopRequest.from_environment()
    stop.install_signal_handlers()
    try:
        calibration = load_calibration(args.calibration)
        with ExitStack() as stack:
            try:
                camera = MacCamera(args.camera, size=calibration.intrinsics[0].size)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            stack.callback(camera.close)
            detector = PoseDetector()
            stack.callback(detector.close)
            measurement = MeasurementSession(
                calibration,
                config=MeasurementConfig(
                    body_mass_kg=args.body_mass, gravity_mode=args.gravity_mode
                ),
                metadata={
                    "cam0_offset_ms": args.cam0_offset_ms,
                    "preview_hz": args.preview_hz,
                },
            )
            link = PhoneLink(
                port=args.port,
                advertise_host=args.advertise_host,
            session=stable_session(renew=args.new_session),
                capture_mode=CaptureMode(args.preview_hz, 640, 70),
                on_pairs=measurement.on_pairs,
                on_landmarks=measurement.on_landmarks,
                on_hello=measurement.check_hello,
                accept_frame=measurement.accept_frame,
                on_tick=measurement.flush,
                on_stop=measurement.close,
            )
            stack.callback(cv.destroyAllWindows)
            link.start()
            stack.callback(link.stop)
            live = LiveSession(
                camera, detector, link, cam0_offset_ms=args.cam0_offset_ms
            )
            print(f"Pixel cam1: {link.url}")
            try:
                while not stop.requested() and not measurement.failed.is_set():
                    live.step()
                    if poll_window() in (27, ord("q")):
                        measurement.stop_reason = "key"
                        break
            except KeyboardInterrupt:
                measurement.stop_reason = "ctrl_c"
            except Exception as exc:
                measurement.error = str(exc)
                measurement.exit_code = 1
                measurement.stop_reason = "error"
                print(str(exc), file=sys.stderr)
            # 記録は、この with を抜けるとき（link.stop → measurement.close）に閉じる。何で止まったかを先に決めておく
            # （GUI の停止ボタンは停止ファイル、端末からは SIGTERM。どちらも stop.requested()）
            if measurement.stop_reason is None:
                measurement.stop_reason = "stop_request" if stop.requested() else "failed"
        if measurement.directory is None:
            print("Pixel から点が届かなかったため、記録はありません")
        else:
            print(f"保存: {measurement.directory}")
        if measurement.error:
            print(measurement.error, file=sys.stderr)
        return measurement.exit_code
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"計測を開始できません: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
