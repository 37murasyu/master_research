"""Calibrated Mac + Pixel measurement CLI.

ゲージ（GUI の Qt 窓）は子の標準出力の ``@@GAUGE {json}`` の行（``app.gauge.protocol``）だけを見て動く。
メインループの 1 周（``live.step()``、Mac のカメラの約 30 Hz）ごとに 1 行を ``sys.stdout.write`` の 1 回で出す
（``GaugeTicker``。間引かない）。値は受信スレッド（``MeasurementSession.on_pairs``）が ``GaugeTracker`` に積む。
重い import とファイルの読み込み（1RM の表・EKF のプロファイル）は、受信が始まる前にメインスレッドで済ませる。
"""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import cv2 as cv
import config
from app.core.stop_request import StopRequest
from app.gauge.thresholds import PARTS, load_one_rm, subject_index
from app.gauge.tracker import GaugeTicker, GaugeTracker
from app.hybrid.calibration_io import load_calibration
from app.hybrid.session import stable_session
from app.hybrid.link import PhoneLink, CaptureMode
from app.hybrid.live import LiveSession
from app.hybrid.mac_camera import MacCamera, default_camera_index
from app.hybrid.measurement import MeasurementSession
from app.hybrid.pose_detector import PoseDetector
from app.runners.hybrid_preview import poll_window
from app.hybrid.ekf import EkfSettings
from app.hybrid.gravity import candidate_axes
from app.runners.network_measure import MeasurementConfig

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _flag(name: str, default: bool) -> bool:
    """``config.env_flag`` と同じ読み方（大文字小文字と前後の空白は問わない）。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return True if value in _TRUE else False if value in _FALSE else default


def one_rm_from_env() -> tuple[str | None, dict | None, str | None]:
    """被験者番号・1RM [kg]・1RM が無い理由。表は ``ONE_RM_CSV``、空なら作業フォルダの ``m_max_all_merged.csv``。

    無ければ帯を出さない（ゲージは値だけ動く）。計測は止めない。
    """
    raw = (os.environ.get("SUBJECT_ID") or "").strip() or None
    subject = subject_index(raw)
    if subject is None:
        return raw, None, f"SUBJECT_ID={raw!r} が被験者番号（数字）でない"
    path = Path((os.environ.get("ONE_RM_CSV") or "").strip() or Path(config.folder_path) / "m_max_all_merged.csv")
    try:
        one_rm = load_one_rm(path, subject)
    except OSError as exc:
        return raw, None, f"1RM の表を読めない（{path}: {exc}）"
    missing = [part for part in PARTS if one_rm.get(part) is None]
    reason = f"1RM の表 {path} に被験者 {subject} の {', '.join(missing)} が無い" if missing else None
    return raw, one_rm, reason


def measurement_config(body_mass_kg: float, gravity_mode: str) -> MeasurementConfig:
    """環境変数（GUI が子へ全件渡す）から混成の計測の設定を作る。"""
    subject, one_rm, reason = one_rm_from_env()
    if reason:
        print(f"[ゲージ] {reason}。その部位は帯（W_0.70〜W_0.85）を出さない", file=sys.stderr)
    level_plane = _flag("GRAVITY_LEVEL_PLANE_ON", False)
    try:
        ambiguity = float(os.environ.get("GRAVITY_AMBIG_DELTA") or 0.08)
    except ValueError:
        ambiguity = 0.08
    return MeasurementConfig(
        body_mass_kg=body_mass_kg,
        gravity_mode=gravity_mode,
        gravity_candidates=tuple(candidate_axes(level_plane, os.environ.get("GRAVITY_LEVEL_PLANE", "YZ"))),
        gravity_ambiguity=ambiguity,
        one_rm=one_rm,
        subject_id=subject,
        dyn_gate=_flag("HYBRID_DYN_GATE", True),
        ekf=EkfSettings.from_env(),
    )


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
            tracker = GaugeTracker(source="measure")
            ticker = GaugeTicker(tracker)
            measurement = MeasurementSession(
                calibration,
                config=measurement_config(args.body_mass, args.gravity_mode),
                metadata={
                    "cam0_offset_ms": args.cam0_offset_ms,
                    "preview_hz": args.preview_hz,
                    "mac_camera": getattr(camera, "controls", None),
                },
                tracker=tracker,
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
                    ticker.tick()
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
        ticker.tick(force=True)
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
