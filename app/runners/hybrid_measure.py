"""Calibrated Mac + Pixel measurement CLI.

ゲージ（GUI の Qt 窓）は子の標準出力の ``@@GAUGE {json}`` の行（``app.gauge.protocol``）だけを見て動く。
メインループの 1 周（``live.step()``、Mac のカメラの約 30 Hz）ごとに 1 行を ``sys.stdout.write`` の 1 回で出す
（``GaugeTicker``。間引かない）。値は受信スレッド（``MeasurementSession.on_pairs``）が ``GaugeTracker`` に積む。
重い import とファイルの読み込み（1RM の表・EKF のプロファイル）は、受信が始まる前にメインスレッドで済ませる。
"""

import argparse
from contextlib import ExitStack
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import cv2 as cv
import config
from app.core.stop_request import StopRequest
from app.gauge.thresholds import PARTS, load_one_rm, subject_index
from app.gauge.tracker import GaugeTicker, GaugeTracker
from app.hybrid.calibration_io import load_calibration, mac_identity
from app.hybrid.session import stable_session
from app.hybrid.link import PhoneLink, CaptureMode
from app.hybrid.live import LiveSession
from app.hybrid.mac_camera import MacCamera, default_camera_index
from app.hybrid.measurement import MeasurementSession
from app.hybrid.pose_detector import PoseDetector
from app.runners.hybrid_preview import poll_window
from app.hybrid.demo_gauge import DemoConfig
from app.hybrid.ekf import EkfSettings
from energy_pipeline import EnergyFilterConfig
from app.hybrid.gravity import candidate_axes
from app.runners.network_measure import MeasurementConfig

# 真偽の設定は USB と同じ読み方（大文字小文字と前後の空白は問わない）
_flag = config.env_flag


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
    except (OSError, ValueError, csv.Error) as exc:
        # 無い・読めない（OSError）だけでなく、文字コード違い（UnicodeDecodeError）や壊れた CSV でも計測は止めない
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
    ambiguity = config.env_float("GRAVITY_AMBIG_DELTA", 0.08)
    return MeasurementConfig(
        body_mass_kg=body_mass_kg,
        gravity_mode=gravity_mode,
        gravity_candidates=tuple(candidate_axes(level_plane, os.environ.get("GRAVITY_LEVEL_PLANE", "YZ"))),
        gravity_ambiguity=ambiguity,
        one_rm=one_rm,
        subject_id=subject,
        dyn_gate=_flag("HYBRID_DYN_GATE", True),
        ekf=EkfSettings.from_env(),
        energy_filter=EnergyFilterConfig.from_env(),
        demo=DemoConfig.from_env() if _flag("DEMO_MONO_GAUGE_ON", False) else None,
        offline_wrist_capture=_flag("OFFLINE_WRIST_CAPTURE", False),
    )


def _default_body_mass(fallback: float = 60.0) -> float:
    """設定 BODY_MASS_KG を読む。数でなければ（NaN・inf も）既定値（起動の段階で落とさない）。"""
    raw = os.environ.get("BODY_MASS_KG")
    if not raw:
        return fallback
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        print(f"BODY_MASS_KG={raw!r} は数ではないため {fallback} kg を使います", file=sys.stderr)
        return fallback
    return value


def check_mac_camera(calibration, index) -> str | None:
    """今開く Mac のカメラ（番号 ``index``）が校正したカメラと同じかを確かめる。違えば止める理由を返す。

    識別子は校正ランナーが meta の ``cameras[0].device_id`` に書いたのと同じ ``mac_identity``（``<機種>:camera<番号>``）。
    Camo や iPhone の連係カメラで番号がずれたまま計測すると、別のカメラの画像に校正を当て、3D とトルクが丸ごと
    狂ったまま記録が complete になる。識別子を取れない（sysctl が無い・失敗した）ときと、meta に識別子が無い
    古い校正は確かめようがないので、警告だけ出して続ける。

    見るのは機種と番号だけなので、同じ番号に別の装置が入れ替わったことまでは分からない（``mac_identity`` の注記）。
    """
    cameras = calibration.meta.get("cameras") or [{}]
    expected = cameras[0].get("device_id")
    if not expected:
        print("[Mac カメラ] 校正の meta に Mac のカメラの識別子が無い（古い校正）ため、"
              "校正したカメラと同じかを確かめられない。そのまま計測します", file=sys.stderr)
        return None
    try:
        current = mac_identity(index)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[Mac カメラ] カメラの識別子を取得できない（{exc}）ため、"
              "校正したカメラと同じかを確かめられない。そのまま計測します", file=sys.stderr)
        return None
    if current != expected:
        return (f"校正した Mac のカメラ（{expected}）と、今開くカメラ（{current}）が違います。"
                "校正したときのカメラ番号を --camera（設定 CAM0）で指定するか、校正し直してください")
    return None


def main(argv=None):
    # 常に実機（Mac のカメラと Pixel）。記録の再生は別の role（app.entry.REPLAY_ROLE）で、環境変数では切り替えない
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
    # NaN は比べるとどれも偽なので、有限かを先に見る（通すと記録の開始で meta.json を書けずに落ちる）
    if not (math.isfinite(args.body_mass) and args.body_mass > 0
            and math.isfinite(args.preview_hz) and args.preview_hz >= 0):
        parser.error("体重は正の値、表示 Hz は0以上の値（どちらも NaN・inf は不可）にしてください")
    stop = StopRequest.from_environment()
    stop.install_signal_handlers()
    try:
        calibration = load_calibration(args.calibration)
        # 開く前に確かめる。記録は Pixel の最初の点で始まるので、ここで止めれば計測フォルダは残らない
        mismatch = check_mac_camera(calibration, args.camera)
        if mismatch:
            print(mismatch, file=sys.stderr)
            return 2
        with ExitStack() as stack:
            try:
                camera = MacCamera(args.camera, size=calibration.intrinsics[0].size)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            stack.callback(camera.close)
            detector = PoseDetector()
            stack.callback(detector.close)
            # 名前を config にすると、main の中ではモジュールの config が隠れる（Python は関数全体で局所とみなす）
            measure_config = measurement_config(args.body_mass, args.gravity_mode)
            tracker = GaugeTracker(source="demo" if measure_config.demo is not None else "measure")
            ticker = GaugeTicker(tracker)
            measurement = MeasurementSession(
                calibration,
                config=measure_config,
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
                # 同期バッファの格子を計測（NetworkMeasurement・EKF・記録）と同じものにする
                grid=measure_config.grid,
                capture_mode=CaptureMode(args.preview_hz, 640, 70),
                on_pairs=measurement.on_pairs,
                on_landmarks=measurement.on_landmarks,
                on_hello=measurement.check_hello,
                accept_frame=measurement.accept_frame,
                on_tick=measurement.flush,
                on_stop=measurement.close,
            )
            # ゲージの接続表示（measurement.flush）は、Pixel が名乗った接続が残っているかも見る
            measurement.remote_connected = lambda: link.remote_role in link.status().devices
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
