"""Guided Mac + Pixel calibration: preview → collect → review/save → 盤を立てて静止（任意）。"""

import argparse
from contextlib import ExitStack
import sys
import time
import cv2 as cv
import numpy as np
import config
from app.core import resources
from app.core.stop_request import StopRequest
from app.hybrid.calibration_io import (
    cache_key,
    mac_identity,
    load_intrinsics,
    save_intrinsics,
    save_calibration,
    update_meta,
)
from app.hybrid.checkerboard import (
    Board,
    calibrate_intrinsics,
    calibrate_stereo,
    max_view_error,
    reprojections,
)
from app.hybrid.collector import REASONS, STILL_PX, BoardCollector
from app.hybrid.display import compose
from app.hybrid.gravity_board import MAX_TILT_DEG, STILL_PX as UPRIGHT_STILL_PX, WARN_TILT_DEG, UprightCollector
from app.hybrid.live import LiveSession
from app.hybrid.session import stable_session
from app.hybrid.link import PhoneLink, PREVIEW, CALIBRATION, OFF
from app.hybrid.mac_camera import MacCamera, default_camera_index
from app.hybrid.pose_detector import PoseDetector
from app.runners.hybrid_preview import poll_window

# 保存済みの内部パラメータ（キャッシュ）を使ってよい再投影誤差の上限 [px]。
# OpenCV のカメラ番号は入れ替わる（Camo や iPhone の連係カメラ）ので、キャッシュの鍵が
# 合っても別のカメラのものを引きうる。超えたら捨てて、単体ビューから求め直す。
CACHE_TOLERANCE_PX = 1.5
# 盤を立てる段階: 既定の時間切れ [s]、標準出力へ進み具合を出す間隔 [s]、省略のキー（Enter）
BOARD_UP_TIMEOUT_S = 30.0
BOARD_UP_REPORT_S = 2.0
ENTER_KEYS = (10, 13)
UPRIGHT_STATUS = {
    "no_board": "盤が見えない",
    "waiting": "静止を確かめ中",
    "moving": f"動いている（{UPRIGHT_STILL_PX:g} px 以下で採用）",
    "tilted": f"傾きすぎ（{MAX_TILT_DEG:g}° 以下で採用）",
    "accepted": "採用",
}


def rejection_summary(collector):
    """ペアを見送った理由を、多い順に短く並べる。使う人が何を直せばよいか分かるように。"""
    top = collector.reasons.most_common(3)
    motion = collector.last_motion_px
    text = "見送り: " + ("・".join(f"{REASONS[k]} {n}" for k, n in top) if top else "なし")
    if motion is not None:
        text += f" / 直近の盤の動き {motion:.1f} px（{STILL_PX:.1f} 以下で採用）"
    return text


def board_up_enabled(choice=None):
    """盤を立てる段階を行うか。引数 ``--board-up on|off`` が優先し、無ければ ``HYBRID_GRAVITY_BOARD``（既定 1）。"""
    if choice is not None:
        return choice == "on"
    return config.env_flag("HYBRID_GRAVITY_BOARD", True)


def board_up_timeout(value=None):
    """盤を立てる段階の時間切れ [s]。引数が優先し、無ければ ``HYBRID_GRAVITY_BOARD_TIMEOUT_S``（既定 30）。"""
    if value is not None:
        return float(value)
    return config.env_float("HYBRID_GRAVITY_BOARD_TIMEOUT_S", BOARD_UP_TIMEOUT_S)


def run_board_up(session, board, intrinsic, directory, stop, *, timeout_s=BOARD_UP_TIMEOUT_S):
    """校正の保存の後、盤を立てて静止させた短辺の上向きを校正フォルダの meta.json に足す。

    記録したら ``checkerboard_short_axis`` の辞書を返す。Enter・n（省略）、q・Esc（中止）、停止の要求、
    時間切れでは記録せず None を返す（どれも校正は保存済みなので終了コードは 0 のまま）。
    """
    collector = UprightCollector(board, intrinsic.K, intrinsic.distortion)
    session.board = board
    print(
        "[盤を立てる] 盤を鉛直に立て（短辺を上下・長辺を水平）、Mac のカメラの正面で静止してください。"
        f"Enter・n: 省略 / q: 中止（{timeout_s:.0f} 秒で時間切れ）。記録すると計測の重力の向きに使います"
    )
    start = last = time.monotonic()
    status = "waiting"
    while True:
        if stop.requested():
            print("[盤を立てる] 停止の要求で終了（盤の向きは記録しない）")
            return None
        now = time.monotonic()
        if now - start >= timeout_s:
            print(f"[盤を立てる] 時間切れ（{timeout_s:.0f} 秒）。盤の向きは記録しない（計測は体幹から重力を決める）")
            return None
        session.step(infer=False, lines=(
            f"盤を立てて静止: 標本 {len(collector.samples)}/{collector.needed}  {UPRIGHT_STATUS[status]}",
            "短辺を上下・長辺を水平に、Mac のカメラの正面で止める / Enter・n: 省略 / q: 中止",
        ))
        status = collector.add(session.local_corners)
        if collector.done:
            entry = collector.result()
            update_meta(directory, checkerboard_short_axis=entry)
            print(
                f"[盤を立てる] 記録: 上向き {entry['up_label_runtime']}、傾き {entry['tilt_deg']:.1f}°、"
                f"ばらつき {entry['spread_deg']:.1f}°（{entry['samples']} 標本）→ {directory}"
            )
            if entry["tilt_deg"] > WARN_TILT_DEG:
                print(
                    f"[盤を立てる][警告] 盤の短辺が Mac のカメラの上向きから {entry['tilt_deg']:.1f}° 傾いている"
                    f"（{WARN_TILT_DEG:g}° 以下が目安）。Mac の天板を鉛直にし、盤をまっすぐ立てて校正をやり直すと確か"
                )
            return entry
        if now - last >= BOARD_UP_REPORT_S:
            last = now
            tilt = collector.last_tilt_deg
            motion = collector.last_motion_px
            print(
                f"[盤を立てる] 標本 {len(collector.samples)}/{collector.needed}、{UPRIGHT_STATUS[status]}"
                + (f"、傾き {tilt:.1f}°" if tilt is not None else "")
                + (f"、動き {motion:.1f} px" if motion is not None else "")
                + f"、残り {max(0.0, timeout_s - (now - start)):.0f} 秒"
            )
        key = poll_window()
        if key in ENTER_KEYS or key == ord("n"):
            print("[盤を立てる] 省略（盤の向きは記録しない。計測は体幹から重力を決める）")
            return None
        if key in (27, ord("q")):
            print("[盤を立てる] 中止（盤の向きは記録しない。校正は保存済み）")
            return None


def board_defaults():
    import yaml

    data = yaml.safe_load(
        (resources.resource_root() / "calibration_settings.yaml").read_text()
    )
    return dict(
        rows=data.get("checkerboard_rows", 4),
        cols=data.get("checkerboard_columns", 7),
        square_cm=data.get("checkerboard_box_size_scale", 3.0),
    )


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    defaults = board_defaults()
    parser = argparse.ArgumentParser(description="混成ステレオ校正（盤寸法は cm）")
    parser.add_argument("--camera", type=int, default=None, help="Mac のカメラ番号（既定は CAM0）")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--advertise-host", help="QR に載せる PC のアドレス。既定は自動判定（VPN 接続中は誤ることがある）")
    parser.add_argument("--new-session", action="store_true", help="session を作り直す（前に QR を読んだ端末は自動でつながらなくなる）")
    parser.add_argument("--rows", type=int, default=defaults["rows"])
    parser.add_argument("--cols", type=int, default=defaults["cols"])
    parser.add_argument("--square-cm", type=float, default=defaults["square_cm"])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--board-up", choices=("on", "off"), default=None,
                        help="保存の後に盤を立てて静止させ、重力の向きを記録する（既定は HYBRID_GRAVITY_BOARD、無ければ on）")
    parser.add_argument("--board-up-timeout", type=float, default=None,
                        help="盤を立てる段階の時間切れ [s]（既定は HYBRID_GRAVITY_BOARD_TIMEOUT_S、無ければ 30）")
    args = parser.parse_args(argv)
    board_up = board_up_enabled(args.board_up)
    board_up_limit = board_up_timeout(args.board_up_timeout)
    if args.camera is None:
        args.camera = default_camera_index()
    board = Board(args.rows, args.cols, args.square_cm)
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
        session = LiveSession(camera, detector, link)
        print(
            f"Pixel cam1: {link.url}\nMac の天板を鉛直に開き、カメラ位置を固定してください。Space: 盤集め開始"
        )
        phase = "preview"
        collector = None
        start = 0
        solution = None
        cached = [None, None]
        keys = [None, None]
        cameras = []
        review = None
        last_report = 0.0

        def restart(message):
            """盤集めを最初からやり直す。例外で落ちると、集めた盤がすべて消える。"""
            nonlocal phase, cameras
            print(message)
            phase = "preview"
            cameras = []
            session.board = None
            link.set_capture_mode(PREVIEW)

        try:
            while not stop.requested():
                if phase == "review":
                    cv.imshow("Mac + Pixel", review)
                else:
                    lines = (
                        (
                            "Space: 盤集め開始 / q: 中止",
                            "Mac の天板を鉛直に。両カメラに盤全体が大きく写るよう向き合わせ",
                        )
                        if phase == "preview"
                        else (
                            f"単体 Mac {len(collector.mono[0])}/15  Pixel {len(collector.mono[1])}/15 / ペア {len(collector.pairs)}/12"
                            f"  （キャッシュ Mac {'あり' if cached[0] else 'なし'}・Pixel {'あり' if cached[1] else 'なし'}）",
                            "両方のカメラに盤全体が写る位置で静止。採用されたら位置・距離・傾きを大きく変えます",
                            rejection_summary(collector),
                        )
                    )
                    if phase == "collect" and time.monotonic() - last_report > 5:
                        last_report = time.monotonic()
                        print(f"[盤集め] ペア {len(collector.pairs)} / {rejection_summary(collector)}")
                    # 返るのは復号できた画像だけ。復号結果と盤の検出結果は session に置かれる
                    capture = session.step(infer=phase == "preview", lines=lines)
                    if phase == "collect" and time.monotonic() - start > 1.2:
                        collector.add_mac(
                            session.local_t_ns,
                            session.local_image,
                            corners=session.local_corners,
                        )
                        if capture is not None:
                            if not cameras:
                                hello = link.status().devices.get("cam1")
                                if hello is None or not hello.device_id:
                                    print(
                                        "端末 ID を取得できません。新しい release 版で QR を読み直してください"
                                    )
                                    return 2
                                cameras = [
                                    dict(
                                        kind="mac",
                                        device=mac_identity(args.camera),
                                        device_id=mac_identity(args.camera),
                                    ),
                                    dict(
                                        kind="pixel",
                                        device=hello.device,
                                        device_id=hello.device_id,
                                    ),
                                ]
                                sizes = [camera.size, (capture.width, capture.height)]
                                keys = [
                                    cache_key(c["kind"], c["device_id"], s)
                                    for c, s in zip(cameras, sizes)
                                ]
                                cached = [
                                    None if args.no_cache else load_intrinsics(k)
                                    for k in keys
                                ]
                                collector.cached = tuple(v is not None for v in cached)
                            try:
                                collector.add_remote(
                                    capture.t_capture_ns,
                                    session.remote_image,
                                    corners=session.remote_corners,
                                )
                            except ValueError as exc:
                                restart(f"{exc}（盤集めを最初からやり直します）")
                    if phase == "collect" and collector.ready:
                        a, b = zip(*collector.pairs)
                        mismatch = {
                            i: max_view_error(board, cached[i], (a, b)[i])
                            for i in (0, 1)
                            if cached[i] is not None
                        }
                        mismatch = {i: e for i, e in mismatch.items() if e > CACHE_TOLERANCE_PX}
                        for i, error in mismatch.items():
                            print(
                                f"cam{i} の保存済み内部パラメータが今回の画像と合いません"
                                f"（再投影誤差 {error:.2f} px）。単体ビューを集めて求め直します"
                            )
                            cached[i] = None
                        if mismatch:
                            # 単体ビューが足りるまで集め続ける（collector.ready が偽に戻る）
                            collector.cached = tuple(v is not None for v in cached)
                        else:
                            try:
                                intrinsics = [
                                    cached[i]
                                    or calibrate_intrinsics(
                                        board, collector.mono[i], collector.sizes[i]
                                    )
                                    for i in (0, 1)
                                ]
                                stereo = calibrate_stereo(board, a, b, *intrinsics)
                            except (ValueError, cv.error) as exc:
                                restart(
                                    f"推定に失敗しました（{exc}）。盤の位置や傾きを大きく変えて集め直します"
                                )
                            else:
                                for i in (0, 1):
                                    cameras[i]["intrinsics_source"] = (
                                        "cache" if cached[i] else "new"
                                    )
                                solution = (*intrinsics, stereo)
                                message = f"RMS {stereo.rms:.3f} px / 基線 {np.linalg.norm(stereo.T):.2f} cm / マス誤差 {stereo.square_error_mm:.3f} mm"
                                print(message)
                                index = stereo.used_indices[-1]
                                images = [
                                    cv.cvtColor(im, cv.COLOR_GRAY2BGR)
                                    for im in collector.pair_images[index]
                                ]
                                overlays = reprojections(
                                    board, a[index], b[index], *intrinsics, stereo
                                )
                                review = compose(
                                    *images,
                                    left_corners=overlays[0],
                                    right_corners=overlays[1],
                                    lines=(message, "s: 保存 / r: やり直し / q: 中止"),
                                )
                                phase = "review"
                                link.set_capture_mode(OFF)
                                if (
                                    max(stereo.rms, *(i.rms for i in intrinsics)) <= 1
                                    and stereo.square_error_mm <= 1
                                ):
                                    cv.imshow("Mac + Pixel", review)
                                    cv.waitKey(1)
                                    directory = save_calibration(
                                        *solution, board, cameras=cameras
                                    )
                                    for key, value in zip(keys, intrinsics):
                                        save_intrinsics(key, value)
                                    print(f"自動保存: {directory}")
                                    if board_up:
                                        run_board_up(session, board, intrinsics[0], directory, stop,
                                                     timeout_s=board_up_limit)
                                    return 0
                key = poll_window()
                if key in (27, ord("q")):
                    break
                if phase == "preview" and key == 32:
                    collector = BoardCollector(board)
                    link.take_capture()
                    link.set_capture_mode(CALIBRATION)
                    phase = "collect"
                    start = time.monotonic()
                    session.board = board
                elif phase == "review" and key == ord("s"):
                    directory = save_calibration(*solution, board, cameras=cameras)
                    for k, v in zip(keys, solution[:2]):
                        save_intrinsics(k, v)
                    print(f"保存: {directory}")
                    if board_up:
                        run_board_up(session, board, solution[0], directory, stop, timeout_s=board_up_limit)
                    return 0
                elif phase == "review" and key == ord("r"):
                    phase = "preview"
                    cameras = []
                    session.board = None
                    link.set_capture_mode(PREVIEW)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
