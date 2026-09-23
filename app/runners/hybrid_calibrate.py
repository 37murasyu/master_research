"""Guided Mac + Pixel calibration: preview → collect → review/save."""

import argparse
from contextlib import ExitStack
import sys
import time
import cv2 as cv
import numpy as np
from app.core import resources
from app.core.stop_request import StopRequest
from app.hybrid.calibration_io import (
    cache_key,
    mac_identity,
    load_intrinsics,
    save_intrinsics,
    save_calibration,
)
from app.hybrid.checkerboard import (
    Board,
    calibrate_intrinsics,
    calibrate_stereo,
    max_view_error,
    reprojections,
)
from app.hybrid.collector import BoardCollector
from app.hybrid.display import compose
from app.hybrid.live import LiveSession
from app.hybrid.link import PhoneLink, PREVIEW, CALIBRATION, OFF
from app.hybrid.mac_camera import MacCamera, default_camera_index
from app.hybrid.pose_detector import PoseDetector
from app.runners.hybrid_preview import poll_window

# 保存済みの内部パラメータ（キャッシュ）を使ってよい再投影誤差の上限 [px]。
# OpenCV のカメラ番号は入れ替わる（Camo や iPhone の連係カメラ）ので、キャッシュの鍵が
# 合っても別のカメラのものを引きうる。超えたら捨てて、単体ビューから求め直す。
CACHE_TOLERANCE_PX = 1.5


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
    parser.add_argument("--rows", type=int, default=defaults["rows"])
    parser.add_argument("--cols", type=int, default=defaults["cols"])
    parser.add_argument("--square-cm", type=float, default=defaults["square_cm"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
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
        link = PhoneLink(port=args.port, capture_mode=PREVIEW)
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
                            f"単体 Mac {len(collector.mono[0])}/15  Pixel {len(collector.mono[1])}/15 / ペア {len(collector.pairs)}/12",
                            "盤を静止させてください。採用後は位置・距離・傾きを大きく変えます",
                            f"キャッシュ Mac: {bool(cached[0])} Pixel: {bool(cached[1])}",
                        )
                    )
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
