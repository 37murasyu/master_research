"""2 台の USB カメラの全フレームを録画する。§6-2・§6-3・§3-2 を、録画を計測に読み込ませて確かめるため。

計測（``master_research_code.py``）自身の録画は、間引きの後のフレームしか書かないうえ fps 30 固定で書くので、
時間が詰まって処理し直せない。そこで録画だけをするツールを別に作った。計測と同じカメラ設定
（``app.core.camera_controls``）で撮り、2 台を並列に grab して全フレームを書く。

出力は ``<out>/<label>_<MMDD_HHMMSS>/``。受け取った ``cameras_raw/<試技>/`` と同じ並びで、
``CALIB_BASE_DIR`` にそのまま渡せる。

- ``cam0_<ts>.avi`` と ``cam1_<ts>.avi``（既定は MJPG。圧縮の劣化で S6 の雑音の推定がずれないように）
- 校正ファイル 4 つのコピー（``--calib`` から）
- ``frames.csv``: フレームごとの grab 完了時刻（``time.monotonic_ns``）と左右の差
- ``meta.json``: 大きさ、容器の fps、実測の fps、録画の穴と左右のずれ（``frame_timing``）、止まった理由など

録画の穴（USB の取りこぼしで間隔が空いた所）と左右のずれは、再生では直せない（再生は動画のフレームを 1 フレームの
間隔で並べ、同じ番号のフレームを組にする）。終わりに数えて警告する。``tools/verify_run.py replay`` の報告にも出す。

止め方: 停止ファイル（``--stop-file`` か ``APP_STOP_FILE``）、SIGTERM、Ctrl-C、プレビューで q か ESC、
``--duration``、``--max-frames``、カメラの失敗。どれで止めても動画を閉じて ``meta.json`` を書く。

使い方（リポジトリ直下で）::

    python -m tools.record_stereo --label S07 --duration 90
    python -m tools.record_stereo --cam0 0 --cam1 1 --calib camera_parameters --out recordings --codec mp4v
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

# pylint: disable=no-member
import cv2 as cv
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # python tools/record_stereo.py でも動くように
    sys.path.insert(0, str(REPO_ROOT))

from app.core.camera_controls import apply_camera_controls  # noqa: E402
from app.core.stop_request import StopRequest  # noqa: E402
from app.core.video_source import parse_spec  # noqa: E402
from app.tuning.raw_capture import git_commit  # noqa: E402
# 計測と同じ開き方にする（Windows ではバックエンドで露出の値の意味やデバイス番号の並びが変わる）。
# video_io は config を import するので、作業フォルダに output_data ができる
from video_io import open_capture_and_read_first  # noqa: E402

CALIB_FILES = ("c0.dat", "c1.dat", "rot_trans_c0.dat", "rot_trans_c1.dat")
CODECS = {"mjpg": ("MJPG", ".avi"), "mp4v": ("mp4v", ".mp4")}
SETTINGS_PATH = REPO_ROOT / "calibration_settings.yaml"
# 実測の fps が容器の fps からこれ以上ずれたら警告する（再生では dt が容器の fps から決まるため）
FPS_TOLERANCE = 0.05
WARMUP_FRAMES = 5
# 隣り合うフレームの間隔が、間隔の中央値のこの倍を超えたら録画の穴（取りこぼし）と数える
GAP_FACTOR = 1.5
PROGRESS_EVERY = 150
EXIT_SETUP_FAILED = 2


class FrameSizeMismatch(RuntimeError):
    """カメラの解像度が校正と違う。"""


def calibration_defaults(path: str | Path = SETTINGS_PATH) -> SimpleNamespace:
    """校正（``calib.py``）と同じカメラ番号・解像度。設定ファイルが無ければ 0/1・1280×720。"""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        data = {}
    return SimpleNamespace(cam0=str(data.get("camera0", 0)), cam1=str(data.get("camera1", 1)),
                           width=int(data.get("frame_width", 1280)), height=int(data.get("frame_height", 720)))


def check_frame_size(actual: tuple[int, int], expected: tuple[int, int], allow: bool) -> None:
    if tuple(actual) == tuple(expected):
        return
    message = (f"カメラの解像度 {actual[0]}x{actual[1]} が校正 {expected[0]}x{expected[1]} と違う。"
               "この映像を校正の行列で三角測量すると 3D が狂う")
    if not allow:
        raise FrameSizeMismatch(message + "（承知で撮るなら --allow-size-mismatch）")
    print(f"[WARN] {message}")


def open_camera(spec: str, index: int, size: tuple[int, int], allow_size_mismatch: bool):
    """カメラを計測（``video_io.open_capture_and_read_first``）と同じ開き方で開き、計測と同じ設定を当て、
    解像度を校正に合わせる。戻り値は (cap, 実際の大きさ)。
    """
    parsed = parse_spec(spec)
    cap, ok, _ = open_capture_and_read_first(parsed.value)
    if not ok:
        raise RuntimeError(f"cam{index}（{spec}）を開けない。USB の接続と、ほかのアプリが使っていないかを確かめる")
    if parsed.supports_camera_controls:
        apply_camera_controls(cap, index)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, size[1])
    actual = None
    for _ in range(WARMUP_FRAMES):  # 最初の数枚は真っ黒・古い大きさのことがある
        ok, frame = cap.read()
        if ok and frame is not None:
            actual = (int(frame.shape[1]), int(frame.shape[0]))
    if actual is None:
        cap.release()
        raise RuntimeError(f"cam{index}（{spec}）から画像が来ない")
    try:
        check_frame_size(actual, size, allow_size_mismatch)
    except FrameSizeMismatch:
        cap.release()
        raise
    return cap, actual


def _backend_name(cap) -> str | None:
    try:
        return str(cap.getBackendName())
    except Exception:  # noqa: BLE001  偽のカメラや古い OpenCV
        return None


def _timed_grab(camera, clock):
    ok = camera.grab()
    return bool(ok), clock()


def record(cameras, writers, on_row, should_stop, *, clock=time.monotonic_ns, preview=None, pool=None):
    """止める理由が出るまで、2 台を並列に grab して全フレームを書く。戻り値は (フレーム数, 理由)。"""
    frames = 0
    try:
        while True:
            reason = should_stop(frames)
            if reason:
                return frames, reason
            if pool is not None:
                future = pool.submit(_timed_grab, cameras[1], clock)
                ok0, t0 = _timed_grab(cameras[0], clock)
                ok1, t1 = future.result()
            else:
                (ok0, t0), (ok1, t1) = _timed_grab(cameras[0], clock), _timed_grab(cameras[1], clock)
            if not (ok0 and ok1):
                return frames, "grab_failed"
            got0, frame0 = cameras[0].retrieve()
            got1, frame1 = cameras[1].retrieve()
            if not (got0 and got1):
                return frames, "retrieve_failed"
            writers[0].write(frame0)
            writers[1].write(frame1)
            on_row(frames, t0, t1)
            frames += 1
            if preview is not None and preview(frame0, frame1):
                return frames, "key"
    except KeyboardInterrupt:
        return frames, "ctrl_c"
    except Exception as error:  # noqa: BLE001  録れた分の数と理由を meta に残す（カメラの切断など）
        return frames, f"error: {type(error).__name__}: {error}"


def _preview(frame0, frame1) -> bool:
    """左右を縮小して並べて出す。q か ESC で True。"""
    small = [cv.resize(f, None, fx=0.5, fy=0.5) for f in (frame0, frame1)]
    if small[0].shape[0] != small[1].shape[0]:
        small[1] = cv.resize(small[1], (small[1].shape[1], small[0].shape[0]))
    cv.imshow("record_stereo (q / ESC で停止)", cv.hconcat(small))
    return (cv.waitKey(1) & 0xFF) in (ord("q"), 27)


def _measured_fps(times_ns: list[int]) -> float | None:
    if len(times_ns) < 2 or times_ns[-1] <= times_ns[0]:
        return None
    return (len(times_ns) - 1) / ((times_ns[-1] - times_ns[0]) / 1e9)


def frame_timing(t0_ns: Sequence[float], skew_ms: Sequence[float]) -> dict[str, Any] | None:
    """``frames.csv`` の時刻から、録画の穴（取りこぼし）と左右のずれを数える。間隔が取れなければ None。

    - 穴: cam0 の grab の間隔が中央値の ``GAP_FACTOR`` 倍を超えた所。取りこぼした数は「間隔 ÷ 中央値 − 1」の四捨五入の和
    - 左右のずれ: cam1 − cam0 の grab 完了時刻の差の絶対値。中央値の間隔の半分を超えたフレームを数える
      （再生は同じ番号のフレームを組にするので、半フレームを超えると隣の時刻と組になっているのに近い）
    """
    t = np.asarray(t0_ns, dtype=float)
    steps = np.diff(t) / 1e6
    positive = steps[np.isfinite(steps) & (steps > 0)]
    if not positive.size:
        return None
    median = float(np.median(positive))
    gaps = positive[positive > GAP_FACTOR * median]
    skew = np.abs(np.asarray(skew_ms, dtype=float))
    skew = skew[np.isfinite(skew)]
    return {
        "median_interval_ms": median,
        "max_interval_ms": float(positive.max()),
        "gaps": int(gaps.size),
        "missing_frames": int(np.maximum(np.rint(gaps / median) - 1, 0).sum()),
        "max_skew_ms": float(skew.max()) if skew.size else None,
        "p95_skew_ms": float(np.percentile(skew, 95)) if skew.size else None,
        "skewed_frames": int(np.sum(skew > median / 2.0)),
    }


def read_frame_timing(session: str | Path) -> dict[str, Any] | None:
    """録画のフォルダの ``frames.csv`` から ``frame_timing`` を数え直す（meta に数の無い古い録画にも使う）。"""
    path = Path(session) / "frames.csv"
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return frame_timing([float(r["t0_ns"]) for r in rows], [float(r["skew_ms"]) for r in rows])


def timing_warnings(timing: dict[str, Any] | None) -> list[str]:
    """``frame_timing`` の警告の文（無ければ空）。"""
    if not timing:
        return []
    warnings = []
    if timing["gaps"]:
        warnings.append(
            f"録画に穴が {timing['gaps']} か所（最大の間隔 {timing['max_interval_ms']:.0f} ms、ふだんは "
            f"{timing['median_interval_ms']:.1f} ms、取りこぼし約 {timing['missing_frames']} フレーム）。再生はフレームを "
            "1 フレームの間隔で並べるので、その区間の時間が詰まる。USB の帯域（MJPG）・ほかの負荷を確かめて撮り直す")
    if timing["skewed_frames"]:
        warnings.append(
            f"左右の撮影時刻のずれが半フレーム（{timing['median_interval_ms'] / 2:.1f} ms）を超えたフレームが "
            f"{timing['skewed_frames']}（最大 {timing['max_skew_ms']:.1f} ms、95% {timing['p95_skew_ms']:.1f} ms）。"
            "再生は同じ番号のフレームを組にするので、その間は左右の時刻が合わない")
    return warnings


def build_parser(defaults: SimpleNamespace) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2 台の USB カメラの全フレームを録画する（計測に読み込ませて確かめる用）")
    parser.add_argument("--cam0", default=defaults.cam0, help="cam0 の番号かパス（既定は calibration_settings.yaml）")
    parser.add_argument("--cam1", default=defaults.cam1, help="cam1 の番号かパス（既定は calibration_settings.yaml）")
    parser.add_argument("--calib", default=str(REPO_ROOT / "camera_parameters"), help="校正ファイル 4 つのフォルダ")
    parser.add_argument("--out", default="recordings", help="録画を置くフォルダ")
    parser.add_argument("--label", default="rec", help="録画のフォルダ名の頭（被験者番号など）")
    parser.add_argument("--duration", type=float, default=0.0, help="録画の長さ [秒]（0 なら止めるまで）")
    parser.add_argument("--max-frames", type=int, default=0, help="録画するフレーム数の上限（0 なら無制限）")
    parser.add_argument("--codec", choices=sorted(CODECS), default="mjpg", help="mjpg（.avi、劣化小）か mp4v（.mp4、小さい）")
    parser.add_argument("--fps", type=float, default=30.0, help="動画に書く fps（カメラの実際の fps と合わせる）")
    parser.add_argument("--width", type=int, default=defaults.width, help="解像度の幅（既定は校正と同じ）")
    parser.add_argument("--height", type=int, default=defaults.height, help="解像度の高さ（既定は校正と同じ）")
    parser.add_argument("--allow-size-mismatch", action="store_true", help="解像度が校正と違っても撮る")
    parser.add_argument("--stop-file", default=None, help="このファイルが現れたら止める（既定は APP_STOP_FILE）")
    parser.add_argument("--no-preview", action="store_true", help="プレビューのウィンドウを出さない")
    return parser


def main(argv=None, open_camera=open_camera) -> int:
    args = build_parser(calibration_defaults()).parse_args(argv)
    calib = Path(args.calib)
    missing = [name for name in CALIB_FILES if not (calib / name).is_file()]
    if missing:
        print(f"[ERROR] 校正ファイルが無い: {calib} に {', '.join(missing)}。先に校正（calib.py）をする", file=sys.stderr)
        return EXIT_SETUP_FAILED
    size = (args.width, args.height)

    cameras, sizes = [], []
    try:
        for index, spec in enumerate((args.cam0, args.cam1)):
            cap, actual = open_camera(spec, index, size, args.allow_size_mismatch)
            cameras.append(cap)
            sizes.append(actual)
    except (RuntimeError, FrameSizeMismatch) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        for cap in cameras:
            cap.release()
        return EXIT_SETUP_FAILED

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    session = Path(args.out) / f"{args.label}_{timestamp}"
    session.mkdir(parents=True, exist_ok=False)
    for name in CALIB_FILES:
        shutil.copy2(calib / name, session / name)
    fourcc, ext = CODECS[args.codec]
    videos = [session / f"cam{i}_{timestamp}{ext}" for i in range(2)]
    writers = [cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*fourcc), args.fps, sizes[i])
               for i, path in enumerate(videos)]

    stop = StopRequest(args.stop_file) if args.stop_file else StopRequest.from_environment()
    if stop.path is not None:
        stop.path.unlink(missing_ok=True)  # 前回の停止ファイルが残っていると、最初の周回で止まる
    previous_handlers = {num: signal.getsignal(num) for num in
                         (getattr(signal, name, None) for name in ("SIGTERM", "SIGBREAK")) if num is not None}
    stop.install_signal_handlers()

    times0: list[int] = []
    skews: list[float] = []   # cam1 − cam0 [ms]（frames.csv の skew_ms と同じ符号付き）
    started = datetime.now().isoformat(timespec="seconds")
    t_start = time.monotonic_ns()
    frames, reason = 0, "setup_failed"
    csv_file = open(session / "frames.csv", "w", newline="", encoding="utf-8")
    try:
        if not all(w.isOpened() for w in writers):
            print(f"[ERROR] 動画を書けない（{fourcc}）。--codec を変えて試す", file=sys.stderr)
            return EXIT_SETUP_FAILED
        rows = csv.writer(csv_file)
        rows.writerow(["index", "t0_ns", "t1_ns", "skew_ms"])

        def on_row(index, t0, t1):
            skew = round((t1 - t0) / 1e6, 3)
            rows.writerow([index, t0, t1, skew])
            times0.append(t0)
            skews.append(skew)
            if index and index % PROGRESS_EVERY == 0:
                fps_now = _measured_fps(times0)
                print(f"[REC] {index} フレーム（{fps_now:.1f} fps）", flush=True)

        def should_stop(n):
            if stop.requested():
                return "stop_request"
            if args.max_frames and n >= args.max_frames:
                return "max_frames"
            if args.duration and (time.monotonic_ns() - t_start) / 1e9 >= args.duration:
                return "duration"
            return None

        print(f"[REC] 録画を始める: {session}（{sizes[0][0]}x{sizes[0][1]}、{fourcc}、止めるには q / ESC / Ctrl-C）",
              flush=True)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="grab1") as pool:
            frames, reason = record(cameras, writers, on_row, should_stop, pool=pool,
                                    preview=None if args.no_preview else _preview)
    finally:
        csv_file.close()
        for writer in writers:
            writer.release()
        for cap in cameras:
            cap.release()
        if not args.no_preview:
            cv.destroyAllWindows()
        for num, handler in previous_handlers.items():
            signal.signal(num, handler)
        measured = _measured_fps(times0)
        fps_off = measured is not None and abs(measured - args.fps) / args.fps > FPS_TOLERANCE
        timing = frame_timing(times0, skews)
        meta = {
            "label": args.label,
            "timestamp": timestamp,
            "cameras": [str(args.cam0), str(args.cam1)],
            "frame_size": list(sizes[0]),
            "frame_sizes": [list(s) for s in sizes],
            "container_fps": float(args.fps),
            "measured_fps": measured,
            "fps_mismatch": bool(fps_off),
            "max_skew_ms": max(abs(s) for s in skews) if skews else None,
            "frame_timing": timing,
            "codec": fourcc,
            "backends": [_backend_name(cap) for cap in cameras],
            "videos": [path.name for path in videos],
            "frames": frames,
            "stop_reason": reason,
            "calibration_source": str(calib.resolve()),
            "camera_env": {k: v for k, v in os.environ.items() if k.startswith("CAM")},
            "git_commit": git_commit(REPO_ROOT),
            "started_at": started,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
        (session / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[REC] 終了: {frames} フレーム、理由 {reason}、実測 {measured or float('nan'):.2f} fps → {session}")
    if fps_off:
        print(f"[WARN] 実測 {measured:.2f} fps が動画の {args.fps:g} fps と {FPS_TOLERANCE:.0%} 以上ずれた。"
              f"再生では DT_SEC={1.0 / measured:.5f} を渡す（tools/verify_run.py replay は自動で渡す）")
    for warning in timing_warnings(timing):
        print(f"[WARN] {warning}")
    if reason.startswith("error"):
        print(f"[ERROR] 録画が途中で止まった: {reason}", file=sys.stderr)
        return 1
    if frames == 0:
        print("[ERROR] 1 フレームも録れなかった", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
