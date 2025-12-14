"""Utility script to inspect and preview connected cameras.

Usage examples:
  python show_camera.py --scan            # list indexes detected per backend
  python show_camera.py --index 1 --backend dshow  # preview specific device
Press 'q' while the preview window is focused to exit.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, Optional, Tuple

import cv2  # type: ignore[attr-defined]

BACKENDS: Dict[str, int] = {
    "auto": 0,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}

# These presets cover common BRIO and generic UVC formats that sometimes
# need to be explicitly requested on Windows to stream video frames.
FORMAT_PRESETS: Tuple[Tuple[str, int, int, int], ...] = (
    ("MJPG", 3840, 2160, 30),
    ("MJPG", 2560, 1440, 30),
    ("MJPG", 1920, 1080, 30),
    ("MJPG", 1280, 720, 30),
    ("MJPG", 960, 540, 30),
    ("YUY2", 1280, 720, 30),
    ("YUY2", 640, 480, 30),
)


def scan(max_index: int) -> None:
    for name, backend in BACKENDS.items():
        found = []
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _frame = cap.read()
                found.append((idx, bool(ret)))
            cap.release()
        print(f"backend={name} ({backend}) -> {found}")


def apply_settings(
    cap: cv2.VideoCapture,
    width: Optional[int],
    height: Optional[int],
    fps: Optional[int],
    fourcc: Optional[str],
) -> None:
    if fourcc:
        if len(fourcc) != 4:
            raise ValueError("fourcc must be 4 characters (e.g. MJPG, YUY2)")
        code = cv2.VideoWriter_fourcc(*fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, code)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


def preview(
    index: int,
    backend_name: str,
    width: Optional[int],
    height: Optional[int],
    fps: Optional[int],
    fourcc: Optional[str],
    retry_presets: bool,
) -> None:
    backend = BACKENDS[backend_name]
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"[ERR] failed to open camera index {index} with backend {backend_name}", file=sys.stderr)
        sys.exit(1)

    try:
        apply_settings(cap, width, height, fps, fourcc)
    except ValueError as err:
        cap.release()
        print(f"[ERR] {err}", file=sys.stderr)
        sys.exit(2)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] opened index={index} backend={backend_name} size={width:.0f}x{height:.0f} fps={fps:.2f}")
    print("[INFO] press 'q' in the preview window to exit")

    preset_iter: Iterable[Tuple[str, int, int, int]] = FORMAT_PRESETS if retry_presets else ()
    preset_iter = iter(preset_iter)
    attempted_presets = False

    while True:
        ret, frame = cap.read()
        if not ret:
            if retry_presets:
                next_preset = next(preset_iter, None)
                if next_preset:
                    attempted_presets = True
                    preset_fourcc, preset_w, preset_h, preset_fps = next_preset
                    print(
                        "[WARN] frame read failed; retrying with preset "
                        f"fourcc={preset_fourcc} size={preset_w}x{preset_h} fps={preset_fps}"
                    )
                    apply_settings(cap, preset_w, preset_h, preset_fps, preset_fourcc)
                    # Give the device a short moment to apply new settings.
                    cv2.waitKey(100)
                    continue
            if attempted_presets:
                print("[ERR] no presets left; camera did not deliver frames", file=sys.stderr)
            else:
                print("[WARN] failed to read frame")
            break
        cv2.imshow(f"camera {index} ({backend_name})", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and preview cameras via OpenCV")
    parser.add_argument("--scan", action="store_true", help="List camera indexes per backend (no preview)")
    parser.add_argument("--max-index", type=int, default=5, help="Highest index to probe when scanning")
    parser.add_argument("--index", type=int, default=0, help="Camera index to preview")
    parser.add_argument(
        "--backend",
        choices=BACKENDS.keys(),
        default="dshow",
        help="Video backend to use when previewing",
    )
    parser.add_argument("--width", type=int, help="Request specific frame width before streaming")
    parser.add_argument("--height", type=int, help="Request specific frame height before streaming")
    parser.add_argument("--fps", type=int, help="Request specific frame rate before streaming")
    parser.add_argument("--fourcc", type=str, help="Request pixel format (e.g. MJPG, YUY2)")
    parser.add_argument(
        "--retry-presets",
        action="store_true",
        help="Cycle through common format presets if frame reads keep failing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scan:
        scan(args.max_index)
    else:
        preview(
            index=args.index,
            backend_name=args.backend,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
            retry_presets=args.retry_presets,
        )


if __name__ == "__main__":
    main()
