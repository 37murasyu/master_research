"""
Video I/O layer separated from the main script.
Provides:
- find_latest_recordings(base_dir)
- find_recording_pairs(base_dir)
- resolve_input_streams()
- open_capture_and_read_first(src)

Relies on config.py for environment-like switches to preserve existing behavior.
"""
from __future__ import annotations

import os
import glob
from typing import Tuple
# pylint: disable=no-member

import cv2 as cv

from config import (
    folder_path,
    input_stream1,
    input_stream2,
    USE_SAMPLE_VIDEOS,
    AUTO_FALLBACK_TO_FILES,
    PREFER_RECORDING_PAIRS,
    IO_DEBUG,
)


def _dbg(*args, **kwargs):
    if IO_DEBUG:
        print("[IODBG]", *args, **kwargs)


def find_latest_recordings(base_dir: str) -> Tuple[str | None, str | None]:
    cam0_list = sorted(glob.glob(os.path.join(base_dir, "cam0_output_*.mp4")), reverse=True)
    cam1_list = sorted(glob.glob(os.path.join(base_dir, "cam1_output_*.mp4")), reverse=True)
    cam0 = cam0_list[0] if cam0_list else None
    cam1 = cam1_list[0] if cam1_list else None
    return cam0, cam1


def find_recording_pairs(base_dir: str) -> list[tuple[str, str]]:
    cam0_list = glob.glob(os.path.join(base_dir, "cam0_output_*.mp4"))
    cam1_list = glob.glob(os.path.join(base_dir, "cam1_output_*.mp4"))

    def suffix(p: str) -> str:
        return os.path.basename(p).replace("cam0_output_", "").replace("cam1_output_", "")

    m0 = {suffix(p): p for p in cam0_list}
    m1 = {suffix(p): p for p in cam1_list}
    common = sorted(set(m0.keys()) & set(m1.keys()), reverse=True)
    return [(m0[s], m1[s]) for s in common]


def resolve_input_streams() -> Tuple[object, object, bool, str]:
    # 1) Force sample
    if USE_SAMPLE_VIDEOS:
        if PREFER_RECORDING_PAIRS:
            pairs = find_recording_pairs(folder_path)
            if pairs:
                cam0_file, cam1_file = pairs[0]
                return cam0_file, cam1_file, True, "USE_SAMPLE_VIDEOS=1 (pair)"
        cam0_file = os.path.join(folder_path, "media", "cam000_test.mp4")
        cam1_file = os.path.join(folder_path, "media", "cam111_test.mp4")
        print(f"[Info] Using sample videos: {cam0_file}, {cam1_file}")
        return cam0_file, cam1_file, True, "USE_SAMPLE_VIDEOS=1 (media)"

    s1, s2 = input_stream1, input_stream2

    def _try_open(pair: Tuple[object, object]) -> bool:
        tmp0 = cv.VideoCapture(pair[0])
        tmp1 = cv.VideoCapture(pair[1])
        ok = tmp0.isOpened() and tmp1.isOpened()
        tmp0.release()
        tmp1.release()
        return ok

    if not (isinstance(s1, int) and isinstance(s2, int)):
        return s1, s2, True, "config: file paths"

    if _try_open((s1, s2)):
        return s1, s2, False, "config: cameras 0/1"

    if AUTO_FALLBACK_TO_FILES:
        cam0_file, cam1_file = find_latest_recordings(folder_path)
        if cam0_file is None:
            cam0_file = os.path.join(folder_path, "media", "cam000_test.mp4")
        if cam1_file is None:
            cam1_file = os.path.join(folder_path, "media", "cam111_test.mp4")
        return cam0_file, cam1_file, True, "AUTO_FALLBACK_TO_FILES=1 (camera open failed)"

    return s1, s2, not (s1 == 0 and s2 == 1), "no fallback"


def open_capture_and_read_first(src: object) -> tuple[cv.VideoCapture, bool, object | None]:
    if isinstance(src, str):
        backends = []
        for b in (getattr(cv, 'CAP_FFMPEG', None), getattr(cv, 'CAP_MSMF', None), getattr(cv, 'CAP_DSHOW', None), cv.CAP_ANY):
            if isinstance(b, int) and b not in backends:
                backends.append(b)
    else:
        backends = [cv.CAP_ANY]
        for b in (getattr(cv, 'CAP_FFMPEG', None), getattr(cv, 'CAP_MSMF', None), getattr(cv, 'CAP_DSHOW', None)):
            if isinstance(b, int) and b not in backends:
                backends.append(b)

    src_to_use = os.path.normpath(src) if isinstance(src, str) else src
    for be in backends:
        try:
            cap = cv.VideoCapture(src_to_use, be) if be != cv.CAP_ANY else cv.VideoCapture(src_to_use)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    try:
                        total = cap.get(cv.CAP_PROP_FRAME_COUNT) if isinstance(src, str) else -1
                        _dbg("opened backend=", be, "src=", src_to_use, "frames=", total)
                    except Exception:
                        pass
                    return cap, True, frame
                cap.release()
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
    cap = cv.VideoCapture()
    return cap, False, None
