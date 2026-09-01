from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from overlay_pose_on_video import POSE_EDGES, load_2d_csv


def _valid_point(p: np.ndarray) -> bool:
    return np.all(np.isfinite(p)) and p[0] >= 0 and p[1] >= 0


def _draw_links(
    img: np.ndarray,
    pts: np.ndarray,
    joint_ids: np.ndarray,
    color: Tuple[int, int, int] = (0, 180, 255),
    thickness: int = 2,
) -> None:
    idx_map: Dict[int, int] = {jid: i for i, jid in enumerate(joint_ids.tolist())}
    for a, b in POSE_EDGES:
        if a not in idx_map or b not in idx_map:
            continue
        pa = pts[idx_map[a]]
        pb = pts[idx_map[b]]
        if not _valid_point(pa) or not _valid_point(pb):
            continue
        cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), color, thickness)


def _draw_link_trails_overlay(
    base: np.ndarray,
    frames: np.ndarray,
    pts2d: np.ndarray,
    joint_ids: np.ndarray,
    frame_start: int,
    frame_end: int,
    color: Tuple[int, int, int] = (0, 180, 255),
    alpha: float = 0.25,
    thickness: int = 2,
) -> np.ndarray:
    overlay = base.copy()
    frame_to_idx = {int(f): i for i, f in enumerate(frames.tolist())}
    for f in range(frame_start, frame_end):
        idx = frame_to_idx.get(f)
        if idx is None:
            continue
        _draw_links(overlay, pts2d[idx], joint_ids, color=color, thickness=thickness)
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot 3-frame pose range on frame 0 background.")
    ap.add_argument("--video", required=True, help="cam0 video path (background frame 0)")
    ap.add_argument("--csv", required=True, help="cam0 2D pose CSV (joint_x/joint_y columns)")
    ap.add_argument("--seconds", type=float, default=5.0, help="duration from start to analyze")
    ap.add_argument("--out", default=None, help="output image path")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame0 = cap.read()
    cap.release()
    if not ok or frame0 is None:
        raise RuntimeError("failed to read frame 0")

    frames, pts2d, joint_ids = load_2d_csv(args.csv)
    frame_start = 0
    frame_end = int(round(fps * args.seconds))

    out_img = frame0.copy()

    # draw links at frame 0 if available
    frame0_idx = {int(f): i for i, f in enumerate(frames.tolist())}.get(0)
    if frame0_idx is not None:
        _draw_links(out_img, pts2d[frame0_idx], joint_ids)

    out_img = _draw_link_trails_overlay(
        out_img,
        frames,
        pts2d,
        joint_ids,
        frame_start,
        frame_end,
    )

    out_path = args.out
    if not out_path:
        vpath = Path(args.video)
        out_path = str(vpath.parent / f"{vpath.stem}_pose_range_5s.png")

    cv2.imwrite(out_path, out_img)
    print(f"[OK] saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
