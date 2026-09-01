import argparse
import os
from typing import Dict, Iterable, List, Tuple
# pylint: disable=no-member
import cv2
import numpy as np
import pandas as pd

# MediaPipe-style skeleton edges
POSE_EDGES: List[Tuple[int, int]] = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
    (23, 24), (11, 23), (12, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (17, 19), (19, 21), (16, 18), (18, 20), (20, 22),
]


def load_2d_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    frames = df["frame"].to_numpy()
    cols = [c for c in df.columns if c != "frame"]

    x_cols = [c for c in cols if c.endswith("_x")]
    y_cols = [c for c in cols if c.endswith("_y")]
    if len(x_cols) != len(y_cols):
        raise ValueError("2D CSV must have the same number of x and y columns")

    joint_ids: List[int] = []
    ordered_x_cols: List[str] = []
    ordered_y_cols: List[str] = []
    for x_col in x_cols:
        try:
            jid = int(x_col.split("_")[1])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"bad column name: {x_col}") from exc
        y_col = f"joint_{jid}_y"
        if y_col not in y_cols:
            raise ValueError(f"missing y column for {x_col}")
        joint_ids.append(jid)
        ordered_x_cols.append(x_col)
        ordered_y_cols.append(y_col)

    data_x = df[ordered_x_cols].to_numpy(dtype=float)
    data_y = df[ordered_y_cols].to_numpy(dtype=float)
    data = np.stack([data_x, data_y], axis=2)  # shape (T, J, 2)
    return frames, data, np.array(joint_ids, dtype=int)


def draw_pose(
    img: np.ndarray,
    pts: np.ndarray,
    joint_ids: np.ndarray,
    edges: Iterable[Tuple[int, int]],
    radius: int = 4,
    thickness: int = 2,
    show_labels: bool = False,
) -> None:
    idx_map: Dict[int, int] = {jid: i for i, jid in enumerate(joint_ids.tolist())}
    # draw edges first for cleaner overlay
    for a, b in edges:
        if a not in idx_map or b not in idx_map:
            continue
        pa = pts[idx_map[a]]
        pb = pts[idx_map[b]]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        if pa[0] < 0 or pb[0] < 0:
            continue
        cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 180, 255), thickness)

    for jid, p in zip(joint_ids, pts):
        if np.any(np.isnan(p)) or p[0] < 0:
            continue
        cv2.circle(img, tuple(p.astype(int)), radius, (0, 255, 0), -1)
        if show_labels:
            cv2.putText(img, str(jid), (int(p[0]) + 4, int(p[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def overlay_video(
    video_path: str,
    csv_path: str,
    out_path: str,
    start: int,
    end: int | None,
    stride: int,
    show_labels: bool,
) -> None:
    frames, pts2d, joint_ids = load_2d_csv(csv_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps / max(stride, 1), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {out_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_frame = total if end is None else min(end, total)

    frame_map: Dict[int, int] = {int(f): i for i, f in enumerate(frames.tolist())}
    target_frames = list(range(start, last_frame, stride))

    for idx, fidx in enumerate(target_frames):
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        if not ok:
            break
        ret, frame = cap.read()
        if not ret:
            break

        pose_idx = frame_map.get(fidx)
        if pose_idx is not None and pose_idx < len(pts2d):
            draw_pose(frame, pts2d[pose_idx], joint_ids, POSE_EDGES, show_labels=show_labels)
        cv2.putText(frame, f"frame {fidx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(frame, f"frame {fidx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        writer.write(frame)
        if (idx + 1) % 50 == 0:
            print(f"wrote {idx + 1}/{len(target_frames)} frames", end="\r")

    cap.release()
    writer.release()
    print(f"overlay saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay 2D pose CSV onto video")
    ap.add_argument("--video", required=True, help="input video path (cam0) ")
    ap.add_argument("--csv", required=True, help="2D pose CSV for the same camera (joint_x/joint_y columns)")
    ap.add_argument("--out", help="output mp4 path")
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--end", type=int, help="end frame index (exclusive)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride for overlay")
    ap.add_argument("--labels", action="store_true", help="draw joint ids")
    args = ap.parse_args()

    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(args.video)
        out_path = base + "_overlay.mp4"

    overlay_video(args.video, args.csv, out_path, start=args.start, end=args.end, stride=max(1, args.stride), show_labels=args.labels)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
