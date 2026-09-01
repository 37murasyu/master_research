"""Estimate stereo extrinsics using pose keypoints as correspondences.
- Detect MediaPipe Pose on both videos.
- Collect matched 2D joints per frame (only joints visible in both views).
- Undistort + normalize, then estimate Essential with RANSAC and recover R,t (unit baseline).
- Saves rot_trans_c0.dat (I,0) and rot_trans_c1.dat (R,t_unit) to output dir.
Limitations: baseline scale is unknown; t is unit-length. Requires synced videos and valid intrinsics/distortion .dat files.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

# pylint: disable=no-member
import cv2 as cv
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    raise ImportError("mediapipe is required for pose keypoints") from exc


def load_intrinsics(dat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with dat_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    mode = None
    k_rows: list[list[float]] = []
    dist_vals: list[list[float]] = []
    for ln in lines:
        low = ln.lower()
        if low.startswith("intrinsic"):
            mode = "K"
            continue
        if low.startswith("distortion"):
            mode = "D"
            continue
        vals = [float(v) for v in ln.split()]
        if mode == "K":
            k_rows.append(vals)
        elif mode == "D":
            dist_vals.append(vals)
    if len(k_rows) != 3:
        raise ValueError(f"invalid K rows in {dat_path}")
    if not dist_vals:
        dist_vals = [[0.0, 0.0, 0.0, 0.0, 0.0]]
    K = np.array(k_rows, dtype=np.float64)
    D = np.array([dist_vals[0]], dtype=np.float64)
    return K, D


def save_rot_trans(R: np.ndarray, t: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["R:"]
    for r in R:
        lines.append(" ".join(str(v) for v in r))
    lines.append("T:")
    for v in t.reshape(-1, 1):
        lines.append(str(v[0]))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def symmetric_epipolar_rms(F: np.ndarray, pts0: np.ndarray, pts1: np.ndarray) -> float:
    pts0_h = cv.convertPointsToHomogeneous(pts0).reshape(-1, 3)
    pts1_h = cv.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    Fx1 = (F @ pts0_h.T).T
    Ftx2 = (F.T @ pts1_h.T).T
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    num = np.abs(np.sum(pts1_h * (F @ pts0_h.T).T, axis=1))
    d = num / np.sqrt(denom + 1e-12)
    return float(math.sqrt(np.mean(d * d)))


def extract_pose_kpts(frame: np.ndarray, pose, target_ids: list[int]) -> np.ndarray:
    h, w = frame.shape[:2]
    res = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    if res.pose_landmarks is None:
        return np.full((len(target_ids), 2), -1.0, dtype=np.float32)
    pts = []
    lm = res.pose_landmarks.landmark
    for pid in target_ids:
        if pid >= len(lm):
            pts.append([-1.0, -1.0])
            continue
        pts.append([lm[pid].x * w, lm[pid].y * h])
    return np.array(pts, dtype=np.float32)


def collect_matches(cam0: Path, cam1: Path, pose_ids: list[int], max_frames: int, stride: int, model_complexity: int) -> tuple[np.ndarray, np.ndarray]:
    cap0 = cv.VideoCapture(str(cam0))
    cap1 = cv.VideoCapture(str(cam1))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("failed to open videos")
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=model_complexity, smooth_landmarks=True)
    pts0_all: list[np.ndarray] = []
    pts1_all: list[np.ndarray] = []
    frame_idx = 0
    collected = 0
    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not ok0 or not ok1:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        k0 = extract_pose_kpts(f0, pose, pose_ids)
        k1 = extract_pose_kpts(f1, pose, pose_ids)
        # keep only joints present in both
        mask = (k0[:, 0] >= 0) & (k1[:, 0] >= 0)
        if mask.any():
            pts0_all.append(k0[mask])
            pts1_all.append(k1[mask])
            collected += mask.sum()
        frame_idx += 1
        if frame_idx >= max_frames:
            break
    cap0.release()
    cap1.release()
    if not pts0_all:
        raise RuntimeError("no pose correspondences collected")
    return np.concatenate(pts0_all, axis=0), np.concatenate(pts1_all, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate stereo extrinsic from pose keypoints")
    ap.add_argument("--c0", required=True, type=Path, help="path to c0.dat")
    ap.add_argument("--c1", required=True, type=Path, help="path to c1.dat")
    ap.add_argument("--cam0", required=True, type=Path, help="cam0 video")
    ap.add_argument("--cam1", required=True, type=Path, help="cam1 video")
    ap.add_argument("--out", required=True, type=Path, help="output directory for rot_trans_*.dat")
    ap.add_argument("--frames", type=int, default=800, help="max frames to sample")
    ap.add_argument("--stride", type=int, default=2, help="sample every Nth frame")
    ap.add_argument("--pose-ids", type=str, default="0,11,12,13,14,15,16,23,24,25,26,27,28", help="comma-separated pose landmark ids to use")
    ap.add_argument("--ransac-thresh", type=float, default=0.01, help="RANSAC threshold for Essential (normalized units)")
    ap.add_argument("--model-complexity", type=int, default=0, choices=[0, 1, 2], help="MediaPipe Pose model complexity (0=fast)")
    args = ap.parse_args()

    pose_ids = [int(x) for x in args.pose_ids.split(",") if x.strip()]

    K0, D0 = load_intrinsics(args.c0)
    K1, D1 = load_intrinsics(args.c1)

    pts0_px, pts1_px = collect_matches(args.cam0, args.cam1, pose_ids, max_frames=args.frames, stride=args.stride, model_complexity=args.model_complexity)
    print(f"[INFO] collected correspondences: {len(pts0_px)}")

    # Undistort to pixel coords, then normalize for Essential estimation
    pts0_ud = cv.undistortPoints(pts0_px.reshape(-1, 1, 2), K0, D0, P=None).reshape(-1, 2)
    pts1_ud = cv.undistortPoints(pts1_px.reshape(-1, 1, 2), K1, D1, P=None).reshape(-1, 2)

    E, mask = cv.findEssentialMat(pts0_ud, pts1_ud, focal=1.0, pp=(0.0, 0.0), method=cv.RANSAC, threshold=args.ransac_thresh, prob=0.999)
    if E is None or mask is None:
        raise RuntimeError("findEssentialMat failed")
    inliers = int(mask.sum())
    print(f"[INFO] inliers: {inliers}/{len(pts0_ud)}")

    _, R, t, pose_mask = cv.recoverPose(E, pts0_ud, pts1_ud, mask=mask)  # pylint: disable=too-many-function-args
    pose_inliers = int(pose_mask.sum()) if pose_mask is not None else inliers
    t_unit = t / (np.linalg.norm(t) + 1e-12)
    print(f"[INFO] recoverPose inliers: {pose_inliers}")

    # Epipolar RMS (using calibrated F)
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    epip_rmse = symmetric_epipolar_rms(F, cv.undistortPoints(pts0_px.reshape(-1, 1, 2), K0, D0, P=K0).reshape(-1, 2), cv.undistortPoints(pts1_px.reshape(-1, 1, 2), K1, D1, P=K1).reshape(-1, 2))
    print(f"[INFO] epipolar RMS (px, undistorted): {epip_rmse:.3f}")

    # Save rot_trans
    save_rot_trans(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64), args.out / "rot_trans_c0.dat")
    save_rot_trans(R, t_unit, args.out / "rot_trans_c1.dat")
    print(f"[DONE] saved rot_trans_* to {args.out}")


if __name__ == "__main__":
    main()
