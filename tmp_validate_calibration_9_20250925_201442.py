"""Compute epipolar residuals for dataset 9_20250925_201442 using existing .dat calibration.
Collect ORB correspondences across sampled frames, filter via RANSAC F, then evaluate
symmetric epipolar distance (pixels) for the provided K/D/R/T.
Also estimates extrinsic via recoverPose for a sanity comparison.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
# pylint: disable=no-member
import cv2 as cv
import numpy as np


def load_intrinsics(dat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with dat_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    mode = None
    K_rows: list[list[float]] = []
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
            K_rows.append(vals)
        elif mode == "D":
            dist_vals.append(vals)
    if len(K_rows) != 3:
        raise ValueError(f"invalid K rows in {dat_path}")
    if not dist_vals:
        dist_vals = [[0.0, 0.0, 0.0, 0.0, 0.0]]
    K = np.array(K_rows, dtype=np.float64)
    D = np.array([dist_vals[0]], dtype=np.float64)
    return K, D


def load_rot_trans(dat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with dat_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    mode = None
    R_rows: list[list[float]] = []
    T_vals: list[float] = []
    for ln in lines:
        low = ln.lower()
        if low.startswith("r:"):
            mode = "R"
            continue
        if low.startswith("t:"):
            mode = "T"
            continue
        vals = [float(v) for v in ln.split()]
        if mode == "R":
            R_rows.append(vals)
        elif mode == "T":
            T_vals.extend(vals)
    if len(R_rows) != 3:
        raise ValueError(f"invalid R rows in {dat_path}")
    if len(T_vals) != 3:
        raise ValueError(f"invalid T values in {dat_path}")
    R = np.array(R_rows, dtype=np.float64)
    t = np.array(T_vals, dtype=np.float64).reshape(3, 1)
    return R, t


def hat(t: np.ndarray) -> np.ndarray:
    tx, ty, tz = t.flatten()
    return np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]], dtype=np.float64)


def symmetric_epipolar_rms(F: np.ndarray, pts0: np.ndarray, pts1: np.ndarray) -> float:
    pts0_h = cv.convertPointsToHomogeneous(pts0).reshape(-1, 3)
    pts1_h = cv.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    Fx1 = (F @ pts0_h.T).T
    Ftx2 = (F.T @ pts1_h.T).T
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    num = np.abs(np.sum(pts1_h * (F @ pts0_h.T).T, axis=1))
    d = num / np.sqrt(denom + 1e-12)
    return float(math.sqrt(np.mean(d * d)))


def gather_correspondences(cam0: Path, cam1: Path, frames: int = 30, max_total: int = 2500):
    cap0 = cv.VideoCapture(str(cam0))
    cap1 = cv.VideoCapture(str(cam1))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("failed to open videos")
    n0 = int(cap0.get(cv.CAP_PROP_FRAME_COUNT))
    n1 = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))
    n = min(n0, n1)
    if n <= 0:
        raise RuntimeError("invalid frame count")
    idx = np.linspace(0, n - 1, num=frames, dtype=np.int32)
    det = cv.ORB_create(4000)
    pts0_all: list[np.ndarray] = []
    pts1_all: list[np.ndarray] = []
    total = 0
    for i in idx:
        cap0.set(cv.CAP_PROP_POS_FRAMES, int(i))
        cap1.set(cv.CAP_PROP_POS_FRAMES, int(i))
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not ok0 or not ok1:
            continue
        kp0, d0 = det.detectAndCompute(f0, None)
        kp1, d1 = det.detectAndCompute(f1, None)
        if d0 is None or d1 is None:
            continue
        matcher = cv.BFMatcher(cv.NORM_HAMMING)
        knn = matcher.knnMatch(d0, d1, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        F, mask = cv.findFundamentalMat(pts0, pts1, cv.FM_RANSAC, 1.0, 0.999)
        if F is None or mask is None:
            continue
        inliers = mask.ravel() == 1
        pts0_in = pts0[inliers].reshape(-1, 2)
        pts1_in = pts1[inliers].reshape(-1, 2)
        if pts0_in.size == 0:
            continue
        pts0_all.append(pts0_in)
        pts1_all.append(pts1_in)
        total += pts0_in.shape[0]
        if total >= max_total:
            break
    cap0.release()
    cap1.release()
    if not pts0_all:
        raise RuntimeError("no correspondences collected")
    return np.concatenate(pts0_all, axis=0), np.concatenate(pts1_all, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", required=True, type=Path)
    ap.add_argument("--c1", required=True, type=Path)
    ap.add_argument("--rt1", required=True, type=Path)
    ap.add_argument("--cam0", required=True, type=Path)
    ap.add_argument("--cam1", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    K0, D0 = load_intrinsics(args.c0)
    K1, D1 = load_intrinsics(args.c1)
    R1, t1 = load_rot_trans(args.rt1)
    t1_unit = t1 / (np.linalg.norm(t1) + 1e-12)
    baseline = float(np.linalg.norm(t1))

    pts0, pts1 = gather_correspondences(args.cam0, args.cam1, frames=args.frames)
    print(f"[INFO] collected matches: {len(pts0)}")

    # undistort to pixel coordinates before evaluating F
    pts0_ud = cv.undistortPoints(pts0.reshape(-1, 1, 2), K0, D0, P=K0).reshape(-1, 2)
    pts1_ud = cv.undistortPoints(pts1.reshape(-1, 1, 2), K1, D1, P=K1).reshape(-1, 2)

    E_calib = hat(t1_unit.flatten()) @ R1
    F_calib = np.linalg.inv(K1).T @ E_calib @ np.linalg.inv(K0)
    epip_rmse_px = symmetric_epipolar_rms(F_calib, pts0_ud, pts1_ud)
    print(f"[INFO] epipolar RMS (px, undistorted): {epip_rmse_px:.3f}")

    # Pose re-estimation for sanity check (normalized coords to allow differing intrinsics)
    pts0_norm = cv.undistortPoints(pts0.reshape(-1, 1, 2), K0, D0).reshape(-1, 2)
    pts1_norm = cv.undistortPoints(pts1.reshape(-1, 1, 2), K1, D1).reshape(-1, 2)
    E_est, maskE = cv.findEssentialMat(pts0_norm, pts1_norm, focal=1.0, pp=(0.0, 0.0), method=cv.RANSAC, threshold=0.01, prob=0.999)  # pylint: disable=too-many-function-args
    pose_inliers = int(maskE.sum()) if maskE is not None else 0
    R_est, t_est = None, None
    if E_est is not None:
        _, R_est, t_est, mask_pose = cv.recoverPose(E_est, pts0_norm, pts1_norm, mask=maskE)  # pylint: disable=too-many-function-args
        pose_inliers = int(mask_pose.sum()) if mask_pose is not None else pose_inliers

    if R_est is not None and t_est is not None:
        R_delta = R_est @ R1.T
        cos_angle = (np.trace(R_delta) - 1.0) / 2.0
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        rot_err_deg = math.degrees(math.acos(cos_angle))
        t_est_unit = t_est / (np.linalg.norm(t_est) + 1e-12)
        dot_t = float(np.clip(np.dot(t_est_unit.flatten(), t1_unit.flatten()), -1.0, 1.0))
        t_err_deg = math.degrees(math.acos(dot_t))
        print(f"[INFO] recoverPose inliers: {pose_inliers}")
        print(f"[INFO] rotation delta: {rot_err_deg:.3f} deg")
        print(f"[INFO] translation dir delta: {t_err_deg:.3f} deg")
    else:
        print("[WARN] recoverPose failed")

    print(f"[INFO] baseline magnitude from calib: {baseline:.3f} (units of .dat T)")


if __name__ == "__main__":
    main()
