import argparse
import os
# pylint: disable=no-member
import cv2 as cv
import numpy as np


def load_intrinsics(dat_path: str):
    with open(dat_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    mode = None
    K = []
    dist = []
    for ln in lines:
        if ln.lower().startswith("intrinsic"):
            mode = "K"
            continue
        if ln.lower().startswith("distortion"):
            mode = "D"
            continue
        vals = list(map(float, ln.split()))
        if mode == "K":
            K.append(vals)
        elif mode == "D":
            dist.append(vals)
    if len(K) != 3:
        raise ValueError(f"invalid K in {dat_path}")
    if not dist:
        dist = [[0, 0, 0, 0, 0]]
    return np.array(K, dtype=np.float64), np.array([dist[0]], dtype=np.float64)


def save_rot_trans(R: np.ndarray, T: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("R:\n")
        for row in R:
            f.write(" ".join(f"{v}" for v in row) + " \n")
        f.write("T:\n")
        for row in T.reshape(-1, 1):
            f.write(f"{row[0]} \n")


def create_detector(name: str = "orb", nfeatures: int | None = None):
    name = (name or "orb").lower()
    if name == "sift" and hasattr(cv, "SIFT_create"):
        return cv.SIFT_create()
    return cv.ORB_create(nfeatures or 2000)


def match_desc(desc1, desc2, ratio: float = 0.75):
    if desc1 is None or desc2 is None:
        return []
    is_float = desc1.dtype == np.float32 or desc1.dtype == np.float64
    if is_float:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=32)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv.BFMatcher(cv.NORM_HAMMING)
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return sorted(good, key=lambda m: m.distance)


def estimate_extrinsic_feature(imgL, imgR, K0, D0, K1, D1, detector="orb", ratio=0.75, ransac_thresh=1.0, baseline_m=None, nfeatures=None):
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    det = create_detector(detector, nfeatures=nfeatures)
    kp1, des1 = det.detectAndCompute(grayL, None)
    kp2, des2 = det.detectAndCompute(grayR, None)
    matches = match_desc(des1, des2, ratio=ratio)
    if len(matches) < 8:
        raise RuntimeError(f"not enough matches: {len(matches)} < 8")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, ransac_thresh, 0.999)
    if F is None or maskF is None:
        raise RuntimeError("findFundamentalMat failed")
    inlier_mask = maskF.ravel() == 1
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]
    if len(pts1_in) < 8:
        raise RuntimeError(f"inliers too few: {len(pts1_in)} < 8")
    E = K1.T @ F @ K0
    _, R, t, pose_mask = cv.recoverPose(E, pts1_in, pts2_in, K0, K1)
    if pose_mask is not None and pose_mask.sum() < 8:
        raise RuntimeError("recoverPose inliers too few")
    t_unit = t / (np.linalg.norm(t) + 1e-12)
    if baseline_m is not None:
        t = t_unit * float(baseline_m)
    else:
        t = t_unit
    return R, t


def grab_frame(video_path: str, frame_idx: int = 0):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    return frame


def main():
    ap = argparse.ArgumentParser(description="Repair extrinsic using feature matches from videos")
    ap.add_argument("--c0", required=True, help="path to c0.dat")
    ap.add_argument("--c1", required=True, help="path to c1.dat")
    ap.add_argument("--cam0", required=True, help="cam0 video path")
    ap.add_argument("--cam1", required=True, help="cam1 video path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--frame", type=int, default=0, help="frame index to sample (default 0)")
    ap.add_argument("--detector", choices=["orb", "sift"], default="orb")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac", type=float, default=1.0)
    ap.add_argument("--baseline", type=float, default=None, help="optional baseline in meters")
    ap.add_argument("--nfeatures", type=int, default=None)
    args = ap.parse_args()

    K0, D0 = load_intrinsics(args.c0)
    K1, D1 = load_intrinsics(args.c1)

    img0 = grab_frame(args.cam0, args.frame)
    img1 = grab_frame(args.cam1, args.frame)

    R, t = estimate_extrinsic_feature(
        img0,
        img1,
        K0,
        D0,
        K1,
        D1,
        detector=args.detector,
        ratio=args.ratio,
        ransac_thresh=args.ransac,
        baseline_m=args.baseline,
        nfeatures=args.nfeatures,
    )

    os.makedirs(args.out, exist_ok=True)
    save_rot_trans(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64), os.path.join(args.out, "rot_trans_c0.dat"))
    save_rot_trans(R, t, os.path.join(args.out, "rot_trans_c1.dat"))
    print("[OK] saved rot_trans_c0.dat and rot_trans_c1.dat")


if __name__ == "__main__":
    main()
