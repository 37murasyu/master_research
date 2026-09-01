import argparse
import math
import os
# pylint: disable=no-member
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares


def load_intrinsics(dat_path: str):
    with open(dat_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    mode = None
    K = []
    dist = []
    for ln in lines:
        low = ln.lower()
        if low.startswith("intrinsic"):
            mode = "K"
            continue
        if low.startswith("distortion"):
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


def save_intrinsics(K: np.ndarray, D: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("intrinsic:\n")
        for row in K:
            f.write(" ".join(f"{v}" for v in row) + "\n")
        f.write("distortion:\n")
        f.write(" ".join(f"{float(v)}" for v in D.reshape(-1)) + "\n")


def load_rot_trans(path: str):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    mode = None
    R = []
    T = []
    for ln in lines:
        low = ln.lower()
        if low.startswith("r:"):
            mode = "R"
            continue
        if low.startswith("t:"):
            mode = "T"
            continue
        vals = list(map(float, ln.split()))
        if mode == "R":
            R.append(vals)
        elif mode == "T":
            T.append(vals)
    if len(R) != 3:
        raise ValueError(f"invalid R in {path}")
    if len(T) != 3:
        raise ValueError(f"invalid T in {path}")
    return np.array(R, dtype=np.float64), np.array(T, dtype=np.float64).reshape(3, 1)


def save_rot_trans(R: np.ndarray, T: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("R:\n")
        for row in R:
            f.write(" ".join(f"{v}" for v in row) + "\n")
        f.write("T:\n")
        for row in T.reshape(-1, 1):
            f.write(f"{row[0]}\n")


def hat(vec3: np.ndarray):
    x, y, z = vec3.ravel()
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def collect_matches(cap0, cap1, frame_indices, detector, ratio=0.75, ransac=1.0, max_total=2000):
    matcher = detector
    pts0_all = []
    pts1_all = []
    for idx in frame_indices:
        cap0.set(cv.CAP_PROP_POS_FRAMES, int(idx))
        cap1.set(cv.CAP_PROP_POS_FRAMES, int(idx))
        ok0, img0 = cap0.read()
        ok1, img1 = cap1.read()
        if not ok0 or img0 is None or not ok1 or img1 is None:
            continue
        gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        kp0, des0 = matcher.detectAndCompute(gray0, None)
        kp1, des1 = matcher.detectAndCompute(gray1, None)
        if des0 is None or des1 is None:
            continue
        is_float = des0.dtype == np.float32 or des0.dtype == np.float64
        if is_float:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=32)
            match_engine = cv.FlannBasedMatcher(index_params, search_params)
        else:
            match_engine = cv.BFMatcher(cv.NORM_HAMMING)
        knn = match_engine.knnMatch(des0, des1, k=2)
        good = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        if len(good) < 8:
            continue
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        F, mask = cv.findFundamentalMat(pts0, pts1, cv.FM_RANSAC, ransac, 0.999)
        if F is None or mask is None:
            continue
        mask = mask.ravel() == 1
        pts0_in = pts0[mask]
        pts1_in = pts1[mask]
        if len(pts0_in) < 8:
            continue
        pts0_all.append(pts0_in.reshape(-1, 2))
        pts1_all.append(pts1_in.reshape(-1, 2))
        if sum(len(p) for p in pts0_all) >= max_total:
            break
    if not pts0_all:
        raise RuntimeError("no matches collected")
    pts0_cat = np.concatenate(pts0_all, axis=0)
    pts1_cat = np.concatenate(pts1_all, axis=0)
    return pts0_cat, pts1_cat


def epipolar_residuals(theta, pts0, pts1, K0, K1, use_k3=False, reg=1e-3):
    k1_0, k2_0, k3_0, k1_1, k2_1, k3_1, rx, ry, rz, tx, ty, tz = theta
    D0 = np.array([k1_0, k2_0, 0.0, 0.0, k3_0 if use_k3 else 0.0], dtype=np.float64)
    D1 = np.array([k1_1, k2_1, 0.0, 0.0, k3_1 if use_k3 else 0.0], dtype=np.float64)
    pts0_u = cv.undistortPoints(pts0.reshape(-1, 1, 2), K0, D0, P=None).reshape(-1, 2)
    pts1_u = cv.undistortPoints(pts1.reshape(-1, 1, 2), K1, D1, P=None).reshape(-1, 2)
    ones = np.ones((pts0_u.shape[0], 1), dtype=np.float64)
    x0 = np.concatenate([pts0_u, ones], axis=1)
    x1 = np.concatenate([pts1_u, ones], axis=1)
    R, _ = cv.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    t = np.array([tx, ty, tz], dtype=np.float64)
    norm_t = np.linalg.norm(t) + 1e-12
    t = t / norm_t
    E = hat(t) @ R
    Ex0 = (E @ x0.T).T
    Etx1 = (E.T @ x1.T).T
    num1 = np.sum(x1 * Ex0, axis=1)
    num0 = np.sum(x0 * Etx1, axis=1)
    den1 = np.linalg.norm(Ex0[:, :2], axis=1) + 1e-12
    den0 = np.linalg.norm(Etx1[:, :2], axis=1) + 1e-12
    r1 = num1 / den1
    r0 = num0 / den0
    reg_term = reg * np.array([k1_0, k2_0, k3_0, k1_1, k2_1, k3_1], dtype=np.float64)
    return np.concatenate([r1, r0, reg_term])


def radial_report(label, k1, k2, k3):
    for r in (0.5, 1.0):
        radial = 1 + k1 * r * r + k2 * r**4 + k3 * r**6
        print(f"{label} r={r}: radial={radial}")


def main():
    ap = argparse.ArgumentParser(description="Refine distortion using synchronized stereo videos")
    ap.add_argument("--c0", required=True, help="path to c0.dat")
    ap.add_argument("--c1", required=True, help="path to c1.dat")
    ap.add_argument("--cam0", required=True, help="cam0 video path")
    ap.add_argument("--cam1", required=True, help="cam1 video path")
    ap.add_argument("--rt1", default=None, help="optional initial rot_trans_c1.dat")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--frames", type=int, default=40, help="number of frames to sample")
    ap.add_argument("--max-matches", type=int, default=2000, help="max total correspondences")
    ap.add_argument("--detector", choices=["orb", "sift"], default="orb")
    ap.add_argument("--ratio", type=float, default=0.75, help="ratio test")
    ap.add_argument("--ransac", type=float, default=1.0, help="RANSAC threshold for F")
    ap.add_argument("--use-k3", action="store_true", help="optimize k3 as well")
    ap.add_argument("--regularization", type=float, default=1e-3, help="L2 weight on distortion")
    args = ap.parse_args()

    K0, D0_init = load_intrinsics(args.c0)
    K1, D1_init = load_intrinsics(args.c1)

    det = cv.ORB_create(4000) if args.detector == "orb" else cv.SIFT_create()
    cap0 = cv.VideoCapture(args.cam0)
    cap1 = cv.VideoCapture(args.cam1)
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("failed to open videos")
    n0 = int(cap0.get(cv.CAP_PROP_FRAME_COUNT))
    n1 = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))
    n = min(n0, n1)
    if n <= 0:
        raise RuntimeError("invalid frame count")
    idx = np.linspace(0, n - 1, num=args.frames, dtype=np.int32)
    pts0, pts1 = collect_matches(cap0, cap1, idx, det, ratio=args.ratio, ransac=args.ransac, max_total=args.max_matches)
    cap0.release()
    cap1.release()

    if args.rt1:
        R_init, t_init = load_rot_trans(args.rt1)
        rvec_init, _ = cv.Rodrigues(R_init)
        t_init = t_init.reshape(3)
    else:
        # fallback: use recoverPose on first inlier set as initial
        pts0_h = pts0.reshape(-1, 1, 2)
        pts1_h = pts1.reshape(-1, 1, 2)
        F_init, mask_init = cv.findFundamentalMat(pts0_h, pts1_h, cv.FM_RANSAC, args.ransac, 0.999)  # pylint: disable=too-many-function-args
        if F_init is None:
            raise RuntimeError("cannot init pose")
        E_init = K1.T @ F_init @ K0
        _, R_init, t_init, pose_mask = cv.recoverPose(E_init, pts0_h[mask_init.ravel() == 1], pts1_h[mask_init.ravel() == 1], K0, K1)  # pylint: disable=too-many-function-args
        rvec_init, _ = cv.Rodrigues(R_init)
        t_init = t_init.reshape(3)

    k1_0, k2_0, k3_0 = float(D0_init[0, 0]), float(D0_init[0, 1]), float(D0_init[0, 4] if D0_init.shape[1] > 4 else 0.0)
    k1_1, k2_1, k3_1 = float(D1_init[0, 0]), float(D1_init[0, 1]), float(D1_init[0, 4] if D1_init.shape[1] > 4 else 0.0)
    x0 = np.array([k1_0, k2_0, k3_0, k1_1, k2_1, k3_1, rvec_init[0, 0], rvec_init[1, 0], rvec_init[2, 0], t_init[0], t_init[1], t_init[2]], dtype=np.float64)
    # Allow wide bounds because initial distortion is large; prevent failure from out-of-bounds init
    bounds_lo = [-2000, -2000, -2000, -2000, -2000, -2000, -20, -20, -20, -20, -20, -20]
    bounds_hi = [2000, 2000, 2000, 2000, 2000, 2000, 20, 20, 20, 20, 20, 20]
    res = least_squares(
        epipolar_residuals,
        x0,
        bounds=(bounds_lo, bounds_hi),
        verbose=1,
        kwargs=dict(pts0=pts0, pts1=pts1, K0=K0, K1=K1, use_k3=args.use_k3, reg=args.regularization),
        loss="huber",
        f_scale=1.0,
        max_nfev=200,
    )

    k1_0, k2_0, k3_0, k1_1, k2_1, k3_1, rx, ry, rz, tx, ty, tz = res.x
    R_refined, _ = cv.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    t_refined = np.array([tx, ty, tz], dtype=np.float64)
    t_refined = t_refined / (np.linalg.norm(t_refined) + 1e-12)

    D0_new = np.array([[k1_0, k2_0, 0.0, 0.0, k3_0 if args.use_k3 else 0.0]], dtype=np.float64)
    D1_new = np.array([[k1_1, k2_1, 0.0, 0.0, k3_1 if args.use_k3 else 0.0]], dtype=np.float64)

    os.makedirs(args.out, exist_ok=True)
    save_intrinsics(K0, D0_new, os.path.join(args.out, "c0_refined.dat"))
    save_intrinsics(K1, D1_new, os.path.join(args.out, "c1_refined.dat"))
    save_rot_trans(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64), os.path.join(args.out, "rot_trans_refined_c0.dat"))
    save_rot_trans(R_refined, t_refined.reshape(3, 1), os.path.join(args.out, "rot_trans_refined_c1.dat"))

    print("--- distortion radial factors (r=0.5,1.0) ---")
    radial_report("c0", k1_0, k2_0, k3_0 if args.use_k3 else 0.0)
    radial_report("c1", k1_1, k2_1, k3_1 if args.use_k3 else 0.0)
    print("residual rms=", math.sqrt(np.mean(res.fun**2)))
    print("saved to", args.out)


if __name__ == "__main__":
    main()