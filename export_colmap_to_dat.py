"""Convert a COLMAP sparse model (TXT) into legacy .dat intrinsics/extrinsics.

Usage:
  python export_colmap_to_dat.py --model path/to/sparse/0_text --out output_dir

Assumptions:
- Two cameras, images names contain 'cam0' and 'cam1' to disambiguate.
- Model already reconstructed via COLMAP and exported to TXT with model_converter.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def _read_cameras(path: Path):
    cams = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cams[cam_id] = {"model": model, "width": width, "height": height, "params": params}
    return cams


def _read_images(path: Path):
    images = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]], dtype=float)
            tvec = np.array([float(x) for x in parts[5:8]], dtype=float)
            cam_id = int(parts[8])
            name = parts[9]
            images[name] = {"image_id": image_id, "qvec": qvec, "tvec": tvec, "camera_id": cam_id}
            # Skip following line with points2D
            next(f, None)
    return images


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    # COLMAP quaternion convention: q = (qw, qx, qy, qz)
    qw, qx, qy, qz = qvec
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    return R


def _cam_params_to_K_dist(model: str, params):
    model = model.upper()
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:  # params: f, cx, cy (, k1)
        f, cx, cy = params[0], params[1], params[2]
        k1 = params[3] if len(params) > 3 else 0.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
        dist5 = [k1, 0.0, 0.0, 0.0, 0.0]
        return K, dist5
    if model == "PINHOLE":  # params: fx, fy, cx, cy
        fx, fy, cx, cy = params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        dist5 = [0.0, 0.0, 0.0, 0.0, 0.0]
        return K, dist5
    if model in {"RADIAL", "RADIAL3"}:  # params: f, cx, cy, k1, k2 (, k3)
        f, cx, cy = params[0], params[1], params[2]
        k1 = params[3] if len(params) > 3 else 0.0
        k2 = params[4] if len(params) > 4 else 0.0
        k3 = params[5] if len(params) > 5 else 0.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
        dist5 = [k1, k2, 0.0, 0.0, k3]
        return K, dist5
    if model == "OPENCV":  # params: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = params[:4]
        k1, k2, p1, p2 = params[4:8]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        dist5 = [k1, k2, p1, p2, 0.0]
        return K, dist5
    raise ValueError(f"Unsupported camera model: {model}")


def _relative_pose(R0: np.ndarray, t0: np.ndarray, R1: np.ndarray, t1: np.ndarray):
    R10 = R1 @ R0.T
    t10 = t1 - R10 @ t0
    return R10, t10


def _save_intrinsic_dat(path: Path, K: np.ndarray, dist5):
    lines = ["intrinsic:"]
    for row in K:
        lines.append(" ".join(str(v) for v in row))
    lines.append("distortion:")
    lines.append(" ".join(str(v) for v in dist5))
    path.write_text("\n".join(lines), encoding="ascii")


def _save_rot_trans_dat(path: Path, R: np.ndarray, t: np.ndarray):
    lines = ["R:"]
    for r in R:
        lines.append(" ".join(str(v) for v in r))
    lines.append("T:")
    for v in t:
        lines.append(str(v))
    path.write_text("\n".join(lines), encoding="ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to COLMAP TXT model directory (contains cameras.txt, images.txt)")
    ap.add_argument("--out", required=True, help="Output directory for .dat files")
    ap.add_argument("--cam0-key", default="cam0", help="Substring to identify cam0 images")
    ap.add_argument("--cam1-key", default="cam1", help="Substring to identify cam1 images")
    args = ap.parse_args()

    model_dir = Path(args.model)
    cameras = _read_cameras(model_dir / "cameras.txt")
    images = _read_images(model_dir / "images.txt")

    img0_name = next((n for n in images if args.cam0_key in n), None)
    img1_name = next((n for n in images if args.cam1_key in n), None)
    if img0_name is None or img1_name is None:
        raise RuntimeError(f"Could not find images containing keys: {args.cam0_key}, {args.cam1_key}")

    img0 = images[img0_name]
    img1 = images[img1_name]

    cam0 = cameras[img0["camera_id"]]
    cam1 = cameras[img1["camera_id"]]

    K0, dist0 = _cam_params_to_K_dist(cam0["model"], cam0["params"])
    K1, dist1 = _cam_params_to_K_dist(cam1["model"], cam1["params"])

    R0 = _qvec_to_rotmat(img0["qvec"])
    t0 = img0["tvec"]
    R1 = _qvec_to_rotmat(img1["qvec"])
    t1 = img1["tvec"]

    R10, t10 = _relative_pose(R0, t0, R1, t1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_intrinsic_dat(out_dir / "c0.dat", K0, dist0)
    _save_intrinsic_dat(out_dir / "c1.dat", K1, dist1)
    _save_rot_trans_dat(out_dir / "rot_trans_c0.dat", np.eye(3), np.zeros(3))
    _save_rot_trans_dat(out_dir / "rot_trans_c1.dat", R10, t10)

    summary = {
        "model_dir": model_dir.as_posix(),
        "cam0_image": img0_name,
        "cam1_image": img1_name,
        "cam0_model": cam0["model"],
        "cam1_model": cam1["model"],
        "K0": K0.tolist(),
        "K1": K1.tolist(),
        "dist0": dist0,
        "dist1": dist1,
        "R10": R10.tolist(),
        "t10": t10.tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
