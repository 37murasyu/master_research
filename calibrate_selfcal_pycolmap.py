"""Self-calibrate two videos (~30s) and emit four .dat files (c0/c1/rot_trans_c0/rot_trans_c1).

Pipeline (pycolmap required):
1) Extract frames from cam0/cam1 at a fixed step.
2) COLMAP feature extraction + exhaustive matching.
3) Incremental SfM + BA.
4) Export intrinsics/extrinsics to .dat compatible with the legacy calib format.

Usage example:
  python calibrate_selfcal_pycolmap.py \
    --cam0 "C:/Users/villa/Desktop/master_Research/cameras_raw/3_20250925_155122/cam0_3_20250925_155122.mp4" \
    --cam1 "C:/Users/villa/Desktop/master_Research/cameras_raw/3_20250925_155122/cam1_3_20250925_155122.mp4" \
    --out  "output_data/calib_selfcal/3_20250925_155122" \
    --step 10

Notes:
- Scale is unknown (mono SfM). To set a physical baseline, scale the poses afterward using a known distance.
- rot_trans expresses cam1 in cam0 coordinates. cam0 is identity (R=I, T=0).
"""

import argparse
import json
import shutil
from pathlib import Path
# pylint: disable=no-member
import cv2
import numpy as np
import pycolmap  # pylint: disable=no-member

# pycolmap 3.x may not expose reconstruction as a submodule depending on build.
try:  # pragma: no cover - import shim for version differences
    import pycolmap.reconstruction as pcl_reconstruction  # type: ignore  # pylint: disable=import-error,no-name-in-module
except Exception:  # pragma: no cover
    pcl_reconstruction = None  # type: ignore


def extract_frames(cam_path: Path, out_dir: Path, step: int) -> dict:
    cap = cv2.VideoCapture(str(cam_path))  # type: ignore[attr-defined]
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cam_path}")
    frame_count_prop = getattr(cv2, "CAP_PROP_FRAME_COUNT", 7)
    fps_prop = getattr(cv2, "CAP_PROP_FPS", 5)
    width_prop = getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)
    height_prop = getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)
    pos_frames_prop = getattr(cv2, "CAP_PROP_POS_FRAMES", 1)
    total = int(cap.get(frame_count_prop))
    fps = cap.get(fps_prop)
    width = int(cap.get(width_prop))
    height = int(cap.get(height_prop))
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx in range(0, total, step):
        cap.set(pos_frames_prop, idx)
        ok, frame = cap.read()
        if not ok:
            break
        name = f"f{idx:06d}.jpg"
        # IMWRITE_JPEG_QUALITY is present in cv2; suppress linter noise.
        quality_flag = int(getattr(cv2, "IMWRITE_JPEG_QUALITY", 1))
        cv2.imwrite(str(out_dir / name), frame, [quality_flag, 95])
        saved += 1
    cap.release()
    return {"frames": saved, "fps": fps, "width": width, "height": height, "total": total}


def run_colmap(image_dir: Path, out_dir: Path, fx_hint: float, cx: float, cy: float):
    db_path = out_dir / "database.db"
    sparse_dir = out_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    # OpenCV model (fx, fy, cx, cy, k1, k2, p1, p2)
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "OPENCV"
    pycolmap.extract_features(  # type: ignore[attr-defined]
        database_path=str(db_path),
        image_path=str(image_dir),
        reader_options=reader_opts,
    )
    pycolmap.match_exhaustive(database_path=str(db_path))  # type: ignore[attr-defined]

    reconstruct_fn = None
    if pcl_reconstruction and hasattr(pcl_reconstruction, "reconstruct"):
        reconstruct_fn = pcl_reconstruction.reconstruct
    elif hasattr(pycolmap, "reconstruct"):
        reconstruct_fn = pycolmap.reconstruct  # type: ignore[attr-defined]

    if reconstruct_fn is None:
        raise ImportError("pycolmap reconstruct API not found; check pycolmap version")

    recs = reconstruct_fn(
        database_path=str(db_path),
        image_path=str(image_dir),
        output_path=str(sparse_dir),
    )
    return recs


def _cam_params_to_intrinsic_dist(cam) -> tuple:
    # OPENCV model params: fx, fy, cx, cy, k1, k2, p1, p2
    p = cam.params
    fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    # Distortion to 5 terms: k1, k2, p1, p2, k3(=0)
    dist5 = [p[4], p[5], p[6], p[7], 0.0] if len(p) >= 8 else [0, 0, 0, 0, 0]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    return K, dist5


def _save_intrinsic_dat(path: Path, K: np.ndarray, dist5):
    lines = ["intrinsic:"]
    for row in K:
        lines.append(" ".join(str(v) for v in row))
    lines.append("distortion:")
    lines.append(" ".join(str(v) for v in dist5))
    path.write_text("\n".join(lines))


def _save_rot_trans_dat(path: Path, R: np.ndarray, t: np.ndarray):
    lines = ["R:"]
    for r in R:
        lines.append(" ".join(str(v) for v in r))
    lines.append("T:")
    for v in t:
        lines.append(str(v))
    path.write_text("\n".join(lines))


def _relative_pose(img_cam0, img_cam1):
    # world->cam transforms: Rcw, tcw. We want cam1 in cam0 frame.
    R0 = img_cam0.rotation_matrix()
    t0 = img_cam0.tvec
    R1 = img_cam1.rotation_matrix()
    t1 = img_cam1.tvec
    R10 = R1 @ R0.T
    t10 = t1 - R10 @ t0
    return R10, t10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam0", required=True)
    ap.add_argument("--cam1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=10, help="frame step for extraction")
    args = ap.parse_args()

    cam0 = Path(args.cam0)
    cam1 = Path(args.cam1)
    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)

    info0 = extract_frames(cam0, out_root / "images" / "cam0", args.step)
    info1 = extract_frames(cam1, out_root / "images" / "cam1", args.step)

    fx_hint = max(info0["width"], info0["height"])
    cx, cy = info0["width"] / 2, info0["height"] / 2

    recs = run_colmap(out_root / "images", out_root, fx_hint, cx, cy)

    summary = {
        "cam0": cam0.as_posix(),
        "cam1": cam1.as_posix(),
        "step": args.step,
        "frames_extracted": {"cam0": info0, "cam1": info1},
        "reconstructions": len(recs),
    }

    if not recs:
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    rec = recs[0]
    summary.update(
        {
            "num_images_registered": rec.num_reg_images(),
            "num_cameras": rec.num_cameras(),
            "num_points3D": rec.num_points3D(),
            "mean_reproj_error": rec.mean_reproj_error(),
        }
    )

    imgs_cam0 = [img for name, img in rec.images.items() if "cam0" in name]
    imgs_cam1 = [img for name, img in rec.images.items() if "cam1" in name]
    if not imgs_cam0 or not imgs_cam1:
        summary["error"] = "cam0 or cam1 images not registered"
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    img0 = imgs_cam0[0]
    img1 = imgs_cam1[0]

    cam0_obj = rec.cameras[img0.camera_id]
    cam1_obj = rec.cameras[img1.camera_id]

    K0, dist0 = _cam_params_to_intrinsic_dist(cam0_obj)
    K1, dist1 = _cam_params_to_intrinsic_dist(cam1_obj)

    R10, t10 = _relative_pose(img0, img1)

    out_root.mkdir(parents=True, exist_ok=True)
    _save_intrinsic_dat(out_root / "c0.dat", K0, dist0)
    _save_intrinsic_dat(out_root / "c1.dat", K1, dist1)
    _save_rot_trans_dat(out_root / "rot_trans_c0.dat", np.eye(3), np.zeros(3))
    _save_rot_trans_dat(out_root / "rot_trans_c1.dat", R10, t10)

    text_dir = out_root / "sparse" / "0" / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    rec.export_to_txt(text_dir)

    summary.update(
        {
            "intrinsic_c0": K0.tolist(),
            "intrinsic_c1": K1.tolist(),
            "dist_c0": dist0,
            "dist_c1": dist1,
            "rot_trans_c1": {"R": R10.tolist(), "t": t10.tolist()},
        }
    )

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
