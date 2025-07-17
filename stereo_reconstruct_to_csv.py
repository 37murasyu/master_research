# pyright: reportAttributeAccessIssue=false
import argparse
import os
import sys
import time
from typing import List

# pylint: disable=no-member
import cv2 as cv
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover - runtime import
    print("[WARN] mediapipe import failed. Please ensure mediapipe is installed.")
    raise

# プロジェクト内ユーティリティ
from config import pose_keypoints  # 12関節のインデックス
from utils import extract_keypoints, calculate_3d_keypoints, get_projection_matrix


def _init_pose(det_conf=0.5, track_conf=0.5):
    mp_pose = mp.solutions.pose
    pose0 = mp_pose.Pose(min_detection_confidence=det_conf, min_tracking_confidence=track_conf)
    pose1 = mp_pose.Pose(min_detection_confidence=det_conf, min_tracking_confidence=track_conf)
    return pose0, pose1


def _open_caps(path0: str, path1: str):
    cap0_ctor = getattr(cv, "VideoCapture")
    cap1_ctor = getattr(cv, "VideoCapture")
    cap0 = cap0_ctor(path0)
    cap1 = cap1_ctor(path1)
    if not cap0.isOpened():
        raise RuntimeError(f"Failed to open video 0: {path0}")
    if not cap1.isOpened():
        raise RuntimeError(f"Failed to open video 1: {path1}")
    return cap0, cap1


def _write_kpts3d_csv(frames_xyz: List[np.ndarray], out_csv: str, float_ndigits: int = 4):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows = []
    for fidx, xyz in enumerate(frames_xyz):
        row = {"frame": fidx}
        if xyz is None:
            # 欠損の場合は -1 を入れる
            for jid in range(12):
                row[f"joint_{jid}_x"] = -1.0
                row[f"joint_{jid}_y"] = -1.0
                row[f"joint_{jid}_z"] = -1.0
        else:
            for jid in range(xyz.shape[0]):
                x, y, z = xyz[jid]
                row[f"joint_{jid}_x"] = round(float(x), float_ndigits)
                row[f"joint_{jid}_y"] = round(float(y), float_ndigits)
                row[f"joint_{jid}_z"] = round(float(z), float_ndigits)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv


def run(
    cam0_path: str,
    cam1_path: str,
    out_csv: str,
    stride: int = 1,
    max_frames: int = -1,
    unit_scale: float = 1.0,
    file_mode: bool = True,
    det_conf: float = 0.5,
    track_conf: float = 0.5,
):
    t0 = time.time()
    print("[INFO] loading projection matrices (camera_parameters)")
    P0 = get_projection_matrix(0, file_mode)
    P1 = get_projection_matrix(1, file_mode)
    if P0 is None or P1 is None:
        raise RuntimeError("Projection matrices not found. Please run stereo calibration to create camera_parameters.")

    print("[INFO] opening videos")
    cap0, cap1 = _open_caps(cam0_path, cam1_path)
    pose0, pose1 = _init_pose(det_conf, track_conf)

    frames_xyz: List[np.ndarray] = []
    fcount = 0
    processed = 0
    skipped = 0

    while True:
        ok0, frame0 = cap0.read()
        ok1, frame1 = cap1.read()
        if not ok0 or not ok1:
            break

        if stride > 1 and (fcount % stride != 0):
            fcount += 1
            skipped += 1
            continue

        # MediapipeはRGB入力
        cvt = getattr(cv, "cvtColor")
        bgr2rgb = getattr(cv, "COLOR_BGR2RGB")
        frame0_rgb = cvt(frame0, bgr2rgb)
        frame1_rgb = cvt(frame1, bgr2rgb)
        frame0_rgb.flags.writeable = False
        frame1_rgb.flags.writeable = False
        res0 = pose0.process(frame0_rgb)
        res1 = pose1.process(frame1_rgb)

        # 2Dキーポイント抽出（ピクセル座標）
        frame0_bgr = frame0  # extract_keypointsは描画にも使用できるが、ここでは未使用
        frame1_bgr = frame1
        kpts0, kpts1 = extract_keypoints(res0, res1, pose_keypoints, frame0_bgr, frame1_bgr)

        # 3D再構成
        xyz = calculate_3d_keypoints(kpts0, kpts1, P0, P1)
        xyz = np.array(xyz, dtype=float)
        if unit_scale != 1.0:
            xyz = xyz * unit_scale

        frames_xyz.append(xyz)
        fcount += 1
        processed += 1

        if (processed % 50) == 0:
            print(f"[TRACE] processed={processed} skipped={skipped}")

        if max_frames > 0 and processed >= max_frames:
            print(f"[INFO] reached max_frames={max_frames}")
            break

    cap0.release()
    cap1.release()
    # poseの明示的クリーンアップ
    try:
        pose0.close()
        pose1.close()
    except AttributeError:
        # older mediapipe versions may not have close()
        pass

    print(f"[INFO] writing CSV -> {out_csv}")
    saved = _write_kpts3d_csv(frames_xyz, out_csv)
    elapsed = time.time() - t0
    print(f"[DONE] frames: {len(frames_xyz)} saved: {saved} elapsed: {elapsed:.2f}s")


def main(argv: List[str]):
    parser = argparse.ArgumentParser(description="Stereo 3D pose reconstruction to CSV from two videos")
    parser.add_argument("--cam0", required=True, help="Camera 0 video file path (left)")
    parser.add_argument("--cam1", required=True, help="Camera 1 video file path (right)")
    parser.add_argument("--out-csv", default="output_data/poses/kpts3d_stereo.csv", help="Output CSV path")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=-1, help="Limit number of frames to process (-1=all)")
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1.0,
        help="Multiply 3D coordinates by this factor (e.g., 0.01 to convert cm->m)",
    )
    parser.add_argument(
        "--file-mode",
        type=int,
        default=1,
        help="Use saved camera_parameters to build projection matrices (1=true,0=false)",
    )
    parser.add_argument("--det-conf", type=float, default=0.5, help="MediaPipe min_detection_confidence")
    parser.add_argument("--track-conf", type=float, default=0.5, help="MediaPipe min_tracking_confidence")
    args = parser.parse_args(argv)

    run(
        cam0_path=args.cam0,
        cam1_path=args.cam1,
        out_csv=args.out_csv,
        stride=args.stride,
        max_frames=args.max_frames,
        unit_scale=args.unit_scale,
        file_mode=bool(args.file_mode),
        det_conf=args.det_conf,
        track_conf=args.track_conf,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
