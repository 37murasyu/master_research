"""stereo_triangulate_pose.py
=================================
2台カメラの同期(想定)動画 + 既存キャリブレーションファイル (c0.dat, c1.dat, rot_trans_c0.dat, rot_trans_c1.dat)
を用い MediaPipe Pose で抽出した2D正規化ランドマークをピクセル座標へ変換し，
投影行列 *DLT (OpenCV triangulatePoints)* で 3D 座標 (T,33,3) を再構成するスクリプト。

前提: 入力ディレクトリ構造例
  <dir>/
    cam0_xxx.mp4
    cam1_xxx.mp4
    c0.dat
    c1.dat
    rot_trans_c0.dat
    rot_trans_c1.dat

出力:
  --out base 名 (例 out_pose.npy) またはデフォルト <dir>/stereo_pose.npy
  3D配列 shape (T,33,3) float32 （未再構成は NaN）
  オプションで 2D キーポイント (cam0/cam1) も保存: *_cam0_2d.npy, *_cam1_2d.npy

同期:
  フレーム数が異なる場合は最小長に合わせる。--stride で間引き。

注意:
  - MediaPipe の normalized landmark (x,y) は [0,1] 領域 (画像左上原点) を前提
  - ピクセル座標 = (x * width, y * height)
  - z は triangulation 結果のスケール (絶対単位ではない) -> 比較には距離正規化を推奨
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List
import csv

import numpy as np
# pylint: disable=no-member
import cv2 as cv  # type: ignore
try:
    import mediapipe as mp  # type: ignore
except Exception as e:  # noqa: BLE001
    mp = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# utils.py 内の機能を局所利用 (依存最小化のため必要部分のみ再実装/インポート)
try:
    from utils import DLT, read_camera_parameters, read_rotation_translation  # type: ignore
except Exception:
    # フォールバック (最小限 DLT のみ再定義)
    def DLT(P1, P2, point1, point2):  # type: ignore
        import numpy as _np
        import cv2 as _cv
        P1 = _np.asarray(P1, dtype=_np.float64)
        P2 = _np.asarray(P2, dtype=_np.float64)
        pts1 = _np.array([[float(point1[0])], [float(point1[1])]], dtype=_np.float64)
        pts2 = _np.array([[float(point2[0])], [float(point2[1])]], dtype=_np.float64)

        func = getattr(_cv, "triangulatePoints", None)
        if func is not None:
            # cv2 stubs may miss triangulatePoints; getattr + ignore silences type checker
            Xh = func(P1, P2, pts1, pts2)  # type: ignore[attr-defined]
            w = float(Xh[3, 0])
            if not _np.isfinite(w) or abs(w) < 1e-12:
                return _np.array([-1.0, -1.0, -1.0], dtype=_np.float64)
            X = (Xh[:3, 0] / w).astype(_np.float64)
        else:
            # Manual DLT fallback (SVD)
            u1, v1 = float(point1[0]), float(point1[1])
            u2, v2 = float(point2[0]), float(point2[1])
            A = _np.vstack([
                u1 * P1[2, :] - P1[0, :],
                v1 * P1[2, :] - P1[1, :],
                u2 * P2[2, :] - P2[0, :],
                v2 * P2[2, :] - P2[1, :],
            ])
            try:
                _, _, Vt = _np.linalg.svd(A)
                Xh = Vt[-1]
                w = Xh[3]
                if not _np.isfinite(w) or abs(w) < 1e-12:
                    return _np.array([-1.0, -1.0, -1.0], dtype=_np.float64)
                X = (Xh[:3] / w).astype(_np.float64)
            except Exception:
                return _np.array([-1.0, -1.0, -1.0], dtype=_np.float64)

        if not _np.all(_np.isfinite(X)):
            return _np.array([-1.0, -1.0, -1.0], dtype=_np.float64)
        return X

    def read_camera_parameters(camera_id, savefolder):  # type: ignore
        path = os.path.join(savefolder, f"c{camera_id}.dat")
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            cmtx = [[float(en) for en in f.readline().split()] for _ in range(3)]
            f.readline()
            dist = [float(en) for en in f.readline().split()]
        return np.array(cmtx), np.array([dist])

    def read_rotation_translation(camera_id, savefolder):  # type: ignore
        path = os.path.join(savefolder, f"rot_trans_c{camera_id}.dat")
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            rot = [[float(en) for en in f.readline().split()] for _ in range(3)]
            f.readline()
            trans = [[float(en) for en in f.readline().split()] for _ in range(3)]
        return np.array(rot), np.array(trans)


def build_projection_matrix(camera_id: int, base: str):
    # utils.read_* は savefolder + filename の単純連結を行う実装のため、末尾セパレータを強制付与
    if not base.endswith(("/", "\\")):
        base = base + os.sep
    K, _ = read_camera_parameters(camera_id, base)
    R, t = read_rotation_translation(camera_id, base)
    P = np.zeros((4, 4), dtype=np.float64)
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1.0
    return K @ P[:3, :]


def parse_args():
    ap = argparse.ArgumentParser(
        description="ステレオ動画から MediaPipe Pose + キャリブで 3D 再構成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-dir", required=True, help="動画とキャリブ(dat)が入ったディレクトリ")
    ap.add_argument("--calib-dir", default=None, help="キャリブ(dat)のディレクトリ (未指定なら --input-dir を使用)")
    ap.add_argument("--cam0", default="cam0_*.mp4", help="左/基準カメラ動画パターン")
    ap.add_argument("--cam1", default="cam1_*.mp4", help="右カメラ動画パターン")
    ap.add_argument("--stride", type=int, default=1, help="フレーム間引き")
    ap.add_argument("--max-frames", type=int, default=None, help="最大フレーム数 (両カメラ共通)")
    ap.add_argument("--out", type=str, default=None, help="出力ファイルパス (.csv か .npy)。未指定なら <dir>/stereo_pose.<fmt>")
    ap.add_argument("--save-format", choices=["csv","npy"], default="csv", help="保存形式。--out の拡張子が .csv/.npy の場合はそちらを優先")
    ap.add_argument("--save-2d", action="store_true", help="2Dランドマークも保存")
    ap.add_argument("--save-failed-mask", action="store_true", help="再構成失敗(全部 -1)フレームマスク保存")
    ap.add_argument("--model-complexity", type=int, choices=[0,1,2], default=1)
    ap.add_argument("--verbose", action="store_true")
    # Optional post-scale: set median length of a joint pair (e.g., 12 14) to target units (e.g., meters)
    ap.add_argument("--scale-pair", type=int, nargs=2, default=None, metavar=("J0","J1"), help="joint indices to measure length for scaling (median)" )
    ap.add_argument("--scale-target", type=float, default=None, help="target length for --scale-pair (same unit as desired output, e.g., meters)")
    return ap.parse_args()


def select_first(pattern: str, base: str) -> Optional[str]:
    import glob
    files = sorted(glob.glob(os.path.join(base, pattern)))
    return files[0] if files else None


def run():
    args = parse_args()
    # 上肢など、CSV出力対象の関節インデックス
    try:
        from config import pose_keypoints as _CSV_JOINT_IDX  # type: ignore
        CSV_JOINT_IDX: List[int] = list(_CSV_JOINT_IDX)
    except Exception:
        CSV_JOINT_IDX = []  # 読み込み失敗時は全関節
    if _IMPORT_ERROR is not None:
        print(f"mediapipe/cv2 import error: {_IMPORT_ERROR}", file=sys.stderr)
        return 1

    v0_path = select_first(args.cam0, args.input_dir)
    v1_path = select_first(args.cam1, args.input_dir)
    if not v0_path or not v1_path:
        print("動画が見つかりません", file=sys.stderr)
        return 1

    if args.verbose:
        print("cam0=", v0_path)
        print("cam1=", v1_path)

    # キャリブディレクトリの決定と存在チェック
    calib_base = args.calib_dir or args.input_dir
    req_files = ["c0.dat", "c1.dat", "rot_trans_c0.dat", "rot_trans_c1.dat"]
    missing = [fn for fn in req_files if not os.path.isfile(os.path.join(calib_base, fn))]
    if missing:
        print("キャリブファイルが見つかりません:", file=sys.stderr)
        for fn in missing:
            print("  not found:", os.path.normpath(os.path.join(calib_base, fn)), file=sys.stderr)
        print("--calib-dir で正しいディレクトリを指定するか、ファイルを --input-dir に配置してください。", file=sys.stderr)
        return 1

    # 投影行列
    P0 = build_projection_matrix(0, calib_base)
    P1 = build_projection_matrix(1, calib_base)
    if args.verbose:
        print("Loaded projection matrices from:", os.path.normpath(calib_base))

    cap0 = cv.VideoCapture(v0_path)
    cap1 = cv.VideoCapture(v1_path)
    if not cap0.isOpened() or not cap1.isOpened():
        print("Video open failed", file=sys.stderr)
        return 1

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames3d: List[np.ndarray] = []
    frames2d_cam0: List[np.ndarray] = []
    frames2d_cam1: List[np.ndarray] = []
    failed_mask: List[bool] = []

    frame_idx = 0
    while True:
        ret0, f0 = cap0.read()
        ret1, f1 = cap1.read()
        if not ret0 or not ret1:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        if args.max_frames is not None and len(frames3d) >= args.max_frames:
            break

        h0, w0 = f0.shape[:2]
        h1, w1 = f1.shape[:2]

        # MediaPipe -> normalized landmarks
        # Some type checkers/stubs for cv2 miss cvtColor; add ignore to silence false positive.
        res0 = pose.process(cv.cvtColor(f0, cv.COLOR_BGR2RGB))  # type: ignore[attr-defined]
        res1 = pose.process(cv.cvtColor(f1, cv.COLOR_BGR2RGB))  # type: ignore[attr-defined]

        def extract_xy(res, w, h):
            if res.pose_landmarks is None:
                return np.full((33, 2), -1.0, dtype=np.float32)
            pts = np.array([[lm.x * w, lm.y * h] for lm in res.pose_landmarks.landmark], dtype=np.float32)
            if pts.shape[0] != 33:
                pts = np.pad(pts, ((0, max(0, 33 - pts.shape[0])), (0, 0)), constant_values=-1.0)[:33]
            return pts

        pts0 = extract_xy(res0, w0, h0)
        pts1 = extract_xy(res1, w1, h1)

        # Triangulation
        frame3d = np.full((33, 3), np.nan, dtype=np.float32)
        invalid_cnt = 0
        for i, (p0, p1) in enumerate(zip(pts0, pts1)):
            if p0[0] < 0 or p1[0] < 0:  # missing
                invalid_cnt += 1
                continue
            X = DLT(P0, P1, p0, p1)
            if X[0] == -1:
                invalid_cnt += 1
                continue
            frame3d[i] = X.astype(np.float32)
        frames3d.append(frame3d)
        if args.save_2d:
            frames2d_cam0.append(pts0)
            frames2d_cam1.append(pts1)
        failed_mask.append(invalid_cnt == 33)

        if args.verbose and frame_idx % (args.stride * 30) == 0:
            print(f"processed frame {frame_idx} -> valid joints {(33-invalid_cnt)}")
        frame_idx += 1

    pose.close()
    cap0.release()
    cap1.release()

    arr3d = np.stack(frames3d, axis=0) if frames3d else np.zeros((0, 33, 3), dtype=np.float32)

    # Optional global scaling to enforce a median segment length
    if args.scale_pair is not None and args.scale_target is not None:
        j0, j1 = args.scale_pair
        if arr3d.size == 0:
            print("warning: no 3D data to scale", file=sys.stderr)
        elif not (0 <= j0 < arr3d.shape[1] and 0 <= j1 < arr3d.shape[1]):
            print("warning: scale-pair indices out of range; skipping scale", file=sys.stderr)
        else:
            seg = arr3d[:, j0, :] - arr3d[:, j1, :]
            seg = seg[np.isfinite(seg).all(axis=1)]
            if seg.size == 0:
                print("warning: no valid segments for scaling; skipping", file=sys.stderr)
            else:
                med = float(np.median(np.linalg.norm(seg, axis=1)))
                if med <= 0:
                    print("warning: median segment length is non-positive; skipping scale", file=sys.stderr)
                else:
                    scale = float(args.scale_target) / med
                    arr3d = arr3d * scale
                    if args.verbose:
                        print(f"scaled 3D by {scale:.6f} so median |joint{j0}-joint{j1}| = {args.scale_target}")

    # 保存形式の決定
    def infer_format_and_path() -> tuple[str, str]:
        # 優先: --out の拡張子、次に --save-format
        if args.out:
            out_path = args.out
            ext = os.path.splitext(out_path)[1].lower()
            if ext in (".csv", ".npy"):
                fmt = "csv" if ext == ".csv" else "npy"
                return fmt, out_path
            else:
                # 拡張子なし/未知なら save-format を付与
                fmt = args.save_format
                out_path = out_path + (".csv" if fmt == "csv" else ".npy")
                return fmt, out_path
        else:
            fmt = args.save_format
            out_path = os.path.join(args.input_dir, f"stereo_pose.{ 'csv' if fmt=='csv' else 'npy'}")
            return fmt, out_path

    def save_3d_csv(path: str, arr: np.ndarray, joint_idx: Optional[List[int]] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        T = arr.shape[0]
        allJ = arr.shape[1] if arr.ndim == 3 else 0
        use_idx = [j for j in (joint_idx or list(range(allJ))) if 0 <= j < allJ]
        header = ["frame"] + [ax for j in use_idx for ax in (f"joint_{j}_x", f"joint_{j}_y", f"joint_{j}_z")]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for t in range(T):
                pts = arr[t, use_idx, :] if use_idx else arr[t]
                row = [t] + [f"{v:.6f}" for v in pts.reshape(-1)]
                w.writerow(row)

    def save_2d_csv(base_path_noext: str, cam0: list[np.ndarray], cam1: list[np.ndarray], joint_idx: Optional[List[int]] = None) -> None:
        p0 = base_path_noext + "_cam0_2d.csv"
        p1 = base_path_noext + "_cam1_2d.csv"
        for path, series in ((p0, cam0), (p1, cam1)):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            T = len(series)
            allJ = series[0].shape[0] if T else 0
            use_idx = [j for j in (joint_idx or list(range(allJ))) if 0 <= j < allJ]
            header = ["frame"] + [f"joint_{j}_x" for j in use_idx] + [f"joint_{j}_y" for j in use_idx]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                for t in range(T):
                    pts = series[t]
                    if use_idx:
                        pts = pts[use_idx, :]
                    row = [t] + [f"{v:.6f}" for v in pts[:,0].tolist()] + [f"{v:.6f}" for v in pts[:,1].tolist()]
                    w.writerow(row)

    fmt, out_path = infer_format_and_path()

    if fmt == "csv":
        save_3d_csv(out_path, arr3d, joint_idx=CSV_JOINT_IDX if CSV_JOINT_IDX else None)
        if args.verbose:
            print("Saved 3D pose CSV:", out_path, arr3d.shape)
        if args.save_2d and frames2d_cam0:
            base_noext = os.path.splitext(out_path)[0]
            save_2d_csv(base_noext, frames2d_cam0, frames2d_cam1, joint_idx=CSV_JOINT_IDX if CSV_JOINT_IDX else None)
        if args.save_failed_mask:
            # 失敗マスクは CSV に 1/0 で保存
            p = os.path.splitext(out_path)[0] + "_failed_mask.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame", "failed"])
                for i, b in enumerate(failed_mask):
                    w.writerow([i, int(bool(b))])
    else:
        # npy 保存
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.save(out_path, arr3d)
        if args.verbose:
            print("Saved 3D pose:", out_path, arr3d.shape)
        if args.save_2d and frames2d_cam0:
            np.save(out_path.replace('.npy', '_cam0_2d.npy'), np.stack(frames2d_cam0, axis=0))
            np.save(out_path.replace('.npy', '_cam1_2d.npy'), np.stack(frames2d_cam1, axis=0))
        if args.save_failed_mask:
            np.save(out_path.replace('.npy', '_failed_mask.npy'), np.array(failed_mask, dtype=np.bool_))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
