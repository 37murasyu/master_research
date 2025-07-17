"""video_pose_extractor.py
=================================
MP4動画群から MediaPipe Pose を用いて 3D (world) ランドマーク時系列を抽出し
npy / (任意) csv 保存するユーティリティ。

基本仕様:
  - 入力: ディレクトリ内の *.mp4 (--pattern で glob 変更可)
  - 出力: (T, 33, 3) の float32 numpy 配列 (world座標) を `<video_stem>_pose.npy`
           可視化用に2D画像座標 (landmark) を同時取得可能 (--save-visibility/--save-2d)
  - 欠損フレーム: ランドマーク未検出の場合は NaN を埋める
  - フレーム間引き: --stride
  - フレーム上限: --max-frames
  - 抽出後, 比較スクリプト `pose_sequence_comparison.py` にそのまま渡せる

依存:
  mediapipe, opencv-python, numpy

使用例:
  python video_pose_extractor.py --input-dir ../cameras_raw --output-dir output_data/poses

  2つの動画を抽出しそのまま比較:
  python video_pose_extractor.py --input-dir ../cameras_raw --pattern 'cam0_*.mp4' --limit 1 \
      && python video_pose_extractor.py --input-dir ../cameras_raw --pattern 'cam1_*.mp4' --limit 1 \
      && python pose_sequence_comparison.py output_data/poses/cam0_xxx_pose.npy output_data/poses/cam1_xxx_pose.npy

注意:
  - world_landmarks は正確なスケールを保証しない(モデル推論スケール)。距離構造比較には有効。
  - 複数人がフレームに存在する場合, 本スクリプトは MediaPipe Pose 単人体モデルを想定。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
# pylint: disable=no-member
import cv2  # type: ignore
import csv
try:
    import mediapipe as mp  # type: ignore
except Exception as e:  # noqa: BLE001
    mp = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass
class ExtractResult:
    path: str
    pose: np.ndarray  # (T,33,3)
    visibility: Optional[np.ndarray] = None  # (T,33)
    pose2d: Optional[np.ndarray] = None  # (T,33,2)


def iter_videos(input_dir: str, pattern: str) -> List[str]:
    pattern_full = os.path.join(input_dir, pattern)
    paths = sorted(glob.glob(pattern_full))
    return [p for p in paths if os.path.isfile(p)]


def extract_pose_from_video(
    video_path: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
    static_image_mode: bool = False,
    upper_body_only: bool = False,
    smooth_landmarks: bool = True,
    model_complexity: int = 1,
    enable_segmentation: bool = False,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    save_visibility: bool = False,
    save_2d: bool = False,
) -> ExtractResult:
    if cv2 is None or mp is None:
        raise RuntimeError(f"mediapipe / opencv import失敗: {_IMPORT_ERROR}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    Pose = mp.solutions.pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    world_series: List[np.ndarray] = []
    vis_series: List[np.ndarray] = []
    lm2d_series: List[np.ndarray] = []

    frame_idx = 0
    grabbed = True
    while grabbed:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        if max_frames is not None and len(world_series) >= max_frames:
            break

        # BGR->RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = Pose.process(rgb)

        # MediaPipe Pose の world 座標は pose_world_landmarks を参照する
        # 直接 world_landmarks にアクセスすると AttributeError になる環境があるため getattr で取得
        wlms = getattr(result, "pose_world_landmarks", None)

        # landmarks (2D) も同様に取得
        lms = getattr(result, "pose_landmarks", None)

        if wlms is not None:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in wlms.landmark], dtype=np.float32)
            if pts.shape[0] != 33:
                # 予期せぬ数
                pts = np.pad(pts, ((0, max(0, 33 - pts.shape[0])), (0, 0)), constant_values=np.nan)[:33]
        else:
            pts = np.full((33, 3), np.nan, dtype=np.float32)

        if save_visibility and lms is not None:
            vis = np.array([lm.visibility for lm in lms.landmark], dtype=np.float32)
            if vis.shape[0] != 33:
                vis = np.pad(vis, (0, max(0, 33 - vis.shape[0])), constant_values=np.nan)[:33]
        else:
            vis = None

        if save_2d and lms is not None:
            pts2d = np.array([[lm.x, lm.y] for lm in lms.landmark], dtype=np.float32)
            if pts2d.shape[0] != 33:
                pts2d = np.pad(pts2d, ((0, max(0, 33 - pts2d.shape[0])), (0, 0)), constant_values=np.nan)[:33]
        else:
            pts2d = None

        world_series.append(pts)
        if save_visibility:
            vis_series.append(vis if vis is not None else np.full((33,), np.nan, dtype=np.float32))
        if save_2d:
            lm2d_series.append(pts2d if pts2d is not None else np.full((33, 2), np.nan, dtype=np.float32))

        frame_idx += 1

    cap.release()
    Pose.close()

    world_arr = np.stack(world_series, axis=0) if world_series else np.zeros((0, 33, 3), dtype=np.float32)
    vis_arr = np.stack(vis_series, axis=0) if save_visibility and vis_series else None
    lm2d_arr = np.stack(lm2d_series, axis=0) if save_2d and lm2d_series else None

    return ExtractResult(path=video_path, pose=world_arr, visibility=vis_arr, pose2d=lm2d_arr)


def save_result(er: ExtractResult, output_dir: str, save_visibility: bool, save_2d: bool, save_format: str = "csv") -> str:
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(er.path))[0]

    # 出力に含める関節インデックス（config.py の pose_keypoints を優先）
    try:
        from config import pose_keypoints as _CSV_JOINT_IDX  # type: ignore
        CSV_JOINT_IDX: Optional[List[int]] = list(_CSV_JOINT_IDX)
    except Exception:
        CSV_JOINT_IDX = None

    def _save_3d_csv(path: str, arr: np.ndarray, joint_idx: Optional[List[int]] = None) -> None:
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

    def _save_2d_csv(path: str, arr2d: np.ndarray, joint_idx: Optional[List[int]] = None) -> None:
        T = arr2d.shape[0]
        allJ = arr2d.shape[1] if arr2d.ndim == 3 else 0
        use_idx = [j for j in (joint_idx or list(range(allJ))) if 0 <= j < allJ]
        header = ["frame"] + [f"joint_{j}_x" for j in use_idx] + [f"joint_{j}_y" for j in use_idx]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for t in range(T):
                pts = arr2d[t, use_idx, :] if use_idx else arr2d[t]
                row = [t] + [f"{v:.6f}" for v in pts[:, 0].tolist()] + [f"{v:.6f}" for v in pts[:, 1].tolist()]
                w.writerow(row)

    save_format = (save_format or "csv").lower()
    if save_format == "csv":
        out_path = os.path.join(output_dir, f"{stem}_pose.csv")
        _save_3d_csv(out_path, er.pose, joint_idx=CSV_JOINT_IDX)
        if save_2d and er.pose2d is not None:
            _save_2d_csv(os.path.join(output_dir, f"{stem}_pose2d.csv"), er.pose2d, joint_idx=CSV_JOINT_IDX)
        if save_visibility and er.visibility is not None:
            vis_path = os.path.join(output_dir, f"{stem}_visibility.csv")
            with open(vis_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                use_idx = CSV_JOINT_IDX or list(range(er.visibility.shape[1]))
                w.writerow(["frame"] + [f"joint_{j}" for j in use_idx])
                for t in range(er.visibility.shape[0]):
                    row = [t] + [f"{er.visibility[t, j]:.4f}" for j in use_idx]
                    w.writerow(row)
        return out_path
    else:
        out_path = os.path.join(output_dir, f"{stem}_pose.npy")
        np.save(out_path, er.pose)
        if save_visibility and er.visibility is not None:
            np.save(os.path.join(output_dir, f"{stem}_visibility.npy"), er.visibility)
        if save_2d and er.pose2d is not None:
            np.save(os.path.join(output_dir, f"{stem}_pose2d.npy"), er.pose2d)
        return out_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MediaPipe Pose による動画 -> (T,33,3) 3Dランドマーク抽出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", required=True, help="動画(.mp4)を含むディレクトリ")
    p.add_argument("--output-dir", default="output_data/poses", help="抽出結果の保存先 (CSV/Npy)")
    p.add_argument("--pattern", default="*.mp4", help="globパターン (例: cam0_*.mp4)")
    p.add_argument("--stride", type=int, default=1, help="フレーム間引き間隔")
    p.add_argument("--max-frames", type=int, default=None, help="各動画の最大処理フレーム数")
    p.add_argument("--limit", type=int, default=None, help="処理する動画ファイル数の上限")
    p.add_argument("--model-complexity", type=int, choices=[0,1,2], default=1, help="MediaPipe Pose model_complexity")
    p.add_argument("--no-smooth", action="store_true", help="landmark smoothing を無効化")
    p.add_argument("--save-visibility", action="store_true", help="visibility 配列も保存")
    p.add_argument("--save-2d", action="store_true", help="2Dランドマーク (正規化座標) も保存")
    p.add_argument("--save-format", choices=["csv", "npy"], default="csv", help="保存形式 (既定: csv)")
    p.add_argument("--verbose", action="store_true", help="進捗を表示")
    p.add_argument("--compare", type=str, default=None, help="既存の .npy (T,33,3) と最後に抽出した結果を即時比較 (δD 平均を表示)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if _IMPORT_ERROR is not None:
        parser.error(f"mediapipe / opencv が読み込めませんでした: {_IMPORT_ERROR}")

    paths = iter_videos(args.input_dir, args.pattern)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        parser.error("該当する動画がありません")

    if args.verbose:
        print(f"Found {len(paths)} video(s)")

    saved_paths: List[str] = []
    start_all = time.time()
    for idx, vp in enumerate(paths):
        t0 = time.time()
        try:
            er = extract_pose_from_video(
                vp,
                stride=args.stride,
                max_frames=args.max_frames,
                static_image_mode=False,
                smooth_landmarks=not args.no_smooth,
                model_complexity=args.model_complexity,
                save_visibility=args.save_visibility,
                save_2d=args.save_2d,
            )
            out_path = save_result(er, args.output_dir, args.save_visibility, args.save_2d, save_format=args.save_format)
            saved_paths.append(out_path)
            if args.verbose:
                print(f"[{idx+1}/{len(paths)}] {os.path.basename(vp)} -> {out_path} frames={er.pose.shape[0]} time={time.time()-t0:.2f}s")
        except Exception as e:  # noqa: BLE001
            print(f"[error] {vp}: {e}", file=sys.stderr)

    if args.verbose:
        print(f"Total time: {time.time()-start_all:.2f}s")
        print("Saved files:")
        for p in saved_paths:
            print("  ", p)
        print("次: pose_sequence_comparison.py file1 file2 で比較 (拡張子は自動判別)" )

    # --compare: 最後に生成したシーケンスと指定シーケンスを簡易比較
    if args.compare and saved_paths:
        try:
            from pose_sequence_comparison import load_pose_file, compute_sequence_delta  # type: ignore
            seq_existing = load_pose_file(args.compare)
            seq_new = load_pose_file(saved_paths[-1])
            res = compute_sequence_delta(seq_existing, seq_new, norm_mode="fro", dtw=False, stride=1, alt_metric=False, progress=False)
            print(f"[compare] mean δD (no DTW): {res['stats']['mean']}")
        except Exception as e:  # noqa: BLE001
            print(f"[compare] 失敗: {e}", file=sys.stderr)
    elif args.compare and not saved_paths:
        print("[compare] 比較対象の抽出結果がありません", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
