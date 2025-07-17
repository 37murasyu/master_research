r"""generate_zero_torque_from_pose.py
pose CSVのframe列に合わせて、トルク列をゼロで満たしたCSV(aim_torque_vec_*.csv 互換)を生成するユーティリティ。

使いどころ:
- 指定セッションでグローバルトルクが未取得/未算出だが、ダミーのトルクCSVが必要な場合。
- 列スキーマは左右別 wrist/elbow/shoulder の18列。

例:
    python generate_zero_torque_from_pose.py --pose-csv .\\output_data\\poses\\stereo_9_20250925_201442_pose.csv \
        --out .\\output_data\\aim_torque_vec_9_20250925_201442.csv
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np


DEFAULT_PART_ORDER: List[str] = [
    "wrist_R",
    "elbow_R",
    "shoulder_R",
    "wrist_L",
    "elbow_L",
    "shoulder_L",
]


def build_cols(parts: List[str]) -> List[str]:
    return [f"{p}_{ax}" for p in parts for ax in ("x", "y", "z")]


def infer_id_from_pose_path(path: str) -> str:
    base = os.path.basename(path)
    # e.g., stereo_9_20250925_201442_pose.csv -> 9_20250925_201442
    if base.startswith("stereo_") and base.endswith("_pose.csv"):
        return base[len("stereo_"):-len("_pose.csv")]
    return os.path.splitext(base)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pose CSVのframeに合わせたゼロトルクCSVを生成")
    ap.add_argument("--pose-csv", required=True)
    ap.add_argument("--out", default=None, help="出力CSVパス。未指定なら output_data/aim_torque_vec_<ID>.csv")
    args = ap.parse_args(argv)

    import pandas as pd
    df = pd.read_csv(args.pose_csv)
    if "frame" not in df.columns:
        raise ValueError("pose CSVに'frame'列がありません")
    frames = df["frame"].to_numpy(dtype=np.int64)
    parts = DEFAULT_PART_ORDER
    torque_cols = build_cols(parts)

    out_df = pd.DataFrame({"frame": frames})
    for c in torque_cols:
        out_df[c] = 0.0

    if args.out is None:
        id_str = infer_id_from_pose_path(args.pose_csv)
        out_dir = os.path.join(os.path.dirname(args.pose_csv), os.pardir)
        out_dir = os.path.normpath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"aim_torque_vec_{id_str}.csv")
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved zero torque CSV: {out_path} frames={len(frames)} cols={len(torque_cols)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
