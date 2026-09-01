from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = [
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\2_stereo_pose_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\3_0stereo_pose_scaled_with2d_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\3_1stereo_pose_scaled_with2d_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\4_0stereo_pose_scaled_with2d_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\5_1stereo_pose_scaled_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\5_stereo_pose_scaled_with2d_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\6_stereo_pose_scaled_with2d_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\7_stereo_pose_scaled_with2d_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\8_stereo_pose_scaled_with2d_lpf.csv"),
    Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf\kpts3d_9_20250925_201442_lpf.csv"),
]

RIGHT_SHOULDER_COL = "joint_12_y"
LEGACY_RIGHT_SHOULDER_COL = "joint_0_y"

for path in files:
    df = pd.read_csv(path)
    col = None
    if RIGHT_SHOULDER_COL in df.columns:
        col = RIGHT_SHOULDER_COL
    elif LEGACY_RIGHT_SHOULDER_COL in df.columns:
        col = LEGACY_RIGHT_SHOULDER_COL
    else:
        print(f"[SKIP] {path.name}: right shoulder y not found")
        continue

    x = df["frame"] if "frame" in df.columns else np.arange(len(df))
    y = df[col]

    plt.figure(figsize=(10, 3))
    plt.plot(x, y, linewidth=0.8)
    plt.title(f"{path.name} {col}")
    plt.xlabel("frame")
    plt.ylabel(col)
    plt.tight_layout()

    out = path.parent / f"{path.stem}_{col}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")
