from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_torque_lpf_recalc")
files = [
    "3_0stereo_pose_scaled_with2d_torque_lpf.csv",
    "3_1stereo_pose_scaled_with2d_torque_lpf.csv",
    "4_0stereo_pose_scaled_with2d_torque_lpf.csv",
    "5_1stereo_pose_scaled_torque_lpf.csv",
    "5_stereo_pose_scaled_with2d_torque_lpf.csv",
    "6_stereo_pose_scaled_with2d_torque_lpf.csv",
    "7_stereo_pose_scaled_with2d_torque_lpf.csv",
    "8_stereo_pose_scaled_with2d_torque_lpf.csv",
]

for name in files:
    path = folder / name
    df = pd.read_csv(path)
    if "elbow_R_local_y" not in df.columns:
        print(f"[SKIP] {name}: elbow_R_local_y not found")
        continue
    x = df["frame"] if "frame" in df.columns else np.arange(len(df))
    y = df["elbow_R_local_y"]
    plt.figure(figsize=(10, 3))
    plt.plot(x, y, linewidth=0.8)
    plt.title(f"{name} elbow_R_local_y")
    plt.xlabel("frame")
    plt.ylabel("elbow_R_local_y")
    plt.tight_layout()
    out = folder / f"{path.stem}_elbow_R_local_y.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")
