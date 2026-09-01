from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = [
    "output_data/cycle_energy/cycle_energy_kpts3d_9_20250925_201442_lpf_s9_R.csv",
    "output_data/cycle_energy/cycle_energy_2_stereo_pose_lpf_s2_L.csv",
    "output_data/cycle_energy/cycle_energy_2_stereo_pose_lpf_s2_R.csv",
    "output_data/cycle_energy/cycle_energy_3_0stereo_pose_scaled_with2d_lpf_s3_L.csv",
    "output_data/cycle_energy/cycle_energy_3_0stereo_pose_scaled_with2d_lpf_s3_R.csv",
    "output_data/cycle_energy/cycle_energy_3_1stereo_pose_scaled_with2d_lpf_s3_L.csv",
    "output_data/cycle_energy/cycle_energy_3_1stereo_pose_scaled_with2d_lpf_s3_R.csv",
    "output_data/cycle_energy/cycle_energy_5_1stereo_pose_scaled_lpf_s5_L.csv",
    "output_data/cycle_energy/cycle_energy_5_1stereo_pose_scaled_lpf_s5_R.csv",
    "output_data/cycle_energy/cycle_energy_6_stereo_pose_scaled_with2d_lpf_s6_L.csv",
    "output_data/cycle_energy/cycle_energy_6_stereo_pose_scaled_with2d_lpf_s6_R.csv",
    "output_data/cycle_energy/cycle_energy_7_stereo_pose_scaled_with2d_lpf_s7_L.csv",
    "output_data/cycle_energy/cycle_energy_7_stereo_pose_scaled_with2d_lpf_s7_R.csv",
    "output_data/cycle_energy/cycle_energy_8_stereo_pose_scaled_with2d_lpf_s8_L.csv",
    "output_data/cycle_energy/cycle_energy_8_stereo_pose_scaled_with2d_lpf_s8_R.csv",
    "output_data/cycle_energy/cycle_energy_kpts3d_9_20250925_201442_lpf_s9_L.csv",
]

root = Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE")

all_vals = []
labels = []
for rel in files:
    path = root / rel
    df = pd.read_csv(path)
    if "ratio_pos_vs_1rm" not in df.columns:
        continue
    vals = df["ratio_pos_vs_1rm"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if vals.size == 0:
        continue
    all_vals.append(vals)
    labels.append(path.stem)

plt.figure(figsize=(10, 6))
if all_vals:
    plt.hist(all_vals, bins=30, stacked=False, alpha=0.5, label=labels)
    plt.legend(fontsize=6, ncol=2)
    plt.xlabel("ratio_pos_vs_1rm")
    plt.ylabel("count")
    plt.title("Histogram of ratio_pos_vs_1rm (all files)")
    plt.tight_layout()
    out_path = root / "output_data" / "cycle_energy" / "ratio_pos_vs_1rm_hist_all.png"
    plt.savefig(out_path, dpi=150)
    print(f"[OK] {out_path}")
else:
    print("[WARN] no data to plot")
