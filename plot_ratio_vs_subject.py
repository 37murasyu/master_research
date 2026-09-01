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

rows = []
for rel in files:
    path = root / rel
    df = pd.read_csv(path)
    if df.empty:
        continue
    if "subject_id" not in df.columns or "ratio_pos_vs_1rm" not in df.columns or "part" not in df.columns:
        continue
    df = df.replace([np.inf, -np.inf], np.nan)
    sid = int(df["subject_id"].iloc[0])
    for part_key, part_label in ("elbow_", "elbow"), ("wrist_", "wrist"):
        sub = df[df["part"].astype(str).str.startswith(part_key)]["ratio_pos_vs_1rm"].dropna().to_numpy(float)
        if sub.size == 0:
            continue
        rows.append({"subject_id": sid, "part": part_label, "ratio": sub})

if not rows:
    print("[WARN] no data to plot")
    raise SystemExit(0)

# aggregate per subject and part
subj = {}
for r in rows:
    key = (r["subject_id"], r["part"])
    subj.setdefault(key, []).append(r["ratio"])

subjects = sorted({k[0] for k in subj.keys()})
parts = ["elbow", "wrist"]

data_by_part = {p: [] for p in parts}
means_by_part = {p: [] for p in parts}
for s in subjects:
    for p in parts:
        arrs = subj.get((s, p), [])
        vals = np.concatenate(arrs) if arrs else np.array([], dtype=float)
        data_by_part[p].append(vals)
        means_by_part[p].append(float(np.nanmean(vals)) if vals.size else np.nan)

plt.figure(figsize=(10, 4.5))
positions = np.arange(len(subjects))
offset = 0.18

bp1 = plt.boxplot(
    data_by_part["elbow"],
    positions=positions - offset,
    widths=0.3,
    patch_artist=True,
    showfliers=False,
)
bp2 = plt.boxplot(
    data_by_part["wrist"],
    positions=positions + offset,
    widths=0.3,
    patch_artist=True,
    showfliers=False,
)

for b in bp1["boxes"]:
    b.set_facecolor("#4C78A8")
for b in bp2["boxes"]:
    b.set_facecolor("#F58518")

plt.plot(
    positions - offset,
    means_by_part["elbow"],
    "D",
    color="#2F4B7C",
    markeredgecolor="white",
    markeredgewidth=1.2,
    label="elbow mean",
)
plt.plot(
    positions + offset,
    means_by_part["wrist"],
    "D",
    color="#1F1F1F",
    markeredgecolor="white",
    markeredgewidth=1.2,
    label="wrist mean",
)

plt.xticks(positions, [str(s) for s in subjects])
plt.xlabel("subject_id")
plt.ylabel("ratio_pos_vs_1rm")
plt.title("ratio_pos_vs_1rm per subject (boxplot)")
plt.legend()
plt.tight_layout()

out_path = root / "output_data" / "cycle_energy" / "ratio_pos_vs_1rm_boxplot_by_subject.png"
plt.savefig(out_path, dpi=150)
print(f"[OK] {out_path}")
