import matplotlib
matplotlib.use('Agg')
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

files = [
    Path('output_data/poses/kpts3d_stereo_24er.csv'),
    Path('output_data/poses/kpts3d_stereo_26eor.csv'),
    Path('output_data/poses/kpts3d_stereo_00er.csv'),
]

fig, axes = plt.subplots(len(files), 1, figsize=(10, 9), sharex=False)

for ax, csv_path in zip(axes, files):
    if not csv_path.exists():
        ax.set_title(f"Missing: {csv_path.name}")
        ax.axis('off')
        print(f"[WARN] skip missing {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    col = 'joint_5_y'
    if col not in df.columns:
        ax.set_title(f"{col} not found: {csv_path.name}")
        ax.axis('off')
        print(f"[WARN] {col} not found in {csv_path}")
        continue

    frames = df['frame'] if 'frame' in df.columns else range(len(df))
    y = df[col]
    ax.plot(frames, y, lw=1.0, label=col)
    ax.set_title(f"{col}: {csv_path.name}")
    ax.set_xlabel('frame')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

fig.tight_layout()
out_png = Path('output_data/poses/kpts3d_joint5_y.png')
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"[SAVED] {out_png}")
