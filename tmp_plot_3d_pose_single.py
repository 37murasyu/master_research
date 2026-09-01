import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("output_data/filtered_pose_lpf/5_1stereo_pose_scaled_lpf.csv")
OUT = Path("fft_plots/pose3d_5_1stereo_lpf_frame100.png")

# MediaPipe IDs used
JOINTS = {
    16: "wrist_R", 14: "elbow_R", 12: "shoulder_R",
    11: "shoulder_L", 13: "elbow_L", 15: "wrist_L",
    24: "hip_R", 23: "hip_L",
    25: "knee_R", 26: "knee_L",
    27: "ankle_R", 28: "ankle_L",
}
EDGES = [
    (12, 14), (14, 16),
    (11, 13), (13, 15),
    (24, 25), (25, 27),
    (23, 26), (26, 28),
    (11, 12), (23, 24),
    (11, 23), (12, 24),
]


def main():
    df = pd.read_csv(CSV)
    frame_idx = min(100, len(df) - 1)
    row = df.iloc[frame_idx]

    pts = {}
    for jid in JOINTS:
        x = row.get(f"joint_{jid}_x")
        y = row.get(f"joint_{jid}_y")
        z = row.get(f"joint_{jid}_z")
        if pd.isna(x) or pd.isna(y) or pd.isna(z):
            continue
        pts[jid] = np.array([x, y, z], dtype=float)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    for jid, p in pts.items():
        ax.scatter(p[0], p[1], p[2], s=20)
        ax.text(p[0], p[1], p[2], str(jid), fontsize=7)

    for a, b in EDGES:
        if a in pts and b in pts:
            pa, pb = pts[a], pts[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], linewidth=1.2)

    ax.set_title(f"3D Pose (frame {frame_idx})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"[OUT] {OUT}")


if __name__ == "__main__":
    main()
