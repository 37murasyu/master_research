import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

CSV = Path("output_data/filtered_pose_lpf/5_1stereo_pose_scaled_lpf.csv")
OUT = Path("fft_plots/pose3d_5_1stereo_lpf_frame100.html")

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

    xs = [pts[j][0] for j in pts]
    ys = [pts[j][1] for j in pts]
    zs = [pts[j][2] for j in pts]
    labels = [str(j) for j in pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers+text", text=labels, textposition="top center", marker=dict(size=4)))

    for a, b in EDGES:
        if a in pts and b in pts:
            pa, pb = pts[a], pts[b]
            fig.add_trace(go.Scatter3d(x=[pa[0], pb[0]], y=[pa[1], pb[1]], z=[pa[2], pb[2]], mode="lines", line=dict(width=3)))

    fig.update_layout(title=f"3D Pose (frame {frame_idx})", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT)
    print(f"[OUT] {OUT}")


if __name__ == "__main__":
    main()
