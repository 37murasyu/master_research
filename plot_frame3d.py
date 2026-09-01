import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def extract_joints(df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    row = df.iloc[0]
    joints: Dict[str, Tuple[float, float, float]] = {}
    for col in df.columns:
        m = re.match(r"joint_(\d+)_(x|y|z)$", col)
        if not m:
            continue
        jid, axis = m.group(1), m.group(2)
        if jid not in joints:
            joints[jid] = [float('nan'), float('nan'), float('nan')]
        idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        joints[jid][idx] = float(row[col])
    return {k: tuple(v) for k, v in joints.items()}


def plot_frame(csv_path: Path, frame: int, out_dir: Path, elev: float, azim: float, interactive: bool) -> Path:
    df = pd.read_csv(csv_path)
    if 'frame' in df.columns:
        target = df[df['frame'] == frame]
        if target.empty:
            raise SystemExit(f"frame {frame} not found in {csv_path}")
        row_df = target.iloc[[0]]
    else:
        if frame >= len(df):
            raise SystemExit(f"frame index {frame} out of range for {csv_path}")
        row_df = df.iloc[[frame]]

    joints = extract_joints(row_df)
    xs, ys, zs, labels = [], [], [], []
    for jid, (x, y, z) in sorted(joints.items(), key=lambda kv: int(kv[0])):
        xs.append(x)
        ys.append(y)
        zs.append(z)
        labels.append(jid)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='tab:blue')
    for x, y, z, lab in zip(xs, ys, zs, labels):
        ax.text(x, y, z, lab, fontsize=8)

    # Draw skeletal connections when available
    connections: List[Tuple[str, str]] = [
        ("11", "12"),  # shoulders
        ("11", "13"), ("13", "15"),  # left arm
        ("12", "14"), ("14", "16"),  # right arm
        ("11", "23"), ("12", "24"),  # shoulders to hips
        ("23", "24"),  # hip line
        ("23", "25"), ("25", "27"),  # left leg
        ("24", "26"), ("26", "28"),  # right leg
    ]

    for a, b in connections:
        if a in joints and b in joints:
            (x1, y1, z1), (x2, y2, z2) = joints[a], joints[b]
            ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", linewidth=1)

    ax.set_xlabel('x (m)')  # horizontal
    ax.set_ylabel('y (m)')  # vertical
    ax.set_zlabel('z (m)')  # depth
    ax.set_title(f'{csv_path.stem} frame {frame}')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    if interactive:
        plt.show()
        return out_dir / f"{csv_path.stem}_frame{frame}_3d.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_frame{frame}_3d.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Plot 3D joint scatter for a single frame')
    ap.add_argument('--csv', nargs='+', required=True, help='Pose CSVs with joint_* columns')
    ap.add_argument('--frame', type=int, default=100, help='Frame index (matching frame column if present)')
    ap.add_argument('--out-dir', default='output_data/plots', help='Output directory for PNGs')
    ap.add_argument('--elev', type=float, default=0.0, help='Elevation angle for view_init (y up)')
    ap.add_argument('--azim', type=float, default=-90.0, help='Azimuth angle for view_init (x right, z depth)')
    ap.add_argument('--interactive', action='store_true', help='Show interactive 3D plot (no save)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for csv_path in args.csv:
        out = plot_frame(Path(csv_path), args.frame, out_dir, args.elev, args.azim, args.interactive)
        if not args.interactive:
            print(f'[OUT] {out}')


if __name__ == '__main__':
    main()