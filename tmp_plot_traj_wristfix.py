import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

CSV = 'output_data/torque/kpts3d_subject9_20250925_201442_gcvspl_trimmed_wristfix_torque.csv'
OUT = 'output_data/plots/subject9_wristfix_traj_xyz.png'

# columns to plot (global components)
PARTS = [
    ('wrist_R', 'tab:orange'),
    ('elbow_R', 'tab:blue'),
    ('shoulder_R', 'tab:green'),
]


def main():
    df = pd.read_csv(CSV)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    for part, color in PARTS:
        x = df[f'{part}_x'].to_numpy(float)
        y = df[f'{part}_y'].to_numpy(float)
        z = df[f'{part}_z'].to_numpy(float)
        ax.plot(x, y, z, color=color, label=part, alpha=0.8)
        ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=20, alpha=0.9, label=f'{part} start')
        ax.scatter(x[-1], y[-1], z[-1], color=color, marker='^', s=30, alpha=0.9, label=f'{part} end')

    # axis labels: x (lateral), y (height), z (depth)
    ax.set_xlabel('X (lateral) [units of input]')
    ax.set_ylabel('Y (height) [units of input]')
    ax.set_zlabel('Z (depth) [units of input]')
    ax.set_title('Subject9 wristfix trajectory (global)')

    # attempt roughly equal aspect
    xs = np.concatenate([df[f'{p}_x'].to_numpy(float) for p, _ in PARTS])
    ys = np.concatenate([df[f'{p}_y'].to_numpy(float) for p, _ in PARTS])
    zs = np.concatenate([df[f'{p}_z'].to_numpy(float) for p, _ in PARTS])
    mx, my, mz = np.mean(xs), np.mean(ys), np.mean(zs)
    max_range = 0.5 * max(xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min())
    ax.set_xlim(mx - max_range, mx + max_range)
    ax.set_ylim(my - max_range, my + max_range)
    ax.set_zlim(mz - max_range, mz + max_range)

    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print('[OUT]', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
