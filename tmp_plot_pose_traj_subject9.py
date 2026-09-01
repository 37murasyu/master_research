import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

CSV = 'output_data/poses/kpts3d_subject9_20250925_201442_gcvspl.csv'
OUT = 'output_data/plots/subject9_pose_traj_xyz.png'
FRAME_START = 0
FRAME_END = None  # None = full length


def detect_joint_ids(df):
    return sorted({int(c.split('_')[1]) for c in df.columns if c.startswith('joint_') and c.endswith('_x')})


def load_series(df, jid):
    return (
        df[f'joint_{jid}_x'].to_numpy(float),
        df[f'joint_{jid}_y'].to_numpy(float),
        df[f'joint_{jid}_z'].to_numpy(float),
    )


def main():
    df = pd.read_csv(CSV)
    if FRAME_END is not None:
        df = df.iloc[FRAME_START:FRAME_END]
    else:
        df = df.iloc[FRAME_START:]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    joint_ids = detect_joint_ids(df)
    cmap = plt.cm.get_cmap('tab20', len(joint_ids))

    for idx, jid in enumerate(joint_ids):
        name = f'j{jid}'
        color = cmap(idx)
        x, y, z = load_series(df, jid)
        ax.plot(x, y, z, color=color, label=name, alpha=0.8)
        ax.scatter(x[0], y[0], z[0], color=color, s=15, marker='o', alpha=0.9)
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=25, marker='^', alpha=0.9)

    # axis labels: x (lateral), y (height), z (depth)
    ax.set_xlabel('X (lateral)')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Z (depth)')
    ax.set_title('Subject9 pose trajectory (global)')

    # equal-ish aspect
    xs = np.concatenate([load_series(df, jid)[0] for jid in joint_ids])
    ys = np.concatenate([load_series(df, jid)[1] for jid in joint_ids])
    zs = np.concatenate([load_series(df, jid)[2] for jid in joint_ids])
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
