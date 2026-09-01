import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

CSV = 'output_data/poses/kpts3d_subject9_20250925_201442_gcvspl.csv'
FRAME = 100


def main():
    df = pd.read_csv(CSV)
    if FRAME >= len(df):
        raise SystemExit(f'frame {FRAME} out of range (len={len(df)})')
    joint_ids = sorted({int(c.split('_')[1]) for c in df.columns if c.startswith('joint_') and c.endswith('_x')})
    coords = []
    for jid in joint_ids:
        x = df.loc[FRAME, f'joint_{jid}_x']
        y = df.loc[FRAME, f'joint_{jid}_y']
        z = df.loc[FRAME, f'joint_{jid}_z']
        coords.append((jid, x, y, z))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for jid, x, y, z in coords:
        ax.scatter(x, y, z, label=f'j{jid}')
        ax.text(x, y, z, f'{jid}', fontsize=8)

    xs = [c[1] for c in coords]; ys = [c[2] for c in coords]; zs = [c[3] for c in coords]
    mx = (min(xs) + max(xs)) / 2
    my = (min(ys) + max(ys)) / 2
    mz = (min(zs) + max(zs)) / 2
    max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) * 0.6
    ax.set_xlim(mx - max_range, mx + max_range)
    ax.set_ylim(my - max_range, my + max_range)
    ax.set_zlim(mz - max_range, mz + max_range)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Frame {FRAME} joints')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()

    out = 'output_data/plots/frame100_joints_subject9.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print('[OUT]', os.path.abspath(out))


if __name__ == '__main__':
    main()
