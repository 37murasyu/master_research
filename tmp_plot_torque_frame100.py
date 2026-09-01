import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

CSV = 'output_data/torque/kpts3d_subject9_20250925_201442_gcvspl_trimmed_wristfix_torque.csv'
OUT = 'output_data/plots/subject9_torque_frame100.png'
FRAME = 100  # 0-based index

PARTS = [
    ('wrist_R', 'tab:orange'),
    ('elbow_R', 'tab:blue'),
    ('shoulder_R', 'tab:green'),
]


def main():
    df = pd.read_csv(CSV)
    if FRAME >= len(df):
        raise SystemExit(f'frame {FRAME} out of range (len={len(df)})')

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    for part, color in PARTS:
        x = df.loc[FRAME, f'{part}_x']
        y = df.loc[FRAME, f'{part}_y']
        z = df.loc[FRAME, f'{part}_z']
        ax.scatter(x, y, z, color=color, s=40, label=part)
        ax.text(x, y, z, part, fontsize=8, color=color)

    ax.set_xlabel('X (lateral)')
    ax.set_ylabel('Y (height)')
    ax.set_zlabel('Z (depth)')
    ax.set_title(f'Subject9 torque CSV frame {FRAME}')

    # set a modest cube around points
    xs = [df.loc[FRAME, f'{p}_x'] for p, _ in PARTS]
    ys = [df.loc[FRAME, f'{p}_y'] for p, _ in PARTS]
    zs = [df.loc[FRAME, f'{p}_z'] for p, _ in PARTS]
    mx, my, mz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
    span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)) * 0.6 or 1.0
    ax.set_xlim(mx - span, mx + span)
    ax.set_ylim(my - span, my + span)
    ax.set_zlim(mz - span, mz + span)

    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print('[OUT]', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
