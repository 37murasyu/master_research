import argparse
import pandas as pd
import matplotlib.pyplot as plt


def pick_cols(df, pref):
    # prefer explicit suffix
    cand = [f'joint_3_{axis}{pref}' for axis in ['x', 'y', 'z']]
    if all(c in df.columns for c in cand):
        return cand
    # fallback: plain names
    cand = [f'joint_3_{axis}' for axis in ['x', 'y', 'z']]
    if all(c in df.columns for c in cand):
        return cand
    raise KeyError('joint_3 columns not found')


def main():
    ap = argparse.ArgumentParser(description='Plot joint_3 3Hz series: gcvspl vs Kalman (CA)')
    ap.add_argument('--gcvspl', default='output_data/poses/kpts3d_subject8_20250925_192700_gcvspl_joint3_3hz.csv')
    ap.add_argument('--kalman', default='output_data/poses/kpts3d_subject8_20250925_192700_joint3_ca_3hz.csv')
    ap.add_argument('--out', default='joint3_gcvspl_vs_kalman_3hz.png')
    args = ap.parse_args()

    df_gc = pd.read_csv(args.gcvspl)
    df_kf = pd.read_csv(args.kalman)

    t_gc = df_gc['time_s'] if 'time_s' in df_gc.columns else df_gc.index / 3.0
    t_kf = df_kf['time_s'] if 'time_s' in df_kf.columns else df_kf.index / 3.0

    gc_cols = pick_cols(df_gc, '')
    kf_cols = pick_cols(df_kf, '_kalman')

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        ax.plot(t_gc, df_gc[gc_cols[i]], label='gcvspl 3Hz', color='#2ca02c', alpha=0.9)
        ax.plot(t_kf, df_kf[kf_cols[i]], label='kalman CA 3Hz', color='#d62728', alpha=0.9)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
