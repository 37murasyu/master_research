import pandas as pd
import matplotlib.pyplot as plt


def load(path):
    return pd.read_csv(path)


def main():
    mid_path = 'output_data/poses/joint3_kf3hz_to30_interp_cv_bpf_mid.csv'
    loose_path = 'output_data/poses/joint3_kf3hz_to30_interp_cv_bpf_loose.csv'
    gcvspl_path = 'output_data/poses/kpts3d_subject8_20250925_192700_gcvspl.csv'
    raw_path = 'output_data/poses/kpts3d_subject8_20250925_192700.csv'

    df_mid = load(mid_path)
    df_loose = load(loose_path)
    df_gc = load(gcvspl_path)
    df_raw = load(raw_path)

    t_mid = df_mid['time_s']
    t_loose = df_loose['time_s']
    t_gc = df_gc['frame'] / 30.0 if 'frame' in df_gc.columns else df_gc.index / 30.0
    t_raw = df_raw['frame'] / 30.0 if 'frame' in df_raw.columns else df_raw.index / 30.0

    cols = ['joint_3_x', 'joint_3_y', 'joint_3_z']
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x', 'y', 'z']

    for i, ax in enumerate(axes):
        ax.plot(t_gc, df_gc[cols[i]], label='gcvspl 30Hz', color='#2ca02c', alpha=0.8, lw=1.0)
        ax.plot(t_raw, df_raw[cols[i]], label='raw 30Hz', color='#555555', alpha=0.8, lw=1.0, zorder=1.2)
        ax.plot(t_mid, df_mid[cols[i] + '_kf3hz_up30'], label='KF mid (q=5e-6,r=7e-4, BPF 0.1-2.2)', color='#1f77b4', alpha=0.9, lw=1.1)
        ax.plot(t_loose, df_loose[cols[i] + '_kf3hz_up30'], label='KF loose (q=1e-5,r=1e-3, BPF 0.1-2.5)', color='#d62728', alpha=0.9, lw=1.1)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig('joint3_compare_mid_loose.png', dpi=200)
    print('Saved joint3_compare_mid_loose.png')

    # raw-only plot (separate)
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axes2):
        ax.plot(t_raw, df_raw[cols[i]], label='raw 30Hz', color='#555555', alpha=0.9, lw=1.0)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes2[-1].set_xlabel('time (s)')
    fig2.tight_layout()
    fig2.savefig('joint3_raw_only.png', dpi=200)
    print('Saved joint3_raw_only.png')


if __name__ == '__main__':
    main()
