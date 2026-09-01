import pandas as pd
import matplotlib.pyplot as plt


def main():
    bpf3 = pd.read_csv('output_data/poses/joint3_kf3hz_states_max_bpf3.csv')
    bpf10 = pd.read_csv('output_data/poses/joint3_kf3hz_states_max_bpf10.csv')

    t0, t1 = 10.0, 15.0
    mask3 = (bpf3['time_s'] >= t0) & (bpf3['time_s'] <= t1)
    mask10 = (bpf10['time_s'] >= t0) & (bpf10['time_s'] <= t1)

    cols = ['joint_3_x', 'joint_3_y', 'joint_3_z']
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        col = cols[i]
        ax.plot(bpf3.loc[mask3, 'time_s'], bpf3.loc[mask3, col + '_kf3hz_up30'], label='BPF 0.1-3.0, KF', color='#d62728')
        ax.plot(bpf10.loc[mask10, 'time_s'], bpf10.loc[mask10, col + '_kf3hz_up30'], label='BPF 0.1-10.0, KF', color='#ff7f0e', alpha=0.9)
        ax.plot(bpf3.loc[mask3, 'time_s'], bpf3.loc[mask3, col + '_raw'], label='raw', color='#1f77b4', alpha=0.4)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig('joint3_bpf3_vs_bpf10_10to15.png', dpi=200)
    print('Saved joint3_bpf3_vs_bpf10_10to15.png')


if __name__ == '__main__':
    main()
