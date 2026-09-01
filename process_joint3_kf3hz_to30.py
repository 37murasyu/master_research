import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from realtime_kalman_filter import KalmanCA1D, Kalman1D


def biquad_lowpass(fc: float, fs: float, q: float = math.sqrt(0.5)):
    k = math.tan(math.pi * fc / fs)
    norm = 1.0 / (1 + k / q + k * k)
    b0 = k * k * norm
    b1 = 2 * b0
    b2 = b0
    a1 = 2 * (k * k - 1) * norm
    a2 = (1 - k / q + k * k) * norm
    return np.array([b0, b1, b2], float), np.array([1.0, a1, a2], float)


def biquad_highpass(fc: float, fs: float, q: float = math.sqrt(0.5)):
    k = math.tan(math.pi * fc / fs)
    norm = 1.0 / (1 + k / q + k * k)
    b0 = 1 * norm
    b1 = -2 * b0
    b2 = b0
    a1 = 2 * (k * k - 1) * norm
    a2 = (1 - k / q + k * k) * norm
    return np.array([b0, b1, b2], float), np.array([1.0, a1, a2], float)


def biquad_filter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    # Direct Form II Transposed
    y = np.zeros_like(x, dtype=float)
    z1 = 0.0
    z2 = 0.0
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    for i in range(x.size):
        w = x[i] - a1 * z1 - a2 * z2
        y[i] = b0 * w + b1 * z1 + b2 * z2
        z2 = z1
        z1 = w
    return y


def apply_bpf(data: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    hp_b, hp_a = biquad_highpass(f_lo, fs)
    lp_b, lp_a = biquad_lowpass(f_hi, fs)
    out = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        x = data[:, j]
        x = biquad_filter(hp_b, hp_a, x)
        x = biquad_filter(lp_b, lp_a, x)
        out[:, j] = x
    return out

def build_time_from_frame(df, fps):
    if 'frame' in df.columns:
        return df['frame'].to_numpy(float) / fps
    return np.arange(len(df), dtype=float) / fps

def run_ca_kf(t, data, q_acc=1e-3, r=1e-3, gate_std=3.0):
    filters = [KalmanCA1D(q_acc=q_acc, r=r, gate_std=gate_std) for _ in range(data.shape[1])]
    out = np.zeros_like(data, dtype=float)
    vel = np.zeros_like(data, dtype=float)
    acc = np.zeros_like(data, dtype=float)
    last_t = t[0]
    for i, ti in enumerate(t):
        dt = max(1e-6, ti - last_t) if i > 0 else t[1] - t[0] if len(t) > 1 else 1/3
        last_t = ti
        for j in range(data.shape[1]):
            out[i, j] = filters[j].step(float(data[i, j]), dt)
            vel[i, j] = filters[j].v
            acc[i, j] = filters[j].a
    return out, vel, acc


def run_cv_kf(t, data, q_pos=1e-4, q_vel=1e-4, r=1e-3, gate_std=3.0):
    d = data.shape[1]
    filters = [Kalman1D(q_pos=q_pos, q_vel=q_vel, r=r, gate_std=gate_std) for _ in range(d)]
    out = np.zeros_like(data, dtype=float)
    vel = np.zeros_like(data, dtype=float)
    last_t = t[0]
    for i, ti in enumerate(t):
        dt = max(1e-6, ti - last_t) if i > 0 else (t[1] - t[0] if len(t) > 1 else 1/3)
        last_t = ti
        for j in range(d):
            out[i, j] = filters[j].step(float(data[i, j]), dt)
            vel[i, j] = filters[j].v
    return out, vel


def run_kf_predict_full(t_meas, meas, t_full, model='ca', q_acc=1e-3, q_pos=1e-4, q_vel=1e-4, r=1e-3, gate_std=3.0):
    """Simulate KF at full timeline: predict every t_full step, update when measurement arrives."""
    d = meas.shape[1]
    if model == 'ca':
        filters = [KalmanCA1D(q_acc=q_acc, r=r, gate_std=gate_std) for _ in range(d)]
    else:
        filters = [Kalman1D(q_pos=q_pos, q_vel=q_vel, r=r, gate_std=gate_std) for _ in range(d)]
    out = np.zeros((t_full.size, d), dtype=float)
    vel = np.zeros((t_full.size, d), dtype=float)
    acc = np.zeros((t_full.size, d), dtype=float)
    # seed with first measurement
    for j in range(d):
        out[0, j] = filters[j].step(float(meas[0, j]), dt=1.0 / 3.0)
        vel[0, j] = getattr(filters[j], 'v', 0.0)
        acc[0, j] = getattr(filters[j], 'a', 0.0)
    last_t = t_full[0]
    idx = 1  # next measurement index
    eps = 1e-9
    for k in range(1, t_full.size):
        t = t_full[k]
        dt = max(1e-6, t - last_t)
        last_t = t
        for j in range(d):
            filters[j].predict(dt)
        if idx < len(t_meas) and t + eps >= t_meas[idx]:
            for j in range(d):
                filters[j].update(float(meas[idx, j]))
            idx += 1
        for j in range(d):
            out[k, j] = filters[j].x
            vel[k, j] = getattr(filters[j], 'v', 0.0)
            acc[k, j] = getattr(filters[j], 'a', 0.0)
    return out, vel, acc

def main():
    ap = argparse.ArgumentParser(description='Downsample raw to 3Hz, Kalman CA, upsample to 30Hz and compare to gcvspl')
    ap.add_argument('--raw', default='output_data/poses/kpts3d_subject8_20250925_192700.csv')
    ap.add_argument('--gcvspl', default='output_data/poses/kpts3d_subject8_20250925_192700_gcvspl.csv')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--target-hz', type=float, default=3.0)
    ap.add_argument('--model', choices=['cv', 'ca'], default='ca', help='KF model for 3Hz filtering and model mode')
    ap.add_argument('--q-acc', type=float, default=1e-3, help='process noise (ca)')
    ap.add_argument('--q-pos', type=float, default=1e-4, help='process noise pos (cv)')
    ap.add_argument('--q-vel', type=float, default=1e-4, help='process noise vel (cv)')
    ap.add_argument('--r', type=float, default=1e-3)
    ap.add_argument('--gate-std', type=float, default=3.0)
    ap.add_argument('--mode', choices=['interp', 'model'], default='interp', help='interp: KF at 3Hz then linear upsample; model: run KF predicting each 30Hz step and updating at 3Hz obs')
    ap.add_argument('--bpf-low', type=float, help='Highpass cutoff Hz applied before downsample/KF')
    ap.add_argument('--bpf-high', type=float, help='Lowpass cutoff Hz applied before downsample/KF')
    ap.add_argument('--bpf-only', action='store_true', help='Skip Kalman; only apply BPF at native fps')
    ap.add_argument('--bpf-only-downsample', action='store_true', help='Apply BPF, downsample to target-hz, then interpolate back to native fps (no Kalman)')
    ap.add_argument('--export-states', action='store_true', help='When using KF, also export velocity/acceleration estimates')
    ap.add_argument('--plot-states', action='store_true', help='Plot velocity/acceleration when exported')
    ap.add_argument('--states-png', help='Optional path to save velocity/acceleration plot (defaults to out-png stem + _states.png)')
    ap.add_argument('--out-csv', default='output_data/poses/joint3_kf3hz_to30.csv')
    ap.add_argument('--out-png', default='joint3_kf3hz_vs_gcvspl_30hz.png')
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    gc = pd.read_csv(args.gcvspl)

    cols = ['joint_3_x', 'joint_3_y', 'joint_3_z']
    t_raw = build_time_from_frame(raw, args.fps)
    t_gc = build_time_from_frame(gc, args.fps)

    data_raw = raw[cols].to_numpy(float)
    if args.bpf_low or args.bpf_high:
        if not (args.bpf_low and args.bpf_high):
            raise ValueError('Specify both --bpf-low and --bpf-high to enable BPF')
        data_raw = apply_bpf(data_raw, fs=args.fps, f_lo=args.bpf_low, f_hi=args.bpf_high)

    if args.bpf_only and args.bpf_only_downsample:
        raise ValueError('Choose either --bpf-only or --bpf-only-downsample, not both')
    if args.export_states and (args.bpf_only or args.bpf_only_downsample):
        raise ValueError('--export-states requires Kalman filtering (disable bpf-only modes)')

    if args.bpf_only:
        kf_full = data_raw
        series_label = 'BPF only 30Hz'
        col_suffix = '_bpf_only'
        vel_full = None
        acc_full = None
    else:
        # downsample to target-hz
        step = 1.0 / args.target_hz
        t_ds = np.arange(t_raw[0], t_raw[-1] + 1e-9, step)
        ds = np.vstack([np.interp(t_ds, t_raw, data_raw[:, i]) for i in range(3)]).T

        if args.bpf_only_downsample:
            kf_full = np.vstack([np.interp(t_raw, t_ds, ds[:, i]) for i in range(3)]).T
            series_label = 'BPF 3Hz -> interp 30Hz'
            col_suffix = '_bpf3hz_interp30'
            vel_full = None
            acc_full = None
        else:
            if args.mode == 'interp':
                if args.model == 'ca':
                    kf_ds, vel_ds, acc_ds = run_ca_kf(t_ds, ds, q_acc=args.q_acc, r=args.r, gate_std=args.gate_std)
                    vel_full = np.vstack([np.interp(t_raw, t_ds, vel_ds[:, i]) for i in range(3)]).T
                    acc_full = np.vstack([np.interp(t_raw, t_ds, acc_ds[:, i]) for i in range(3)]).T
                else:
                    kf_ds, vel_ds = run_cv_kf(t_ds, ds, q_pos=args.q_pos, q_vel=args.q_vel, r=args.r, gate_std=args.gate_std)
                    vel_full = np.vstack([np.interp(t_raw, t_ds, vel_ds[:, i]) for i in range(3)]).T
                    acc_full = None
                kf_full = np.vstack([np.interp(t_raw, t_ds, kf_ds[:, i]) for i in range(3)]).T
            else:
                # predict every 30Hz step, update only at 3Hz measurements
                kf_full, vel_full, acc_full = run_kf_predict_full(
                    t_ds, ds, t_raw,
                    model=args.model,
                    q_acc=args.q_acc,
                    q_pos=args.q_pos,
                    q_vel=args.q_vel,
                    r=args.r,
                    gate_std=args.gate_std,
                )
            series_label = 'Kalman from 3Hz -> 30Hz'
            col_suffix = '_kf3hz_up30'

    out_df = pd.DataFrame({'time_s': t_raw})
    for i, c in enumerate(cols):
        out_df[c + col_suffix] = kf_full[:, i]
        out_df[c + '_raw'] = raw[c].to_numpy(float)
        if args.export_states:
            if vel_full is None or (args.model != 'ca' and args.model != 'cv'):
                pass
            else:
                out_df[c + col_suffix + '_vel'] = vel_full[:, i]
                if acc_full is not None:
                    out_df[c + col_suffix + '_acc'] = acc_full[:, i]
    out_df.to_csv(args.out_csv, index=False)

    # Plot vs gcvspl
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        ax.plot(t_gc, gc[cols[i]], label='gcvspl 30Hz', color='#2ca02c', alpha=0.8)
        ax.plot(t_raw, kf_full[:, i], label=series_label, color='#d62728', alpha=0.8)
        ax.plot(t_raw, data_raw[:, i], label='raw 30Hz (after optional BPF)', color='#1f77b4', alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200)
    print(f'Saved {args.out_csv} and {args.out_png}')

    if args.plot_states and args.export_states and vel_full is not None:
        has_acc = acc_full is not None
        rows = 3 + (3 if has_acc else 0)
        fig2, axes2 = plt.subplots(rows, 1, figsize=(10, 10 if has_acc else 7), sharex=True)
        label_components = ['x', 'y', 'z']
        for i in range(3):
            ax = axes2[i]
            ax.plot(t_raw, vel_full[:, i], color='#9467bd', label='velocity (KF)')
            ax.set_ylabel(f'v_{label_components[i]}')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        if has_acc:
            for i in range(3):
                ax = axes2[3 + i]
                ax.plot(t_raw, acc_full[:, i], color='#8c564b', label='acceleration (KF)')
                ax.set_ylabel(f'a_{label_components[i]}')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
        axes2[-1].set_xlabel('time (s)')
        fig2.tight_layout()
        states_png = args.states_png if args.states_png else (args.out_png[:-4] + '_states.png' if args.out_png.lower().endswith('.png') else args.out_png + '_states.png')
        fig2.savefig(states_png, dpi=200)
        print(f'Saved {states_png}')


if __name__ == '__main__':
    main()
