"""Plot wrist_L tau_y and dtheta time series with detected cycle timings.

Usage:
  python plot_wrist_series_with_cycles.py \
    --forearm-npy 0925_115504_forearm_L_0925_115504.npy \
    --tau-npy 0925_115504_tau_wrist_L_0925_115504.npy \
    --out output_data/wrist_L_series_0925_115504.png \
    --offset 6 --clip-dtheta 0.35 --cycle-min-amp 0.10 --cycle-min-length 10 \
    --cycle-detect-mode auto

This script reuses core logic from offline_wrist_energy.py for reconstructing
angles, computing dtheta, and detecting cycles. It then plots tau_y (N·m) and
dtheta (rad) with vertical spans for detected cycles.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    dot = float(np.dot(a, b)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    crossn = np.linalg.norm(np.cross(a/na, b/nb))
    import math as _m
    return _m.atan2(crossn, dot)


def reconstruct_theta_angle_ref(vecs: np.ndarray) -> np.ndarray:
    if len(vecs) == 0:
        return np.array([])
    u = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    ref = u[0]
    return np.array([angle_between(ref, ui) for ui in u], dtype=float)


def reconstruct_theta_angle_diff(vecs: np.ndarray, clip_dth: float) -> np.ndarray:
    if len(vecs) == 0:
        return np.array([])
    u = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    th = np.zeros(len(u))
    for i in range(1, len(u)):
        dth = angle_between(u[i-1], u[i])
        dth = max(-clip_dth, min(clip_dth, dth))
        th[i] = th[i-1] + dth
    return th


def detect_cycles_peak_valley(signal: np.ndarray, min_amp: float, min_len: int):
    if len(signal) < min_len * 2:
        return []
    t = np.asarray(signal)
    dt = np.diff(t)
    sign = np.sign(dt)
    extrema = []
    for i in range(1, len(sign)):
        if sign[i-1] > 0 and sign[i] <= 0:
            extrema.append(('peak', i))
        elif sign[i-1] < 0 and sign[i] >= 0:
            extrema.append(('valley', i))
    cycles = []
    for j in range(1, len(extrema)-1):
        t0, i0 = extrema[j-1]
        t1, i1 = extrema[j]
        t2, i2 = extrema[j+1]
        if t0 == t2 or t0 == t1 or t1 == t2:
            continue
        start = i0
        end = i2
        if end - start + 1 < min_len:
            continue
        seg = t[start:end+1]
        amp = float(np.max(seg) - np.min(seg))
        if amp < min_amp:
            continue
        cycles.append((start, end, amp))
    return cycles


def detect_cycles_tau(tau: np.ndarray, min_amp: float, min_len: int):
    if len(tau) < min_len * 2:
        return []
    t = np.asarray(tau)
    if len(t) >= 5:
        kernel = np.ones(5)/5.0
        t_s = np.convolve(t, kernel, mode='same')
    else:
        t_s = t
    sign = np.sign(t_s)
    zero_cross = []
    for i in range(1, len(sign)):
        if sign[i] != sign[i-1]:
            zero_cross.append(i)
    cycles = []
    for a, b in zip(zero_cross, zero_cross[1:]):
        if b - a + 1 < min_len:
            continue
        seg = t_s[a:b+1]
        amp = float(np.max(seg) - np.min(seg))
        if amp < min_amp * 0.5:
            continue
        cycles.append((a, b, amp))
    return cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forearm-npy', required=True)
    ap.add_argument('--tau-npy', required=True)
    ap.add_argument('--offset', type=int, default=6)
    ap.add_argument('--clip-dtheta', type=float, default=0.35)
    ap.add_argument('--cycle-min-amp', type=float, default=0.10)
    ap.add_argument('--cycle-min-length', type=int, default=10)
    ap.add_argument('--cycle-detect-mode', type=str, default='auto', choices=['auto','angle_ref','angle_diff','tau'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--tau-clamp-abs', type=float, default=1000.0, help='Clamp tau to +/- this value for plotting and detection.')
    args = ap.parse_args()

    vecs = np.load(args.forearm_npy)
    tau = np.load(args.tau_npy)
    if args.tau_clamp_abs and args.tau_clamp_abs > 0:
        tau = np.clip(tau, -args.tau_clamp_abs, args.tau_clamp_abs)
    N = min(len(vecs), len(tau))
    vecs = vecs[:N]; tau = tau[:N]

    # build angle series
    theta_ref = reconstruct_theta_angle_ref(vecs)
    theta_diff = reconstruct_theta_angle_diff(vecs, args.clip_dtheta)

    # dtheta series (per frame) for plotting
    u = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    dtheta = np.zeros(N)
    for i in range(1, N):
        dtheta[i] = angle_between(u[i-1], u[i])
    dtheta = np.clip(dtheta, -args.clip_dtheta, args.clip_dtheta)

    # cycle detection
    modes = ['angle_ref','angle_diff','tau'] if args.cycle_detect_mode == 'auto' else [args.cycle_detect_mode]
    cycles = []
    used_mode = None
    for m in modes:
        if m == 'angle_ref':
            cycles = detect_cycles_peak_valley(theta_ref, args.cycle_min_amp, args.cycle_min_length)
        elif m == 'angle_diff':
            cycles = detect_cycles_peak_valley(theta_diff, args.cycle_min_amp, args.cycle_min_length)
        else:
            cycles = detect_cycles_tau(tau, args.cycle_min_amp, args.cycle_min_length)
        if cycles:
            used_mode = m
            break

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(tau, label='tau_y (N·m)', color='tab:red')
    ax[1].plot(dtheta, label='dtheta (rad)', color='tab:blue')

    for (s, e, _amp) in cycles:
        ax[0].axvspan(s, e, color='tab:gray', alpha=0.2)
        ax[1].axvspan(s, e, color='tab:gray', alpha=0.2)

    ax[0].set_ylabel('tau_y (N·m)')
    ax[1].set_ylabel('dtheta (rad)')
    ax[1].set_xlabel('frame')
    ttl = f'wrist_L: cycles={len(cycles)} mode={used_mode}' if used_mode else 'wrist_L: cycles=0'
    fig.suptitle(ttl)
    for a in ax:
        a.legend(loc='upper right')
        a.grid(True, ls=':')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(args.out, dpi=150)
    print(f'[OUT] plot -> {args.out}')


if __name__ == '__main__':
    main()
