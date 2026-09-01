import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import compute_local_torque


def build_links_from_pose(df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    joint_ids = []
    for c in df.columns:
        if c.startswith('joint_') and c.split('_')[1].isdigit() and c.split('_')[2] in ('x','y','z'):
            joint_ids.append(int(c.split('_')[1]))
    joint_ids = sorted(set(joint_ids))
    cols = []
    use_ids = []
    for jid in joint_ids:
        triplet = []
        for ax in ('x','y','z'):
            cand_f = f'joint_{jid}_{ax}_f'
            cand = f'joint_{jid}_{ax}'
            if cand_f in df.columns:
                triplet.append(cand_f)
            elif cand in df.columns:
                triplet.append(cand)
            else:
                triplet = []
                break
        if triplet:
            use_ids.append(jid)
            cols.extend(triplet)
    if not use_ids:
        raise ValueError('No complete joint triplets found in pose CSV')
    data = df[cols].to_numpy(dtype=float).reshape(-1, len(use_ids), 3)
    return data, use_ids


def build_link_vectors(p3d: np.ndarray, joint_ids: List[int]) -> Dict[str, np.ndarray]:
    id2row = {jid: i for i, jid in enumerate(joint_ids)}
    def g(j):
        return p3d[id2row[j]] if j in id2row else np.array([np.nan, np.nan, np.nan])
    out = {}
    if all(k in id2row for k in (16,14)):
        out['forearm_R'] = g(16) - g(14)
    if all(k in id2row for k in (14,12)):
        out['upper_R'] = g(14) - g(12)
    if all(k in id2row for k in (12,11)):
        out['shoulder_R'] = g(12) - g(11)
    if all(k in id2row for k in (15,13)):
        out['forearm_L'] = g(15) - g(13)
    if all(k in id2row for k in (13,11)):
        out['upper_L'] = g(13) - g(11)
    if all(k in id2row for k in (11,12)):
        out['shoulder_L'] = g(11) - g(12)
    return out


def detect_torque_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    parts = ['wrist_R','elbow_R','shoulder_R','wrist_L','elbow_L','shoulder_L']
    cols = []
    for p in parts:
        for ax in ('x','y','z'):
            col = f'{p}_{ax}'
            if col not in df.columns:
                raise ValueError(f'Missing torque column {col}')
            cols.append(col)
    return parts, cols


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


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    dot = float(np.dot(a, b)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    crossn = np.linalg.norm(np.cross(a/na, b/nb))
    import math as _m
    return _m.atan2(crossn, dot)


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
        start = i0; end = i2
        if end - start + 1 < min_len:
            continue
        seg = t[start:end+1]
        amp = float(np.max(seg) - np.min(seg))
        if amp < min_amp:
            continue
        cycles.append((start, end, amp))
    return cycles


def trapezoid_work(tau: np.ndarray, theta: np.ndarray) -> float:
    if len(tau) < 2 or len(theta) < 2:
        return 0.0
    tau = np.asarray(tau); theta = np.asarray(theta)
    dtheta = np.diff(theta)
    # tau length = N, dtheta length = N-1 -> use average tau over segment
    tau_mid = 0.5 * (tau[1:] + tau[:-1])
    return float(np.sum(tau_mid * dtheta))


def resample_time_series(t: np.ndarray, arr: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    out = np.zeros((len(t_new), arr.shape[1]), dtype=float) if arr.ndim == 2 else np.zeros(len(t_new), dtype=float)
    for i in range(arr.shape[1] if arr.ndim == 2 else 1):
        if arr.ndim == 2:
            out[:, i] = np.interp(t_new, t, arr[:, i])
        else:
            out = np.interp(t_new, t, arr)
            break
    return out


def main():
    ap = argparse.ArgumentParser(description='Recompute local torques with custom elbow frame and cycle/work at low rate')
    ap.add_argument('--pose-csv', required=True)
    ap.add_argument('--torque-csv', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--target-hz', type=float, default=3.0)
    ap.add_argument('--clip-dtheta', type=float, default=0.35)
    ap.add_argument('--cycle-min-amp', type=float, default=0.10)
    ap.add_argument('--cycle-min-length', type=int, default=6)
    ap.add_argument('--out-csv', default='output_data/torque_local_custom.csv')
    ap.add_argument('--out-csv-ds', default='output_data/torque_local_custom_3hz.csv')
    ap.add_argument('--out-cycles', default='output_data/torque_local_custom_cycles.json')
    args = ap.parse_args()

    pose_df = pd.read_csv(args.pose_csv)
    torque_df = pd.read_csv(args.torque_csv)

    pose_arr, joint_ids = build_links_from_pose(pose_df)
    parts, torque_cols = detect_torque_cols(torque_df)
    tau_g = torque_df[torque_cols].to_numpy(dtype=float).reshape(-1, len(parts), 3)

    T = min(len(pose_arr), len(tau_g))
    pose_arr = pose_arr[:T]
    tau_g = tau_g[:T]
    frames = np.arange(T)
    t = frames / args.fps

    # link vectors per frame
    local_out = np.zeros_like(tau_g)
    for i in range(T):
        links = build_link_vectors(pose_arr[i], joint_ids)
        for p_idx, part in enumerate(parts):
            if part == 'wrist_R':
                link = links.get('forearm_R', np.array([np.nan,np.nan,np.nan]))
                parent = None
            elif part == 'elbow_R':
                link = links.get('upper_R', np.array([np.nan,np.nan,np.nan]))
                parent = links.get('forearm_R')
            elif part == 'shoulder_R':
                link = links.get('shoulder_R', np.array([np.nan,np.nan,np.nan]))
                parent = links.get('upper_R')
            elif part == 'wrist_L':
                link = links.get('forearm_L', np.array([np.nan,np.nan,np.nan]))
                parent = None
            elif part == 'elbow_L':
                link = links.get('upper_L', np.array([np.nan,np.nan,np.nan]))
                parent = links.get('forearm_L')
            else:  # shoulder_L
                link = links.get('shoulder_L', np.array([np.nan,np.nan,np.nan]))
                parent = links.get('upper_L')
            local_out[i, p_idx, :] = compute_local_torque(tau_g[i, p_idx, :], link, parent)

    # Save full-rate local
    out_df = pd.DataFrame({'time_s': t})
    for p_idx, part in enumerate(parts):
        for ax_i, ax in enumerate(('x','y','z')):
            out_df[f'{part}_local_custom_{ax}'] = local_out[:, p_idx, ax_i]
    out_df.to_csv(args.out_csv, index=False)

    # Downsample to target-hz (interp on time)
    step = 1.0 / args.target_hz
    t_ds = np.arange(t[0], t[-1] + 1e-9, step)
    tau_y_ds = resample_time_series(t, local_out[:, parts.index('wrist_R'), 1], t_ds)
    forearm_R = np.array([build_link_vectors(p, joint_ids).get('forearm_R', np.array([np.nan,np.nan,np.nan])) for p in pose_arr])
    forearm_R_ds = resample_time_series(t, forearm_R, t_ds)
    theta_ds = reconstruct_theta_angle_diff(forearm_R_ds, args.clip_dtheta)

    # cycle detection on theta
    cycles = detect_cycles_peak_valley(theta_ds, args.cycle_min_amp, args.cycle_min_length)

    # work per cycle using tau_y (local) and theta
    import json
    cycle_out = []
    for (s, e, amp) in cycles:
        work = trapezoid_work(tau_y_ds[s:e+1], theta_ds[s:e+1])
        cycle_out.append({'start_idx': int(s), 'end_idx': int(e), 'amp': float(amp), 'work': float(work)})
    with open(args.out_cycles, 'w', encoding='utf-8') as f:
        json.dump({'target_hz': args.target_hz, 'cycles': cycle_out}, f, ensure_ascii=False, indent=2)

    # save downsampled series
    out_ds = pd.DataFrame({'time_s': t_ds, 'theta': theta_ds, 'tau_y': tau_y_ds})
    out_ds.to_csv(args.out_csv_ds, index=False)

    print(f'Saved local torque (custom axes) -> {args.out_csv}')
    print(f'Saved downsampled series -> {args.out_csv_ds}')
    print(f'Saved cycles -> {args.out_cycles} (count={len(cycle_out)})')


if __name__ == '__main__':
    main()
