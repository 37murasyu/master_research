from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ensure non-interactive backend
import matplotlib.pyplot as plt

from offline_wrist_energy import compute_wrist_energy


def compute_1rm_value(M: float, mmax: float) -> float:
    return (6.225e-3 * M + mmax) * 5.01


def aggregate_cycle_work(frames: np.ndarray, work_inc: np.ndarray, cycle_index_map: dict[int,int], offset: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for i, w in enumerate(work_inc):
        f = int(frames[offset + i])
        cidx = int(cycle_index_map.get(f, -1))
        if cidx <= 0:
            continue
        out[cidx] = out.get(cidx, 0.0) + float(w)
    return out


def main():
    ap = argparse.ArgumentParser(description='Quick per-cycle wrist work computation and 1RM comparison')
    ap.add_argument('--cycles-csv', required=True)
    ap.add_argument('--forearm-npy', required=True)
    ap.add_argument('--tau-npy', required=True)
    ap.add_argument('--forearm-npy-left', required=True)
    ap.add_argument('--tau-npy-left', required=True)
    ap.add_argument('--body-mass', type=float, required=True)
    ap.add_argument('--mmax-all-csv', required=True)
    ap.add_argument('--subject-id', type=int, required=True)
    ap.add_argument('--offset', type=int, default=6)
    ap.add_argument('--clip-dtheta', type=float, default=0.35)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-png', required=True)
    args = ap.parse_args()

    # cycles mapping
    df_cyc = pd.read_csv(args.cycles_csv)
    if 'frame' not in df_cyc.columns or 'cycle_index' not in df_cyc.columns:
        raise SystemExit('cycles CSV must have frame and cycle_index columns')
    frames = df_cyc['frame'].to_numpy(int)
    cycle_index_map = {int(f): int(c) for f, c in zip(df_cyc['frame'].to_numpy(int), df_cyc['cycle_index'].to_numpy(int))}

    # load NPYs
    fa_R = np.load(args.forearm_npy)
    tau_R = np.load(args.tau_npy)
    fa_L = np.load(args.forearm_npy_left)
    tau_L = np.load(args.tau_npy_left)

    # compute per-frame positive work (lag-compensated)
    res_R = compute_wrist_energy(fa_R, tau_R, _dt=1/30.0, offset=args.offset, clip_dtheta=args.clip_dtheta, E_low=0.0, E_high=0.0)
    res_L = compute_wrist_energy(fa_L, tau_L, _dt=1/30.0, offset=args.offset, clip_dtheta=args.clip_dtheta, E_low=0.0, E_high=0.0)

    # aggregate per cycle
    E_by_cycle_R = aggregate_cycle_work(frames, res_R.work_inc, cycle_index_map, args.offset)
    E_by_cycle_L = aggregate_cycle_work(frames, res_L.work_inc, cycle_index_map, args.offset)

    # 1RM lookup
    df_m = pd.read_csv(args.mmax_all_csv)
    row = df_m.loc[df_m['subject_id'] == args.subject_id]
    if row.empty:
        raise SystemExit(f'subject_id={args.subject_id} not found in {args.mmax_all_csv}')
    mL = float(row.iloc[0]['wrist_L'])
    mR = float(row.iloc[0]['wrist_R'])
    one_rm_L = compute_1rm_value(args.body_mass, mL)
    one_rm_R = compute_1rm_value(args.body_mass, mR)

    # export CSV
    rows = []
    for cidx, E in sorted(E_by_cycle_R.items()):
        rows.append({'cycle_index': cidx, 'part': 'wrist_R', 'E_cycle_J': E, 'one_rm': one_rm_R, 'ratio_vs_1rm': E / one_rm_R if one_rm_R > 0 else np.nan})
    for cidx, E in sorted(E_by_cycle_L.items()):
        rows.append({'cycle_index': cidx, 'part': 'wrist_L', 'E_cycle_J': E, 'one_rm': one_rm_L, 'ratio_vs_1rm': E / one_rm_L if one_rm_L > 0 else np.nan})
    out_df = pd.DataFrame(rows)
    out_df.sort_values(['part','cycle_index'], inplace=True)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f'[OUT] cycle work CSV -> {args.out_csv}')

    # percent plot
    all_cycles = sorted(set(E_by_cycle_R.keys()).union(set(E_by_cycle_L.keys())))
    x = np.arange(len(all_cycles))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    yR = [(E_by_cycle_R.get(c, 0.0) / one_rm_R * 100.0) if one_rm_R > 0 else 0.0 for c in all_cycles]
    yL = [(E_by_cycle_L.get(c, 0.0) / one_rm_L * 100.0) if one_rm_L > 0 else 0.0 for c in all_cycles]
    ax.bar(x - width/2, yR, width, label='wrist_R')
    ax.bar(x + width/2, yL, width, label='wrist_L')
    ax.axhline(100.0, color='gray', linestyle='--', label='1RM=100%')
    ax.set_ylabel('Energy per cycle (% of 1RM)')
    ax.set_xlabel('cycle_index')
    ax.set_title('Wrist work per cycle (percent of 1RM)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in all_cycles])
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    print(f'[OUT] plot -> {args.out_png}')


if __name__ == '__main__':
    main()
