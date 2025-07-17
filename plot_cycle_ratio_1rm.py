"""Plot ratio_1RM_percent per cycle from a per-cycle CSV.

Usage:
  python plot_cycle_ratio_1rm.py \
    --csv output_data/wrist_cycles_bilateral_9_20250925_201442_with1RM_s9_clean.csv \
    --out output_data/plots/ratio_1rm_percent_9_20250925_201442.png

The script expects columns including: cycle_index, part, ratio_1RM_percent.
It groups by part (wrist_R, wrist_L) and plots cycle_index vs ratio_1RM_percent.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='per-cycle CSV path')
    ap.add_argument('--out', required=False, help='output PNG path')
    ap.add_argument('--ymax', type=float, default=None, help='optional y-axis upper limit (percent)')
    ap.add_argument('--ymin', type=float, default=None, help='optional y-axis lower limit; must be > 0 for log scale')
    ap.add_argument('--yscale', type=str, default='linear', choices=['linear','log'], help='y-axis scale')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Normalize and sanitize ratio column
    if 'ratio_1RM_percent' not in df.columns:
        raise SystemExit('ratio_1RM_percent column not found in CSV')
    df['ratio_1RM_percent'] = pd.to_numeric(df['ratio_1RM_percent'], errors='coerce')
    df = df[np.isfinite(df['ratio_1RM_percent'])]
    if args.yscale == 'log':
        # Drop non-positive values for log scale
        df = df[df['ratio_1RM_percent'] > 0]

    # Some CSVs may interleave parts with cycle_index restarting per part; that's OK.
    parts = sorted(df['part'].dropna().unique())
    colors = {
        'wrist_R': 'tab:red',
        'wrist_L': 'tab:blue',
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    for p in parts:
        sdf = df[df['part'] == p].copy()
        if sdf.empty:
            continue
        # Ensure cycle_index is integer-like
        sdf['cycle_index'] = pd.to_numeric(sdf['cycle_index'], errors='coerce').astype('Int64')
        sdf = sdf.dropna(subset=['cycle_index'])
        # Sort by cycle_index for nicer lines
        sdf = sdf.sort_values('cycle_index')
        yvals = sdf['ratio_1RM_percent'].to_numpy()
        if args.yscale == 'log':
            # keep only positive values
            keep = yvals > 0
            sdf = sdf.iloc[np.where(keep)[0]]
            yvals = sdf['ratio_1RM_percent'].to_numpy()
        ax.plot(
            sdf['cycle_index'].astype(int).to_numpy(),
            yvals,
            marker='o', linestyle='-', label=p, color=colors.get(p, None), alpha=0.9
        )

    ax.set_xlabel('cycle_index (per part)')
    ax.set_ylabel('ratio_1RM_percent (%)')
    ttl = os.path.basename(args.csv)
    ax.set_title(f'Per-cycle ratio_1RM_percent\n{ttl}')
    ax.grid(True, ls=':')
    ax.legend(loc='upper right')

    # Axis scale and limits
    ax.set_yscale(args.yscale)
    if args.yscale == 'log':
        # For log scale, lower bound must be > 0
        vals = df['ratio_1RM_percent'].to_numpy()
        if vals.size:
            ymin = args.ymin if (args.ymin is not None and args.ymin > 0) else max(1e-1, float(np.nanpercentile(vals, 1)) * 0.9)
            if args.ymax is not None and args.ymax > 0:
                ax.set_ylim(ymin, args.ymax)
            else:
                p99 = float(np.nanpercentile(vals, 99))
                ymax = max(p99 * 1.1, np.nanmax(vals) * 1.02)
                ax.set_ylim(ymin, ymax)
    else:
        if args.ymax is not None and args.ymax > 0:
            low = args.ymin if (args.ymin is not None) else 0.0
            ax.set_ylim(low, args.ymax)
        else:
            vals = df['ratio_1RM_percent'].to_numpy()
            if vals.size:
                p99 = float(np.nanpercentile(vals, 99))
                ymax = max(p99 * 1.1, np.nanmax(vals) * 1.02)
                ymin = args.ymin if (args.ymin is not None) else min(0.0, np.nanmin(vals) * 0.98)
                ax.set_ylim(ymin, ymax)

    out_path = args.out
    if not out_path:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = os.path.join('output_data', 'plots', f'ratio_1rm_percent_{base}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'[OUT] plot -> {out_path}')


if __name__ == '__main__':
    main()
