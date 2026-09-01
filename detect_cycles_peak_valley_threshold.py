import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_peaks_and_valleys(x: np.ndarray, peak_thr: float, valley_thr: float) -> Tuple[List[int], List[int]]:
    peaks, valleys = [], []
    n = len(x)
    if n < 3:
        return peaks, valleys
    for i in range(1, n - 1):
        if x[i] >= peak_thr and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            peaks.append(i)
        if x[i] <= valley_thr and x[i] <= x[i - 1] and x[i] <= x[i + 1]:
            valleys.append(i)
    return peaks, valleys


def build_cycles(peaks: List[int], valleys: List[int]) -> List[Tuple[int, int, int]]:
    cycles: List[Tuple[int, int, int]] = []
    vp = 0
    for pi in range(len(peaks) - 1):
        p1 = peaks[pi]
        p2 = peaks[pi + 1]
        # find valley between p1 and p2
        while vp < len(valleys) and valleys[vp] <= p1:
            vp += 1
        if vp >= len(valleys):
            break
        v = valleys[vp]
        if v >= p2:
            continue
        cycles.append((p1, v, p2))
    return cycles


def plot_series(x: np.ndarray, cycles: List[Tuple[int, int, int]], out_path: Path, title: str):
    plt.figure(figsize=(12, 4))
    plt.plot(x, label='signal', color='#1f77b4')
    for (p1, v, p2) in cycles:
        plt.axvspan(p1, p2, color='gray', alpha=0.2)
        plt.plot(p1, x[p1], 'ro', markersize=4)
        plt.plot(v, x[v], 'ko', markersize=4)
        plt.plot(p2, x[p2], 'ro', markersize=4)
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main():
    ap = argparse.ArgumentParser(description='Detect cycles by peak>thr and valley<thr (peak-valley-peak).')
    ap.add_argument('--csv', required=True, help='Input CSV path')
    ap.add_argument('--column', default='joint_0_y_f', help='Column to analyze')
    ap.add_argument('--peak-th', type=float, default=0.025, help='Minimum peak value')
    ap.add_argument('--valley-th', type=float, default=-0.05, help='Maximum valley value')
    ap.add_argument('--out-json', default='cycles.json', help='Output JSON for cycles')
    ap.add_argument('--out-png', default='cycles.png', help='Output plot PNG')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.column not in df.columns:
        raise SystemExit(f"Column {args.column} not found in CSV")
    x = df[args.column].to_numpy(float)

    peaks, valleys = find_peaks_and_valleys(x, args.peak_th, args.valley_th)
    cycles = build_cycles(peaks, valleys)

    data = []
    for (p1, v, p2) in cycles:
        data.append({
            'start_peak_idx': int(p1),
            'valley_idx': int(v),
            'end_peak_idx': int(p2),
            'peak1': float(x[p1]),
            'valley': float(x[v]),
            'peak2': float(x[p2]),
            'amp_drop': float(x[p1] - x[v]),
            'amp_rise': float(x[p2] - x[v]),
            'duration_frames': int(p2 - p1),
        })

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'cycles': data, 'peak_thr': args.peak_th, 'valley_thr': args.valley_th, 'column': args.column}, f, ensure_ascii=False, indent=2)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot_series(x, cycles, out_png, title=f'{args.column}: peak>{args.peak_th}, valley<{args.valley_th}, cycles={len(cycles)}')

    print(f'Saved cycles JSON -> {out_json.as_posix()} (count={len(cycles)})')
    print(f'Saved plot -> {out_png.as_posix()}')


if __name__ == '__main__':
    main()
