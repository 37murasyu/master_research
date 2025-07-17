from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

# MediaPipe IDs
RIGHT_SHOULDER = 12
LEFT_SHOULDER = 11


def resolve_y(df: pd.DataFrame, jid: int) -> np.ndarray | None:
    for col in (f"joint_{jid}_y_f", f"joint_{jid}_y"):
        if col in df.columns:
            return df[col].to_numpy(float)
    return None


def mavg(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return x
    w = int(w)
    if w % 2 == 0:
        w += 1
    if len(x) < w:
        return x
    k = np.ones(w) / float(w)
    return np.convolve(x, k, mode='same')


def detrend(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return x
    w = int(w)
    if w % 2 == 0:
        w += 1
    if len(x) < w:
        return x - float(np.median(x))
    kb = np.ones(w) / float(w)
    baseline = np.convolve(x, kb, mode='same')
    return x - baseline


def detect_cycles_valley(signal: np.ndarray, min_len: int, min_amp: float) -> List[tuple[int,int,float]]:
    if signal is None or len(signal) < max(5, 2*min_len):
        return []
    t = np.asarray(signal, dtype=float)
    dt = np.diff(t)
    sign = np.sign(dt)
    valleys = []
    for i in range(1, len(sign)):
        if sign[i-1] < 0 and sign[i] >= 0:
            valleys.append(i)
    out = []
    for a, b in zip(valleys, valleys[1:]):
        if b - a + 1 < min_len:
            continue
        seg = t[a:b+1]
        amp = float(np.max(seg) - np.min(seg))
        if amp < min_amp:
            continue
        out.append((a, b, amp))
    return out


def main():
    ap = argparse.ArgumentParser(description='肩(11,12)のY極小(谷)でサイクル検出し、CSVの末尾に cycle_index 列を付与')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=False)
    ap.add_argument('--unit', choices=['auto','m','cm','mm'], default='auto')
    ap.add_argument('--merge', choices=['auto','avg','right','left'], default='auto')
    ap.add_argument('--smooth-window', type=int, default=31)
    ap.add_argument('--detrend-window', type=int, default=301)
    ap.add_argument('--min-len', type=int, default=60)
    ap.add_argument('--min-amp', type=float, default=0.02)
    ap.add_argument('--roi-start', type=int, default=None)
    ap.add_argument('--roi-end', type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    y_r = resolve_y(df, RIGHT_SHOULDER)
    y_l = resolve_y(df, LEFT_SHOULDER)
    if y_r is None and y_l is None:
        raise SystemExit('shoulder Y columns not found (joint_11_y[_f], joint_12_y[_f])')

    def scale(series: np.ndarray) -> np.ndarray:
        if series is None:
            return None
        if args.unit == 'm':
            return series
        med = float(np.nanmedian(np.abs(series))) if series.size else 0.0
        if args.unit == 'cm' or (args.unit == 'auto' and med > 1.0 and med <= 100.0):
            return series * 0.01
        if args.unit == 'mm' or (args.unit == 'auto' and med > 100.0):
            return series * 0.001
        return series

    y_r = scale(y_r)
    y_l = scale(y_l)

    # merge strategy
    if args.merge == 'avg' and y_r is not None and y_l is not None:
        y = 0.5 * (y_r + y_l)
    elif args.merge == 'right' and y_r is not None:
        y = y_r
    elif args.merge == 'left' and y_l is not None:
        y = y_l
    else:
        # auto: 右優先で存在する方
        y = y_r if y_r is not None else y_l

    # ROI
    T = len(y)
    s = int(args.roi_start) if args.roi_start is not None else 0
    e = int(args.roi_end) if args.roi_end is not None else T
    s = max(0, s); e = max(s+1, min(T, e))
    y_roi = y[s:e]

    # preprocess
    y_s = mavg(y_roi, args.smooth_window)
    y_s = detrend(y_s, args.detrend_window)

    cycles = detect_cycles_valley(y_s, args.min_len, args.min_amp)

    # build cycle_index per frame (0,1,2,... across cycles; -1 elsewhere)
    cyc_idx = np.full(T, -1, dtype=int)
    idx = 0
    for a, b, _ in cycles:
        # map back to original indexing range with offset s
        aa, bb = a + s, b + s
        cyc_idx[aa:bb+1] = idx
        idx += 1

    out_df = df.copy()
    out_df['cycle_index'] = cyc_idx

    out_path = args.output or os.path.splitext(args.input)[0] + '_with_cycles.csv'
    out_df.to_csv(out_path, index=False)
    print(f"[OUT] saved -> {out_path}  (cycles={len(cycles)})")


if __name__ == '__main__':
    main()
