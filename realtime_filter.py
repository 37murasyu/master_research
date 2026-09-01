"""Lightweight causal noise filter for 30 Hz pose data to ~4 Hz output.

- Causal 5-point median for outlier suppression (optional if buffer not full yet).
- Causal 1st-order IIR low-pass with configurable cutoff `fc` (Hz).
- Optional downsample to target rate (e.g., 4 Hz) by linear interpolation of filtered signal.

Usage (offline simulation of real-time path):
  python realtime_filter.py \
    --input output_data/poses/kpts3d_subject8_20250925_192700.csv \
    --columns joint_3_x,joint_3_y,joint_3_z \
    --fs 30 --fc 2.0 --median-win 5 \
    --output filtered_fullrate.csv \
    --output-ds filtered_4hz.csv --target-hz 4

In real-time, call CausalMedianIIR.step(x, dt) per sample; for 4 Hz output, emit on a 0.25 s timer using the latest filtered values.
"""

from __future__ import annotations
import argparse
import math
from typing import List, Optional

import numpy as np
import pandas as pd


class CausalMedianIIR:
    def __init__(self, fc: float = 2.0, median_win: int = 5):
        self.fc = fc
        self.median_win = median_win if median_win % 2 == 1 else median_win + 1
        self.buf = [0.0] * self.median_win
        self.count = 0
        self.idx = 0
        self.y = 0.0

    def _median(self) -> float:
        if self.count < 1:
            return 0.0
        if self.count < self.median_win:
            data = self.buf[: self.count]
        else:
            data = self.buf
        s = sorted(data)
        return s[len(s) // 2]

    def step(self, x: float, dt: float) -> float:
        # ring buffer update
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.median_win
        self.count = min(self.count + 1, self.median_win)

        xm = self._median()
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.fc * dt)
        self.y = alpha * xm + (1.0 - alpha) * self.y
        return self.y


def _build_time(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=float) / fs


def apply_filter(data: np.ndarray, time_s: np.ndarray, fc: float, median_win: int) -> np.ndarray:
    n, d = data.shape
    out = np.zeros_like(data, dtype=float)
    filters = [CausalMedianIIR(fc=fc, median_win=median_win) for _ in range(d)]
    last_t = time_s[0]
    for i in range(n):
        t = time_s[i]
        dt = max(1e-6, t - last_t) if i > 0 else 1.0 / 30.0
        last_t = t
        for j in range(d):
            out[i, j] = filters[j].step(float(data[i, j]), dt)
    return out


def downsample_linear(time_s: np.ndarray, data: np.ndarray, target_hz: float) -> tuple[np.ndarray, np.ndarray]:
    if target_hz <= 0:
        return time_s, data
    t_start, t_end = float(time_s[0]), float(time_s[-1])
    step = 1.0 / target_hz
    t_out = np.arange(t_start, t_end + 1e-9, step)
    ds = np.zeros((t_out.size, data.shape[1]), dtype=float)
    for j in range(data.shape[1]):
        ds[:, j] = np.interp(t_out, time_s, data[:, j])
    return t_out, ds


def main() -> int:
    ap = argparse.ArgumentParser(description="Causal median+IIR filter and 4 Hz downsample")
    ap.add_argument('--input', required=True, help='Input CSV path')
    ap.add_argument('--columns', required=True, help='Comma-separated column names to filter')
    ap.add_argument('--fs', type=float, default=30.0, help='Sampling rate if no time column (Hz)')
    ap.add_argument('--time-col', help='Time column name in seconds (optional)')
    ap.add_argument('--fc', type=float, default=2.0, help='Low-pass cutoff (Hz)')
    ap.add_argument('--median-win', type=int, default=5, help='Odd window size for median')
    ap.add_argument('--output', required=True, help='Output CSV for full-rate filtered data')
    ap.add_argument('--output-ds', help='Output CSV for downsampled data (e.g., 4 Hz)')
    ap.add_argument('--target-hz', type=float, default=4.0, help='Target rate for downsample')
    args = ap.parse_args()

    cols = [c.strip() for c in args.columns.split(',') if c.strip()]
    usecols: List[str] = cols.copy()
    if args.time_col:
        usecols = [args.time_col] + cols
    df = pd.read_csv(args.input, usecols=usecols)

    if args.time_col:
        time_s = df[args.time_col].to_numpy(float)
    else:
        time_s = _build_time(len(df), args.fs)
    data = df[cols].to_numpy(float)

    filt = apply_filter(data, time_s, fc=args.fc, median_win=args.median_win)
    df_out = pd.DataFrame({ 'time_s': time_s })
    for i, c in enumerate(cols):
        df_out[c + '_rtfilt'] = filt[:, i]
    df_out.to_csv(args.output, index=False)

    if args.output_ds:
        t_ds, d_ds = downsample_linear(time_s, filt, target_hz=args.target_hz)
        df_ds = pd.DataFrame({ 'time_s': t_ds })
        for i, c in enumerate(cols):
            df_ds[c + '_rtfilt'] = d_ds[:, i]
        df_ds.to_csv(args.output_ds, index=False)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
