"""Lightweight 1D Kalman filter (position + velocity) for real-time pose streams.

- Constant-velocity model (state = [x, v]).
- Per-axis filter; O(1) per sample.
- Optional gating: skip/soften updates when innovation is too large (outlier suppression).
- Optional downsample to target rate (e.g., 3–4 Hz) by linear interpolation after filtering.

Usage example:
  python realtime_kalman_filter.py \
    --input output_data/poses/kpts3d_subject8_20250925_192700.csv \
    --columns joint_3_x,joint_3_y,joint_3_z \
    --fs 30 --output kalman_full.csv \
    --output-ds kalman_3hz.csv --target-hz 3
"""

from __future__ import annotations
import argparse
import math
from typing import List

import numpy as np
import pandas as pd


class Kalman1D:
    """Constant-velocity Kalman: state [x, v]."""

    def __init__(
        self,
        q_pos: float = 1e-4,
        q_vel: float = 1e-3,
        r: float = 1e-3,
        gate_std: float = 3.0,
    ):
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r = r
        self.gate_std = gate_std
        # state
        self.x = 0.0
        self.v = 0.0
        # covariance P (2x2) symmetric
        self.p_xx = 1.0
        self.p_xv = 0.0
        self.p_vv = 1.0
        self.initialized = False

    def predict(self, dt: float) -> None:
        # F = [[1, dt],[0,1]] ; Q = diag(q_pos, q_vel)
        x_new = self.x + self.v * dt
        v_new = self.v
        p_xx = self.p_xx + dt * (self.p_xv + self.p_xv) + dt * dt * self.p_vv + self.q_pos
        p_xv = self.p_xv + dt * self.p_vv
        p_vv = self.p_vv + self.q_vel
        self.x, self.v = x_new, v_new
        self.p_xx, self.p_xv, self.p_vv = p_xx, p_xv, p_vv

    def update(self, z: float) -> None:
        # H = [1, 0]
        y = z - self.x  # innovation
        s = self.p_xx + self.r  # innovation covariance
        if s <= 0:
            return
        if self.gate_std > 0:
            if abs(y) > self.gate_std * math.sqrt(s):
                # skip update (outlier)
                return
        kx = self.p_xx / s
        kv = self.p_xv / s
        self.x += kx * y
        self.v += kv * y
        # Joseph form (simplified for H=[1,0])
        self.p_xv -= kx * self.p_xv
        self.p_vv -= kv * self.p_xv  # note: uses updated p_xv
        self.p_xx = (1 - kx) * self.p_xx

    def step(self, z: float, dt: float) -> float:
        if not self.initialized:
            self.x = float(z)
            self.v = 0.0
            self.p_xx, self.p_xv, self.p_vv = 1.0, 0.0, 1.0
            self.initialized = True
            return self.x
        self.predict(dt)
        self.update(z)
        return self.x


class KalmanCA1D:
    """Constant-acceleration Kalman: state [x, v, a]."""

    def __init__(
        self,
        q_acc: float = 1e-3,
        r: float = 1e-3,
        gate_std: float = 3.0,
    ):
        self.q_acc = q_acc  # continuous-time accel noise intensity
        self.r = r
        self.gate_std = gate_std
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0
        # covariance P (3x3) symmetric elements
        self.p_xx = 1.0; self.p_xv = 0.0; self.p_xa = 0.0
        self.p_vv = 1.0; self.p_va = 0.0; self.p_aa = 1.0
        self.initialized = False

    def predict(self, dt: float) -> None:
        dt2 = dt * dt
        dt3 = dt2 * dt
        # state
        x_new = self.x + self.v * dt + 0.5 * self.a * dt2
        v_new = self.v + self.a * dt
        a_new = self.a
        # covariance propagation using closed-form F, Q
        # F
        # [[1, dt, 0.5 dt^2], [0,1,dt], [0,0,1]]
        p_xx = self.p_xx + dt * (self.p_xv + self.p_xv) + dt2 * self.p_vv + dt2 * self.p_xa + dt3 * self.p_va + 0.25 * dt2 * dt2 * self.p_aa
        p_xv = self.p_xv + dt * self.p_vv + 0.5 * dt2 * self.p_va + dt * self.p_xa + 0.5 * dt2 * self.p_aa
        p_xa = self.p_xa + dt * self.p_va + 0.5 * dt2 * self.p_aa
        p_vv = self.p_vv + dt * (self.p_va + self.p_va) + dt2 * self.p_aa
        p_va = self.p_va + dt * self.p_aa
        p_aa = self.p_aa
        # add Q(dt) for continuous white accel noise intensity q_acc
        q = self.q_acc
        q11 = q * (dt ** 5) / 20.0
        q12 = q * (dt ** 4) / 8.0
        q13 = q * dt3 / 6.0
        q22 = q * dt3 / 3.0
        q23 = q * dt2 / 2.0
        q33 = q * dt
        p_xx += q11
        p_xv += q12
        p_xa += q13
        p_vv += q22
        p_va += q23
        p_aa += q33

        self.x, self.v, self.a = x_new, v_new, a_new
        self.p_xx, self.p_xv, self.p_xa = p_xx, p_xv, p_xa
        self.p_vv, self.p_va, self.p_aa = p_vv, p_va, p_aa

    def update(self, z: float) -> None:
        # H = [1,0,0]
        y = z - self.x
        s = self.p_xx + self.r
        if s <= 0:
            return
        if self.gate_std > 0 and abs(y) > self.gate_std * math.sqrt(s):
            return
        kx = self.p_xx / s
        kv = self.p_xv / s
        ka = self.p_xa / s
        self.x += kx * y
        self.v += kv * y
        self.a += ka * y
        # update cov; Joseph simplified for H=[1,0,0]
        self.p_xx = (1 - kx) * self.p_xx
        self.p_xv -= kx * self.p_xv
        self.p_xa -= kx * self.p_xa
        self.p_vv -= kv * self.p_xv
        self.p_va -= kv * self.p_xa
        self.p_aa -= ka * self.p_xa

    def step(self, z: float, dt: float) -> float:
        if not self.initialized:
            self.x = float(z)
            self.v = 0.0
            self.a = 0.0
            self.p_xx, self.p_xv, self.p_xa = 1.0, 0.0, 0.0
            self.p_vv, self.p_va, self.p_aa = 1.0, 0.0, 1.0
            self.initialized = True
            return self.x
        self.predict(dt)
        self.update(z)
        return self.x


def _build_time(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=float) / fs


def apply_kalman(data: np.ndarray, time_s: np.ndarray, cfg) -> np.ndarray:
    n, d = data.shape
    out = np.zeros_like(data, dtype=float)
    if cfg.model == 'ca':
        filters = [KalmanCA1D(q_acc=cfg.q_acc, r=cfg.r, gate_std=cfg.gate_std) for _ in range(d)]
    else:
        filters = [Kalman1D(q_pos=cfg.q_pos, q_vel=cfg.q_vel, r=cfg.r, gate_std=cfg.gate_std) for _ in range(d)]
    last_t = time_s[0]
    for i in range(n):
        t = time_s[i]
        dt = max(1e-6, t - last_t) if i > 0 else 1.0 / cfg.fs
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


def parse_args():
    ap = argparse.ArgumentParser(description="Lightweight 1D Kalman filter (cv or ca) for pose CSV with variable dt")
    ap.add_argument('--input', required=True, help='Input CSV path')
    ap.add_argument('--columns', required=True, help='Comma-separated column names')
    ap.add_argument('--fs', type=float, default=30.0, help='Sampling rate if no time column (Hz)')
    ap.add_argument('--time-col', help='Time column name in seconds (optional)')
    ap.add_argument('--model', choices=['cv', 'ca'], default='ca', help='Model: cv=const-vel, ca=const-acc')
    ap.add_argument('--q-pos', type=float, default=1e-4, help='Process noise variance for position (cv)')
    ap.add_argument('--q-vel', type=float, default=1e-3, help='Process noise variance for velocity (cv)')
    ap.add_argument('--q-acc', type=float, default=1e-3, help='Accel noise intensity (ca, continuous)')
    ap.add_argument('--r', type=float, default=1e-3, help='Measurement noise variance')
    ap.add_argument('--gate-std', type=float, default=3.0, help='Outlier gate in sigma (<=0 to disable)')
    ap.add_argument('--output', required=True, help='Output CSV for full-rate filtered data')
    ap.add_argument('--output-ds', help='Output CSV for downsampled data')
    ap.add_argument('--target-hz', type=float, default=3.0, help='Target rate for downsample')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
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

    class CFG:
        fs = args.fs
        q_pos = args.q_pos
        q_vel = args.q_vel
        q_acc = args.q_acc
        r = args.r
        gate_std = args.gate_std
        model = args.model
    filt = apply_kalman(data, time_s, CFG)

    df_out = pd.DataFrame({'time_s': time_s})
    for i, c in enumerate(cols):
        df_out[c + '_kalman'] = filt[:, i]
    df_out.to_csv(args.output, index=False)

    if args.output_ds:
        t_ds, d_ds = downsample_linear(time_s, filt, target_hz=args.target_hz)
        df_ds = pd.DataFrame({'time_s': t_ds})
        for i, c in enumerate(cols):
            df_ds[c + '_kalman'] = d_ds[:, i]
        df_ds.to_csv(args.output_ds, index=False)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
