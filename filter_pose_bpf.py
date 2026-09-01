import argparse
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd


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


def apply_bpf(series: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    hp_b, hp_a = biquad_highpass(f_lo, fs)
    lp_b, lp_a = biquad_lowpass(f_hi, fs)
    x = series.astype(float)
    x = biquad_filter(hp_b, hp_a, x)
    x = biquad_filter(lp_b, lp_a, x)
    return x


def detect_joint_columns(df: pd.DataFrame) -> Dict[int, Tuple[str, str, str]]:
    joints = {}
    for col in df.columns:
        if not col.startswith('joint_'):
            continue
        parts = col.split('_')
        if len(parts) < 3:
            continue
        if not parts[1].isdigit():
            continue
        axis = parts[2]
        if axis not in ('x', 'y', 'z'):
            continue
        jid = int(parts[1])
        base_raw = f"joint_{jid}_{axis}"
        base_f = base_raw + "_f"
        chosen = None
        if base_raw in df.columns:
            chosen = base_raw
        elif base_f in df.columns:
            chosen = base_f
        else:
            continue
        if jid not in joints:
            joints[jid] = [None, None, None]
        idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        joints[jid][idx] = chosen
    # ensure complete triplets
    complete = {jid: tuple(cols) for jid, cols in joints.items() if all(cols)}
    return complete


def main():
    ap = argparse.ArgumentParser(description='Apply BPF to pose CSV and write *_f columns')
    ap.add_argument('--pose-csv', required=True)
    ap.add_argument('--out-pose', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--bpf-low', type=float, default=0.1)
    ap.add_argument('--bpf-high', type=float, default=4.0)
    args = ap.parse_args()

    df = pd.read_csv(args.pose_csv)
    joints = detect_joint_columns(df)
    if not joints:
        raise ValueError('No joint_* triplets found')

    fs = args.fps
    for jid, cols in joints.items():
        for axis, col in zip(('x', 'y', 'z'), cols):
            filt = apply_bpf(df[col].to_numpy(dtype=float), fs=fs, f_lo=args.bpf_low, f_hi=args.bpf_high)
            out_col = f"joint_{jid}_{axis}_f"
            df[out_col] = filt
    df.to_csv(args.out_pose, index=False)
    print(f'Saved filtered pose to {args.out_pose}')


if __name__ == '__main__':
    main()
