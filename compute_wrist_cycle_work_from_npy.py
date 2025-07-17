"""
Compute per-cycle positive work from wrist local torque (NPY) and forearm vectors (NPY),
auto-detect cycles from the arm angle, and save CSV/PNG. Designed to finalize outputs
even when pose/torque CSVs are not readily available.

Inputs (auto-detected under output_data/wrist_inputs):
  - forearm_R_<id>.npy, forearm_L_<id>.npy : shape (T, 3)
  - tau_wrist_R_<id>.npy, tau_wrist_L_<id>.npy : shape (T,)

Outputs:
  - output_data/wrist_cycle_work_<id>.csv
  - output_data/wrist_cycle_work_<id>_<mode>.png   # mode in {percent, absolute}

Notes:
  - Angle theta is defined as the angle between forearm vector and the chosen up-axis (default: y).
  - Cycle detection is done on smoothed theta using valley-to-valley segmentation with spacing guards.
  - 1RM is computed from m_max_all.csv and the given subject id.
"""
from __future__ import annotations

import argparse
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # headless-safe backend
import matplotlib.pyplot as plt

try:
    from scipy.signal import butter, filtfilt
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# Try to import fps from config (fallbacks if not present)
try:
    from config import fps as _FPS_CFG
except Exception:
    _FPS_CFG = 30


def _butter_lowpass_filtfilt(x: np.ndarray, fs: float, fc: float, order: int) -> np.ndarray:
    if len(x) < max(8, 3*order+1):
        return x.copy()
    if not _SCIPY_OK:
        # moving average fallback
        k = max(3, min(9, (len(x)//10)*2+1))
        return np.convolve(x, np.ones(k)/k, mode='same')
    nyq = 0.5 * fs
    wn = min(0.99, max(1e-3, fc / nyq))
    b, a = butter(order, wn, btype='low', analog=False)
    try:
        return filtfilt(b, a, x, method='gust')
    except Exception:
        return filtfilt(b, a, x)


def _angle_to_up(v: np.ndarray, up_axis: str = 'y') -> np.ndarray:
    # returns angle [rad] between vector v(t) and the up-axis
    up_map = {
        'x': np.array([1.0, 0.0, 0.0], dtype=np.float64),
        'y': np.array([0.0, 1.0, 0.0], dtype=np.float64),
        'z': np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    u = up_map.get(up_axis.lower(), up_map['y'])
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("forearm vectors must have shape (T,3)")
    vn = np.linalg.norm(v, axis=1) + 1e-12
    cosang = np.clip(np.dot(v, u) / vn, -1.0, 1.0)
    return np.arccos(cosang)


def _detect_cycles_from_theta(theta: np.ndarray, fps: float, fc: float = 1.2, order: int = 2,
                              min_dur_sec: float = 0.5) -> List[Tuple[int, int]]:
    """
    Detect cycles from angle series. Returns list of (start_idx, end_idx) as valley-to-valley windows.
    """
    n = len(theta)
    if n < 5:
        return []
    th = _butter_lowpass_filtfilt(np.unwrap(theta.astype(float)), fs=fps, fc=fc, order=order)
    # find valleys by derivative sign change: d'<0 -> d'>0
    d = np.diff(th)
    sign = np.sign(d)
    # zero handling: forward-fill zeros
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i-1]
    valley_idx = []
    for i in range(1, len(sign)):
        if sign[i-1] < 0 and sign[i] > 0:
            valley_idx.append(i)
    # spacing guard
    min_gap = max(2, int(min_dur_sec * fps))
    valley2 = []
    last = -10**9
    for v in valley_idx:
        if v - last >= min_gap:
            valley2.append(v)
            last = v
    cycles: List[Tuple[int, int]] = []
    for a, b in zip(valley2, valley2[1:]):
        if b - a >= min_gap:
            cycles.append((a, b))
    return cycles


def _compute_cycle_energy(theta: np.ndarray, tau: np.ndarray, dt_sec: float,
                          fc: float = 1.2, order: int = 2, resample_n: int = 80,
                          max_dth: float = 0.25) -> Tuple[float, float]:
    """Simplified version of compute_cycle_energy_filtered.
    Returns (E_pos, E_neg).
    """
    n = len(theta)
    if n < 3:
        return 0.0, 0.0
    fs = 1.0 / max(1e-6, dt_sec)
    th = np.unwrap(theta.astype(float))
    th_f = _butter_lowpass_filtfilt(th, fs, fc, order)
    tau_f = _butter_lowpass_filtfilt(tau.astype(float), fs, fc, order)
    t = np.arange(n, dtype=float) * dt_sec
    # uniform resample
    t0, t1 = t[0], t[-1]
    if t1 <= t0:
        t1 = t0 + dt_sec * max(1, n-1)
    ti = np.linspace(t0, t1, max(resample_n, 10))
    th_i = np.interp(ti, t, th_f)
    tau_i = np.interp(ti, t, tau_f)
    dth = np.diff(th_i)
    dth = np.clip(dth, -max_dth, max_dth)
    tau_mid = 0.5 * (tau_i[1:] + tau_i[:-1])
    c = tau_mid * dth
    e_pos = float(np.sum(np.maximum(c, 0.0)))
    e_neg = float(np.sum(np.maximum(-c, 0.0)))
    return e_pos, e_neg


@dataclass
class SeriesPack:
    theta: np.ndarray
    tau: np.ndarray


def _load_latest_id(base_dir: str) -> Tuple[str, dict]:
    """Scan output_data/wrist_inputs and pick the most recent id having all four files.
    Returns (id_str, paths_dict).
    paths_dict keys: forearm_R, forearm_L, tau_R, tau_L
    """
    wi = os.path.join(base_dir, 'output_data', 'wrist_inputs')
    assert os.path.isdir(wi), f"not found: {wi}"
    files = glob.glob(os.path.join(wi, '*.npy'))
    # group by id suffix (after the first 2 tokens)
    groups: dict[str, dict[str, str]] = {}
    for p in files:
        name = os.path.basename(p)
        # patterns:
        #  forearm_R_<id>.npy / forearm_L_<id>.npy
        #  tau_wrist_R_<id>.npy / tau_wrist_L_<id>.npy
        if name.startswith('forearm_R_'):
            sid = name[len('forearm_R_'):-4]
            groups.setdefault(sid, {})['forearm_R'] = p
        elif name.startswith('forearm_L_'):
            sid = name[len('forearm_L_'):-4]
            groups.setdefault(sid, {})['forearm_L'] = p
        elif name.startswith('tau_wrist_R_'):
            sid = name[len('tau_wrist_R_'):-4]
            groups.setdefault(sid, {})['tau_R'] = p
        elif name.startswith('tau_wrist_L_'):
            sid = name[len('tau_wrist_L_'):-4]
            groups.setdefault(sid, {})['tau_L'] = p
    # pick the group that has all four
    candidates: List[Tuple[str, dict]] = []
    for sid, d in groups.items():
        if all(k in d for k in ('forearm_R','forearm_L','tau_R','tau_L')):
            # mtime as recency
            mt = max(os.path.getmtime(v) for v in d.values())
            candidates.append((sid, {'paths': d, 'mtime': mt}))
    if not candidates:
        raise FileNotFoundError("No complete set of NPYs found under output_data/wrist_inputs")
    candidates.sort(key=lambda x: x[1]['mtime'], reverse=True)
    sid = candidates[0][0]
    return sid, candidates[0][1]['paths']


def _load_id_paths(base_dir: str, id_str: str) -> dict:
    wi = os.path.join(base_dir, 'output_data', 'wrist_inputs')
    paths = {
        'forearm_R': os.path.join(wi, f'forearm_R_{id_str}.npy'),
        'forearm_L': os.path.join(wi, f'forearm_L_{id_str}.npy'),
        'tau_R': os.path.join(wi, f'tau_wrist_R_{id_str}.npy'),
        'tau_L': os.path.join(wi, f'tau_wrist_L_{id_str}.npy'),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing file: {p}")
    return paths


def _load_series(paths: dict, up_axis: str = 'y') -> Tuple[SeriesPack, SeriesPack]:
    fr = np.load(paths['forearm_R'])
    fl = np.load(paths['forearm_L'])
    tr = np.load(paths['tau_R'])
    tl = np.load(paths['tau_L'])
    if fr.ndim != 2 or fr.shape[1] != 3:
        raise ValueError("forearm_R shape must be (T,3)")
    if fl.ndim != 2 or fl.shape[1] != 3:
        raise ValueError("forearm_L shape must be (T,3)")
    if tr.ndim != 1 or tl.ndim != 1:
        raise ValueError("tau series must be 1-D")
    if not (len(fr) == len(tr) and len(fl) == len(tl)):
        raise ValueError("length mismatch between forearm and tau series")
    th_r = _angle_to_up(fr, up_axis=up_axis)
    th_l = _angle_to_up(fl, up_axis=up_axis)
    return SeriesPack(theta=th_r, tau=tr), SeriesPack(theta=th_l, tau=tl)


def _load_mmax(base_dir: str, subject_id: int) -> Tuple[float, float, float]:
    mpath = os.path.join(base_dir, 'm_max_all.csv')
    df = pd.read_csv(mpath)
    # expect columns: subject_id, m_max_L, m_max_R (or similar); fallback to a single m_max
    sidcol = 'subject_id' if 'subject_id' in df.columns else 'subject'
    row = df[df[sidcol] == subject_id]
    if row.empty:
        # fallback: pick first row
        row = df.iloc[[0]]
    if {'m_max_L','m_max_R'}.issubset(row.columns):
        mL = float(row['m_max_L'].values[0])
        mR = float(row['m_max_R'].values[0])
    elif 'm_max' in row.columns:
        mL = mR = float(row['m_max'].values[0])
    else:
        # conservative fallback
        mL = mR = float(row.select_dtypes(include='number').iloc[0, 1])
    # 1RM formula per prior notes: (6.225e-3 * M + m_max) * 5.01
    # We don't know body mass M from here; assume 60 as per config default
    M = 60.0
    oneRM_L = (6.225e-3 * M + mL) * 5.01
    oneRM_R = (6.225e-3 * M + mR) * 5.01
    return M, oneRM_L, oneRM_R


def main():
    ap = argparse.ArgumentParser(description='Compute per-cycle wrist work from NPY wrist inputs and save CSV/PNG.')
    ap.add_argument('--base-dir', default='.', help='Workspace root (default: .)')
    ap.add_argument('--id', default=None, help='Suffix id used in NPY filenames (auto if omitted)')
    ap.add_argument('--subject', type=int, default=8, help='Subject id for 1RM lookup (default: 8)')
    ap.add_argument('--up-axis', default='y', choices=['x','y','z'], help='Up-axis for angle definition (default: y)')
    ap.add_argument('--mode', default='percent', choices=['percent','absolute'], help='Plot y-axis mode (default: percent)')
    ap.add_argument('--out-dir', default='output_data', help='Output directory (default: output_data)')
    ap.add_argument('--fc', type=float, default=1.2, help='Low-pass cutoff for filtering (Hz)')
    ap.add_argument('--order', type=int, default=2, help='Low-pass filter order')
    ap.add_argument('--min-sec', type=float, default=0.5, help='Minimum cycle duration (s)')
    args = ap.parse_args()

    base = os.path.abspath(args.base_dir)
    os.makedirs(os.path.join(base, args.out_dir), exist_ok=True)
    # pick id and load paths
    if args.id:
        id_str = args.id
        paths = _load_id_paths(base, id_str)
    else:
        id_str, paths = _load_latest_id(base)

    R, L = _load_series(paths, up_axis=args.up_axis)
    # time base
    fps = float(_FPS_CFG or 30)
    dt = 1.0 / fps

    # cycles per side (detect from theta)
    cyc_R = _detect_cycles_from_theta(R.theta, fps=fps, fc=args.fc, order=args.order, min_dur_sec=args.min_sec)
    cyc_L = _detect_cycles_from_theta(L.theta, fps=fps, fc=args.fc, order=args.order, min_dur_sec=args.min_sec)

    # energy per cycle
    rows = []
    for side, pack, cycles in [('R', R, cyc_R), ('L', L, cyc_L)]:
        for ci, (a, b) in enumerate(cycles, start=1):
            epos, eneg = _compute_cycle_energy(pack.theta[a:b+1], pack.tau[a:b+1], dt_sec=dt)
            rows.append({'side': side, 'cycle_index': ci, 'start': a, 'end': b, 'E_pos': epos, 'E_neg': eneg})
    df = pd.DataFrame(rows)

    # 1RM lookup
    try:
        M, oneRM_L, oneRM_R = _load_mmax(base, args.subject)
    except Exception as e:
        # proceed without percent if failed
        M, oneRM_L, oneRM_R = 60.0, np.nan, np.nan
        print(f"[warn] 1RM lookup failed: {e}")

    # percent column
    perc = []
    for _, r in df.iterrows():
        d = r['E_pos']
        denom = oneRM_R if r['side'] == 'R' else oneRM_L
        perc.append(float(d) / float(denom) * 100.0 if (denom and denom == denom and denom > 0) else np.nan)
    df['percent_of_1RM'] = perc

    out_csv = os.path.join(base, args.out_dir, f'wrist_cycle_work_{id_str}.csv')
    df.to_csv(out_csv, index=False)
    print(f"[OUT] CSV: {out_csv}  (N={len(df)})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for side in ['R','L']:
        sub = df[df['side'] == side]
        x = np.arange(1, len(sub)+1)
        y = sub['percent_of_1RM'].values if args.mode == 'percent' else sub['E_pos'].values
        ax.bar(x + (0.2 if side=='R' else -0.2), y, width=0.4, label=f'{side}')
    if args.mode == 'percent':
        ax.set_ylabel('Work per cycle [% of 1RM]')
    else:
        ax.set_ylabel('Work per cycle [J]')
    ax.set_xlabel('Cycle index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(base, args.out_dir, f'wrist_cycle_work_{id_str}_{args.mode}.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OUT] PNG: {out_png}")


if __name__ == '__main__':
    main()
