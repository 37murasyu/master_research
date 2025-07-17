from __future__ import annotations

import argparse
from typing import List, Tuple
import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    dot = float(np.dot(a, b)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    crossn = np.linalg.norm(np.cross(a/na, b/nb))
    import math
    return math.atan2(crossn, dot)


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


def detect_cycles_peak_valley(signal: np.ndarray, min_amp: float, min_len: int) -> List[Tuple[int,int,float]]:
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


def smooth_mavg(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return x
    w = int(w)
    if w % 2 == 0:
        w += 1
    if len(x) >= w:
        k = np.ones(w)/float(w)
        return np.convolve(x, k, mode='same')
    return x


def detect_cycles_tau(tau: np.ndarray, min_amp: float, min_len: int, smooth_window: int = 11) -> List[Tuple[int,int,float]]:
    if len(tau) < min_len * 2:
        return []
    t = np.asarray(tau)
    # simple moving average smoothing to stabilize zero-cross detection
    t_s = smooth_mavg(t, smooth_window)
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


def detect_cycles_tau_env(tau: np.ndarray, min_amp: float, min_len: int, smooth_window: int = 101, peak_prom: float = 20.0) -> List[Tuple[int,int,float]]:
    """Detect cycles using peaks of |tau| envelope. Each adjacent peak pair forms a cycle window.
    - min_amp: threshold on envelope difference within the window
    - min_len: minimum frames between peaks
    - smooth_window: moving average window for envelope smoothing
    - peak_prom: simple prominence-like threshold relative to median(abs(tau))
    """
    if len(tau) < min_len * 2:
        return []
    t = np.asarray(tau)
    env = np.abs(t)
    env_s = smooth_mavg(env, smooth_window)
    med = float(np.median(env_s))
    thr = med + peak_prom
    # naive peak detection
    peaks: List[int] = []
    for i in range(1, len(env_s)-1):
        if env_s[i] > env_s[i-1] and env_s[i] >= env_s[i+1] and env_s[i] >= thr:
            if peaks and i - peaks[-1] < min_len:
                # keep the higher one within the refractory window
                if env_s[i] > env_s[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    cycles: List[Tuple[int,int,float]] = []
    for a, b in zip(peaks, peaks[1:]):
        if b - a + 1 < min_len:
            continue
        seg = t[a:b+1]
        amp = float(np.max(seg) - np.min(seg))
        if amp < min_amp:
            continue
        cycles.append((a, b, amp))
    return cycles


def main():
    ap = argparse.ArgumentParser(description='Sweep cycle detection params and report counts')
    ap.add_argument('--forearm-npy', required=True)
    ap.add_argument('--tau-npy', required=True)
    ap.add_argument('--clip-dtheta', type=float, default=0.35)
    ap.add_argument('--tau-clamp-abs', type=float, default=1000.0)
    ap.add_argument('--target', type=int, default=8)
    ap.add_argument('--window-frames', type=int, default=0, help='If >0, scan windows of this length (frames) to find ROI with counts closest to target')
    ap.add_argument('--step-frames', type=int, default=120, help='Step size for window scan (frames)')
    args = ap.parse_args()

    vecs = np.load(args.forearm_npy)
    tau = np.load(args.tau_npy)
    if args.tau_clamp_abs and args.tau_clamp_abs > 0:
        tau = np.clip(tau, -args.tau_clamp_abs, args.tau_clamp_abs)
    N = min(len(vecs), len(tau))
    vecs = vecs[:N]; tau = tau[:N]

    theta_ref = reconstruct_theta_angle_ref(vecs)
    theta_diff = reconstruct_theta_angle_diff(vecs, args.clip_dtheta)

    modes = ['angle_ref', 'angle_diff', 'tau', 'tau_env']
    # Angle modes (radian scale)
    ang_min_amps = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    ang_min_lens = [60, 90, 120, 150, 180, 240, 300, 360]
    ang_windows = [1, 51, 101, 201, 301]
    # Tau mode (N·m scale)
    tau_min_amps = [30, 60, 100, 150, 200, 300, 500]
    tau_min_lens = [45, 60, 75, 90, 120, 150, 180, 210, 240]
    tau_windows = [5, 11, 21, 31, 51, 101, 201, 301]
    tau_env_proms = [10.0, 20.0, 40.0, 80.0]

    def sweep_on_segment(seg_slice: slice):
        res = []
        thetar = theta_ref[seg_slice]
        thetad = theta_diff[seg_slice]
        taur = tau[seg_slice]
        for m in modes:
            if m in ('angle_ref', 'angle_diff'):
                for a in ang_min_amps:
                    for L in ang_min_lens:
                        for w in ang_windows:
                            sig = smooth_mavg(thetar if m=='angle_ref' else thetad, w)
                            cyc = detect_cycles_peak_valley(sig, a, L)
                            cnt = len(cyc)
                            score = abs(cnt - args.target)
                            res.append((score, cnt, m, a, L, w, seg_slice.start or 0, (seg_slice.stop or len(theta_ref))-1))
            else:
                for a in tau_min_amps:
                    for L in tau_min_lens:
                        for w in tau_windows:
                            if m == 'tau':
                                cyc = detect_cycles_tau(taur, a, L, smooth_window=w)
                                cnt = len(cyc)
                                score = abs(cnt - args.target)
                                res.append((score, cnt, m, a, L, w, seg_slice.start or 0, (seg_slice.stop or len(theta_ref))-1))
                            else:  # tau_env
                                for prom in tau_env_proms:
                                    cyc = detect_cycles_tau_env(taur, a, L, smooth_window=w, peak_prom=prom)
                                    cnt = len(cyc)
                                    score = abs(cnt - args.target)
                                    res.append((score, cnt, m, a, L, (w, prom), seg_slice.start or 0, (seg_slice.stop or len(theta_ref))-1))
        return res

    results = []
    if args.window_frames and args.window_frames > 0:
        W = int(args.window_frames)
        step = max(1, int(args.step_frames))
        N = len(theta_ref)
        for s in range(0, max(1, N-W+1), step):
            e = min(N, s+W)
            results.extend(sweep_on_segment(slice(s, e)))
    else:
        results = sweep_on_segment(slice(None, None))

    results.sort(key=lambda x: (x[0], -x[1]))
    print('# best 15 settings (score=|count-target|)')
    for r in results[:15]:
        suffix = '' if r[2] != 'tau' else f'  smooth_window={r[5]}'
        roi = f'  roi=[{r[6]}:{r[7]}]'
        print(f'score={r[0]}  count={r[1]}  mode={r[2]}  min_amp={r[3]:.2f}  min_len={r[4]}{suffix}{roi}')

    # also print any exact matches for quick pick
    exact = [r for r in results if r[0] == 0]
    if exact:
        print('\n# exact matches (count == target)')
        for r in exact[:10]:
            suffix = '' if r[2] != 'tau' else f'  smooth_window={r[5]}'
            roi = f'  roi=[{r[6]}:{r[7]}]'
            print(f'count={r[1]}  mode={r[2]}  min_amp={r[3]:.2f}  min_len={r[4]}{suffix}{roi}')


if __name__ == '__main__':
    main()
