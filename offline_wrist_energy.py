"""Offline wrist energy analysis.

Usage (example):
    python offline_wrist_energy.py \
        --video-dir ..\\cameras_raw\\9_20250925_201442 \
        --mmax-json m_max_part_9.json \
        --body-mass 60 \
        --offset 6 \
        --out wrist_energy.csv

Simplifications:
- Wrist angle change approximated by successive change of forearm direction (no hand landmark).
- Positive work only: max(tau_y * dtheta, 0)
- tau_y lag compensated by shifting torque series forward by --offset frames.
- r_x is uniform (default 0.30 m) unless overridden.

Future improvements:
- Introduce hand landmarks for true wrist flexion/extension.
- Apply filtering (Butterworth) mirroring compute_cycle_energy_filtered.
- Separate pronation/supination vs flexion/extension axes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Reuse constants / helper formulas (duplicated minimal subset to avoid heavy import of main)
CONST_K = math.sqrt(3)/2 + 1.0
G = 9.80665

# ---- Minimal m1 per part (copied logic) ----
# (coefficients from existing code: wrist: upper_arm(0.026)+upper_limb(0.276+0.19)+thigh(0.123)=0.615,
#  elbow: upper_limb(0.276+0.19)+thigh(0.123)=0.589 )
_DEF_COEFFS = {
    'wrist_R': 0.615,
    'wrist_L': 0.615,
    'elbow_R': 0.589,
    'elbow_L': 0.589,
}


def compute_m1_map(body_mass: float) -> Dict[str, float]:
    return {k: body_mass * v for k, v in _DEF_COEFFS.items()}


def compute_energy_thresholds(m_max_part: Dict[str, float], m1_map: Dict[str, float], r_x: float) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for part, mmax in m_max_part.items():
        m1p = float(m1_map.get(part, 0.0))
        base_coeff = 0.42 * m1p
        e_low = r_x * G * (base_coeff + 0.3 * mmax) * CONST_K
        e_high = r_x * G * (base_coeff + 0.7 * mmax) * CONST_K
        out[part] = (float(e_low), float(e_high))
    return out


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        return 0.0
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    dot = float(np.dot(a, b)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    crossn = np.linalg.norm(np.cross(a/na, b/nb))
    return math.atan2(crossn, dot)


@dataclass
class SeriesResult:
    dtheta: np.ndarray
    tau: np.ndarray
    work_inc: np.ndarray
    E_cum: np.ndarray
    E_low: float
    E_high: float


def compute_wrist_energy(forearm_vecs: np.ndarray, tau_local_y: np.ndarray, _dt: float, offset: int, clip_dtheta: float, E_low: float, E_high: float) -> SeriesResult:  # _dt kept for backward compatibility
    N = min(len(forearm_vecs), len(tau_local_y))
    _ = _dt  # explicitly ignore unused legacy param
    if N < 3:
        return SeriesResult(np.array([]), np.array([]), np.array([]), np.array([]), E_low, E_high)
    f = forearm_vecs[:N]
    tau = tau_local_y[:N]
    # sanitize non-finite values to prevent NaN propagation
    if np.isnan(tau).any() or ~np.isfinite(tau).all():
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    u = f / norms
    dtheta = np.zeros(N)
    for i in range(1, N):
        dtheta[i] = angle_between(u[i-1], u[i])
    dtheta = np.clip(dtheta, -clip_dtheta, clip_dtheta)
    # ensure angle increments are finite
    if np.isnan(dtheta).any() or ~np.isfinite(dtheta).all():
        dtheta = np.nan_to_num(dtheta, nan=0.0, posinf=0.0, neginf=0.0)
    # lag compensation (pose ahead of torque) -> shift torque forward
    if offset > 0 and N > offset:
        tau_sync = tau[offset:]
        dtheta_sync = dtheta[:-offset]
    else:
        tau_sync = tau
        dtheta_sync = dtheta
    M = min(len(tau_sync), len(dtheta_sync))
    tau_sync = tau_sync[:M]
    dtheta_sync = dtheta_sync[:M]
    work_inc = tau_sync * dtheta_sync
    work_inc[work_inc < 0] = 0.0
    E_cum = np.cumsum(work_inc)
    return SeriesResult(dtheta_sync, tau_sync, work_inc, E_cum, E_low, E_high)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-dir', type=str, required=False, help='(Optional) Video directory – currently unused placeholder.')
    ap.add_argument('--forearm-npy', type=str, help='(Right) forearm vector sequence (N,3) .npy')
    ap.add_argument('--tau-npy', type=str, help='(Right) local torque y sequence (N,) .npy')
    ap.add_argument('--forearm-npy-left', type=str, help='(Left) forearm vector sequence (N,3) .npy')
    ap.add_argument('--tau-npy-left', type=str, help='(Left) local torque y sequence (N,) .npy')
    ap.add_argument('--mmax-json', type=str, required=True, help='m_max_part JSON file (subset keys wrist/elbow).')
    ap.add_argument('--body-mass', type=float, required=True)
    ap.add_argument('--r-x', type=float, default=0.30, help='Effective radius r_x (m) for wrist/elbow (uniform).')
    ap.add_argument('--offset', type=int, default=6, help='Frame offset (pose leads torque).')
    ap.add_argument('--dt', type=float, default=1/30.0)
    ap.add_argument('--clip-dtheta', type=float, default=0.35, help='Clip for per-frame angle change (rad).')
    ap.add_argument('--out', type=str, default='wrist_energy.csv')
    ap.add_argument('--part', type=str, default='wrist_R', choices=['wrist_R','wrist_L','both'], help='both: export bilateral if left data present')
    ap.add_argument('--mode', type=str, default='per-frame', choices=['per-frame','per-cycle'], help='per-cycle: aggregate energy per detected movement cycle')
    ap.add_argument('--cycle-min-amp', type=float, default=0.15, help='Minimum cumulative angle excursion (rad) to accept a cycle (peak-valley).')
    ap.add_argument('--cycle-min-length', type=int, default=12, help='Minimum frames in a cycle window.')
    ap.add_argument('--include-partial', action='store_true', help='Include last incomplete cycle if it has enough frames.')
    ap.add_argument('--debug-cycles', action='store_true', help='Print debug info about cycle detection and optionally dump theta/tau.')
    ap.add_argument('--cycle-detect-mode', type=str, default='auto', choices=['auto','angle_ref','angle_diff','tau','tau_env','shoulder_z','shoulder_y'], help='Cycle detection source: angle_ref/diff (forearm), tau/tau_env (wrist torque), shoulder_z/y (shoulder vertical component). Auto prefers shoulder_y > shoulder_z > angle > tau.')
    ap.add_argument('--debug-dump-prefix', type=str, default='', help='If set, dump per-part debug CSV: <prefix>_<part>_debug.csv')
    # Optional ROI for detection (frame indices [start, end], inclusive of start, exclusive of end)
    ap.add_argument('--roi-start', type=int, default=None, help='Start frame index to analyze (inclusive).')
    ap.add_argument('--roi-end', type=int, default=None, help='End frame index to analyze (exclusive).')
    # 1RM comparison options
    ap.add_argument('--mmax-all-csv', type=str, default='m_max_all.csv', help='CSV containing m_max (kg) per subject: columns subject_id,wrist_L,wrist_R,...')
    ap.add_argument('--subject-id', type=int, default=None, help='Subject ID to pick wrist_L/wrist_R from m_max_all.csv for 1RM calculation')
    # Tau clamp for noise rejection
    ap.add_argument('--tau-clamp-abs', type=float, default=1000.0, help='Clamp tau_y to +/- this value (N·m). Use 0 to disable.')
    # tau/tau_env detection parameters
    ap.add_argument('--tau-smooth-window', type=int, default=101, help='Moving average window for tau-based detection (odd).')
    ap.add_argument('--tau-env-peak-prom', type=float, default=20.0, help='Prominence-like threshold over median(|tau|) for tau_env peaks.')
    # shoulder_z inputs (optional, for cycle detection only)
    ap.add_argument('--shoulder-z-npy', type=str, default=None, help='Right shoulder Z series .npy (length N).')
    ap.add_argument('--shoulder-z-npy-left', type=str, default=None, help='Left shoulder Z series .npy (length N).')
    ap.add_argument('--shoulder-z-unit', type=str, default='m', choices=['auto','m','cm','mm'], help='Unit of shoulder-Z arrays; will be converted to meters.')
    ap.add_argument('--shoulder-z-smooth-window', type=int, default=31, help='Moving average window (odd) for shoulder-Z smoothing (short).')
    ap.add_argument('--shoulder-z-valley-prom', type=float, default=0.03, help='Minimum valley prominence (m) for shoulder-Z cycle detection.')
    ap.add_argument('--shoulder-z-detrend-window', type=int, default=301, help='Moving average window (odd) for detrending shoulder-Z (long baseline). Use 0 to disable.')
    ap.add_argument('--shoulder-z-max-right', type=float, default=None, help='Noise threshold for Z_R (meters). If --shoulder-z-remove-from-rise, remove data from first rising crossing onward.')
    ap.add_argument('--shoulder-z-max-left', type=float, default=None, help='Noise threshold for Z_L (meters). If --shoulder-z-remove-from-rise, remove data from first rising crossing onward.')
    ap.add_argument('--shoulder-z-remove-from-rise', action='store_true', help='If set, remove samples from the first rising threshold crossing onward (per side).')
    ap.add_argument('--shoulder-z-rise-confirm', type=int, default=3, help='Consecutive frames above threshold required to confirm rising crossing.')
    ap.add_argument('--shoulder-z-smooth-peaks', action='store_true', help='If set, smooth only spike segments above threshold using linear interpolation between boundary points (per side). Thresholds are taken from --shoulder-z-max-right/left.')
    # shoulder_y inputs (optional, for cycle detection only)
    ap.add_argument('--shoulder-y-npy', type=str, default=None, help='Right shoulder Y series .npy (length N, meters).')
    ap.add_argument('--shoulder-y-npy-left', type=str, default=None, help='Left shoulder Y series .npy (length N, meters).')
    ap.add_argument('--shoulder-y-unit', type=str, default='m', choices=['auto','m','cm','mm'], help='Unit of shoulder-Y arrays; will be converted to meters.')
    ap.add_argument('--shoulder-y-smooth-window', type=int, default=31, help='Moving average window (odd) for shoulder-Y smoothing (short).')
    ap.add_argument('--shoulder-y-valley-prom', type=float, default=0.03, help='Minimum valley prominence (m) for shoulder-Y cycle detection.')
    ap.add_argument('--shoulder-y-detrend-window', type=int, default=301, help='Moving average window (odd) for detrending shoulder-Y (long baseline). Use 0 to disable.')
    ap.add_argument('--shoulder-y-merge', type=str, default='auto', choices=['auto','avg','right','left'], help='If both left/right shoulder-Y are provided, how to merge for detection. auto: use part side; avg: average both; right/left: force select.')
    return ap.parse_args()


def main():
    args = parse_args()
    if args.part != 'both' and (not args.forearm_npy or not args.tau_npy):
        print('[INFO] --forearm-npy と --tau-npy を指定すると実データ計算が可能。今回は閾値のみ計算するモード。')
    with open(args.mmax_json, 'r', encoding='utf-8') as f:
        m_max_map = json.load(f)
    m1_map = compute_m1_map(args.body_mass)
    thr_map = compute_energy_thresholds(m_max_map, m1_map, args.r_x)
    if args.mode == 'per-cycle':
        # Per-cycle path handled after threshold map computed below (we still need thr_map)
        pass

    if args.part != 'both' and args.mode != 'per-cycle':
        if args.part not in thr_map:
            raise ValueError(f'part {args.part} not in m_max map.')
        E_low, E_high = thr_map[args.part]
        print(f'[THRESH] {args.part}: E_low={E_low:.3f} J  E_high={E_high:.3f} J')
        if args.forearm_npy and args.tau_npy and os.path.exists(args.forearm_npy) and os.path.exists(args.tau_npy):
            forearm_vecs = np.load(args.forearm_npy)
            tau_y = np.load(args.tau_npy)
            res = compute_wrist_energy(forearm_vecs, tau_y, args.dt, args.offset, args.clip_dtheta, E_low, E_high)
            if res.E_cum.size == 0:
                print('[WARN] Not enough samples to compute energy.')
                return
            ratio_low = res.E_cum / (E_low + 1e-12)
            ratio_high = res.E_cum / (E_high + 1e-12)
            with open(args.out, 'w', newline='', encoding='utf-8') as fw:
                w = csv.writer(fw)
                w.writerow(['frame','part','dtheta_rad','tau_y','work_inc_J','E_cum_J','ratio_low','ratio_high','E_low','E_high'])
                for i in range(len(res.E_cum)):
                    w.writerow([i, args.part, res.dtheta[i] if i < len(res.dtheta) else '', res.tau[i] if i < len(res.tau) else '', res.work_inc[i] if i < len(res.work_inc) else '', res.E_cum[i], ratio_low[i], ratio_high[i], E_low, E_high])
            print(f'[OUT] Wrote {args.out}  final_E={res.E_cum[-1]:.3f}  ratio_low={ratio_low[-1]:.2f}  ratio_high={ratio_high[-1]:.2f}')
        else:
            print('[INFO] データファイル未指定のため閾値計算のみ。')
        return
    # Bilateral mode (per-frame)
    need_R = args.forearm_npy and args.tau_npy and os.path.exists(args.forearm_npy) and os.path.exists(args.tau_npy)
    need_L = args.forearm_npy_left and args.tau_npy_left and os.path.exists(args.forearm_npy_left) and os.path.exists(args.tau_npy_left)
    if args.mode != 'per-cycle' and not (need_R or need_L):
        print('[INFO] bilateral指定だが入力ファイルが不足: 閾値一覧のみ')
        for p, (el, eh) in thr_map.items():
            print(f'[THRESH] {p}: E_low={el:.3f} E_high={eh:.3f}')
        return
    if args.mode == 'per-frame':
        rows = []
        header = ['frame','part','dtheta_rad','tau_y','work_inc_J','E_cum_J','ratio_low','ratio_high','E_low','E_high']
        if need_R:
            ElR, EhR = thr_map['wrist_R']
            fr = np.load(args.forearm_npy)
            tr = np.load(args.tau_npy)
            if args.tau_clamp_abs and args.tau_clamp_abs > 0:
                tr = np.clip(tr, -args.tau_clamp_abs, args.tau_clamp_abs)
            resR = compute_wrist_energy(fr, tr, args.dt, args.offset, args.clip_dtheta, ElR, EhR)
            ratio_low_R = resR.E_cum / (ElR + 1e-12)
            ratio_high_R = resR.E_cum / (EhR + 1e-12)
            for i in range(len(resR.E_cum)):
                rows.append([i, 'wrist_R', resR.dtheta[i] if i < len(resR.dtheta) else '', resR.tau[i] if i < len(resR.tau) else '', resR.work_inc[i] if i < len(resR.work_inc) else '', resR.E_cum[i], ratio_low_R[i], ratio_high_R[i], ElR, EhR])
        if need_L:
            ElL, EhL = thr_map['wrist_L']
            fl = np.load(args.forearm_npy_left)
            tl = np.load(args.tau_npy_left)
            if args.tau_clamp_abs and args.tau_clamp_abs > 0:
                tl = np.clip(tl, -args.tau_clamp_abs, args.tau_clamp_abs)
            resL = compute_wrist_energy(fl, tl, args.dt, args.offset, args.clip_dtheta, ElL, EhL)
            ratio_low_L = resL.E_cum / (ElL + 1e-12)
            ratio_high_L = resL.E_cum / (EhL + 1e-12)
            for i in range(len(resL.E_cum)):
                rows.append([i, 'wrist_L', resL.dtheta[i] if i < len(resL.dtheta) else '', resL.tau[i] if i < len(resL.tau) else '', resL.work_inc[i] if i < len(resL.work_inc) else '', resL.E_cum[i], ratio_low_L[i], ratio_high_L[i], ElL, EhL])
        with open(args.out, 'w', newline='', encoding='utf-8') as fw:
            wcsv = csv.writer(fw)
            wcsv.writerow(header)
            for r in rows:
                wcsv.writerow(r)
        print(f'[OUT] bilateral CSV -> {args.out} (rows={len(rows)})')
        return

    # ---------- Per-cycle mode implementation ----------
    if args.mode == 'per-cycle':
        # helper: load 1RM values per wrist side if subject specified
        rm1_map: Dict[str, float] = {}
        if args.subject_id is not None and args.mmax_all_csv and os.path.exists(args.mmax_all_csv):
            with open(args.mmax_all_csv, 'r', encoding='utf-8') as fcsv:
                rdr = csv.DictReader(fcsv)
                for row in rdr:
                    sid_str = row.get('subject_id')
                    try:
                        sid = int(sid_str) if sid_str is not None else None
                    except ValueError:
                        sid = None
                    if sid is None or sid != int(args.subject_id):
                        continue
                    # parse floats safely
                    valL = row.get('wrist_L')
                    valR = row.get('wrist_R')
                    try:
                        mL = float(valL) if valL is not None else float('nan')
                    except ValueError:
                        mL = float('nan')
                    try:
                        mR = float(valR) if valR is not None else float('nan')
                    except ValueError:
                        mR = float('nan')
                    base = 6.225e-3 * float(args.body_mass)
                    if np.isfinite(mR):
                        rm1_map['wrist_R'] = (base + mR) * 5.01
                    if np.isfinite(mL):
                        rm1_map['wrist_L'] = (base + mL) * 5.01
                    break

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

        def detect_cycles_peak_valley(signal: np.ndarray, min_amp: float, min_len: int):
            if len(signal) < min_len*2:
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
                t1, _i1 = extrema[j]  # _i1 retained for readability
                _ = _i1  # mark as used
                t2, i2 = extrema[j+1]
                if t0 == t2 or t0 == t1 or t1 == t2:
                    continue
                start = i0
                end = i2
                if end - start + 1 < min_len:
                    continue
                seg = t[start:end+1]
                amp = float(np.max(seg) - np.min(seg))
                if amp < min_amp:
                    continue
                cycles.append((start, end, amp))
            return cycles

        def _smooth_mavg(x: np.ndarray, w: int) -> np.ndarray:
            if w is None or w <= 1:
                return x
            w = int(w)
            if w % 2 == 0:
                w += 1
            if len(x) >= w:
                k = np.ones(w)/float(w)
                return np.convolve(x, k, mode='same')
            return x

        def detect_cycles_tau(tau: np.ndarray, min_amp: float, min_len: int):
            # Use torque sign changes and envelope.
            if len(tau) < min_len*2:
                return []
            t = np.asarray(tau)
            # smoothing to reduce noise
            t_s = _smooth_mavg(t, max(5, args.tau_smooth_window))
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
                if amp < min_amp * 0.5:  # torque amplitude threshold looser
                    continue
                cycles.append((a, b, amp))
            return cycles

        def detect_cycles_tau_env(tau: np.ndarray, min_amp: float, min_len: int, smooth_window: int = 101, peak_prom: float = 20.0):
            """Detect cycles using peaks of |tau| envelope. Each adjacent peak pair forms a cycle window."""
            if len(tau) < min_len*2:
                return []
            t = np.asarray(tau)
            env = np.abs(t)
            env_s = _smooth_mavg(env, smooth_window)
            med = float(np.median(env_s))
            thr = med + peak_prom
            peaks = []
            for i in range(1, len(env_s)-1):
                if env_s[i] > env_s[i-1] and env_s[i] >= env_s[i+1] and env_s[i] >= thr:
                    if peaks and i - peaks[-1] < min_len:
                        # keep higher peak within refractory
                        if env_s[i] > env_s[peaks[-1]]:
                            peaks[-1] = i
                    else:
                        peaks.append(i)
            cycles = []
            for a, b in zip(peaks, peaks[1:]):
                if b - a + 1 < min_len:
                    continue
                seg = t[a:b+1]
                amp = float(np.max(seg) - np.min(seg))
                if amp < min_amp:
                    continue
                cycles.append((a, b, amp))
            return cycles

        def detect_cycles_shoulder_z(z: np.ndarray, min_amp: float, min_len: int, smooth_window: int, valley_prom: float, detrend_window: int):
            """Detect cycles from shoulder Z using valley-to-valley segmentation with prominence.
            - Smooth series with moving average of given window
            - Find local minima (valleys)
            - Keep valleys whose prominence (min(left_max - val, right_max - val)) >= valley_prom
            - Enforce refractory distance >= min_len between valleys (keep deeper valley)
            - Form cycles as adjacent accepted valleys with amplitude>=min_amp and length>=min_len
            """
            if z is None or len(z) < max(min_len*2, 5):
                return []
            t = np.asarray(z, dtype=float)
            # smoothing (short)
            w = int(smooth_window) if smooth_window and smooth_window > 1 else 0
            if w > 0:
                if w % 2 == 0:
                    w += 1
                if len(t) >= w:
                    k = np.ones(w)/float(w)
                    t = np.convolve(t, k, mode='same')
            # detrend (long baseline)
            dw = int(detrend_window) if detrend_window and detrend_window > 1 else 0
            if dw > 0:
                if dw % 2 == 0:
                    dw += 1
                if len(t) >= dw:
                    kb = np.ones(dw)/float(dw)
                    baseline = np.convolve(t, kb, mode='same')
                    t = t - baseline
                else:
                    # fallback: subtract global median
                    t = t - float(np.median(t))
            # local minima candidates
            candidates = []
            for i in range(1, len(t)-1):
                if t[i] <= t[i-1] and t[i] <= t[i+1]:
                    candidates.append(i)
            if not candidates:
                return []
            # compute prominence for each candidate using local left/right maxima search
            def local_max_left(idx: int):
                m = t[idx]
                j = idx - 1
                while j > 0 and t[j] <= t[j-1]:
                    j -= 1
                # climb to left max
                while j > 0 and t[j] >= t[j-1]:
                    j -= 1
                # find actual local max around j..idx
                left = np.max(t[max(0, j):idx]) if idx - max(0, j) > 0 else t[idx]
                return float(left)
            def local_max_right(idx: int):
                m = t[idx]
                j = idx + 1
                while j < len(t)-1 and t[j] <= t[j+1]:
                    j += 1
                while j < len(t)-1 and t[j] >= t[j+1]:
                    j += 1
                right = np.max(t[idx+1:min(len(t), j+1)]) if min(len(t), j+1) - (idx+1) > 0 else t[idx]
                return float(right)
            scored = []
            for i in candidates:
                left_max = local_max_left(i)
                right_max = local_max_right(i)
                prom = min(left_max - t[i], right_max - t[i])
                scored.append((i, prom))
            # keep those above prominence
            filt = [i for i, p in scored if p >= valley_prom]
            if not filt:
                return []
            # enforce refractory distance >= min_len; keep deeper valleys
            accepted = []
            for i in filt:
                if not accepted:
                    accepted.append(i)
                else:
                    if i - accepted[-1] < min_len:
                        # keep lower (deeper) valley
                        if t[i] < t[accepted[-1]]:
                            accepted[-1] = i
                    else:
                        accepted.append(i)
            cycles = []
            for a, b in zip(accepted, accepted[1:]):
                if b - a + 1 < min_len:
                    continue
                seg = t[a:b+1]
                amp = float(np.max(seg) - np.min(seg))
                if amp < min_amp:
                    continue
                cycles.append((a, b, amp))
            return cycles

        def cycle_energy(vecs: np.ndarray, tau: np.ndarray, E_low: float, E_high: float, offset: int, clip_dth: float, start: int, end: int):
            sub_v = vecs[start:end+1]
            sub_tau = tau[start:end+1]
            # reuse compute_wrist_energy on slice then take final E_cum
            res = compute_wrist_energy(sub_v, sub_tau, args.dt, offset, clip_dth, E_low, E_high)
            E = float(res.E_cum[-1]) if res.E_cum.size else 0.0
            return E

        out_rows = []
        header = ['cycle_index','part','start_frame','end_frame','frames','duration_s','amp_rad','E_cycle_J','ratio_low','ratio_high','E_low','E_high','RM1_value','ratio_1RM_percent']
        parts_to_do = []
        if args.part == 'both':
            if need_R: parts_to_do.append('wrist_R')
            if need_L: parts_to_do.append('wrist_L')
        else:
            parts_to_do.append(args.part)
        cycle_index_map = {p:0 for p in parts_to_do}
        for part in parts_to_do:
            if part == 'wrist_R':
                vec_path = args.forearm_npy
                tau_path = args.tau_npy
            else:
                vec_path = args.forearm_npy_left
                tau_path = args.tau_npy_left
            if not (vec_path and tau_path and os.path.exists(vec_path) and os.path.exists(tau_path)):
                print(f'[WARN] {part}: 必要ファイルが存在しないためスキップ')
                continue
            vecs = np.load(vec_path)
            tau = np.load(tau_path)
            if args.tau_clamp_abs and args.tau_clamp_abs > 0:
                tau = np.clip(tau, -args.tau_clamp_abs, args.tau_clamp_abs)
            # sanitize non-finite torque samples before detection/integration
            if np.isnan(tau).any() or ~np.isfinite(tau).all():
                tau = np.nan_to_num(tau, nan=0.0, posinf=args.tau_clamp_abs if args.tau_clamp_abs else 0.0, neginf=-(args.tau_clamp_abs if args.tau_clamp_abs else 0.0))
            # Shoulder Z/Y signal if provided
            z_sig = None
            y_sig_r = None
            y_sig_l = None
            if part == 'wrist_R' and args.shoulder_z_npy and os.path.exists(args.shoulder_z_npy):
                try:
                    z_sig = np.load(args.shoulder_z_npy)
                except Exception:
                    z_sig = None
            if part == 'wrist_L' and args.shoulder_z_npy_left and os.path.exists(args.shoulder_z_npy_left):
                try:
                    z_sig = np.load(args.shoulder_z_npy_left)
                except Exception:
                    z_sig = None
            # Load shoulder Y if provided
            if args.shoulder_y_npy and os.path.exists(args.shoulder_y_npy):
                try:
                    y_sig_r = np.load(args.shoulder_y_npy)
                except Exception:
                    y_sig_r = None
            if args.shoulder_y_npy_left and os.path.exists(args.shoulder_y_npy_left):
                try:
                    y_sig_l = np.load(args.shoulder_y_npy_left)
                except Exception:
                    y_sig_l = None
            # Truncate all signals to common length and apply ROI consistently
            cand_lengths = [len(vecs), len(tau)]
            if z_sig is not None:
                cand_lengths.append(len(z_sig))
            if y_sig_r is not None:
                cand_lengths.append(len(y_sig_r))
            if y_sig_l is not None:
                cand_lengths.append(len(y_sig_l))
            N = min(cand_lengths) if cand_lengths else 0
            vecs = vecs[:N]
            tau = tau[:N]
            if z_sig is not None:
                z_sig = z_sig[:N]
                # scale to meters if needed
                def _unit_scale_to_m(unit: str, series: np.ndarray) -> float:
                    if unit == 'm':
                        return 1.0
                    if unit == 'cm':
                        return 0.01
                    if unit == 'mm':
                        return 0.001
                    # auto detect: use median absolute value heuristic
                    med = float(np.nanmedian(np.abs(series))) if series.size else 0.0
                    if med > 10.0:
                        return 0.001  # mm -> m
                    if med > 1.0:
                        return 0.01   # cm -> m
                    return 1.0        # assume meters
                scale_z = _unit_scale_to_m(args.shoulder_z_unit, z_sig)
                if args.debug_cycles:
                    print(f"[DEBUG] {part} shoulder-Z unit scale -> {scale_z}")
                if scale_z != 1.0:
                    z_sig = z_sig * scale_z
            # scale shoulder-Y to meters
            if y_sig_r is not None:
                def _unit_scale_to_m_y(unit: str, series: np.ndarray) -> float:
                    if unit == 'm':
                        return 1.0
                    if unit == 'cm':
                        return 0.01
                    if unit == 'mm':
                        return 0.001
                    med = float(np.nanmedian(np.abs(series))) if series.size else 0.0
                    if med > 10.0:
                        return 0.001
                    if med > 1.0:
                        return 0.01
                    return 1.0
                scale_y_r = _unit_scale_to_m_y(args.shoulder_y_unit, y_sig_r[:N])
                if scale_y_r != 1.0:
                    y_sig_r = y_sig_r[:N] * scale_y_r
                else:
                    y_sig_r = y_sig_r[:N]
            if y_sig_l is not None:
                scale_y_l = _unit_scale_to_m_y(args.shoulder_y_unit, y_sig_l[:N])
                if scale_y_l != 1.0:
                    y_sig_l = y_sig_l[:N] * scale_y_l
                else:
                    y_sig_l = y_sig_l[:N]
            # optional ROI crop (apply same [s:e] to all)
            if args.roi_start is not None or args.roi_end is not None:
                s = max(0, int(args.roi_start)) if args.roi_start is not None else 0
                e = min(N, int(args.roi_end)) if args.roi_end is not None else N
                if s < e:
                    vecs = vecs[s:e]
                    tau = tau[s:e]
                    if z_sig is not None:
                        z_sig = z_sig[s:e]
                    if y_sig_r is not None:
                        y_sig_r = y_sig_r[s:e]
                    if y_sig_l is not None:
                        y_sig_l = y_sig_l[s:e]
            # optional spike smoothing: replace segments above threshold by linear interpolation between boundaries
            def _smooth_spike_segments(x: np.ndarray, thr: float) -> tuple[np.ndarray, int]:
                if x is None or thr is None:
                    return x, 0
                y = x.copy()
                above = y > thr
                count = 0
                i = 0
                n = len(y)
                while i < n:
                    if above[i]:
                        j = i
                        while j < n and above[j]:
                            j += 1
                        # segment is [i, j-1]
                        left = i - 1
                        right = j
                        if left >= 0 and right < n:
                            v0 = y[left]
                            v1 = y[right]
                            seg_len = right - left - 1
                            if seg_len > 0:
                                y[left+1:right] = np.linspace(v0, v1, seg_len+2)[1:-1]
                                count += 1
                        elif left >= 0 and right >= n:
                            # tail segment: hold last valid value
                            y[left+1:] = y[left]
                            count += 1
                        elif left < 0 and right < n:
                            # head segment: hold right boundary value
                            y[:right] = y[right]
                            count += 1
                        i = j
                    else:
                        i += 1
                return y, count

            if args.shoulder_z_smooth_peaks and z_sig is not None:
                thr = args.shoulder_z_max_right if part == 'wrist_R' else args.shoulder_z_max_left
                if thr is not None:
                    z_new, nseg = _smooth_spike_segments(z_sig, float(thr))
                    if args.debug_cycles:
                        print(f"[DEBUG] {part} smooth-peaks: smoothed {nseg} segments (thr={thr})")
                    z_sig = z_new

            # optional noise removal: cut from rising threshold crossing
            def _find_rising_index(x: np.ndarray, thr: float, confirm: int) -> int | None:
                if x is None or thr is None:
                    return None
                k = max(1, int(confirm))
                for i in range(1, len(x)):
                    if x[i] > thr and x[i-1] <= thr:
                        if i + k - 1 < len(x) and np.all(x[i:i+k] > thr):
                            return i
                return None
            if args.shoulder_z_remove_from_rise and z_sig is not None:
                thr = args.shoulder_z_max_right if part == 'wrist_R' else args.shoulder_z_max_left
                if thr is not None:
                    idx = _find_rising_index(z_sig, float(thr), args.shoulder_z_rise_confirm)
                    if idx is not None and idx > 0:
                        if args.debug_cycles:
                            print(f"[DEBUG] {part} remove-from-rise: cutting at {idx} (thr={thr})")
                        vecs = vecs[:idx]
                        tau = tau[:idx]
                        z_sig = z_sig[:idx]
                        N = len(vecs)
            # Build candidate signals depending on mode (after final cropping)
            theta_ref = reconstruct_theta_angle_ref(vecs)
            theta_diff = reconstruct_theta_angle_diff(vecs, args.clip_dtheta)
            cycles = []
            detect_mode_used = None
            def _log(msg):
                if args.debug_cycles:
                    print(msg)
            modes_to_try = []
            if args.cycle_detect_mode == 'auto':
                # Prefer shoulder_y > shoulder_z > angles > torque
                modes_to_try = ['shoulder_y','shoulder_z','angle_ref','angle_diff','tau','tau_env']
            else:
                modes_to_try = [args.cycle_detect_mode]
            for mode in modes_to_try:
                if mode == 'angle_ref':
                    cycles = detect_cycles_peak_valley(theta_ref, args.cycle_min_amp, args.cycle_min_length)
                elif mode == 'angle_diff':
                    cycles = detect_cycles_peak_valley(theta_diff, args.cycle_min_amp, args.cycle_min_length)
                elif mode == 'tau':
                    cycles = detect_cycles_tau(tau, args.cycle_min_amp, args.cycle_min_length)
                elif mode == 'tau_env':
                    cycles = detect_cycles_tau_env(tau, args.cycle_min_amp, args.cycle_min_length, smooth_window=args.tau_smooth_window, peak_prom=args.tau_env_peak_prom)
                elif mode == 'shoulder_z':
                    if z_sig is None:
                        cycles = []
                    else:
                        cycles = detect_cycles_shoulder_z(z_sig, args.cycle_min_amp, args.cycle_min_length, args.shoulder_z_smooth_window, args.shoulder_z_valley_prom, args.shoulder_z_detrend_window)
                elif mode == 'shoulder_y':
                    # Build detection signal from shoulder-Y per settings
                    y_sig = None
                    if y_sig_r is not None or y_sig_l is not None:
                        if args.shoulder_y_merge == 'avg' and y_sig_r is not None and y_sig_l is not None:
                            y_sig = 0.5 * (y_sig_r + y_sig_l)
                        elif args.shoulder_y_merge == 'right' and y_sig_r is not None:
                            y_sig = y_sig_r
                        elif args.shoulder_y_merge == 'left' and y_sig_l is not None:
                            y_sig = y_sig_l
                        else:
                            # auto: use side matching part, fallback to whichever is available
                            if part == 'wrist_R' and y_sig_r is not None:
                                y_sig = y_sig_r
                            elif part == 'wrist_L' and y_sig_l is not None:
                                y_sig = y_sig_l
                            else:
                                y_sig = y_sig_r if y_sig_r is not None else y_sig_l
                    if y_sig is None:
                        cycles = []
                    else:
                        cycles = detect_cycles_shoulder_z(y_sig, args.cycle_min_amp, args.cycle_min_length, args.shoulder_y_smooth_window, args.shoulder_y_valley_prom, args.shoulder_y_detrend_window)
                if cycles:
                    detect_mode_used = mode
                    break
            if not cycles:
                _log(f'[INFO] {part}: cycle not detected (tried {modes_to_try})')
            else:
                _log(f'[INFO] {part}: detected {len(cycles)} cycles using mode={detect_mode_used}')
            if args.debug_cycles and args.debug_dump_prefix:
                dump_path = f"{args.debug_dump_prefix}_{part}_debug.csv"
                with open(dump_path, 'w', newline='', encoding='utf-8') as fwdbg:
                    wdbg = csv.writer(fwdbg)
                    wdbg.writerow(['frame','theta_ref','theta_diff','tau'])
                    for i in range(len(theta_ref)):
                        td = theta_diff[i] if i < len(theta_diff) else ''
                        tv = tau[i] if i < len(tau) else ''
                        wdbg.writerow([i, theta_ref[i], td, tv])
                _log(f'[DEBUG] dumped debug series -> {dump_path}')
            El, Eh = thr_map[part]
            for (start, end, amp) in cycles:
                E = cycle_energy(vecs, tau, El, Eh, args.offset, args.clip_dtheta, start, end)
                ratio_l = E / (El + 1e-12)
                ratio_h = E / (Eh + 1e-12)
                frames = end - start + 1
                dur = frames * args.dt
                cidx = cycle_index_map[part]
                rm1_val = rm1_map.get(part) if rm1_map else None
                ratio_rm1 = (E / (rm1_val + 1e-12) * 100.0) if rm1_val and np.isfinite(rm1_val) and rm1_val != 0 else ''
                out_rows.append([cidx, part, start, end, frames, dur, amp, E, ratio_l, ratio_h, El, Eh, rm1_val if rm1_val else '', ratio_rm1])
                cycle_index_map[part] += 1
        with open(args.out, 'w', newline='', encoding='utf-8') as fw:
            wcsv = csv.writer(fw)
            wcsv.writerow(header)
            for r in out_rows:
                wcsv.writerow(r)
        print(f'[OUT] per-cycle CSV -> {args.out} (cycles={len(out_rows)})')
        return

if __name__ == '__main__':
    main()
