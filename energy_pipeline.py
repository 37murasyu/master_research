"""
Energy (work per cycle) pipeline utilities.
- Environment-tunable LPF and resampling
- Robust angle/torque smoothing and trapezoidal integration (positive/negative split)

Public API:
- compute_cycle_energy_filtered(theta: np.ndarray, tau: np.ndarray, dt_sec: float) -> tuple[float, float, dict]

Environment variables:
- E_FC, E_LPF_ORDER, E_RESAMPLE_N, E_MAX_DTH, E_WLOW, E_WHIGH, E_DEBUG
"""
from __future__ import annotations

import os
import math
from typing import Tuple

import numpy as np

try:
    # Recommended: SciPy filter & interpolation
    from scipy.signal import butter, filtfilt
    from scipy.interpolate import PchipInterpolator
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False
    butter = filtfilt = PchipInterpolator = None  # type: ignore

# ===== Environment-configurable parameters =====
# (Assuming ~5 Hz motion; fc ~1.0–1.5 Hz recommended.)
E_FC = float(os.getenv('E_FC', '1.2'))
E_LPF_ORDER = int(os.getenv('E_LPF_ORDER', '2'))  # 2–4
E_RESAMPLE_N = int(os.getenv('E_RESAMPLE_N', '80'))  # 50–100
E_MAX_DTH = float(os.getenv('E_MAX_DTH', '0.25'))  # per-step angle clamp [rad]
E_WINSOR_PCTL_LOW = float(os.getenv('E_WLOW', '5'))  # torque winsorize low percentile
E_WINSOR_PCTL_HIGH = float(os.getenv('E_WHIGH', '95'))
E_DEBUG = os.getenv('E_DEBUG', '0') in ('1', 'true', 'True')


def _butter_lowpass_filtfilt(x: np.ndarray, fs: float, fc: float, order: int) -> np.ndarray:
    if len(x) < max(8, 3 * order + 1):
        return x.copy()
    if not _SCIPY_OK:
        # Simple moving-average fallback
        k = max(3, min(9, (len(x) // 10) * 2 + 1))
        return np.convolve(x, np.ones(k) / k, mode='same')
    nyq = 0.5 * fs
    wn = min(0.99, max(1e-3, fc / nyq))
    b, a = butter(order, wn, btype='low', analog=False)
    try:
        return filtfilt(b, a, x, method='gust')
    except Exception:
        return filtfilt(b, a, x)


def _interp_uniform(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    # x: time (monotonic), y: values, n: samples
    if len(x) < 2:
        ui = np.linspace(0.0, 1.0, max(2, n))
        base = y[0] if len(y) else 0.0
        return ui, np.full_like(ui, base, dtype=float)
    x0, x1 = float(x[0]), float(x[-1])
    if x1 <= x0:
        x1 = x0 + 1e-6
    ui = np.linspace(0.0, 1.0, n)
    ti = x0 + ui * (x1 - x0)
    if _SCIPY_OK:
        try:
            pchip_fn = PchipInterpolator(x, y, extrapolate=True)
            yi = pchip_fn(ti)
            return ui, yi
        except Exception:
            pass
    yi = np.interp(ti, x, y)
    return ui, yi


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    # Stable angle [rad] between two 3D vectors
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    dot = float(np.dot(a, b)) / (na * nb)
    dot = np.clip(dot, -1.0, 1.0)
    crossn = np.linalg.norm(np.cross(a / na, b / nb))
    return math.atan2(crossn, dot)

# Backwards alias (if any external still refers to the old private name)
_angle_between = angle_between

# Optional explicit exports
__all__ = [
    'compute_cycle_energy_filtered',
    'angle_between',
]


def _winsorize(y: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    if len(y) < 4:
        return y.copy()
    lo = np.percentile(y, p_low)
    hi = np.percentile(y, p_high)
    return np.clip(y, lo, hi)


def compute_cycle_energy_filtered(theta: np.ndarray, tau: np.ndarray, dt_sec: float) -> tuple[float, float, dict]:
    """Filter theta/tau, resample, and compute trapezoidal positive/negative work.
    Returns: (E_pos, E_neg, info)
    """
    n = len(theta)
    if n < 3:
        return 0.0, 0.0, {'status': 'too_few', 'n': n}
    fs = 1.0 / max(1e-6, dt_sec)
    # 1) unwrap + LPF
    th = np.unwrap(np.asarray(theta, dtype=np.float64))
    th_f = _butter_lowpass_filtfilt(th, fs, E_FC, E_LPF_ORDER)
    tau_f = _butter_lowpass_filtfilt(np.asarray(tau, dtype=np.float64), fs, E_FC, E_LPF_ORDER)
    # 2) time normalize (0..T -> 0..1)
    t = np.arange(n, dtype=np.float64) * dt_sec
    ui, th_u = _interp_uniform(t, th_f, E_RESAMPLE_N)
    _, tau_u = _interp_uniform(t, tau_f, E_RESAMPLE_N)
    # winsorize outliers
    tau_u = _winsorize(tau_u, E_WINSOR_PCTL_LOW, E_WINSOR_PCTL_HIGH)
    dth = np.diff(th_u)
    dth = np.clip(dth, -E_MAX_DTH, E_MAX_DTH)
    # 3) trapezoidal integration (split +/−)
    tau_mid = 0.5 * (tau_u[1:] + tau_u[:-1])
    contrib = tau_mid * dth
    e_pos = float(np.sum(np.maximum(contrib, 0.0)))
    e_neg = float(np.sum(np.maximum(-contrib, 0.0)))
    info = {'status': 'ok', 'n_u': int(len(th_u))}
    if len(th_u) < 30:
        info['low_conf'] = True
    if E_DEBUG:
        print(f"[EPIPE] n={n}->{len(th_u)} e+={e_pos:.4f} e-={e_neg:.4f}")
    return e_pos, e_neg, info
