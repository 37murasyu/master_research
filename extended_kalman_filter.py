"""Extended Kalman filter for 1D axes (position, velocity, acceleration).

- State: [x, v, a]
- Process: constant-acceleration with continuous white accel noise intensity q_acc.
- Measurement: configurable; defaults to direct position observation.
- Optional gating on innovation to suppress outliers.
- Designed to be drop-in with existing band-pass filters: you can pass a prefilter
  callable (time_s, data) -> filtered_data before EKF updates.

Example (numpy arrays):
    cfg = EKFConfig(q_acc=1e-3, r=5e-4, gate_std=3.0)
    pos, vel, acc = run_ekf(data, time_s, cfg)

    # with band-pass filter
    def my_bpf(t, d):
        return apply_bandpass(d, fs=30.0)  # user-provided
    pos, vel, acc = run_ekf(data, time_s, cfg, prefilter=my_bpf)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EKFConfig:
    q_acc: float = 1e-3  # continuous accel noise intensity
    r: float = 1e-3      # measurement noise variance
    gate_std: float = 3.0  # outlier gate in sigma (<=0 to disable)

    # custom measurement model (optional)
    h_fn: Optional[Callable[[np.ndarray], float]] = None
    h_jac_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


class ExtendedKalman1D:
    """1D EKF with constant-acceleration process and configurable measurement."""

    def __init__(self, cfg: EKFConfig):
        self.cfg = cfg
        self.x = np.zeros(3, dtype=float)  # [x, v, a]
        self.P = np.eye(3, dtype=float)
        self.initialized = False

    def _predict(self, dt: float) -> None:
        dt2 = dt * dt
        dt3 = dt2 * dt
        F = np.array(
            [[1.0, dt, 0.5 * dt2],
             [0.0, 1.0, dt],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
        q = self.cfg.q_acc
        Q = q * np.array(
            [[dt3 * dt2 / 20.0, dt3 * dt / 8.0, dt3 / 6.0],
             [dt3 * dt / 8.0, dt3 / 3.0, dt2 / 2.0],
             [dt3 / 6.0, dt2 / 2.0, dt]],
            dtype=float,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _measure(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.cfg.h_fn is not None and self.cfg.h_jac_fn is not None:
            z_pred = float(self.cfg.h_fn(x))
            H = np.asarray(self.cfg.h_jac_fn(x), dtype=float).reshape(1, 3)
        else:
            z_pred = float(x[0])
            H = np.array([[1.0, 0.0, 0.0]], dtype=float)
        return z_pred, H

    def step(self, z: Optional[float], dt: float) -> Tuple[float, float, float]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not self.initialized:
            if z is None:
                return float("nan"), float("nan"), float("nan")
            self.x[:] = 0.0
            self.x[0] = float(z)
            self.P = np.eye(3, dtype=float)
            self.initialized = True
            return float(self.x[0]), float(self.x[1]), float(self.x[2])

        self._predict(dt)
        if z is None:
            return float(self.x[0]), float(self.x[1]), float(self.x[2])

        z_pred, H = self._measure(self.x)
        y = float(z) - z_pred
        S = float(H @ self.P @ H.T + self.cfg.r)
        if S <= 0:
            return float(self.x[0]), float(self.x[1]), float(self.x[2])
        if self.cfg.gate_std > 0 and abs(y) > self.cfg.gate_std * np.sqrt(S):
            return float(self.x[0]), float(self.x[1]), float(self.x[2])

        K = (self.P @ H.T) / S  # (3x1)
        self.x = self.x + (K[:, 0] * y)
        I = np.eye(3, dtype=float)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K * self.cfg.r * K.T
        return float(self.x[0]), float(self.x[1]), float(self.x[2])


class ExtendedKalmanND:
    """Apply ExtendedKalman1D independently to each axis."""

    def __init__(self, dim: int, cfg: EKFConfig):
        self.filters = [ExtendedKalman1D(cfg) for _ in range(dim)]

    def step(self, z: Sequence[float] | None, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if z is None:
            meas = None
        else:
            arr = np.asarray(z, dtype=float)
            if arr.shape[0] != len(self.filters):
                raise ValueError("Measurement dimension mismatch")
            meas = arr
        pos = np.zeros(len(self.filters), dtype=float)
        vel = np.zeros(len(self.filters), dtype=float)
        acc = np.zeros(len(self.filters), dtype=float)
        for i, f in enumerate(self.filters):
            z_i = None if meas is None else float(meas[i])
            px, pv, pa = f.step(z_i, dt)
            pos[i], vel[i], acc[i] = px, pv, pa
        return pos, vel, acc


def run_ekf(
    data: np.ndarray,
    time_s: np.ndarray,
    cfg: EKFConfig,
    prefilter: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run EKF per axis over data (shape: n x d).

    Arguments
    ---------
    data : ndarray (n, d)
        Measurements per axis.
    time_s : ndarray (n,)
        Time stamps in seconds (monotonic).
    cfg : EKFConfig
        Filter configuration.
    prefilter : callable, optional
        If provided, called as prefilter(time_s, data) before EKF updates.
        Use this to plug in the existing band-pass filter.

    Returns
    -------
    pos, vel, acc : ndarrays of shape (n, d)
    """
    if data.ndim != 2:
        raise ValueError("data must be 2D (n, d)")
    if time_s.ndim != 1 or time_s.shape[0] != data.shape[0]:
        raise ValueError("time_s must be shape (n,) and aligned with data")
    if prefilter is not None:
        data = np.asarray(prefilter(time_s, data), dtype=float)
        if data.shape != (time_s.shape[0], data.shape[1]):
            raise ValueError("prefilter must return array with same shape as input data")

    n, d = data.shape
    pos = np.zeros_like(data, dtype=float)
    vel = np.zeros_like(data, dtype=float)
    acc = np.zeros_like(data, dtype=float)
    filt = ExtendedKalmanND(d, cfg)

    last_t = float(time_s[0])
    for i in range(n):
        t = float(time_s[i])
        dt = max(1e-6, t - last_t) if i > 0 else 1e-3
        last_t = t
        p, v, a = filt.step(data[i], dt)
        pos[i], vel[i], acc[i] = p, v, a
    return pos, vel, acc


__all__ = [
    "EKFConfig",
    "ExtendedKalman1D",
    "ExtendedKalmanND",
    "run_ekf",
]
