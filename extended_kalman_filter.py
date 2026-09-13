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

try:
    # Only LandmarkEKF's optional band-pass prefilter needs SciPy; the EKF itself does not.
    from scipy.signal import butter, lfilter, lfilter_zi
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


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
        if (cfg.h_fn is None) != (cfg.h_jac_fn is None):
            missing = "h_jac_fn" if cfg.h_jac_fn is None else "h_fn"
            raise ValueError(
                f"h_fn and h_jac_fn must be given together (missing {missing}); "
                "otherwise the measurement model silently falls back to identity"
            )
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
        # H P H^T は (1,1) 配列。NumPy 2 は float() に 0 次元以外を渡すと TypeError にする
        S = float((H @ self.P @ H.T)[0, 0] + self.cfg.r)
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


class LandmarkEKF:
    """Streaming EKF (per-axis) with optional band-pass prefilter for 3D landmarks."""

    def __init__(
        self,
        n_points: int,
        fs: float,
        cfg: EKFConfig,
        bpf_low: float = 0.0,
        bpf_high: float = 0.0,
        bpf_order: int = 2,
        vectorized: bool = True,
    ) -> None:
        self.n_points = int(n_points)
        self.cfg = cfg
        # ベクトル化パス: (n_points*3) 本の独立スカラーEKFを配列で保持
        #   _X: (N,3) 状態 [x,v,a] / _P: (N,3,3) 共分散 / _init: (N,) 初期化済みフラグ
        # 逐次パス（従来）: ExtendedKalman1D のリスト
        self.vectorized = bool(vectorized)
        if self.vectorized:
            _n = self.n_points * 3
            self._I3 = np.eye(3, dtype=float)
            self._X = np.zeros((_n, 3), dtype=float)
            self._P = np.tile(self._I3, (_n, 1, 1))
            self._init = np.zeros(_n, dtype=bool)
            # F/Q は dt にしか依存しないので (dt, F, Q) を1件キャッシュする
            self._fq = None
            # Joseph 形式の (I - KH)。列1,2 は常に単位行列のままなので使い回す
            self._A = np.tile(self._I3, (_n, 1, 1))
            self.filters = None
        else:
            self.filters = [[ExtendedKalman1D(cfg) for _ in range(3)] for _ in range(self.n_points)]
        # streaming band-pass (optional)
        self._bpf_enabled = False
        self._bpf_b = None
        self._bpf_a = None
        self._bpf_state = None
        if _SCIPY_OK and bpf_low > 0 and bpf_high > 0 and bpf_high > bpf_low:
            nyq = 0.5 * fs
            low = max(1e-3, bpf_low / nyq)
            high = min(0.99, bpf_high / nyq)
            if low < high:
                self._bpf_b, self._bpf_a = butter(bpf_order, [low, high], btype='band')
                zi = lfilter_zi(self._bpf_b, self._bpf_a)
                self._bpf_state = np.tile(zi, (self.n_points, 3, 1))
                self._bpf_enabled = True

    def _apply_bpf(self, arr: np.ndarray) -> np.ndarray:
        if not self._bpf_enabled:
            return arr
        out = np.array(arr, dtype=float, copy=True)
        for i in range(self.n_points):
            for j in range(3):
                x = arr[i, j]
                if not np.isfinite(x):
                    continue
                y, zf = lfilter(self._bpf_b, self._bpf_a, [x], zi=self._bpf_state[i, j])
                self._bpf_state[i, j] = zf
                out[i, j] = y[-1]
        return out

    def _step_vectorized(self, arr: np.ndarray, dt: float):
        """ExtendedKalman1D.step を (n_points*3) 本まとめて配列演算で実行する。

        逐次版と同一の分岐を再現する:
          - 未初期化 かつ 欠測      -> NaN を返し、未初期化のまま
          - 未初期化 かつ 観測あり  -> x=[z,0,0], P=I で初期化（predict も update もしない）
          - 初期化済み              -> predict。欠測 / S<=0 / ゲート外 なら predict のみ
        """
        cfg = self.cfg
        X, P, init = self._X, self._P, self._init
        z = arr.reshape(-1)
        valid = np.isfinite(z)

        # 1) 今フレームで初期化されるもの
        new = (~init) & valid
        if new.any():
            X[new] = 0.0
            X[new, 0] = z[new]
            P[new] = self._I3
            init[new] = True

        # 2) 既に初期化済みだったものだけ predict（今回初期化した分は除く）
        pred = init & ~new
        pi = np.flatnonzero(pred)
        if pi.size:
            fq = self._fq
            if fq is None or fq[0] != dt:
                dt2 = dt * dt
                dt3 = dt2 * dt
                F = np.array([[1.0, dt, 0.5 * dt2],
                              [0.0, 1.0, dt],
                              [0.0, 0.0, 1.0]], dtype=float)
                Q = cfg.q_acc * np.array(
                    [[dt3 * dt2 / 20.0, dt3 * dt / 8.0, dt3 / 6.0],
                     [dt3 * dt / 8.0, dt3 / 3.0, dt2 / 2.0],
                     [dt3 / 6.0, dt2 / 2.0, dt]], dtype=float)
                fq = self._fq = (dt, F, Q)
            _, F, Q = fq
            X[pi] = X[pi] @ F.T
            P[pi] = F @ P[pi] @ F.T + Q

            # 3) 観測がある行だけ update（H = [1,0,0] なので行列積は不要）
            ui = pi[valid[pi]]
            if ui.size:
                y = z[ui] - X[ui, 0]
                S = P[ui, 0, 0] + cfg.r
                ok = S > 0.0
                if cfg.gate_std > 0:
                    ok[ok] &= np.abs(y[ok]) <= cfg.gate_std * np.sqrt(S[ok])
                si = ui[ok]
                if si.size:
                    Ps = P[si]
                    K = Ps[:, :, 0] / S[ok][:, None]          # (m,3) = P H^T / S
                    X[si] = X[si] + K * y[ok][:, None]
                    # Joseph 形式: (I-KH) P (I-KH)^T + K r K^T
                    # H = [1,0,0] なので I-KH は列0だけが単位行列と異なる。
                    # 事前確保した _A の列1,2 は単位行列のまま使い回し、列0だけ書き換える
                    A = self._A[:si.size]
                    np.negative(K, out=A[:, :, 0])
                    A[:, 0, 0] += 1.0
                    P[si] = A @ Ps @ np.transpose(A, (0, 2, 1)) \
                        + cfg.r * (K[:, :, None] * K[:, None, :])

        # 未初期化のものは NaN（逐次版と同じ）
        out = X if init.all() else np.where(init[:, None], X, np.nan)
        n = self.n_points
        return (out[:, 0].reshape(n, 3).copy(),
                out[:, 1].reshape(n, 3).copy(),
                out[:, 2].reshape(n, 3).copy())

    def step(self, meas: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if meas.shape != (self.n_points, 3):
            raise ValueError(f"meas shape must be {(self.n_points, 3)}, got {meas.shape}")
        if dt <= 0:
            dt = 1e-3
        arr = np.asarray(meas, dtype=float)
        arr = self._apply_bpf(arr)
        if self.vectorized:
            return self._step_vectorized(arr, dt)
        pos = np.zeros_like(arr)
        vel = np.zeros_like(arr)
        acc = np.zeros_like(arr)
        for i in range(self.n_points):
            for j in range(3):
                z_val = arr[i, j]
                meas_val = None if not np.isfinite(z_val) else float(z_val)
                px, pv, pa = self.filters[i][j].step(meas_val, dt)
                pos[i, j], vel[i, j], acc[i, j] = px, pv, pa
        return pos, vel, acc


__all__ = [
    "EKFConfig",
    "ExtendedKalman1D",
    "ExtendedKalmanND",
    "LandmarkEKF",
    "run_ekf",
]
