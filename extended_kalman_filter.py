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
    # 門の外の観測を捨てずに、分散を S' = y²/c² に膨らませて取り込む（Huber 型の縮小）。
    # 捨てる方式は、ずれるほど捨て続けて戻れない（設計メモ 欠陥 3）。既存の run_ekf の挙動を
    # 変えないよう既定はオフ。LandmarkEKF の実行時は master_research_code.py がオンにする。
    robust_gate: bool = False
    # 欠測がこの秒数を超えたら外挿をやめて NaN を返し、次の観測で初期化し直す（欠陥 4）。0 は無制限。
    max_gap_s: float = 0.0

    # custom measurement model (optional)
    h_fn: Optional[Callable[[np.ndarray], float]] = None
    h_jac_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None


@dataclass(eq=False)  # holds ndarrays, so the generated __eq__ would raise
class SeriesNoise:
    """Per-series (q_acc, r, gate_std) for LandmarkEKF: one entry per point and axis.

    LandmarkEKF runs n_points*3 independent scalar filters. A single constant cannot fit
    them all: fitting real recordings puts the spread at 432x in r and 205x in q_acc.
    Series are ordered as ``point * 3 + axis``, matching the (n_points, 3) measurement array.

    There is deliberately no h_fn here: LandmarkEKF observes position directly, and a
    measurement model that cannot be set cannot be silently ignored.
    """

    q_acc: np.ndarray
    r: np.ndarray
    gate_std: np.ndarray

    @classmethod
    def uniform(cls, n_series: int, cfg: EKFConfig) -> "SeriesNoise":
        """Same scalar values for every series (what the runtime used before tuning)."""
        ones = np.ones(int(n_series), dtype=float)
        return cls(q_acc=ones * cfg.q_acc, r=ones * cfg.r, gate_std=ones * cfg.gate_std)

    def as_arrays(self, n_series: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = tuple(np.asarray(v, dtype=float).reshape(-1) for v in (self.q_acc, self.r, self.gate_std))
        for name, arr in zip(("q_acc", "r", "gate_std"), arrays):
            if arr.shape != (n_series,):
                raise ValueError(f"{name} must have {n_series} entries, got {arr.shape[0]}")
        return arrays


def constant_acceleration_model(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """State transition F and unit-intensity process noise Q (q_acc = 1) for one step.

    The process is constant acceleration driven by continuous white jerk, so a filter
    uses ``q_acc * Q``. The tuning code (``app/tuning``) calls this same function, so an
    estimated ``q_acc`` means exactly what the runtime filter will do with it.
    """
    dt2 = dt * dt
    dt3 = dt2 * dt
    F = np.array(
        [[1.0, dt, 0.5 * dt2],
         [0.0, 1.0, dt],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    q_unit = np.array(
        [[dt3 * dt2 / 20.0, dt3 * dt / 8.0, dt3 / 6.0],
         [dt3 * dt / 8.0, dt3 / 3.0, dt2 / 2.0],
         [dt3 / 6.0, dt2 / 2.0, dt]],
        dtype=float,
    )
    return F, q_unit


def gap_expired(gap_s, max_gap_s: float):
    """欠測の長さが上限を超えたか。0 以下の上限は無制限。

    欠測の長さは dt の足し上げなので、ちょうど上限のところで丸め誤差に揺れないよう少し余裕を持たせる。
    """
    if max_gap_s <= 0:
        return np.zeros_like(gap_s, dtype=bool) if isinstance(gap_s, np.ndarray) else False
    return gap_s > max_gap_s * (1.0 + 1e-9)


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
        self._gap_s = 0.0  # 連続した欠測の長さ [s]

    def _predict(self, dt: float) -> None:
        F, q_unit = constant_acceleration_model(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.cfg.q_acc * q_unit

    def _measure(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.cfg.h_fn is not None and self.cfg.h_jac_fn is not None:
            z_pred = float(self.cfg.h_fn(x))
            H = np.asarray(self.cfg.h_jac_fn(x), dtype=float).reshape(1, 3)
        else:
            z_pred = float(x[0])
            H = np.array([[1.0, 0.0, 0.0]], dtype=float)
        return z_pred, H

    def _start(self, z: Optional[float]) -> None:
        """観測 ``z`` で状態を初期化する（位置 = z、速度・加速度 0、P = I）。

        初めて観測したとき（``step``）と、外から作り直すとき（``LandmarkEKF.reset_series``）で共有する。
        ``z`` が None なら作った直後の未初期化の状態に戻す（次の観測で初期化する）。
        """
        self.x[:] = 0.0
        self.P = np.eye(3, dtype=float)
        self._gap_s = 0.0
        if z is None:
            self.initialized = False
            return
        self.x[0] = float(z)
        self.initialized = True

    def step(self, z: Optional[float], dt: float) -> Tuple[float, float, float]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not self.initialized:
            if z is None:
                return float("nan"), float("nan"), float("nan")
            self._start(z)
            return float(self.x[0]), float(self.x[1]), float(self.x[2])

        self._predict(dt)
        if z is None:
            self._gap_s += dt
            if gap_expired(self._gap_s, self.cfg.max_gap_s):
                self.initialized = False
                return float("nan"), float("nan"), float("nan")
            return float(self.x[0]), float(self.x[1]), float(self.x[2])
        self._gap_s = 0.0

        z_pred, H = self._measure(self.x)
        y = float(z) - z_pred
        # H P H^T は (1,1) 配列。NumPy 2 は float() に 0 次元以外を渡すと TypeError にする
        HPH = float((H @ self.P @ H.T)[0, 0])
        S = HPH + self.cfg.r
        if S <= 0:
            return float(self.x[0]), float(self.x[1]), float(self.x[2])
        r_eff = self.cfg.r
        if self.cfg.gate_std > 0 and abs(y) > self.cfg.gate_std * np.sqrt(S):
            if not self.cfg.robust_gate:
                return float(self.x[0]), float(self.x[1]), float(self.x[2])
            # 門の外: 正規化イノベーションがちょうど c になるまで観測分散を膨らませて取り込む
            S = y * y / (self.cfg.gate_std * self.cfg.gate_std)
            r_eff = S - HPH

        K = (self.P @ H.T) / S  # (3x1)
        self.x = self.x + (K[:, 0] * y)
        I = np.eye(3, dtype=float)
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K * r_eff * K.T
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
        robust_gate: bool = False,
        max_gap_s: float = 0.0,
    ) -> None:
        if isinstance(cfg, EKFConfig) and (cfg.h_fn is not None or cfg.h_jac_fn is not None):
            raise ValueError(
                "LandmarkEKF observes position directly; h_fn / h_jac_fn are not supported "
                "(the vectorised path used to ignore them silently)"
            )
        self.n_points = int(n_points)
        self.cfg = cfg
        # 門の外の扱い（捨てる／分散を膨らませて取り込む）と、欠測の上限 [s]。EKFConfig の同名の項目を参照
        self.robust_gate = bool(robust_gate)
        self.max_gap_s = float(max_gap_s)
        # 系列（点 × 軸）ごとの (q_acc, r, gate_std)。スカラー設定なら全系列同値にする。
        # 並びは point*3 + axis で、観測の (n_points, 3) を reshape(-1) した順と一致する。
        n_series = self.n_points * 3
        noise = cfg if isinstance(cfg, SeriesNoise) else SeriesNoise.uniform(n_series, cfg)
        self._q_acc, self._r, self._gate_std = noise.as_arrays(n_series)
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
            self._gap = np.zeros(_n, dtype=float)  # 連続した欠測の長さ [s]
            # F/Q は dt にしか依存しないので (dt, F, Q) を1件キャッシュする
            self._fq = None
            # Joseph 形式の (I - KH)。列1,2 は常に単位行列のままなので使い回す
            self._A = np.tile(self._I3, (_n, 1, 1))
            self.filters = None
        else:
            self.filters = [
                [ExtendedKalman1D(self._series_config(i * 3 + j)) for j in range(3)]
                for i in range(self.n_points)
            ]
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

    @property
    def bandpass_enabled(self) -> bool:
        """前処理のバンドパスが実際に有効か（SciPy が無い・帯域が不正なら無効）。"""
        return self._bpf_enabled

    def _series_config(self, k: int) -> EKFConfig:
        """逐次版の系列 k の設定。ベクトル化版と同じ値・同じ門の扱いにする。"""
        return EKFConfig(
            q_acc=float(self._q_acc[k]), r=float(self._r[k]), gate_std=float(self._gate_std[k]),
            robust_gate=self.robust_gate, max_gap_s=self.max_gap_s,
        )

    def set_noise(self, noise) -> None:
        """雑音パラメータ（SeriesNoise か EKFConfig）を差し替える。状態はそのまま引き継ぐ。

        実行時に基準長の比（L_run / L_cal）が分かった時点で、較正値を体格に合わせて掛け直すのに使う。
        """
        n_series = self.n_points * 3
        noise = noise if isinstance(noise, SeriesNoise) else SeriesNoise.uniform(n_series, noise)
        self._q_acc, self._r, self._gate_std = noise.as_arrays(n_series)
        self._fq = None  # Q は q_acc に比例するので作り直す
        if not self.vectorized:
            for i in range(self.n_points):
                for j in range(3):
                    self.filters[i][j].cfg = self._series_config(i * 3 + j)

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

    def _start_rows(self, rows: np.ndarray, z: np.ndarray) -> None:
        """ベクトル化版の系列 ``rows``（(N,) の真偽）を観測 ``z``（(N,)）で初期化する。

        ``ExtendedKalman1D._start`` と同じ値（位置 = z、速度・加速度 0、P = I、欠測の長さ 0）。``z`` が有限でない
        系列は未初期化に戻す（次の観測で初期化する。未初期化の間は位置を使わない）。初めて観測したとき
        （``_step_vectorized`` の 1)。``z`` はどれも有限）と ``reset_series`` で共有する。
        """
        X = self._X
        X[rows] = 0.0
        X[rows, 0] = z[rows]
        self._P[rows] = self._I3
        self._init[rows] = np.isfinite(z[rows])
        self._gap[rows] = 0.0

    def reset_series(self, mask: np.ndarray, z: np.ndarray) -> None:
        """``mask``（(n_points,) の真偽）の点の状態を観測 ``z``（(n_points, 3)）で初期化し直す。

        その点を新しく観測し始めたとき（未初期化の系列に初めて観測が来た ``step``）と同じ状態にする:
        位置 = z、速度・加速度 0、P = I、欠測の長さ 0。雑音パラメータは引き継ぐ。点の中で z が NaN の軸は
        未初期化に戻す（次の観測で初期化する。``step`` はそれまで NaN を返す）。``mask`` の外の点は触らない。

        ``z`` は EKF が追う値として使う。前処理の BPF（``bandpass_enabled``）は通さず、その状態も触らない。
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_points,):
            raise ValueError(f"mask shape must be {(self.n_points,)}, got {mask.shape}")
        z = np.asarray(z, dtype=float)
        if z.shape != (self.n_points, 3):
            raise ValueError(f"z shape must be {(self.n_points, 3)}, got {z.shape}")
        if self.vectorized:
            self._start_rows(np.repeat(mask, 3), z.reshape(-1))
            return
        for i in np.flatnonzero(mask):
            for j in range(3):
                z_val = z[i, j]
                self.filters[i][j]._start(float(z_val) if np.isfinite(z_val) else None)

    def _step_vectorized(self, arr: np.ndarray, dt: float):
        """ExtendedKalman1D.step を (n_points*3) 本まとめて配列演算で実行する。

        逐次版と同一の分岐を再現する:
          - 未初期化 かつ 欠測      -> NaN を返し、未初期化のまま
          - 未初期化 かつ 観測あり  -> x=[z,0,0], P=I で初期化（predict も update もしない）
          - 初期化済み              -> predict。欠測 / S<=0 / ゲート外 なら predict のみ
        """
        X, P, init, gap = self._X, self._P, self._init, self._gap
        z = arr.reshape(-1)
        valid = np.isfinite(z)

        # 1) 今フレームで初期化されるもの
        new = (~init) & valid
        if new.any():
            self._start_rows(new, z)

        # 2) 既に初期化済みだったものだけ predict（今回初期化した分は除く）
        pred = init & ~new
        pi = np.flatnonzero(pred)
        if pi.size:
            fq = self._fq
            if fq is None or fq[0] != dt:
                F, q_unit = constant_acceleration_model(dt)
                # Q は系列ごとに違うので (N,3,3)。添字を付け忘れると形は合って値だけ静かに間違う
                fq = self._fq = (dt, F, self._q_acc[:, None, None] * q_unit)
            _, F, Q = fq
            X[pi] = X[pi] @ F.T
            P[pi] = F @ P[pi] @ F.T + Q[pi]

            # 欠測の上限（秒）。超えたら未初期化に戻し、NaN を返して次の観測でやり直す
            if self.max_gap_s > 0:
                missing = pi[~valid[pi]]
                gap[missing] += dt
                gap[pi[valid[pi]]] = 0.0
                expired = missing[gap_expired(gap[missing], self.max_gap_s)]
                init[expired] = False

            # 3) 観測がある行だけ update（H = [1,0,0] なので行列積は不要）
            ui = pi[valid[pi]]
            if ui.size:
                y = z[ui] - X[ui, 0]
                HPH = P[ui, 0, 0]
                S = HPH + self._r[ui]
                r_eff = self._r[ui].copy()
                ok = S > 0.0
                gate = self._gate_std[ui]
                gated = ok & (gate > 0.0)  # gate_std <= 0 の系列はゲート無効（逐次版と同じ）
                if gated.any():
                    outside = np.zeros_like(ok)
                    outside[gated] = np.abs(y[gated]) > gate[gated] * np.sqrt(S[gated])
                    if self.robust_gate:
                        # 門の外: 正規化イノベーションがちょうど c になるまで観測分散を膨らませる
                        S = np.where(outside, y * y / np.where(outside, gate * gate, 1.0), S)
                        r_eff = np.where(outside, S - HPH, r_eff)
                    else:
                        ok &= ~outside
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
                        + r_eff[ok][:, None, None] * (K[:, :, None] * K[:, None, :])

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
    "SeriesNoise",
    "constant_acceleration_model",
    "gap_expired",
    "run_ekf",
]
