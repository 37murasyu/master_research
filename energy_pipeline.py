"""肘のサイクルエネルギー E± の前処理（USB 経路の本体と同じ計算）。

``master_research_code.py`` に直書きされている次の定義と同じ計算を、混成の経路からも使えるようにしたもの:

- ``compute_cycle_energy_filtered``: 有限値の選別 → unwrap → LPF → 0..1 に正規化して ``resample_n`` 点へ
  再標本化（PCHIP）→ τ を分位で切る → dθ を ±``max_dth`` に制限 → 台形で ∫τdθ を正負に分ける
- ``OnlineF0Estimator``・``fc_scheduler``（本体の ``_fc_scheduler``）: θ の支配周波数 f0 を Welch で推定し、
  fc = k·f0 を [fc_min, fc_max] に切って EMA で追う（適応カットオフ、``E_FC_ADAPTIVE_ON``）
- ``AdaptiveCutoff``: 本体の 3270〜3303 行の数え方（毎フレーム f0 に供給し、``E_FC_UPDATE_HZ`` ごとに更新）

本体と数値が一致することは ``tests/test_energy_pipeline.py`` が本体を AST で読んで確かめる。

設定は ``EnergyFilterConfig.from_env``（E_FC, E_LPF_ORDER, E_RESAMPLE_N, E_MAX_DTH, E_WLOW, E_WHIGH,
E_LPF_NATIVE_ON, E_FC_ADAPTIVE_ON, E_FC_MIN, E_FC_MAX, E_FC_K, E_F0_WIN_SEC, E_FC_EMA_BETA, E_FC_UPDATE_HZ,
E_F0_FMIN, E_F0_SNR_THRESHOLD, E_FPS_MIN, E_FPS_MAX, E_DEBUG）。``E_LPF_NATIVE_ON`` の既定は 0（Butterworth）。
本体の既定は 1 だが GUI は 0 を渡す（1 だと 1 次指数フィルタになり、既発表の数値と比べられない）。

``angle_between`` は本体が import している公開名なので残す。
"""

from __future__ import annotations

import collections
import math
import os
import warnings
from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import butter, filtfilt, welch

    _SCIPY_OK = True
except Exception:  # SciPy が無い環境では移動平均と線形補間で代える（本体と同じ）
    _SCIPY_OK = False
    butter = filtfilt = welch = PchipInterpolator = None  # type: ignore

__all__ = [
    "AdaptiveCutoff",
    "EnergyFilterConfig",
    "OnlineF0Estimator",
    "angle_between",
    "compute_cycle_energy_filtered",
    "fc_scheduler",
]

_ENV_TRUE = ("1", "true", "yes", "on")
_ENV_FALSE = ("0", "false", "no", "off")


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    """``config.env_flag`` と同じ読み方（空・知らない値は既定）。"""
    raw = env.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    return default


def _number(env: Mapping[str, str], name: str, default, kind):
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
        return int(value) if kind is int else value
    except ValueError:
        warnings.warn(f"{name}={raw!r} を数として読めないので既定の {default} を使う", RuntimeWarning, stacklevel=3)
        return default


@dataclass(frozen=True)
class EnergyFilterConfig:
    """E± の前処理の設定（名前は本体の E_* に対応）。"""

    fc: float = 1.2  # E_FC [Hz]
    lpf_order: int = 2  # E_LPF_ORDER（ネイティブのときは往復の回数）
    resample_n: int = 80  # E_RESAMPLE_N
    max_dth: float = 0.25  # E_MAX_DTH [rad/点]
    winsor_low: float = 5.0  # E_WLOW [%]
    winsor_high: float = 95.0  # E_WHIGH [%]
    lpf_native_on: bool = False  # E_LPF_NATIVE_ON（本体の既定は 1、GUI の既定は 0）
    fc_adaptive_on: bool = False  # E_FC_ADAPTIVE_ON
    fc_min: float = 2.1  # E_FC_MIN [Hz]
    fc_max: float = 6.0  # E_FC_MAX [Hz]
    fc_k: float = 6.0  # E_FC_K（fc = k·f0）
    f0_win_sec: float = 4.0  # E_F0_WIN_SEC [s]
    fc_ema_beta: float = 0.15  # E_FC_EMA_BETA
    fc_update_hz: float = 1.0  # E_FC_UPDATE_HZ
    f0_fmin: float = 0.3  # E_F0_FMIN [Hz]
    f0_snr_threshold: float = 3.0  # E_F0_SNR_THRESHOLD [dB]
    fps_min: float = 5.0  # E_FPS_MIN
    fps_max: float = 120.0  # E_FPS_MAX
    debug: bool = False  # E_DEBUG

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EnergyFilterConfig":
        """環境変数から読む。数として読めない値は既定に戻して警告する。"""
        env = os.environ if env is None else env
        d = cls()
        return cls(
            fc=_number(env, "E_FC", d.fc, float),
            lpf_order=_number(env, "E_LPF_ORDER", d.lpf_order, int),
            resample_n=_number(env, "E_RESAMPLE_N", d.resample_n, int),
            max_dth=_number(env, "E_MAX_DTH", d.max_dth, float),
            winsor_low=_number(env, "E_WLOW", d.winsor_low, float),
            winsor_high=_number(env, "E_WHIGH", d.winsor_high, float),
            lpf_native_on=_flag(env, "E_LPF_NATIVE_ON", d.lpf_native_on),
            fc_adaptive_on=_flag(env, "E_FC_ADAPTIVE_ON", d.fc_adaptive_on),
            fc_min=_number(env, "E_FC_MIN", d.fc_min, float),
            fc_max=_number(env, "E_FC_MAX", d.fc_max, float),
            fc_k=_number(env, "E_FC_K", d.fc_k, float),
            f0_win_sec=_number(env, "E_F0_WIN_SEC", d.f0_win_sec, float),
            fc_ema_beta=_number(env, "E_FC_EMA_BETA", d.fc_ema_beta, float),
            fc_update_hz=_number(env, "E_FC_UPDATE_HZ", d.fc_update_hz, float),
            f0_fmin=_number(env, "E_F0_FMIN", d.f0_fmin, float),
            f0_snr_threshold=_number(env, "E_F0_SNR_THRESHOLD", d.f0_snr_threshold, float),
            fps_min=_number(env, "E_FPS_MIN", d.fps_min, float),
            fps_max=_number(env, "E_FPS_MAX", d.fps_max, float),
            debug=_flag(env, "E_DEBUG", d.debug),
        )


# ===================== 適応カットオフ（本体の OnlineF0Estimator・_fc_scheduler） =====================


class OnlineF0Estimator:
    """短時間の Welch で θ の支配周波数 f0 を推定する（本体と同じ）。"""

    def __init__(self, fps: float, win_sec: float = 4.0, fmin: float = 0.3, *,
                 fps_min: float = 5.0, fps_max: float = 120.0):
        self.fps = fps
        self.win_sec = float(win_sec)
        self.win_len = max(64, int(fps * win_sec))  # FFT に最小 64 サンプル
        self.fmin = fmin
        self.fps_min = fps_min
        self.fps_max = fps_max
        self.buffer = collections.deque(maxlen=self.win_len)
        self.update_counter = 0

    def set_fps(self, fps_new: float) -> None:
        """実効 fps の更新（窓長も秒で追従する）。"""
        if (not np.isfinite(fps_new)) or fps_new <= 1e-6:
            return
        fps_use = float(np.clip(fps_new, self.fps_min, self.fps_max))
        if abs(fps_use - self.fps) < 1e-6:
            return
        self.fps = fps_use
        new_win_len = max(64, int(round(self.fps * self.win_sec)))
        if new_win_len != self.win_len:
            self.win_len = new_win_len
            self.buffer = collections.deque(list(self.buffer), maxlen=self.win_len)

    def step(self, theta_sample: float) -> None:
        if np.isfinite(theta_sample):
            self.buffer.append(float(theta_sample))
        self.update_counter += 1

    def estimate(self) -> tuple[float, float]:
        """(f0 [Hz], 信頼度 [dB])。標本が 64 に満たなければ (0, 0)。"""
        if len(self.buffer) < 64:
            return 0.0, 0.0
        th_arr = np.unwrap(np.array(list(self.buffer), dtype=np.float64))
        try:
            if _SCIPY_OK:
                freqs, psd = welch(th_arr, fs=self.fps, window="hann", nperseg=min(256, len(th_arr)), noverlap=None)
            else:
                win = np.hanning(len(th_arr))
                fft_val = np.abs(np.fft.rfft(th_arr * win)) ** 2
                freqs = np.fft.rfftfreq(len(th_arr), 1.0 / self.fps)
                psd = fft_val / np.sum(win ** 2)
        except Exception:
            return 0.0, 0.0
        mask = freqs >= self.fmin
        if not np.any(mask):
            return 0.0, 0.0
        freqs_m = freqs[mask]
        psd_m = psd[mask]
        if len(psd_m) == 0:
            return 0.0, 0.0
        idx_peak = np.argmax(psd_m)
        f0 = float(freqs_m[idx_peak])
        bg_power = np.median(psd_m)
        peak_power = psd_m[idx_peak]
        snr_db = 10.0 * np.log10(max(1e-9, peak_power / max(1e-12, bg_power)))
        return f0, snr_db


def fc_scheduler(f0_hat: float, confidence_db: float, fc_prev: float, fc_min: float, fc_max: float,
                 fc_k: float, ema_beta: float, snr_threshold: float) -> float:
    """次の fc [Hz]。信頼度が閾値未満か f0 ≤ 0 なら前の fc のまま（本体の ``_fc_scheduler``）。"""
    if confidence_db < snr_threshold or f0_hat <= 0:
        return fc_prev
    fc_clipped = np.clip(fc_k * f0_hat, fc_min, fc_max)
    return float((1.0 - ema_beta) * fc_prev + ema_beta * fc_clipped)


class AdaptiveCutoff:
    """毎フレーム ``step(theta)`` を呼ぶと、``fc_update_hz`` ごとに ``fc`` を更新する（適応がオフなら固定）。

    本体と同じく、θ は左右の肘角の平均（片方が NaN ならもう片方）を渡す想定。
    """

    def __init__(self, config: EnergyFilterConfig | None = None, fps: float = 30.0):
        self.config = config or EnergyFilterConfig.from_env()
        self.fc = float(self.config.fc)
        self.fps = self._clip(fps)
        self.last_f0 = 0.0
        self.last_conf_db = 0.0
        self._estimator: OnlineF0Estimator | None = None
        self._counter = 0

    def _clip(self, fps) -> float:
        value = float(fps) if fps is not None and np.isfinite(fps) else 30.0
        return float(np.clip(value, self.config.fps_min, self.config.fps_max))

    def step(self, theta: float, fps: float | None = None) -> float:
        """θ を 1 つ供給して、今の fc を返す。``fps`` を渡すと実効 fps を更新する。"""
        cfg = self.config
        if not cfg.fc_adaptive_on:
            return self.fc
        if fps is not None:
            self.fps = self._clip(fps)
        if self._estimator is None:
            self._estimator = OnlineF0Estimator(self.fps, cfg.f0_win_sec, cfg.f0_fmin,
                                                fps_min=cfg.fps_min, fps_max=cfg.fps_max)
        else:
            self._estimator.set_fps(self.fps)
        interval = max(1, int(round(self.fps / max(cfg.fc_update_hz, 1e-6))))
        self._estimator.step(theta)
        self._counter += 1
        if self._counter >= interval:
            self.last_f0, self.last_conf_db = self._estimator.estimate()
            self.fc = fc_scheduler(self.last_f0, self.last_conf_db, self.fc, cfg.fc_min, cfg.fc_max, cfg.fc_k,
                                   cfg.fc_ema_beta, cfg.f0_snr_threshold)
            if cfg.debug:
                print(f"[ADAPTIVE_FC] fps={self.fps:.2f} f0={self.last_f0:.2f}Hz "
                      f"conf={self.last_conf_db:.1f}dB fc={self.fc:.3f}Hz")
            self._counter = 0
        return self.fc


# ===================== 濾波・再標本化（本体と同じ） =====================


def _butter_lowpass_filtfilt(x: np.ndarray, fs: float, fc: float, order: int, *, native: bool = False) -> np.ndarray:
    if len(x) < max(8, 3 * order + 1):
        return x.copy()
    if native:
        try:
            # utils_dynamic は matplotlib・cv2 を読むので、使うときだけ import する
            from utils_dynamic import compute_lpf_exp_fb_native

            dt_local = 1.0 / max(1e-6, float(fs))
            passes = max(1, int(order))
            return compute_lpf_exp_fb_native(np.asarray(x, dtype=np.float64), dt_local, float(fc), passes=passes)
        except Exception:
            pass
    if not _SCIPY_OK:
        k = max(3, min(9, len(x) // 10 * 2 + 1))
        return np.convolve(x, np.ones(k) / k, mode="same")
    nyq = 0.5 * fs
    wn = min(0.99, max(1e-3, fc / nyq))
    b, a = butter(order, wn, btype="low", analog=False)
    try:
        return filtfilt(b, a, x, method="gust")
    except Exception:
        return filtfilt(b, a, x)


def _interp_uniform(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """時刻 x（単調増加）の y を 0..1 の等間隔 n 点へ（PCHIP、無ければ線形）。"""
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
            return ui, PchipInterpolator(x, y, extrapolate=True)(ti)
        except Exception:
            pass
    return ui, np.interp(ti, x, y)


def _winsorize(y: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    if len(y) < 4:
        return y.copy()
    lo = np.percentile(y, p_low)
    hi = np.percentile(y, p_high)
    return np.clip(y, lo, hi)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """2 つの 3 次元ベクトルのなす角 [rad]（数値的に安定な atan2 の形）。"""
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


# 古い私的な名前で参照しているものが残っていても動くように
_angle_between = angle_between


def compute_cycle_energy_filtered(theta: np.ndarray, tau: np.ndarray, dt_sec: float, fc_override: float | None = None,
                                  *, config: EnergyFilterConfig | None = None) -> tuple[float, float, dict]:
    """1 サイクルの肘角 θ [rad] とトルク τ [N·m] から (E⁺, E⁻, info) を返す（本体と同じ計算）。

    ``fc_override`` は適応カットオフがオンのときだけ効く（本体と同じ。オフなら ``config.fc``）。
    ``config`` を省くと呼んだ時点の環境変数から読む。``info`` には使った ``fc`` も入れる。
    """
    cfg = config if config is not None else EnergyFilterConfig.from_env()
    th_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    tau_arr = np.asarray(tau, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(th_arr) & np.isfinite(tau_arr)
    th_arr = th_arr[finite_mask]
    tau_arr = tau_arr[finite_mask]

    if cfg.fc_adaptive_on and fc_override is not None and fc_override > 0:
        fc_use = fc_override
    else:
        fc_use = cfg.fc

    n = len(th_arr)
    if n < 3:
        return 0.0, 0.0, {"status": "too_few", "n": n, "n_valid": int(np.sum(finite_mask)), "fc": float(fc_use)}
    fs = 1.0 / max(1e-6, dt_sec)

    # 1) unwrap + LPF
    th = np.unwrap(th_arr)
    th_f = _butter_lowpass_filtfilt(th, fs, fc_use, cfg.lpf_order, native=cfg.lpf_native_on)
    tau_f = _butter_lowpass_filtfilt(tau_arr, fs, fc_use, cfg.lpf_order, native=cfg.lpf_native_on)
    # 2) 時間を 0..1 に正規化して再標本化
    t = np.arange(n, dtype=np.float64) * dt_sec
    _, th_u = _interp_uniform(t, th_f, cfg.resample_n)
    _, tau_u = _interp_uniform(t, tau_f, cfg.resample_n)
    # 外れの抑制
    tau_u = _winsorize(tau_u, cfg.winsor_low, cfg.winsor_high)
    dth = np.clip(np.diff(th_u), -cfg.max_dth, cfg.max_dth)
    # 3) 台形積分（正負を分ける）
    tau_mid = 0.5 * (tau_u[1:] + tau_u[:-1])
    contrib = tau_mid * dth
    e_pos = float(np.sum(np.maximum(contrib, 0.0)))
    e_neg = float(np.sum(np.maximum(-contrib, 0.0)))
    info = {"status": "ok", "n_u": int(len(th_u)), "n_valid": int(n), "fc": float(fc_use)}
    if len(th_u) < 30:
        info["low_conf"] = True
    if cfg.debug:
        print(f"[EPIPE] n={n}->{len(th_u)} e+={e_pos:.4f} e-={e_neg:.4f}")
    return e_pos, e_neg, info
