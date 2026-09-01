# %%
import logging
import threading
import tracemalloc
import atexit
import traceback
import os
import time
import importlib  # Pylint E0601 対策: 後段で importlib.util を参照するため先行 import
try:
    import serial
except Exception:
    serial = None
import sys
import datetime
import collections

# pylint: disable=no-member
import cv2 as cv
import japanize_matplotlib  # pylint: disable=unused-import # 日本語表示のサポート
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import csv
import json
from extended_kalman_filter import EKFConfig, ExtendedKalman1D
from pose_runtime import PoseEstimator
from logging_setup import setup_logging, get_logger
from video_io import (
    resolve_input_streams,
    open_capture_and_read_first,
    find_recording_pairs,
)
from energy_pipeline import angle_between

from body_part_storage_module import BodyPartDataStorage
from config import (
    PADDING,
    dt,
    folder_path,
    fps,
    g,
    input_stream1,
    input_stream2,
    m1,
    m2,
    m4,

    pose_keypoints,
    rm_path,
    save_dir,
    timestamp,
    w,
    SKIP_FRAMES,
    WHILE_COUNT,
    z_value,
    part_keys,
    impulse_records,
    current_torque_history,
    prev_cycle_frame,
    min_history_len,
    detector,
    gauge,
    current_impulses,
    USE_SAMPLE_VIDEOS,
    AUTO_FALLBACK_TO_FILES,
    HEADLESS,
    IO_DEBUG,
    PREFER_RECORDING_PAIRS,
)
from link_vector_calculator_module import LinkVectorCalculator
from utils import (
    DLT,
    extract_keypoints,
    get_projection_matrix,
    put_text_jp,
    compute_local_torque,
    PushCycleDetector,
)
from utils_dynamic import (
    calculate_individual_torques,
    calculate_inertia_tensor,
    calculate_M_and_F,
    compute_tau_chain_native,
    compute_lpf_exp_fb_native,
    compute_triangulate_transform_native,
)
try:
    from Gauge_display import GaugeDisplay
except Exception:
    GaugeDisplay = None
try:
    import py_native_overlay as _native_overlay
except Exception:
    _native_overlay = None
try:
    import py_native_pose as _native_pose
except Exception:
    _native_pose = None
import math
from typing import Dict, Tuple
import argparse
import glob
from typing import Optional, Tuple
try:
    # 推奨: SciPy フィルタと補間
    from scipy.signal import butter, filtfilt, lfilter, lfilter_zi, welch
    from scipy.interpolate import PchipInterpolator
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# ===================== ロギング初期化 =====================
_root_logger = setup_logging()
logger = get_logger(__name__)

# ===================== サイクルE（ゲージ用）フィルタパイプライン設定 =====================
# 5 Hz 前提: fc は 1.0〜1.5 Hz 推奨。env で微調整可。
E_FC = float(os.getenv('E_FC', '1.2'))
E_LPF_ORDER = int(os.getenv('E_LPF_ORDER', '2'))  # 2〜4
E_RESAMPLE_N = int(os.getenv('E_RESAMPLE_N', '80'))  # 50〜100
E_MAX_DTH = float(os.getenv('E_MAX_DTH', '0.25'))  # 角度ステップの上限 [rad]
E_WINSOR_PCTL_LOW = float(os.getenv('E_WLOW', '5'))  # トルクの下側ウィンズライジング
E_WINSOR_PCTL_HIGH = float(os.getenv('E_WHIGH', '95'))
E_DEBUG = os.getenv('E_DEBUG', '0') in ('1','true','True')
E_LPF_NATIVE_ON = os.getenv('E_LPF_NATIVE_ON', '1') in ('1','true','True')
TRIANG_NATIVE_ON = os.getenv('TRIANG_NATIVE_ON', '1') in ('1','true','True')

# ===================== 重力向き検出/管理 設定 =====================
GRAVITY_AUTO_DETECT = os.getenv('GRAVITY_AUTO_DETECT', '1') in ('1','true','True')
GRAVITY_FROM_CHECKERBOARD_SHORT = os.getenv('GRAVITY_FROM_CHECKERBOARD_SHORT', '1') in ('1','true','True')
GRAVITY_CHECKERBOARD_AXIS_FILE = os.getenv('GRAVITY_CHECKERBOARD_AXIS_FILE', 'camera_parameters/checkerboard_short_axis.json')
GRAVITY_DETECT_FRAMES = int(os.getenv('GRAVITY_DETECT_FRAMES', '90'))  # 3秒@30fps目安
GRAVITY_PREFERRED = os.getenv('GRAVITY_PREFERRED', 'Y-')  # フォールバック表示用
GRAVITY_TAG_IN_CSV = os.getenv('GRAVITY_TAG_IN_CSV', '1') in ('1','true','True')
# Webカメラは必ずしも水平でないため、平面拘束は既定OFF（必要時のみON）
GRAVITY_LEVEL_PLANE_ON = os.getenv('GRAVITY_LEVEL_PLANE_ON', '0') in ('1','true','True')
GRAVITY_LEVEL_PLANE = os.getenv('GRAVITY_LEVEL_PLANE', 'YZ').upper()  # XY/YZ/XZ
GRAVITY_AMBIG_DELTA = float(os.getenv('GRAVITY_AMBIG_DELTA', '0.08'))  # 近接時はpreferred優先
GRAVITY_LEVEL_PLANE_WEBCAM_OK = os.getenv('GRAVITY_LEVEL_PLANE_WEBCAM_OK', '0') in ('1','true','True')

# ===================== 適応的LPF設定（リアルタイムf0追跡）=====================
E_FC_ADAPTIVE_ON = int(os.getenv('E_FC_ADAPTIVE_ON', '0'))  # 0=固定fc, 1=適応fc
E_FC_MIN = float(os.getenv('E_FC_MIN', '2.1'))  # fc下限 [Hz]
E_FC_MAX = float(os.getenv('E_FC_MAX', '6.0'))  # fc上限 [Hz]
E_FC_K = float(os.getenv('E_FC_K', '6.0'))  # f0→fc倍数（オフライン統計値）
E_F0_WIN_SEC = float(os.getenv('E_F0_WIN_SEC', '4.0'))  # FFT窓長 [秒]
E_FC_EMA_BETA = float(os.getenv('E_FC_EMA_BETA', '0.15'))  # fc平滑度 [0-1]
E_FC_UPDATE_HZ = float(os.getenv('E_FC_UPDATE_HZ', '1.0'))  # fc更新レート [Hz]
E_F0_FMIN = float(os.getenv('E_F0_FMIN', '0.3'))  # 最小検出周波数 [Hz]
E_F0_SNR_THRESHOLD = float(os.getenv('E_F0_SNR_THRESHOLD', '3.0'))  # 信頼度 [dB]
E_FPS_EMA_BETA = float(os.getenv('E_FPS_EMA_BETA', '0.20'))  # 実効fps追従EMA [0-1]
E_FPS_MIN = float(os.getenv('E_FPS_MIN', '5.0'))  # 実効fps下限（異常値抑制）
E_FPS_MAX = float(os.getenv('E_FPS_MAX', '120.0'))  # 実効fps上限（異常値抑制）

# 拡張カルマンフィルタ設定（ランドマーク位置/速度/加速度）
EKF_ENABLE = os.getenv('EKF_ENABLE', '1') in ('1', 'true', 'True')
EKF_Q_ACC = float(os.getenv('EKF_Q_ACC', '1e-3'))
EKF_R = float(os.getenv('EKF_R', '1e-3'))
EKF_GATE_STD = float(os.getenv('EKF_GATE_STD', '3.0'))
# バンドパス（任意）: low/high Hz を設定すると IIR で逐次前処理
EKF_BPF_LOW = float(os.getenv('EKF_BPF_LOW', '0'))
EKF_BPF_HIGH = float(os.getenv('EKF_BPF_HIGH', '0'))
EKF_BPF_ORDER = int(os.getenv('EKF_BPF_ORDER', '2'))

# 肘の角度・トルクのフレーム蓄積バッファ（1サイwクル分）
_E_buffers = {
    'elbow_R': {'theta': [], 'tau': []},
    'elbow_L': {'theta': [], 'tau': []},
}

# ===================== 適応的LPFのグローバル状態 =====================
_f0_estimator = None  # lazy init (OnlineF0Estimator instance)
_fc_current = E_FC  # 現在のfc値（Hz）
_fc_update_counter = 0  # 更新フレームカウンタ
_fc_update_interval = None  # lazy set: int(fps / E_FC_UPDATE_HZ)
_lpf_fps_ema = None  # 実効fps推定（loop_dt由来EMA）
_e_dt_sec_current = float(dt) if (isinstance(dt, (int, float)) and dt > 0) else (1.0 / 30.0)

# ===================== 重力推定のグローバル状態 =====================
_grav_up_samples = collections.deque(maxlen=max(10, GRAVITY_DETECT_FRAMES))
_gravity_label = GRAVITY_PREFERRED  # 重力向きラベル（例: Y-）
_gravity_set = False
_gravity_level_plane_on_runtime = GRAVITY_LEVEL_PLANE_ON

# ===================== 適応的f0推定器 =====================
class OnlineF0Estimator:
    """短時間FFTでθから支配周波数(f0)を推定。"""
    def __init__(self, fps: float, win_sec: float = 4.0, fmin: float = 0.3):
        """
        Args:
            fps: フレームレート (Hz)
            win_sec: 分析窓長 (秒)
            fmin: 最小検出周波数 (Hz)
        """
        self.fps = fps
        self.win_sec = float(win_sec)
        self.win_len = max(64, int(fps * win_sec))  # FFTに最小64サンプル確保
        self.fmin = fmin
        self.buffer = collections.deque(maxlen=self.win_len)
        self.update_counter = 0

    def set_fps(self, fps_new: float) -> None:
        """実効fps更新（窓長も秒ベースで追従）。"""
        if (not np.isfinite(fps_new)) or fps_new <= 1e-6:
            return
        fps_use = float(np.clip(fps_new, E_FPS_MIN, E_FPS_MAX))
        if abs(fps_use - self.fps) < 1e-6:
            return
        self.fps = fps_use
        new_win_len = max(64, int(round(self.fps * self.win_sec)))
        if new_win_len != self.win_len:
            self.win_len = new_win_len
            self.buffer = collections.deque(list(self.buffer), maxlen=self.win_len)
    
    def step(self, theta_sample: float) -> None:
        """新しいθサンプルを受け取り、バッファに追加"""
        if np.isfinite(theta_sample):
            self.buffer.append(float(theta_sample))
        self.update_counter += 1
    
    def estimate(self) -> tuple[float, float]:
        """
        FFT/Welchで周波数推定。
        Returns: (f0_hz, confidence_db)
        """
        if len(self.buffer) < 64:
            return 0.0, 0.0
        
        th_arr = np.array(list(self.buffer), dtype=np.float64)
        th_arr = np.unwrap(th_arr)
        
        # Welch: 50% overlap, Hann window (or fallback to simple FFT)
        try:
            if _SCIPY_OK:
                freqs, psd = welch(th_arr, fs=self.fps, window='hann', nperseg=min(256, len(th_arr)), noverlap=None)
            else:
                # Fallback: simple FFT (Hann windowed)
                win = np.hanning(len(th_arr))
                th_w = th_arr * win
                fft_val = np.abs(np.fft.rfft(th_w)) ** 2
                freqs = np.fft.rfftfreq(len(th_arr), 1.0 / self.fps)
                psd = fft_val / np.sum(win ** 2)  # Normalize
        except Exception:
            return 0.0, 0.0
        
        # fmin以上のピークを検索
        mask = freqs >= self.fmin
        if not np.any(mask):
            return 0.0, 0.0
        
        freqs_m = freqs[mask]
        psd_m = psd[mask]
        
        if len(psd_m) == 0:
            return 0.0, 0.0
        
        idx_peak = np.argmax(psd_m)
        f0 = float(freqs_m[idx_peak])
        
        # SNR推定（ピーク vs. 平均背景）
        bg_power = np.median(psd_m)
        peak_power = psd_m[idx_peak]
        snr_db = 10.0 * np.log10(max(1e-9, peak_power / max(1e-12, bg_power)))
        
        return f0, snr_db

def _fc_scheduler(f0_hat: float, confidence_db: float, fc_prev: float, 
                  fc_min: float, fc_max: float, fc_k: float, 
                  ema_beta: float, snr_threshold: float) -> float:
    """適応的fcスケジューラ。
    
    Args:
        f0_hat: 推定周波数 (Hz)
        confidence_db: 信頼度（SNR dB）
        fc_prev: 前フレームのfc
        fc_min, fc_max: fc範囲
        fc_k: f0→fc倍数
        ema_beta: 平滑係数 [0-1]
        snr_threshold: 信頼度閾値（dB）
    
    Returns: 次のfc値 (Hz)
    """
    # 低信頼度ならfc_prevを維持
    if confidence_db < snr_threshold or f0_hat <= 0:
        return fc_prev
    
    # fc_raw = k * f0
    fc_raw = fc_k * f0_hat
    
    # クリップ
    fc_clipped = np.clip(fc_raw, fc_min, fc_max)
    
    # EMA平滑 (jitter低減)
    fc_next = (1.0 - ema_beta) * fc_prev + ema_beta * fc_clipped
    
    return float(fc_next)

def _axis_vec_from_label(label: str) -> np.ndarray:
    mapping = {
        'X+': np.array([ 1.0, 0.0, 0.0]), 'X-': np.array([-1.0, 0.0, 0.0]),
        'Y+': np.array([ 0.0, 1.0, 0.0]), 'Y-': np.array([ 0.0,-1.0, 0.0]),
        'Z+': np.array([ 0.0, 0.0, 1.0]), 'Z-': np.array([ 0.0, 0.0,-1.0]),
    }
    return mapping.get(label, np.array([0.0, -1.0, 0.0]))


def _opposite_axis_label(label: str) -> str:
    if label.endswith('+'):
        return label[:-1] + '-'
    if label.endswith('-'):
        return label[:-1] + '+'
    return label


def _candidate_axis_labels() -> list[str]:
    if not _gravity_level_plane_on_runtime:
        return ['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-']
    plane = GRAVITY_LEVEL_PLANE
    if plane == 'YZ':
        return ['Y+', 'Y-', 'Z+', 'Z-']
    if plane == 'XZ':
        return ['X+', 'X-', 'Z+', 'Z-']
    if plane == 'XY':
        return ['X+', 'X-', 'Y+', 'Y-']
    return ['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-']


def _pick_axis_from_vector(v: np.ndarray) -> tuple[str, np.ndarray, float]:
    """肩→腰ベクトル(=上方向)から最も近い軸と符号を選ぶ。
    GRAVITY_LEVEL_PLANE_ON=1 のとき、候補軸を指定平面に制限する。
    Returns: (up_label like 'Y+', up_axis_unit_vec, cosine_abs)
    """
    pref_up_label = _opposite_axis_label(GRAVITY_PREFERRED)
    pref_up_axis = _axis_vec_from_label(pref_up_label)
    if v is None or not np.all(np.isfinite(v)):
        return pref_up_label, pref_up_axis, 0.0
    vn = np.array(v, dtype=float)
    n = float(np.linalg.norm(vn))
    if n < 1e-9:
        return pref_up_label, pref_up_axis, 0.0
    vn /= n

    candidates = _candidate_axis_labels()
    scored = []
    for lab in candidates:
        ax = _axis_vec_from_label(lab)
        cabs = abs(float(np.dot(vn, ax)))
        scored.append((cabs, lab, ax))
    scored.sort(key=lambda t: t[0], reverse=True)

    best_val, best_label, best_axis = scored[0]
    # 上位2候補が近い場合は preferred を優先（軸ラベル一貫性を維持）
    if len(scored) > 1 and (best_val - scored[1][0]) < GRAVITY_AMBIG_DELTA:
        for _, lab, ax in scored:
            if lab == pref_up_label:
                return lab, ax, abs(float(np.dot(vn, ax)))
    return best_label, best_axis, best_val


def _load_checkerboard_short_axis_runtime(path: str) -> np.ndarray | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        cand = d.get('vector_runtime', None)
        if cand is None:
            cand = d.get('vector_cam0', None)
            if cand is not None and len(cand) == 3:
                cand = [-float(cand[0]), -float(cand[2]), -float(cand[1])]
        if cand is None:
            return None
        v = np.asarray(cand, dtype=float).reshape(3)
        if not np.all(np.isfinite(v)):
            return None
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
        return v / n
    except Exception:
        return None

def _butter_lowpass_filtfilt(x: np.ndarray, fs: float, fc: float, order: int) -> np.ndarray:
    if len(x) < max(8, 3*order+1):
        return x.copy()
    if E_LPF_NATIVE_ON:
        try:
            dt_local = 1.0 / max(1e-6, float(fs))
            passes = max(1, int(order))
            return compute_lpf_exp_fb_native(np.asarray(x, dtype=np.float64), dt_local, float(fc), passes=passes)
        except Exception:
            pass
    if not _SCIPY_OK:
        # 簡易フォールバック: 移動平均
        k = max(3, min(9, len(x)//10*2+1))
        return np.convolve(x, np.ones(k)/k, mode='same')
    nyq = 0.5 * fs
    wn = min(0.99, max(1e-3, fc / nyq))
    b, a = butter(order, wn, btype='low', analog=False)
    try:
        return filtfilt(b, a, x, method='gust')
    except Exception:
        return filtfilt(b, a, x)

def _interp_uniform(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    # x: 時刻 (単調増加), y: 値, n: サンプル数
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
    # 線形フォールバック
    yi = np.interp(ti, x, y)
    return ui, yi

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    # ベクトル間の角度 [rad]（数値安定版）
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
    crossn = np.linalg.norm(np.cross(a/na, b/nb))
    return math.atan2(crossn, dot)


def _kp2d_valid(kpts, idx: int) -> Optional[np.ndarray]:
    try:
        x, y = kpts[idx]
        x = float(x)
        y = float(y)
        if (x < 0) or (y < 0) or (not np.isfinite(x)) or (not np.isfinite(y)):
            return None
        return np.array([x, y], dtype=np.float64)
    except Exception:
        return None


def _elbow_angle_deg_cam0(kpts_cam0, side: str) -> Optional[float]:
    if side == 'R':
        sh, el, wr = 2, 1, 0
    else:
        sh, el, wr = 3, 4, 5
    p_sh = _kp2d_valid(kpts_cam0, sh)
    p_el = _kp2d_valid(kpts_cam0, el)
    p_wr = _kp2d_valid(kpts_cam0, wr)
    if p_sh is None or p_el is None or p_wr is None:
        return None
    v1 = p_sh - p_el
    v2 = p_wr - p_el
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cs = float(np.dot(v1, v2) / (n1 * n2))
    cs = float(np.clip(cs, -1.0, 1.0))
    return float(np.degrees(np.arccos(cs)))


def _shoulder_world_y_m(results0, side: str) -> Optional[float]:
    try:
        if not getattr(results0, 'pose_world_landmarks', None):
            return None
        idx = 12 if side == 'R' else 11
        lm = results0.pose_world_landmarks.landmark[idx]
        y = float(lm.y)
        if not np.isfinite(y):
            return None
        return y
    except Exception:
        return None

def _winsorize(y: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    if len(y) < 4:
        return y.copy()
    lo = np.percentile(y, p_low)
    hi = np.percentile(y, p_high)
    return np.clip(y, lo, hi)


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
    ) -> None:
        self.n_points = int(n_points)
        self.cfg = cfg
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

    def step(self, meas: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if meas.shape != (self.n_points, 3):
            raise ValueError(f"meas shape must be {(self.n_points, 3)}, got {meas.shape}")
        if dt <= 0:
            dt = 1e-3
        arr = np.asarray(meas, dtype=float)
        arr = self._apply_bpf(arr)
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

def compute_cycle_energy_filtered(theta: np.ndarray, tau: np.ndarray, dt_sec: float, fc_override: float = None) -> tuple[float, float, dict]:
    """推奨パイプラインで E⁺/E⁻ を返す。
    
    Args:
        theta: 肘角度配列 [rad]
        tau: トルク配列 [N·m]
        dt_sec: サンプリング間隔 [秒]
        fc_override: フィルタ周波数上書き (Hz, デフォルト=None → E_FCを使用)
    
    Returns: (E_pos, E_neg, info)
    """
    th_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    tau_arr = np.asarray(tau, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(th_arr) & np.isfinite(tau_arr)
    th_arr = th_arr[finite_mask]
    tau_arr = tau_arr[finite_mask]

    n = len(th_arr)
    if n < 3:
        return 0.0, 0.0, {'status': 'too_few', 'n': n, 'n_valid': int(np.sum(finite_mask))}
    fs = 1.0 / max(1e-6, dt_sec)
    
    # ===== 適応fc選択 =====
    if E_FC_ADAPTIVE_ON and fc_override is not None and fc_override > 0:
        fc_use = fc_override
    else:
        fc_use = E_FC
    
    # 1) unwrap + LPF
    th = np.unwrap(th_arr)
    th_f = _butter_lowpass_filtfilt(th, fs, fc_use, E_LPF_ORDER)
    tau_f = _butter_lowpass_filtfilt(tau_arr, fs, fc_use, E_LPF_ORDER)
    # 2) 時間正規化（0..T を 0..1 に）
    t = np.arange(n, dtype=np.float64) * dt_sec
    ui, th_u = _interp_uniform(t, th_f, E_RESAMPLE_N)
    _, tau_u = _interp_uniform(t, tau_f, E_RESAMPLE_N)
    # 外れ抑制
    tau_u = _winsorize(tau_u, E_WINSOR_PCTL_LOW, E_WINSOR_PCTL_HIGH)
    dth = np.diff(th_u)
    dth = np.clip(dth, -E_MAX_DTH, E_MAX_DTH)
    # 3) 台形積分（正負分離）
    tau_mid = 0.5 * (tau_u[1:] + tau_u[:-1])
    contrib = tau_mid * dth
    e_pos = float(np.sum(np.maximum(contrib, 0.0)))
    e_neg = float(np.sum(np.maximum(-contrib, 0.0)))
    info = {'status': 'ok', 'n_u': int(len(th_u)), 'n_valid': int(n)}
    if len(th_u) < 30:
        info['low_conf'] = True
    if E_DEBUG:
        print(f"[EPIPE] n={n}->{len(th_u)} e+={e_pos:.4f} e-={e_neg:.4f}")
    return e_pos, e_neg, info

# ========= MediaPipe Pose Landmarker (Lite) 切替対応 =========
# USE_POSE_LANDMARKER=1 かつ モデルファイルが存在すれば Tasks API を使用。なければ従来の Solutions Pose を使用。
USE_POSE_LANDMARKER = str(os.getenv('USE_POSE_LANDMARKER', '1')).lower() in ('1', 'true', 'yes')
USE_NATIVE_POSE = str(os.getenv('USE_NATIVE_POSE', '0')).lower() in ('1', 'true', 'yes')
DEFAULT_TASK_MODEL = os.path.join(os.path.dirname(__file__), 'pose_landmarker_lite.task')
POSE_TASK_MODEL = os.getenv('POSE_TASK_MODEL', DEFAULT_TASK_MODEL)
# 置かれた .task を自動検出（環境変数未指定 or 既定パスが存在しない場合）
if not os.path.exists(POSE_TASK_MODEL):
    try:
        _dir = os.path.dirname(__file__)
        _candidates = [p for p in glob.glob(os.path.join(_dir, '*.task'))]
        if _candidates:
            # 'pose' を含むものを優先
            _candidates.sort(key=lambda p: (0 if 'pose' in os.path.basename(p).lower() else 1, len(os.path.basename(p))))
            POSE_TASK_MODEL = _candidates[0]
            if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                logger.debug("[Pose] Auto-detected model: %s", POSE_TASK_MODEL)
    except Exception:
        pass
def _detect_threads(default: int = 8) -> int:
    cores = None
    try:
        # psutil があれば物理コア数優先で取得（無ければ動的 import をスキップ）
        import importlib.util
        if importlib.util.find_spec('psutil') is not None:  # type: ignore[attr-defined]
            import importlib
            _psutil = importlib.import_module('psutil')  # type: ignore
            cores = _psutil.cpu_count(logical=False) or _psutil.cpu_count(logical=True)
    except Exception:
        # 取得失敗時は後段のフォールバックへ
        cores = None
    if not cores:
        cores = os.cpu_count() or default
    # 16 物理コア環境なら 12〜16 程度が無難。上限/下限を設ける
    return max(2, min(int(cores), 32))

# XNNPACK スレッド数（環境変数 MP_THREADS があれば優先）
try:
    MP_THREADS = int(os.getenv('MP_THREADS', '').strip()) if os.getenv('MP_THREADS') else _detect_threads(16)
except Exception:
    MP_THREADS = _detect_threads(16)

# MediaPipe に渡す入力スケール（既定 0.5 = 1/2 サイズ）
try:
    MP_INPUT_SCALE = float(os.getenv('MP_INPUT_SCALE', '0.5').strip())
except Exception:
    MP_INPUT_SCALE = 0.5
MP_INPUT_SCALE = max(0.25, min(MP_INPUT_SCALE, 1.0))

# 前フレーム近傍ROIでPose推論を軽量化（人体が急変しない前提）
POSE_ROI_ON = os.getenv('POSE_ROI_ON', '1') in ('1', 'true', 'True')
POSE_ROI_MARGIN_RATIO = float(os.getenv('POSE_ROI_MARGIN_RATIO', '0.25'))
POSE_ROI_MIN_SIDE_RATIO = float(os.getenv('POSE_ROI_MIN_SIDE_RATIO', '0.45'))
POSE_ROI_MIN_VALID_KPTS = int(os.getenv('POSE_ROI_MIN_VALID_KPTS', '4'))
POSE_ROI_MAX_MISS = int(os.getenv('POSE_ROI_MAX_MISS', '4'))
POSE_ROI_MISS_GROW_RATIO = float(os.getenv('POSE_ROI_MISS_GROW_RATIO', '0.25'))
POSE_X_CROP_MARGIN = int(os.getenv('POSE_X_CROP_MARGIN', '140'))
POSE_X_CROP_MIN_WIDTH_RATIO = float(os.getenv('POSE_X_CROP_MIN_WIDTH_RATIO', '0.85'))
DRAW_KEYPOINTS_ON = os.getenv('DRAW_KEYPOINTS', '1') not in ('0', 'false', 'False')
KPS_FAST_ON = os.getenv('KPS_FAST_ON', '0') in ('1', 'true', 'True')

# デバッグ出力: 使用コア数・モデルパスなど（POSE_DEBUG=1 で有効）
if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
    try:
        import importlib.util as _ilu
        _phy = _log = None
        if _ilu.find_spec('psutil') is not None:  # type: ignore[attr-defined]
            import importlib as _il
            _ps = _il.import_module('psutil')  # type: ignore
            _phy = _ps.cpu_count(logical=False)
            _log = _ps.cpu_count(logical=True)
    except Exception:
        _phy = _log = None
    try:
        _cv_cpus = cv.getNumberOfCPUs() if hasattr(cv, 'getNumberOfCPUs') else None
        _cv_threads = cv.getNumThreads() if hasattr(cv, 'getNumThreads') else None
    except Exception:
        _cv_cpus = _cv_threads = None
    logger.debug("[CPU] MP_THREADS=%s  physical=%s  logical=%s  os.cpu_count()=%s", MP_THREADS, _phy, _log, os.cpu_count())
    logger.debug("[CPU] OpenCV: numCPUs=%s  numThreads=%s", _cv_cpus, _cv_threads)
    logger.debug("[Pose] USE_POSE_LANDMARKER=%s  model=%s  exists=%s", USE_POSE_LANDMARKER, POSE_TASK_MODEL, os.path.exists(POSE_TASK_MODEL))
    logger.debug("[Pose] MP_INPUT_SCALE=%s", MP_INPUT_SCALE)
    # 任意: OpenCV スレッド数を明示設定（OPENCV_THREADS）
    _ocv_thr_env = os.getenv('OPENCV_THREADS', '').strip()
    if _ocv_thr_env:
        try:
            _ocv_thr = max(1, int(_ocv_thr_env))
            if hasattr(cv, 'setNumThreads'):
                cv.setNumThreads(_ocv_thr)
                logger.debug("[CPU] OpenCV setNumThreads -> %s", (cv.getNumThreads() if hasattr(cv, 'getNumThreads') else _ocv_thr))
        except Exception as _e:
            logger.debug("[CPU] OpenCV setNumThreads failed: %s", _e)

def _tasks_imports():
    # moved to pose_runtime.PoseEstimator. Kept as a placeholder for backward compat if referenced elsewhere.
    try:
        from pose_runtime import _tasks_imports as _imp  # type: ignore
        return _imp()
    except Exception:
        return None, None, None, None

# ================= HX711 Recorder (M5StampS3) 連携 追加インポート (オプション) =================
HX711_ENABLE = os.getenv('HX711_ENABLE', '0') in ('1', 'true', 'True')
HX_RECORDER_AVAILABLE = False
BLE_RECORDER_AVAILABLE = False
RecorderClient = None  # type: ignore
BLERecorderClientSync = None  # type: ignore
if HX711_ENABLE:
    try:
        # 依存: requests / bleak 等が内部で必要な場合がある
        import importlib.util  # noqa: F401
        from hx711_recorder import RecorderClient, BLERecorderClientSync  # type: ignore  # pylint: disable=import-error
        HX_RECORDER_AVAILABLE = True
        BLE_RECORDER_AVAILABLE = True
    except Exception as _hx_e:
        # print(f"[HX711] インポート失敗(一次): {_hx_e}")
        # 環境変数 HX711_CLIENT_DIR で外部パス指定可
        try:
            HX_CLIENT_DIR = os.getenv('HX711_CLIENT_DIR') or os.path.join(os.path.dirname(__file__), '..', 'wokwi', 'stamps3_force_logger1', 'python_client')
            candidate_path = os.path.join(HX_CLIENT_DIR, 'hx711_recorder.py')
            if os.path.isfile(candidate_path):
                spec = importlib.util.spec_from_file_location('hx711_recorder', candidate_path)
                if spec and spec.loader:
                    mod = spec.loader.load_module()  # type: ignore[attr-defined]
                    RecorderClient = getattr(mod, 'RecorderClient', None)
                    BLERecorderClientSync = getattr(mod, 'BLERecorderClientSync', None)
                    if RecorderClient:
                        HX_RECORDER_AVAILABLE = True
                        BLE_RECORDER_AVAILABLE = BLERecorderClientSync is not None
                        # print('[HX711] 直接パス読み込みで RecorderClient 利用可能')
            else:
                # print('[HX711] 外部クライアントパスが見つかりませんでした')
                pass
        except Exception as _hx_e2:  # pragma: no cover
            # print(f"[HX711] Fallback インポート失敗: {_hx_e2}")
            pass

# オプション依存 requests を遅延インポート (Pylint import-error 抑止用)
def _lazy_requests():  # pragma: no cover - 単純ヘルパ
    try:  # noqa: SIM105
        requests_mod = importlib.import_module('requests')  # type: ignore
        return requests_mod
    except Exception as e:  # ModuleNotFoundError 他
        if 'No module named' in str(e):
            # print(f"[HX711] requests 未インストール: {e}")
            pass
        else:
            # print(f"[HX711] requests 利用不可: {e}")
            pass
        return None

# ================= エネルギー閾値用 m_max_part テンプレ =================
DEFAULT_M_MAX_PART = {
    # 上腕・前腕・手首/肘トルク計測対象を想定した最大保持重量 (kg) の初期値例
    'elbow_R': 6.0,
    'wrist_R': 4.0,
    'elbow_L': 6.0,
    'wrist_L': 4.0,
}

def load_or_create_m_max_part(path: str | None) -> dict:
    """m_max_part を JSON から読み込む。無ければテンプレを書き出し既定値を返す。

    Parameters
    ----------
    path : str | None
        JSON ファイルパス。None の場合は既定辞書を返す。
    """
    if path is None:
        return DEFAULT_M_MAX_PART.copy()
    try:
        if os.path.isfile(path):
            import json as _json
            with open(path, 'r', encoding='utf-8') as f:
                data = _json.load(f)
            if isinstance(data, dict):
                # 欠損キーを補完
                out = DEFAULT_M_MAX_PART.copy()
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        out[k] = float(v)
                return out
        # 無い場合テンプレ書き込み
        import json as _json
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump(DEFAULT_M_MAX_PART, f, ensure_ascii=False, indent=2)
        # print(f"[m_max_part] テンプレ生成: {path}")
    
        return DEFAULT_M_MAX_PART.copy()
    except Exception as e:  # 失敗時は既定返し
        # print(f"[m_max_part] 読込失敗 fallback 既定値利用: {e}")
        return DEFAULT_M_MAX_PART.copy()

# ================= CLI 引数: 被験者番号 ＆ m_max_part 読み込み =================
# できるだけ既存の引数体系に干渉しないため add_help=False で部分解析
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--subject", "-s", dest="_subject_id", default=None, help="被験者番号 (例: S001)")
_partial_args, _ = _ap.parse_known_args()
SUBJECT_ID = _partial_args._subject_id or os.getenv("SUBJECT_ID")
if not SUBJECT_ID:
    try:
        SUBJECT_ID = input("被験者番号を入力してください (例: S001): ").strip()
    except Exception:
        SUBJECT_ID = None
if SUBJECT_ID:
    M_MAX_JSON_PATH = f"m_max_part_{SUBJECT_ID}.json"
else:
    M_MAX_JSON_PATH = None

M_MAX_PART = load_or_create_m_max_part(M_MAX_JSON_PATH)
if SUBJECT_ID and M_MAX_JSON_PATH:
    # print(f"[m_max_part] {SUBJECT_ID} 用の最大保持重量を読み込み: {M_MAX_JSON_PATH} -> {M_MAX_PART}")
    pass
else:
    # print("[m_max_part] SUBJECT_ID 未指定のため既定値を使用します。")
    pass

# ================= HX711 Recorder (M5StampS3) 連携 追加インポート =================
# 別プロジェクト: ../wokwi/stamps3_force_logger1/python_client/hx711_recorder.py
HX_CLIENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'wokwi', 'stamps3_force_logger1', 'python_client'))
if os.path.isdir(HX_CLIENT_DIR) and HX_CLIENT_DIR not in sys.path:
    sys.path.append(HX_CLIENT_DIR)
if HX711_ENABLE:
    try:  # noqa: SIM105
        from hx711_recorder import RecorderClient, BLERecorderClientSync  # type: ignore  # pylint: disable=import-error
    except Exception as _hx_e:  # pragma: no cover
        RecorderClient = None  # type: ignore
        BLERecorderClientSync = None  # type: ignore
        # print(f"[HX711] インポート失敗(一次): {_hx_e}")

        # --- Fallback: 直接ファイルパスからロードを試みる ---
        try:
            import importlib.util
            candidate_path = os.path.join(HX_CLIENT_DIR, 'hx711_recorder.py')
            if os.path.isfile(candidate_path):
                spec = importlib.util.spec_from_file_location('hx711_recorder', candidate_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore[arg-type]
                    RecorderClient = getattr(module, 'RecorderClient', None)  # type: ignore
                    BLERecorderClientSync = getattr(module, 'BLERecorderClientSync', None)  # type: ignore
                    if RecorderClient is not None:
                        # print('[HX711] 直接パス読み込みで RecorderClient 利用可能')
                        pass
        except Exception as _hx_e2:  # noqa: BLE001
            # print(f"[HX711] Fallback インポート失敗: {_hx_e2}")
            pass

def _iso8601_utc_ms() -> str:
    """UTC 現在時刻を ISO8601 (ミリ秒) 文字列で返す。例: 2025-09-22T13:45:12.345Z"""
    try:
        # Python 3.12+: datetime.UTC
        now = datetime.datetime.now(datetime.UTC)  # type: ignore[attr-defined]
    except Exception:
        # 互換: timezone-aware UTC
        now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime('%Y-%m-%dT%H:%M:%S.') + f"{int(now.microsecond/1000):03d}Z"


# ================= 追加: ローカル座標軸デバッグ用フラグ =================
ENABLE_AXES_DEBUG = False  # 特殊入力で True になる

# ================= 追加: 部位別エネルギー閾値計算ユーティリティ =================
CONST_K = (math.sqrt(3) / 2.0) + 1.0  # (√3 / 2 + 1)

def _compute_m1_per_part_from_bodymass(body_mass_kg: float) -> Dict[str, float]:
    """ユーザー要望の有効質量合算に基づく部位別 m1 を返す。

    wrist: 上腕 0.026 + 上肢(0.276+0.19) + 太もも 0.123
    elbow: 上肢(0.276+0.19) + 太もも 0.123
    肩は仕様未定のため 0（必要なら拡張）。
    """
    coeff_upper_arm = 0.026
    coeff_upper_limb = 0.276 + 0.19
    coeff_thigh = 0.123
    m_wrist = body_mass_kg * (coeff_upper_arm + coeff_upper_limb + coeff_thigh)
    m_elbow = body_mass_kg * (coeff_upper_limb + coeff_thigh)
    return {
        'wrist_R': m_wrist,
        'wrist_L': m_wrist,
        'elbow_R': m_elbow,
        'elbow_L': m_elbow,
        'shoulder_R': 0.0,
        'shoulder_L': 0.0,
    }

def compute_energy_thresholds(m_max_part: Dict[str, float], m1_val: float | Dict[str, float], g_scalar: float, r_x_map: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """各部位の (E_low, E_high) を計算して返す。

    数式: 
        E_low  = r_x * g * (0.42*m1 + 0.3*m_max) * K
        E_high = r_x * g * (0.42*m1 + 0.7*m_max) * K
      K = (√3 / 2 + 1)

    m1_val は float（全パート共通）または dict（部位ごと）を受け付ける。
    """
    out: Dict[str, Tuple[float, float]] = {}
    for part, mmax in m_max_part.items():
        # m1 を部位別に解決
        if isinstance(m1_val, dict):
            m1p = float(m1_val.get(part, 0.0))
        else:
            m1p = float(m1_val)
        base_coeff = 0.42 * m1p  # ← ご指摘の係数はここで効いています
        r = float(r_x_map.get(part, 0.3))  # 未定義なら 0.3m 仮置き
        E_low = r * g_scalar * (base_coeff + 0.3 * float(mmax)) * CONST_K
        E_high = r * g_scalar * (base_coeff + 0.7 * float(mmax)) * CONST_K
        out[part] = (float(E_low), float(E_high))
    return out

# ----------------- 追加: ローカル座標軸計算 & 描画ヘルパ -----------------
def _compute_local_rotation_from_link(link_vec):
    """compute_local_torque と同じ回転を再現し R を返す (global->local)。

    Returns
    -------
    R : (3,3) ndarray  回転行列。行ベクトルがローカル基底の global 表現。
    """
    if link_vec is None or not np.all(np.isfinite(link_vec)):
        return None
    n2 = np.dot(link_vec, link_vec)
    if n2 < 1e-12:
        return None
    lx, ly, lz = link_vec
    phi = math.atan2(ly, lx)
    rho = math.sqrt(lx * lx + ly * ly)
    theta = math.atan2(rho, lz)
    cφ, sφ = math.cos(phi), math.sin(phi)
    cθ, sθ = math.cos(theta), math.sin(theta)
    Rz = np.array([[cφ, sφ, 0], [-sφ, cφ, 0], [0, 0, 1]])
    Ry = np.array([[cθ, 0, -sθ], [0, 1, 0], [sθ, 0, cθ]])
    return Ry @ Rz


def _project_point(P, p3):
    """3D点 p3 (3,) を投影行列 P(3x4) で 2D へ (整数座標)。"""
    if p3 is None or not np.all(np.isfinite(p3)):
        return None
    hp = P @ np.array([p3[0], p3[1], p3[2], 1.0])
    if abs(hp[2]) < 1e-9:
        return None
    x = hp[0] / hp[2]
    y = hp[1] / hp[2]
    if not np.all(np.isfinite([x, y])):
        return None
    return int(round(x)), int(round(y))


def _draw_axes_for_link(frame, P, origin3d, R, scale=0.2, color_thickness=2, x_offset: int = 0):
    """1リンクのローカル座標軸をフレームへ描画。

    Parameters
    ----------
    frame : ndarray (BGR)
    P : (3,4) 投影行列
    origin3d : (3,) ローカル原点 (近位関節 or 肩等)
    R : (3,3) global->local 回転行列 (行が local 基底)
    scale : float  軸長のスケール (リンク長に掛ける割合) or 絶対長
    """
    if R is None or origin3d is None:
        return
    # local軸( global 表現 ) : 行ベクトルが local 基底
    ex_g = R[0]
    ey_g = R[1]
    ez_g = R[2]
    # 軸長決定（リンク長相当を取得できない場合は固定長）
    link_len = 1.0
    # 行列 R を作った元ベクトルは ez_g を逆回転したときの? ここでは norm(ez_g) ≈1
    # 適度な視認性のため固定長 * scale * 150(px換算近似) とする。
    L = 150 * scale
    pts3 = {
        'x': origin3d + ex_g * L * 0.01,
        'y': origin3d + ey_g * L * 0.01,
        'z': origin3d + ez_g * L * 0.01,
    }
    o2d = _project_point(P, origin3d)
    if o2d is None:
        return
    for axis_key, col in [('x',(0,0,255)), ('y',(0,255,0)), ('z',(255,0,0))]:
        p2 = _project_point(P, pts3[axis_key])
        if p2 is None:
            continue
        # x オフセット補正（トリミングされた場合に表示位置を一致させる）
        o2d_off = (o2d[0] - x_offset, o2d[1])
        p2_off = (p2[0] - x_offset, p2[1])
        cv.line(frame, o2d_off, p2_off, col, color_thickness, cv.LINE_AA)


def draw_all_local_axes(frame0, frame1, P0, P1, transformed_p3ds, links, x_offset: int = 0, draw_legend: bool = True):
    """全ての上肢リンクについてローカル座標軸を描画 (frame0 に描画。必要あれば両方)。"""
    if transformed_p3ds is None or links is None:
        return
    # 各リンクの近位関節(原点)とベクトルから回転取得
    # 安全のためキーが存在するか確認しつつ進める
    # インデックス対応は links 構築ロジックに追随 (近位 = distal - link_vec)
    for name, vec in links.items():
        if vec is None or not np.all(np.isfinite(vec)):
            continue
        # 原点推定: distal = proximal + vec なので proximal = distal - vec
        # ここではヒューリスティック: 肘/手首などで distal 推定に曖昧さあるため
        # 既知: wrist_R link = hand - elbow (近位=肘), など個別条件化
        origin = None
        try:
            if name == 'wrist_R':
                origin = transformed_p3ds[2]
            elif name == 'elbow_R':
                origin = transformed_p3ds[0]
            elif name == 'shoulder_R':
                origin = transformed_p3ds[0]  # 右肩基準
            elif name == 'wrist_L':
                origin = transformed_p3ds[3]
            elif name == 'elbow_L':
                origin = transformed_p3ds[1]
            elif name == 'shoulder_L':
                origin = transformed_p3ds[1]
        except Exception:
            origin = None
        R = _compute_local_rotation_from_link(vec)
        _draw_axes_for_link(frame0, P0, origin, R, scale=0.25, x_offset=x_offset)
        # 必要なら frame1 にも描画: _draw_axes_for_link(frame1, P1, origin, R)
    if draw_legend:
        cv.rectangle(frame0, (5,5), (150,70), (0,0,0), -1)
        cv.putText(frame0, 'Axes', (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv.putText(frame0, 'X', (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv.putText(frame0, 'Y', (40,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        cv.putText(frame0, 'Z', (70,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)



class SafeSerialController:
    """
    M5StampS3 用のシリアル(DTR)制御ラッパー。
    - 対象ポートを自動検出（見つからなければ無効化）
    - DTR の ON/OFF とパルス（False→True）提供
    """

    def __init__(self, baudrate: int = 115200, timeout: float = 0):
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.enabled = False

    def _detect_port(self) -> str | None:
        # 優先: 環境変数で明示指定
        env_port = os.getenv("M5_PORT")
        if env_port:
            return env_port
        # 候補: M5/ESP32/CP210/CH910/USB Serial などの記述を持つポートを優先
        try:
            from serial.tools import list_ports  # type: ignore
        except ImportError:
            return None
        candidates = []
        for p in list_ports.comports():
            meta = f"{p.device} {p.description} {p.manufacturer} {p.hwid}".lower()
            if any(s in meta for s in [
                "m5", "esp32", "cp210", "silicon labs", "wch", "ch910", "usb serial"
            ]):
                candidates.append(p.device)
        # 見つかったら最初の候補を採用（必要なら M5_PORT で上書き可）
        if candidates:
            return candidates[0]
        # 明示候補が無い場合でも、ユーザー要望により COM3 を優先的に採用（存在確認）
        try:
            for p in list_ports.comports():
                if str(p.device).upper() == "COM3":
                    return "COM3"
        except Exception:
            pass
        return None

    def open(self) -> None:
        if serial is None:
            print("[Serial] pyserial 未導入のためシリアル制御を無効化します。")
            self.enabled = False
            self.ser = None
            return
        port = self._detect_port()
        if not port:
            print("[Serial] 対象デバイスが見つかりませんでした。シリアル制御を無効化します。")
            self.enabled = False
            return
        try:
            self.ser = serial.Serial(port, self.baudrate, timeout=self.timeout)
            # 既知状態に初期化
            self.ser.dtr = False
            self.enabled = True
            print(f"[Serial] 接続: {port} @ {self.baudrate}bps")
        except (getattr(serial, 'SerialException', Exception), OSError, ValueError) as e:
            print(f"[Serial] オープン失敗: {e}. シリアル制御を無効化します。")
            self.enabled = False
            self.ser = None

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            finally:
                self.enabled = False
                self.ser = None

    def set_dtr(self, on: bool) -> None:
        if self.enabled and self.ser:
            self.ser.dtr = bool(on)

    def pulse_dtr(self, delay: float = 0.05) -> None:
        """False→True のパルスを与える（リセット/トリガ用途）"""
        if self.enabled and self.ser:
            time.sleep(delay)
            self.ser.dtr = False
            self.ser.dtr = True


# シリアル制御の初期化（見つからなければ無効化）
serial_ctrl = SafeSerialController(baudrate=115200, timeout=0)
if HX711_ENABLE:
    serial_ctrl.open()

# ================= 追加: Gauge 詳細デバッグダンプ =================
# 目的: ゲージの初期化→毎フレーム更新→描画の各段階で、
#       値としきい値、角度、ウォームアップ状態、Figure 状態などを可視化。
_GAUGE_TRACE = os.getenv('GAUGE_TRACE', '0') not in ('0', 'false', 'False')
_GAUGE_LOG_INT = int(os.getenv('GAUGE_LOG_INT', '1'))  # 何フレ毎にログするか（既定: 毎フレ）

def _gauge_log_state(tag: str = "", force: bool = False) -> None:
    """ゲージの現在状態をダンプする（重くならない範囲で）。"""
    try:
        if not _GAUGE_TRACE:
            return
        # 間引き
        _wc = globals().get('WHILE_COUNT', 0)
        if (not force) and _GAUGE_LOG_INT > 1 and (_wc % _GAUGE_LOG_INT != 0):
            return
        gobj = globals().get('gauge', None)
        if gobj is None:
            print(f"[GaugeTrace] ({tag}) gauge=None")
            return
        # UI の可視性チェック（PyQtGraph ウィンドウ）
        try:
            win = getattr(gobj, 'win', None)
            fig_ok = bool(win.isVisible()) if win is not None else True
        except Exception:
            fig_ok = True

        # 角度を先に取得（内部ウォームアップロジックを通る）
        try:
            angles = gobj.get_angles()
        except Exception as _e:
            angles = f"err:{_e}"

        # しきい値とインパルスのスナップショット
        try:
            thr_items = {k: tuple(map(float, v)) for k, v in getattr(gobj, 'energy_thresholds', {}).items()}
        except Exception:
            thr_items = getattr(gobj, 'energy_thresholds', {})
        try:
            imp_items = {k: float(v) for k, v in getattr(gobj, 'current_impulses', {}).items()}
        except Exception:
            imp_items = getattr(gobj, 'current_impulses', {})

        print(
            "[GaugeTrace]",
            f"tag={tag}",
            f"wc={_wc}",
            f"frame_idx={getattr(gobj, '_frame_index', 'n/a')}",
            f"warmup={getattr(gobj, 'warmup_frames', 'n/a')}",
            f"uiAlive={fig_ok}",
        )
        print(f"[GaugeTrace] part_keys={getattr(gobj, 'part_keys', [])}")
        print(f"[GaugeTrace] impulses={{{', '.join(f'{k}:{v:.2f}' for k,v in imp_items.items())}}}")
        # 閾値は少し長いので1行ずつ
        if thr_items:
            for _k, _tp in thr_items.items():
                try:
                    _l, _h = _tp
                    print(f"[GaugeTrace] threshold[{_k}] low={_l:.3f} high={_h:.3f}")
                except Exception:
                    print(f"[GaugeTrace] threshold[{_k}] = {_tp}")
        print(f"[GaugeTrace] angles={angles}")
    except Exception as _gt_e:
        print(f"[GaugeTrace] dump failed: {_gt_e}")

 


"""
単一モード化: 旧『監修/非監修モード』選択は廃止しました。
かつて監修モード(True)/非監修モード(False) を選択し統計集計有無を切替えていましたが、
運用簡素化のため常時『旧 非監修モード』相当のリアルタイムゲージ表示 + 既存統計読込のみ行います。
必要であれば将来 self-supervised 集計を別スクリプトに切り出してください。
"""
# supervision_mode 変数は廃止（単一モード化済み）

# 統計ファイルのパス（同ディレクトリに置く）
stats_file = os.path.join(os.path.dirname(__file__), "supervision_stats.csv")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# 姿勢推定のためのdetectorオブジェクトを作成（Tasks対応ラッパー）
try:
    POSE_MIN_DET = float(os.getenv('POSE_MIN_DET', '0.5'))
except Exception:
    POSE_MIN_DET = 0.5
try:
    POSE_MIN_TRACK = float(os.getenv('POSE_MIN_TRACK', '0.5'))
except Exception:
    POSE_MIN_TRACK = 0.5
pose0 = PoseEstimator(USE_POSE_LANDMARKER, POSE_TASK_MODEL, min_det=POSE_MIN_DET, min_track=POSE_MIN_TRACK, num_threads=MP_THREADS)
pose1 = PoseEstimator(USE_POSE_LANDMARKER, POSE_TASK_MODEL, min_det=POSE_MIN_DET, min_track=POSE_MIN_TRACK, num_threads=MP_THREADS)
print("✅ Mediapipe・モデル準備 完了")
## Gauge / Matplotlib 初期化（環境変数 DISABLE_MPL=1 で完全無効化可能）
gauge = None  # type: ignore
if (os.getenv('DISABLE_MPL', '0') not in ('1','true','True')) and not HEADLESS:
    # 統計ファイルが無ければ空テンプレを生成
    if not os.path.exists(stats_file):
        print(f"[Gauge] 統計ファイルが無いためテンプレ生成: {stats_file}")
        with open(stats_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['part', 'mean', 'std'])
            for p in part_keys:
                writer.writerow([p, 0.0, 1.0])
    df_stat = pd.read_csv(stats_file)
    stats = {row["part"]: (row["mean"], row["std"]) for _, row in df_stat.iterrows()}
    config_path = os.path.join(os.path.dirname(__file__), "gauge_layout.json")  # 旧: positions.json
    with open(config_path, "r", encoding="utf-8") as f:
        ui_conf = json.load(f)
    _g_debug = os.getenv('GAUGE_DEBUG', '0') not in ('0','false','False')
    _g_warm = int(os.getenv('GAUGE_WARMUP', '1'))  # 約 30フレーム = 1秒想定
    try:
        gauge = GaugeDisplay(config_path, stats, image_path="wheelchair_user.png", debug=_g_debug, warmup_frames=_g_warm)
        print(f"[CFG] Gauge debug={_g_debug} warmup_frames={_g_warm}")
        # PyQtGraph ウィンドウのタイトル設定と表示（非ブロッキング）
        try:
            gauge.show(x=50, y=50, w=640, h=480, on_top=True, title='Gauge Display')
        except Exception:
            pass
        # --- 表示部位を両上腕(=elbow_R/L), 両前腕(=wrist_R/L) の4つに限定 ---
        gauge.filter_parts(['wrist_R', 'elbow_R', 'wrist_L', 'elbow_L'])
        # impulse_records / current_impulses をフィルタ後の part_keys に合わせて再初期化
        current_impulses = {k: 0.0 for k in gauge.part_keys}
        impulse_records = {k: [] for k in gauge.part_keys}
        current_impulses = {k: 0.0 for k in gauge.part_keys}
        impulse_records = {k: [] for k in gauge.part_keys}
        energy_thresh = ui_conf.get("energy_thresholds") if isinstance(ui_conf, dict) else None
        if isinstance(energy_thresh, dict):
            angle_map = {}
            for pk, vals in energy_thresh.items():
                if not (isinstance(vals, (list, tuple)) and len(vals) == 2):
                    continue
                Elow, Ehigh = float(vals[0]), float(vals[1])
                mu, sigma = stats.get(pk, (0.0, 1.0))
                sigma_safe = sigma if abs(sigma) > 1e-9 else 1.0
                a_low = 120 + 60 * ((Elow - mu) / sigma_safe)
                a_high = 120 + 60 * ((Ehigh - mu) / sigma_safe)
                angle_map[pk] = [float(np.clip(a_low, 0, 180)), float(np.clip(a_high, 0, 180))]
            if angle_map:
                gauge.set_band_angles(angle_map)

        # 追加: 自動エネルギーしきい値（GAUGE_THRESH_AUTO=1 で有効）
        try:
            # 既定を ON にする（未指定なら自動適用）
            if str(os.getenv('GAUGE_THRESH_AUTO', '1')).strip() in ('1', 'true', 'True'):
                body_mass = float(os.getenv('BODY_MASS_KG', '65'))
                # r_x（有効半径[m]）の既定。必要に応じて環境変数や外部ファイル化を検討
                R_X_MAP = {
                    'wrist_R': 0.30, 'wrist_L': 0.30,
                    'elbow_R': 0.28, 'elbow_L': 0.28,
                    'shoulder_R': 0.25, 'shoulder_L': 0.25,
                }
                g_scalar = float(os.getenv('G_SCALAR', '9.80665'))
                m1_map = _compute_m1_per_part_from_bodymass(body_mass)
                thr_map = compute_energy_thresholds(M_MAX_PART, m1_map, g_scalar, R_X_MAP)
                # Gauge に適用（プロパティ直書き）
                gauge.energy_thresholds = thr_map
                # しきい帯も合わせて再描画（統計 mu/sigma から角度に変換）
                auto_angle_map = {}
                for pk, (Elow, Ehigh) in thr_map.items():
                    mu, sigma = stats.get(pk, (0.0, 1.0))
                    sigma_safe = sigma if abs(sigma) > 1e-9 else 1.0
                    a_low = 120 + 60 * ((Elow - mu) / sigma_safe)
                    a_high = 120 + 60 * ((Ehigh - mu) / sigma_safe)
                    auto_angle_map[pk] = [float(np.clip(a_low, 0, 180)), float(np.clip(a_high, 0, 180))]
                if auto_angle_map:
                    gauge.set_band_angles(auto_angle_map)
                print(f"[Gauge] GAUGE_THRESH_AUTO 適用: BODY_MASS_KG={body_mass} -> {thr_map}")
                try:
                    print(f"[Gauge] 最終 energy_thresholds: {gauge.energy_thresholds}")
                except Exception:
                    pass
            else:
                print("[Gauge] GAUGE_THRESH_AUTO=0 → JSONの energy_thresholds を使用します")
        except Exception as _auto_e:
            print(f"[Gauge] GAUGE_THRESH_AUTO 適用失敗: {_auto_e}")
        # 追加: 初期化直後のゲージ状態を詳細ダンプ
        try:
            _gauge_log_state(tag="init", force=True)
        except Exception:
            pass
    except Exception as _ginit_e:  # noqa: BLE001
        print(f"[Gauge] 初期化失敗 -> 無効化します: {_ginit_e}")
        gauge = None
else:
    print('[CFG] DISABLE_MPL=1 -> Gauge / Matplotlib を無効化')
# ESC長押し検知用（環境変数 ESC_HOLD_FRAMES で上書き可能）
try:
    ESC_HOLD_FRAMES = int(os.getenv('ESC_HOLD_FRAMES', '10'))
except Exception:
    ESC_HOLD_FRAMES = 10
esc_count = 0

# ESCを即時終了キーとして扱うかのフラグ（デフォルト: 有効）。
# もし誤検出で勝手に終了するなら IMMEDIATE_ESC_BREAK=0 を環境変数で設定してください。
IMMEDIATE_ESC_BREAK = os.getenv('IMMEDIATE_ESC_BREAK', '1') == '1'
# ── 2) 監修モード／非監修モードごとの準備 ──────────────────
# %%

logging.basicConfig(filename="app.log", level=logging.DEBUG)
logging.debug("This message should go to the log file.")
if not HEADLESS:
    cv.namedWindow("MyWindow", cv.WINDOW_NORMAL)

# CSVファイルを読み込む
df_rm = pd.read_csv(rm_path)

# rm_method の列名ゆらぎ対策（例: 『反復回数』/『ダンベル反復回数』など）
# 後続で統一的に参照できるよう、REP_COL_NAME を決定しておく
REP_COL_NAME = None
for cand in ["反復回数", "ダンベル反復回数", "reps", "回数"]:
    if cand in df_rm.columns:
        REP_COL_NAME = cand
        break
if REP_COL_NAME is None:
    # 簡易あいまい一致（日本語/英語の表記ゆれ想定）
    for _c in df_rm.columns:
        try:
            _lc = _c.lower()
        except Exception:
            _lc = str(_c)
        if ("反復" in _c) or ("回数" in _c) or ("rep" in _lc):
            REP_COL_NAME = _c
            break

if REP_COL_NAME is None:
    raise KeyError(
        f"rm_method: 反復回数の列が見つかりません。列候補: {list(df_rm.columns)}\n"
        "例: ヘッダに『反復回数』または『ダンベル反復回数』を含めてください。"
    )
# ウィンドウの位置を設定 (x=100, y=100)

if not HEADLESS:
    cv.moveWindow("MyWindow", 0, 0)

# ================= 追加: ヘルスモニタ & 例外診断 =================
tracemalloc.start(25)
_HEALTH_LOG_INTERVAL = int(os.getenv('HEALTH_INTERVAL', '150'))
_GAUGE_UPDATE_INTERVAL = int(os.getenv('GAUGE_UPDATE_INTERVAL', '2'))
# print(f"[CFG] HEALTH_INTERVAL={_HEALTH_LOG_INTERVAL} GAUGE_UPDATE_INTERVAL={_GAUGE_UPDATE_INTERVAL}")

def _health_snapshot(frame_idx: int):
    try:
        snap = tracemalloc.take_snapshot()
        top_stats = snap.statistics('lineno')[:3]
        mem_str = '; '.join(f"{st.traceback[0].filename.split(os.sep)[-1]}:{st.traceback[0].lineno} {st.size/1024:.1f}KB" for st in top_stats)
    except Exception:
        mem_str = 'n/a'
    th_cnt = len(threading.enumerate())
    gauge_angles = None
    try:
        if gauge is not None:
            gauge_angles = gauge.get_angles()
    except Exception as e:  # noqa: BLE001
        gauge_angles = f'err:{e}'
    # print(f"[HEALTH] frame={frame_idx} threads={th_cnt} mem(top)={mem_str} angles={gauge_angles}")

_fatal_state = {"exception": None}

_CLEANUP_RAN = False  # 再入防止フラグ
def _cleanup_resources():
    """終了時リソース解放 (atexit + finally)。

    デバッガ終了フェーズ (Python finalizing) でネイティブ拡張経由の GIL 操作が走ると
    『PyEval_RestoreThread ... GIL released』 のようなクラッシュを誘発することがある。
    以下の安全策を講じる:
      - 再入防止: 既に実行済なら即 return
      - sys.is_finalizing() を検出したら最小限 (VideoCapture 等 release) のみに限定
      - 例外は握りつぶさずデバッグ環境変数 ENABLE_VERBOSE_CLEANUP=1 で詳細表示
    """
    global _CLEANUP_RAN
    if _CLEANUP_RAN:
        return
    _CLEANUP_RAN = True

    import sys as _sys
    verbose = os.getenv('ENABLE_VERBOSE_CLEANUP', '0') not in ('0','false','False')
    finalizing = getattr(_sys, 'is_finalizing', lambda: False)()
    def _log(msg):
        if verbose:
            print(f"[CLEANUP] {msg}")
    try:
        # 1) ウィンドウ / GUI (finalizing 中は避ける)
        if not finalizing:
            try:
                cv.destroyAllWindows()
                _log('destroyAllWindows ok')
            except Exception as e:
                _log(f'destroyAllWindows err: {e}')
        # 2) Pose インスタンス
        for _p_name in ('pose0','pose1'):
            _p = globals().get(_p_name)
            if _p is None:
                continue
            close_fn = getattr(_p, 'close', None)
            if callable(close_fn):
                try:
                    close_fn()
                    _log(f'{_p_name}.close ok')
                except Exception as e:
                    _log(f'{_p_name}.close err: {e}')
            globals()[_p_name] = None
        # 3) VideoCapture / Writer
        for name in ('cap0','cap1'):
            obj = globals().get(name)
            if obj is not None:
                try:
                    obj.release()
                    _log(f'{name}.release ok')
                except Exception as e:
                    _log(f'{name}.release err: {e}')
                globals()[name] = None
        for name in ('writer0','writer1'):
            obj = globals().get(name)
            if obj is not None:
                try:
                    obj.release()
                    _log(f'{name}.release ok')
                except Exception as e:
                    _log(f'{name}.release err: {e}')
                globals()[name] = None
        # 4) シリアル
        try:
            serial_ctrl.close()
            _log('serial closed')
        except Exception as e:
            _log(f'serial close err: {e}')
        # 4b) HX711 Serial クライアントのクローズ
        try:
            _hx_cli = globals().get('hx_serial_client')
            if _hx_cli is not None:
                close_fn = getattr(_hx_cli, 'close', None)
                if callable(close_fn):
                    close_fn()
                    _log('hx_serial_client.close ok')
                globals()['hx_serial_client'] = None
        except Exception as e:
            _log(f'hx_serial_client close err: {e}')
        # 5) 旧 Matplotlib 図のクローズ処理は不要（PyQtGraph に移行）
    finally:
        if _fatal_state["exception"] and verbose:
            print("[CLEANUP] Fatal exception recorded earlier:")
            print(_fatal_state["exception"])  # traceback text

atexit.register(_cleanup_resources)

# ========= 既存メイン処理はこの直後の while ループ =========

THRESHOLD = None



aim_torque = []
I1 = I2 = I3 = I4 = I5 = I6 = I7 = None

storage = BodyPartDataStorage()
# 部位ごとの計算設定を辞書に格納
part_calculations = {
    "upper_arm_R": {"start": 3, "end": 1},
    "forearm_R": {"start": 5, "end": 3},
    "both_shoulder": {"start": 0, "end": 1},
    "both_hip": {"start": 6, "end": 7},
    "up_arm_l": {"start": 2, "end": 0},
    "forearm_L": {"start": 4, "end": 2},
    "upper_Leg_R": {"start": 7, "end": 9},
    "upper_Leg_L": {"start": 6, "end": 8},
}
# LinkVectorCalculatorのインスタンスを辞書に保持
calculators = {
    part: LinkVectorCalculator(s["start"], s["end"])
    for part, s in part_calculations.items()
}
print("Calculators created")

logger.info("Video loaded")
local_input_stream1, local_input_stream2, file_mode, resolve_reason = resolve_input_streams()
print(f"Input streams resolved: file_mode={file_mode}, reason={resolve_reason}")

# 平面拘束のランタイム有効化判定
if GRAVITY_LEVEL_PLANE_ON:
    _gravity_level_plane_on_runtime = bool(file_mode or GRAVITY_LEVEL_PLANE_WEBCAM_OK)
else:
    _gravity_level_plane_on_runtime = False
if E_DEBUG:
    print(
        f"[GRAVITY] level_plane_runtime={_gravity_level_plane_on_runtime} "
        f"(cfg_on={GRAVITY_LEVEL_PLANE_ON}, plane={GRAVITY_LEVEL_PLANE}, "
        f"file_mode={file_mode}, webcam_ok={GRAVITY_LEVEL_PLANE_WEBCAM_OK})"
    )

# チェッカーボード短辺方向を優先した重力定義（初期キャリブレーション結果）
if GRAVITY_FROM_CHECKERBOARD_SHORT and (not _gravity_set):
    _v_cb = _load_checkerboard_short_axis_runtime(GRAVITY_CHECKERBOARD_AXIS_FILE)
    if _v_cb is not None:
        up_label, axis_unit, cosabs = _pick_axis_from_vector(_v_cb)
        g_label = _opposite_axis_label(up_label)
        g_mag = float(np.linalg.norm(g)) if np.all(np.isfinite(g)) else 9.80665
        new_g = -axis_unit * g_mag
        globals()['g'] = np.array(new_g, dtype=float)
        globals()['_gravity_label'] = g_label
        globals()['_gravity_set'] = True
        print(f"[GRAVITY] checkerboard_short up={up_label} g={g_label} vec={new_g.tolist()} file='{GRAVITY_CHECKERBOARD_AXIS_FILE}'")
    elif E_DEBUG:
        print(f"[GRAVITY] checkerboard short-axis file not found/invalid: '{GRAVITY_CHECKERBOARD_AXIS_FILE}' -> fallback auto-detect")

def _dbg(*args, **kwargs):
    if IO_DEBUG:
        logger.debug(" ".join(str(a) for a in args))

 # moved to video_io.open_capture_and_read_first

reps = 10
# REP_COL_NAME を使用して 1RM% を取得
one_rm_row = df_rm.loc[df_rm[REP_COL_NAME] == reps]
if one_rm_row.empty:
    raise KeyError(
        f"rm_method: {REP_COL_NAME} に {reps} 行が見つかりません。利用可能な値: {sorted(df_rm[REP_COL_NAME].dropna().unique().tolist())}"
    )
one_rm_percentage = one_rm_row["1RM%"].values[0] / 100
# ファイルを読み込みモードで開く
with open(folder_path + "\\max_value.txt", "r", encoding="utf-8") as file:
    # ファイルからデータを一行読み込む
    line = file.readline().strip()  # strip()で余計な空白や改行を除去
    if line == "":
        print("No data in file")
    else:
        # 読み込んだデータをfloatに変換#X
        max_value = float(line)
        #    print(f"ファイル内の最大値: {max_value}")
        THRESHOLD = max_value / one_rm_percentage
# get projection matrices
# --- ユーザーに「準備OKで再開してね」と促す ---
CALIB_BASE_DIR = os.getenv('CALIB_BASE_DIR', '').strip()
if CALIB_BASE_DIR:
    P0 = get_projection_matrix(0, file_mode, base_dir=CALIB_BASE_DIR)
    P1 = get_projection_matrix(1, file_mode, base_dir=CALIB_BASE_DIR)
    print(f"[CALIB] using CALIB_BASE_DIR={CALIB_BASE_DIR}")
else:
    P0 = get_projection_matrix(0, file_mode)
    P1 = get_projection_matrix(1, file_mode)
P0_F64 = np.asarray(P0, dtype=np.float64)
P1_F64 = np.asarray(P1, dtype=np.float64)
print("Projection matrices loaded")
# %%
# 参照順の都合で、デモ用フラグをここでも安全に初期化
if 'DEMO_MONO_GAUGE_ON' not in globals():
    DEMO_MONO_GAUGE_ON = os.getenv('DEMO_MONO_GAUGE_ON', '1') in ('1', 'true', 'True')
if 'DEMO_MONO_CAM0_ONLY' not in globals():
    DEMO_MONO_CAM0_ONLY = os.getenv('DEMO_MONO_CAM0_ONLY', '1') in ('1', 'true', 'True')

save_path0 = f"cam0_output_{timestamp}.mp4"
save_path1 = f"cam1_output_{timestamp}.mp4"

cap0, ok0, frame0 = open_capture_and_read_first(local_input_stream1)
if DEMO_MONO_GAUGE_ON and DEMO_MONO_CAM0_ONLY:
    cap1, ok1 = None, ok0
    frame1 = (None if frame0 is None else frame0.copy())
else:
    cap1, ok1, frame1 = open_capture_and_read_first(local_input_stream2)

if not (ok0 and ok1) or frame0 is None or frame1 is None:
    if DEMO_MONO_GAUGE_ON and DEMO_MONO_CAM0_ONLY:
        print("❌ cam0 の入力読み込みに失敗しました（mono mode）。")
        sys.exit(2)
    print("❌ 入力の読み込みに失敗しました。解決理由:", resolve_reason)
    opened = False
    # 優先順に従って試す
    if PREFER_RECORDING_PAIRS:
        # 1) 録画ペア
        pairs = find_recording_pairs(folder_path)
        if pairs:
            print(f"Trying pairs: {len(pairs)} found (newest first)")
        tried = 0
        for p0, p1 in pairs:
            tried += 1
            print(f"→ Try pair[{tried}]: {os.path.basename(p0)} , {os.path.basename(p1)}")
            try:
                cap0.release(); cap1.release()
            except Exception:
                pass
            cap0, ok0, frame0 = open_capture_and_read_first(p0)
            cap1, ok1, frame1 = open_capture_and_read_first(p1)
            if ok0 and ok1 and frame0 is not None and frame1 is not None:
                print("OK: opened recording pair")
                opened = True
                break
        # 2) サンプル
        if not opened:
            cam0_file = os.path.join(folder_path, "media", "cam000_test.mp4")
            cam1_file = os.path.join(folder_path, "media", "cam111_test.mp4")
            print("→ Try media samples:", cam0_file, cam1_file)
            try:
                cap0.release(); cap1.release()
            except Exception:
                pass
            cap0, ok0, frame0 = open_capture_and_read_first(cam0_file)
            cap1, ok1, frame1 = open_capture_and_read_first(cam1_file)
            opened = ok0 and ok1 and frame0 is not None and frame1 is not None
    else:
        # 1) サンプル
        cam0_file = os.path.join(folder_path, "media", "cam000_test.mp4")
        cam1_file = os.path.join(folder_path, "media", "cam111_test.mp4")
        print("→ Try media samples:", cam0_file, cam1_file)
        try:
            cap0.release(); cap1.release()
        except Exception:
            pass
        cap0, ok0, frame0 = open_capture_and_read_first(cam0_file)
        cap1, ok1, frame1 = open_capture_and_read_first(cam1_file)
        opened = ok0 and ok1 and frame0 is not None and frame1 is not None
        # 2) 録画ペア
        if not opened:
            pairs = find_recording_pairs(folder_path)
            if pairs:
                print(f"Trying pairs: {len(pairs)} found (newest first)")
            tried = 0
            for p0, p1 in pairs:
                tried += 1
                print(f"→ Try pair[{tried}]: {os.path.basename(p0)} , {os.path.basename(p1)}")
                try:
                    cap0.release(); cap1.release()
                except Exception:
                    pass
                cap0, ok0, frame0 = open_capture_and_read_first(p0)
                cap1, ok1, frame1 = open_capture_and_read_first(p1)
                if ok0 and ok1 and frame0 is not None and frame1 is not None:
                    print("OK: opened recording pair")
                    opened = True
                    break
    if not opened:
        # 見つかったペア一覧を提示
        pairs = find_recording_pairs(folder_path)
        if pairs:
            print("Tried these pairs (all failed):")
            for p0, p1 in pairs:
                print(" -", os.path.basename(p0), ",", os.path.basename(p1))
        else:
            print("No cam0/cam1 recording pairs found under:", folder_path)
        print("❌ 利用可能な入力ソースを開けませんでした。終了します。")
        sys.exit(2)
    else:
        # フォールバック後に新しい cap へ差し替えた場合、file_mode では先頭へ巻き戻す
        if file_mode:
            try:
                cap0.set(cv.CAP_PROP_POS_FRAMES, 0)
                cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
                print("[Input] file_mode (post-fallback): rewind to frame 0")
            except Exception as _rew2_e:
                print(f"[Input] rewind (post-fallback) failed (non-fatal): {_rew2_e}")

_dbg("resolved streams:", local_input_stream1, local_input_stream2, "file_mode=", file_mode)
if isinstance(local_input_stream1, str):
    _dbg("src1 exists=", os.path.exists(local_input_stream1), "size=", os.path.getsize(local_input_stream1) if os.path.exists(local_input_stream1) else -1)
if isinstance(local_input_stream2, str):
    _dbg("src2 exists=", os.path.exists(local_input_stream2), "size=", os.path.getsize(local_input_stream2) if os.path.exists(local_input_stream2) else -1)

if frame0 is None or frame1 is None:
    print("❌ 初期フレームが取得できませんでした。終了します。")
    sys.exit(1)

h0, w0 = frame0.shape[:2]
h1, w1 = frame1.shape[:2]
_dbg("first frame sizes:", (w0, h0), (w1, h1))
try:
    total0 = cap0.get(cv.CAP_PROP_FRAME_COUNT)
    total1 = (cap1.get(cv.CAP_PROP_FRAME_COUNT) if cap1 is not None else total0)
    fps0 = cap0.get(cv.CAP_PROP_FPS)
    fps1 = (cap1.get(cv.CAP_PROP_FPS) if cap1 is not None else fps0)
    pos0 = cap0.get(cv.CAP_PROP_POS_FRAMES)
    pos1 = (cap1.get(cv.CAP_PROP_POS_FRAMES) if cap1 is not None else pos0)
    print(f"[Input] stats: frames0={total0}, fps0={fps0}, pos0={pos0} | frames1={total1}, fps1={fps1}, pos1={pos1}")
except Exception:
    pass

# ファイル入力の場合は、初期フレーム消費後に先頭へ巻き戻してからメインループに入る
if file_mode:
    try:
        cap0.set(cv.CAP_PROP_POS_FRAMES, 0)
        if cap1 is not None:
            cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
        print("[Input] file_mode: rewind to frame 0")
    except Exception as _rew_e:
        print(f"[Input] rewind failed (non-fatal): {_rew_e}")

# FPS（必要なら動画から取得する）


# 保存用VideoWriterの初期化（frameの実サイズに合わせる）
fourcc = cv.VideoWriter_fourcc(*"mp4v")
writer0 = cv.VideoWriter(save_path0, fourcc, fps, (w0, h0))
writer1 = cv.VideoWriter(save_path1, fourcc, fps, (w1, h1))
_dbg("writers opened:", writer0.isOpened(), writer1.isOpened(), "paths:", save_path0, save_path1)

# 動作チェック
if not writer0.isOpened():
    print(f"❌ writer0 の初期化に失敗しました（サイズ: {w0}x{h0}）")
if not writer1.isOpened():
    print(f"❌ writer1 の初期化に失敗しました（サイズ: {w1}x{h1}）")


kpts_3d = []  # 3Dキーポイントデータを格納するリスト
mono3d_records = []  # 単眼(Mediapipe world) 3D 座標の記録

landmark_ekf = None
if EKF_ENABLE:
    try:
        _ekf_cfg = EKFConfig(q_acc=EKF_Q_ACC, r=EKF_R, gate_std=EKF_GATE_STD)
        landmark_ekf = LandmarkEKF(len(pose_keypoints), fs=fps, cfg=_ekf_cfg, bpf_low=EKF_BPF_LOW, bpf_high=EKF_BPF_HIGH, bpf_order=EKF_BPF_ORDER)
        print(f"[EKF] enabled: q_acc={EKF_Q_ACC} r={EKF_R} gate={EKF_GATE_STD} bpf=({EKF_BPF_LOW},{EKF_BPF_HIGH})")
    except Exception as _ekf_init_e:
        print(f"[EKF] disabled (init failed): {_ekf_init_e}")
        landmark_ekf = None

DEBUG_LOGS = os.getenv('DEBUG_LOGS', '0') not in ('0','false','False')
# 追加: 動力学＆姿勢デバッグの詳細トグル（必要時のみON）
TRACE_DYN = os.getenv('TRACE_DYN', '0') not in ('0','false','False')
TRACE_POSE = os.getenv('TRACE_POSE', '0') not in ('0','false','False')
try:
    TRACE_EVERY = max(1, int(os.getenv('TRACE_EVERY', '10')))
except Exception:
    TRACE_EVERY = 10
print("Starting loop")

# 姿勢推定のための初期フレーム取得（再実行してOK）
ret0, frame0 = cap0.read()
if cap1 is not None:
    ret1, frame1 = cap1.read()
else:
    ret1, frame1 = ret0, (None if frame0 is None else frame0.copy())

# Mediapipe用にRGB化 & 推定
if frame0 is None or frame1 is None:
    print("⚠️ 初期フレームが None です。再試行します。")
    # 1) 数回リトライ
    recovered = False
    for _i in range(10):
        ret0, frame0 = cap0.read()
        if cap1 is not None:
            ret1, frame1 = cap1.read()
        else:
            ret1, frame1 = ret0, (None if frame0 is None else frame0.copy())
        if ret0 and ret1 and frame0 is not None and frame1 is not None:
            recovered = True
            break
        time.sleep(0.05)
    # 2) キャプチャを再作成
    if not recovered:
        try:
            cap0.release(); cap1.release()
        except Exception:
            pass
        cap0, ok0, frame0 = open_capture_and_read_first(local_input_stream1)
        cap1, ok1, frame1 = open_capture_and_read_first(local_input_stream2)
        recovered = ok0 and ok1 and frame0 is not None and frame1 is not None
    # 3) サンプル／録画ペアへフォールバック
    if not recovered:
        print("[Input] Fallback: try sample media or recording pairs")
        opened = False
        if PREFER_RECORDING_PAIRS:
            pairs = find_recording_pairs(folder_path)
            for p0, p1 in pairs:
                try:
                    cap0.release(); cap1.release()
                except Exception:
                    pass
                cap0, ok0, frame0 = open_capture_and_read_first(p0)
                cap1, ok1, frame1 = open_capture_and_read_first(p1)
                if ok0 and ok1 and frame0 is not None and frame1 is not None:
                    opened = True
                    break
        if not opened:
            cam0_file = os.path.join(folder_path, "media", "cam000_test.mp4")
            cam1_file = os.path.join(folder_path, "media", "cam111_test.mp4")
            try:
                cap0.release(); cap1.release()
            except Exception:
                pass
            cap0, ok0, frame0 = open_capture_and_read_first(cam0_file)
            cap1, ok1, frame1 = open_capture_and_read_first(cam1_file)
            opened = ok0 and ok1 and frame0 is not None and frame1 is not None
        if not opened:
            print("❌ 初期フレームが取得できませんでした。終了します。")
            sys.exit(1)
frame0_rgb = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
frame1_rgb = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
# MediaPipe 入力を縮小（推論専用バッファ）
if MP_INPUT_SCALE < 1.0:
    mp0 = cv.resize(frame0_rgb, None, fx=MP_INPUT_SCALE, fy=MP_INPUT_SCALE, interpolation=cv.INTER_AREA)
    mp1 = cv.resize(frame1_rgb, None, fx=MP_INPUT_SCALE, fy=MP_INPUT_SCALE, interpolation=cv.INTER_AREA)
else:
    mp0, mp1 = frame0_rgb, frame1_rgb
results0 = pose0.process(mp0)
results1 = pose1.process(mp1)

# キーポイント抽出（描画OFFでもOK）
frame0_kpts, frame1_kpts = extract_keypoints(
    results0, results1, pose_keypoints, frame0, frame1
)

# --- Poseヒット率の簡易診断（TRACE_EVERYフレごと、TRACE_POSE=1のとき） ---
if TRACE_POSE and (WHILE_COUNT % TRACE_EVERY == 0):
    def _count_valid(kpts):
        try:
            return int(sum(1 for (x, y) in kpts if x >= 0 and y >= 0))
        except Exception:
            return 0
    v0 = _count_valid(frame0_kpts)
    v1 = _count_valid(frame1_kpts)
    print(f"[POSE] frame={WHILE_COUNT} valid2D: cam0={v0} cam1={v1}")
    # 詳細を見たい場合は下記を一時的に解除
    # print('[POSE] kpts0=', frame0_kpts)
    # print('[POSE] kpts1=', frame1_kpts)


# 有効なX座標のみ抽出（フレーム幅を安全に参照）
def get_valid_x_range(kpts, frame_width):
    valid = [x for x, y in kpts if x >= 0]
    if not valid:
        return 0, frame_width  # 検出なしなら該当フレームのフル幅
    local_x_min = max(0, int(min(valid)))
    local_x_max = min(frame_width, int(max(valid)))
    return local_x_min, local_x_max


def _resize_mp_input(frame_rgb):
    if MP_INPUT_SCALE < 1.0:
        return cv.resize(frame_rgb, None, fx=MP_INPUT_SCALE, fy=MP_INPUT_SCALE, interpolation=cv.INTER_AREA)
    return frame_rgb


def _pose_has_landmarks(results) -> bool:
    try:
        pl = getattr(results, 'pose_landmarks', None)
        return (pl is not None) and hasattr(pl, 'landmark') and (len(pl.landmark) > 0)
    except Exception:
        return False


def _roi_from_keypoints(kpts, frame_shape):
    h, w = frame_shape[:2]
    valid = [(int(x), int(y)) for x, y in kpts if x >= 0 and y >= 0]
    if len(valid) < max(1, POSE_ROI_MIN_VALID_KPTS):
        return None

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    margin_x = int(round((x1 - x0 + 1) * POSE_ROI_MARGIN_RATIO))
    margin_y = int(round((y1 - y0 + 1) * POSE_ROI_MARGIN_RATIO))
    x0 -= margin_x
    x1 += margin_x
    y0 -= margin_y
    y1 += margin_y

    min_side = int(round(max(1, min(w, h)) * POSE_ROI_MIN_SIDE_RATIO))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = max(min_side // 2, (x1 - x0 + 1) // 2, (y1 - y0 + 1) // 2)

    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _expand_roi(roi, frame_shape, grow_ratio=0.25):
    if roi is None:
        return None
    h, w = frame_shape[:2]
    x0, y0, x1, y1 = roi
    rw = max(1, int(x1 - x0))
    rh = max(1, int(y1 - y0))
    grow_x = int(round(rw * max(0.0, grow_ratio)))
    grow_y = int(round(rh * max(0.0, grow_ratio)))

    nx0 = max(0, x0 - grow_x)
    ny0 = max(0, y0 - grow_y)
    nx1 = min(w, x1 + grow_x)
    ny1 = min(h, y1 + grow_y)
    if nx1 <= nx0 or ny1 <= ny0:
        return None
    return int(nx0), int(ny0), int(nx1), int(ny1)


def _remap_landmarks_to_fullframe(results, roi, full_shape):
    if not _pose_has_landmarks(results):
        return False
    x0, y0, x1, y1 = roi
    h, w = full_shape[:2]
    rw = max(1, int(x1 - x0))
    rh = max(1, int(y1 - y0))
    try:
        for lm in results.pose_landmarks.landmark:
            lx = float(lm.x)
            ly = float(lm.y)
            lm.x = (x0 + lx * rw) / float(w)
            lm.y = (y0 + ly * rh) / float(h)
        return True
    except Exception:
        return False


def _pose_process_with_roi(pose_estimator, frame_rgb, roi):
    if (not POSE_ROI_ON) or (roi is None):
        return pose_estimator.process(_resize_mp_input(frame_rgb)), False

    x0, y0, x1, y1 = roi
    crop = frame_rgb[y0:y1, x0:x1]
    if crop is None or crop.size == 0:
        return pose_estimator.process(_resize_mp_input(frame_rgb)), False

    res = pose_estimator.process(_resize_mp_input(crop))
    if _pose_has_landmarks(res):
        _remap_landmarks_to_fullframe(res, roi, frame_rgb.shape)
    return res, True


def _triangulate_points_batch(P0_f64, P1_f64, keypoints0, keypoints1):
    n = len(keypoints0)
    out = np.full((n, 3), -1.0, dtype=np.float64)
    valid_idx = [i for i, (uv0, uv1) in enumerate(zip(keypoints0, keypoints1)) if uv0[0] != -1 and uv1[0] != -1]
    if not valid_idx:
        return out

    pts0 = np.array([[float(keypoints0[i][0]), float(keypoints0[i][1])] for i in valid_idx], dtype=np.float64).T
    pts1 = np.array([[float(keypoints1[i][0]), float(keypoints1[i][1])] for i in valid_idx], dtype=np.float64).T

    Xh = cv.triangulatePoints(P0_f64, P1_f64, pts0, pts1)
    wv = Xh[3, :]

    for j, idx in enumerate(valid_idx):
        wj = float(wv[j])
        if (not np.isfinite(wj)) or abs(wj) < 1e-12:
            continue
        Xj = (Xh[:3, j] / wj).astype(np.float64)
        if np.all(np.isfinite(Xj)):
            out[idx] = Xj
    return out


def _triangulate_transform_batch(P0_f64, P1_f64, keypoints0, keypoints1):
    k0 = np.asarray(keypoints0, dtype=np.float64)
    k1 = np.asarray(keypoints1, dtype=np.float64)

    if TRIANG_NATIVE_ON:
        try:
            out_native = compute_triangulate_transform_native(P0_f64, P1_f64, k0, k1, scale=0.01)
            if isinstance(out_native, np.ndarray) and out_native.ndim == 2 and out_native.shape[1] == 3:
                return out_native
        except Exception as _tri_e:  # noqa: BLE001
            if DEBUG_LOGS and (WHILE_COUNT % 120 == 0):
                print(f"[TRIANG] native failed; fallback to python batch: {_tri_e}")

    frame_p3ds = _triangulate_points_batch(P0_f64, P1_f64, keypoints0, keypoints1)
    temp_np = np.asarray(frame_p3ds, dtype=np.float64).reshape((-1, 3)) * 0.01
    transformed_p3ds = np.empty_like(temp_np)
    transformed_p3ds[:, 0] = -temp_np[:, 0]
    transformed_p3ds[:, 1] = -temp_np[:, 2]
    transformed_p3ds[:, 2] = -temp_np[:, 1]

    # Fallback path uses -1 sentinel for missing points; convert to NaN for finite checks.
    invalid_mask = np.all(np.isclose(frame_p3ds, -1.0), axis=1) | (~np.all(np.isfinite(frame_p3ds), axis=1))
    transformed_p3ds[invalid_mask] = np.nan
    return transformed_p3ds


def _extract_keypoints_fast_single(results, pose_ids, frame_shape):
    try:
        pl = getattr(results, 'pose_landmarks', None)
        if pl is None:
            return [[-1, -1]] * len(pose_ids)
        lm_list = pl.landmark
        if lm_list is None or len(lm_list) == 0:
            return [[-1, -1]] * len(pose_ids)
    except Exception:
        return [[-1, -1]] * len(pose_ids)

    h, w = frame_shape[:2]
    out = []
    for pid in pose_ids:
        lm = lm_list[pid]
        out.append([int(round(float(lm.x) * w)), int(round(float(lm.y) * h))])
    return out


def _extract_keypoints_pair(results0, results1, pose_ids, frame0_bgr, frame1_bgr):
    if KPS_FAST_ON and (not DRAW_KEYPOINTS_ON):
        return (
            _extract_keypoints_fast_single(results0, pose_ids, frame0_bgr.shape),
            _extract_keypoints_fast_single(results1, pose_ids, frame1_bgr.shape),
        )
    return extract_keypoints(results0, results1, pose_ids, frame0_bgr, frame1_bgr)


x0_min, x0_max = get_valid_x_range(frame0_kpts, frame0.shape[1])
x1_min, x1_max = get_valid_x_range(frame1_kpts, frame1.shape[1])

x_min = min(x0_min, x1_min)
x_max = max(x0_max, x1_max)
x_margin = max(0, int(POSE_X_CROP_MARGIN))

# 2カメラで共通に安全なトリミング範囲を設定
width_min = min(frame0.shape[1], frame1.shape[1])
x_start = max(0, x_min - x_margin)
x_end = min(width_min, x_max + x_margin)

# 推定範囲が狭すぎると検出ロストを誘発するため、最低幅を確保
_min_crop_w = int(round(width_min * max(0.3, min(1.0, POSE_X_CROP_MIN_WIDTH_RATIO))))
if (x_end - x_start) < _min_crop_w:
    _cx = (x_start + x_end) // 2
    _half = _min_crop_w // 2
    x_start = max(0, _cx - _half)
    x_end = min(width_min, _cx + _half)

if x_end <= x_start:
    # フォールバック: 全幅
    x_start, x_end = 0, width_min
print(f"✅ トリミング範囲: x = {x_start} ～ {x_end}")

# Pose ROIトラッキング状態（2カメラ）
_pose_roi0 = _roi_from_keypoints(frame0_kpts, frame0.shape) if POSE_ROI_ON else None
_pose_roi1 = _roi_from_keypoints(frame1_kpts, frame1.shape) if POSE_ROI_ON else None
_pose_roi0_miss = 0
_pose_roi1_miss = 0


# %%
# M/F 計算をまとめて走らせる小ユーティリティ
def run_specs(specs):
    Ms, Fs, Parts = [], [], []
    # ネイティブ一括（環境変数でON）
    USE_NATIVE_DYNAMICS = os.getenv('USE_NATIVE_DYNAMICS', '1') in ('1','true','True')
    if USE_NATIVE_DYNAMICS and specs:
        try:
            # バッチ入力を収集
            from utils_dynamic import compute_MF_batch_native
            I_batch = []
            m_batch = []
            omega = []
            dot_omega = []
            ddpg = []
            parts = []
            native_possible = True
            for I, mass, data_seq, kwargs in specs:
                # calculate_M_and_F が参照するデータ形状: data_seq[-1]['omega'], ['dot_omega'], ['dot_dot_pg'], ['part_name']
                if not data_seq:
                    native_possible = False
                    break
                last = data_seq[-1]
                I_use = I if I is not None else np.zeros((3,3))
                m_eff = float(mass) if mass is not None else 0.0
                wv = np.array(last.get('omega', np.zeros(3)), dtype=np.float64)
                dwv = np.array(last.get('dot_omega', np.zeros(3)), dtype=np.float64)
                ag = np.array(last.get('dot_dot_pg', np.zeros(3)), dtype=np.float64)

                # Imode 補正（utils_dynamic.calculate_M_and_F と一致）
                Imode = (kwargs or {}).get('Imode', None)
                if Imode == 3:
                    Info_I3 = (kwargs or {}).get('Info_I3', None)
                    add_part_data = (kwargs or {}).get('add_part_data', None)
                    condition = (kwargs or {}).get('condition', None)
                    if Info_I3 is None or add_part_data is None or not add_part_data:
                        native_possible = False
                        break
                    A1 = float(np.linalg.norm((np.array(Info_I3[1][:2]) + np.array(Info_I3[0][:2])) * 0.5 - np.array(Info_I3[5][:2])))
                    A0 = float(np.linalg.norm((np.array(Info_I3[1][:2]) + np.array(Info_I3[0][:2])) * 0.5 - np.array(Info_I3[4][:2])))
                    ag = (ag * 3.0 + np.array(add_part_data[-1]['dot_dot_pg'], dtype=np.float64)) * 0.25
                    if condition == 1:
                        m_eff = w * 0.276 * A0 / max(A0 + A1, 1e-12)
                        wv = -wv
                        dwv = -dwv
                    elif condition == 0:
                        m_eff = w * 0.276 * A1 / max(A0 + A1, 1e-12)
                elif Imode == 4:
                    add_part_data = (kwargs or {}).get('add_part_data', None)
                    if add_part_data is None or not add_part_data:
                        native_possible = False
                        break
                    wv = np.zeros(3, dtype=np.float64)
                    dwv = np.zeros(3, dtype=np.float64)
                    ag = (ag * 3.0 + np.array(add_part_data[-1]['dot_dot_pg'], dtype=np.float64)) * 0.25

                I_batch.append(np.array(I_use, dtype=np.float64))
                m_batch.append(m_eff)
                omega.append(wv)
                dot_omega.append(dwv)
                ddpg.append(ag)
                parts.append(last.get('part_name', 'unknown'))
            if native_possible and I_batch:
                I_b = np.ascontiguousarray(np.stack(I_batch, axis=0), dtype=np.float64)
                m_b = np.ascontiguousarray(np.array(m_batch, dtype=np.float64))
                w_b = np.ascontiguousarray(np.stack(omega, axis=0), dtype=np.float64)
                dw_b = np.ascontiguousarray(np.stack(dot_omega, axis=0), dtype=np.float64)
                a_b = np.ascontiguousarray(np.stack(ddpg, axis=0), dtype=np.float64)
                g_v = np.ascontiguousarray(np.array(g, dtype=np.float64))
                M_b, F_b = compute_MF_batch_native(I_b, m_b, w_b, dw_b, a_b, g_v)
                if TRACE_DYN and (WHILE_COUNT % TRACE_EVERY == 0):
                    # 入出力のノルム統計でゼロ化を監査
                    def _nz(a):
                        return int(np.count_nonzero(np.isfinite(a) & (np.abs(a) > 1e-12)))
                    print(f"[DYN:MF] N={I_b.shape[0]} | w_nz={_nz(w_b)} dw_nz={_nz(dw_b)} acc_nz={_nz(a_b)} m_sum={float(np.sum(m_b)):.3f}")
                    print(f"[DYN:MF] out | M_norm={float(np.linalg.norm(M_b)):.6f} F_norm={float(np.linalg.norm(F_b)):.6f}")
                    # 詳細を見たい場合は下記を一時的に解除
                    # print('[DYN:MF] M_b=', M_b)
                    # print('[DYN:MF] F_b=', F_b)
                Ms = [M_b[i] for i in range(M_b.shape[0])]
                Fs = [F_b[i] for i in range(F_b.shape[0])]
                Parts = list(parts)
                return Ms, Fs, Parts
        except (RuntimeError, ValueError, TypeError, AttributeError) as _e:
            if os.getenv('POSE_DEBUG','0') in ('1','true','True'):
                print(f"[Dyn] native MF fallback due to: {_e}")
            # フォールバックで個別計算へ
    # フォールバック: 個別にPythonで計算
    for I, mass, data_seq, kwargs in specs:
        M, F, Part = calculate_M_and_F(I, mass, data_seq, g, **kwargs)
        Ms.append(M)
        Fs.append(F)
        Parts.append(Part)
    return Ms, Fs, Parts

# フレームスキップ管理（設定値を変えずローカルで制御）
skip_counter = 0
# SKIP_FRAMES <= 0 の場合は無間引き (=1)
skip_mod = SKIP_FRAMES if isinstance(SKIP_FRAMES, int) and SKIP_FRAMES > 0 else 1
# シリアルが有効なら起動時に DTR パルス
if serial_ctrl.enabled:
    serial_ctrl.pulse_dtr(delay=0.05)
    # パルス後はポートを即時解放し、以降の RecorderClient で同ポートを使用できるようにする
    try:
        serial_ctrl.close()
        print("[Serial] DTR パルス後にポートを解放しました")
    except Exception:
        pass

# ================= HX711 ロガー開始処理 (メイン while 前) =================
hx_start_iso: str | None = None
hx_recording_started = False
hx_serial_client: object | None = None
HX_BASE_URL = os.getenv('HX711_BASE_URL', 'http://192.168.4.1')  # Wi-Fi 経由ダウンロード用 (任意)

# 追加: HX711 クイック診断設定（最初の使用箇所より前に配置）
HX_DIAG = os.getenv('HX_DIAG', '1') in ('1','true','True')

if HX711_ENABLE and (RecorderClient is not None):
    # シリアルポート推定: まずは環境変数 M5_PORT、無ければ COM3 を採用
    _m5_port = os.getenv('M5_PORT') or 'COM3'
    if _m5_port:
        try:
            hx_serial_client = RecorderClient(serial_port=_m5_port, baud=115200, base_url=HX_BASE_URL)
            # DTRリセット等の直後はデバイス初期化待ちが必要な場合があるため、短い待機を入れる
            try:
                time.sleep(0.8)
            except Exception:
                pass
            # まずは EPOCH(ms) で開始（ファームが ISO 未対応の可能性に備える）
            epoch_ms = int(time.time() * 1000)
            ok = hx_serial_client.start_with_epoch(epoch_ms)
            if ok:
                hx_start_iso = _iso8601_utc_ms()
                hx_recording_started = True
                print(f"[HX711] START_EPOCH {epoch_ms} -> OK (port={_m5_port})")
                # 診断モード時は開始直後にも STATUS/HELP を取得してプロトコルとコマンド群を可視化
                if HX_DIAG:
                    try:
                        st = hx_serial_client.status_serial().strip()
                        print(f"[HX711] STATUS(serial): {st}")
                    except Exception as _se:
                        print(f"[HX711] STATUS 取得失敗: {_se}")
                    try:
                        getattr(hx_serial_client, '_send_line')('help')
                        help_lines = []
                        t_end = time.time() + 1.5
                        while time.time() < t_end:
                            chunk = getattr(hx_serial_client, '_read_until')('\n', 0.2)
                            if chunk:
                                help_lines.append(chunk)
                            else:
                                break
                        help_text = ''.join(help_lines).strip()
                        if help_text:
                            print('[HX711] help full:\n' + help_text)
                    except Exception:
                        pass
            else:
                # フォールバック: ISO8601
                hx_start_iso = _iso8601_utc_ms()
                ok2 = hx_serial_client.start_with_iso(hx_start_iso)
                hx_recording_started = ok2
                print(f"[HX711] START {hx_start_iso} -> {'OK' if ok2 else 'NG'} (port={_m5_port})")
                if not ok2:
                    try:
                        st = hx_serial_client.status_serial().strip()
                        print(f"[HX711] STATUS(serial): {st}")
                    except Exception as _se:
                        print(f"[HX711] STATUS 取得失敗: {_se}")
                    # 追加診断: 'help' を送って応答の先頭行を確認
                    try:
                        # _send_line/_read_until は hx711_recorder のローレベル関数
                        getattr(hx_serial_client, '_send_line')('help')
                        # ヘルプ全文を短時間収集
                        help_lines = []
                        t_end = time.time() + 1.5
                        while time.time() < t_end:
                            chunk = getattr(hx_serial_client, '_read_until')('\n', 0.2)
                            if chunk:
                                help_lines.append(chunk)
                            else:
                                break
                        help_text = ''.join(help_lines).strip()
                        if help_text:
                            print('[HX711] help full:\n' + help_text)
                        # 念のため小文字 'status' も試す
                        getattr(hx_serial_client, '_send_line')('status')
                        low_st = getattr(hx_serial_client, '_read_until')('\n', 1.0)
                        if low_st:
                            print('[HX711] status>', low_st.strip())
                    except Exception:
                        pass
        except Exception as e:  # noqa: BLE001
            print(f"[HX711] 開始失敗: {e}")
    else:
        print('[HX711] シリアルポート不明のため開始スキップ')
else:
    if HX711_ENABLE:
        print('[HX711] モジュール未インポートのため開始スキップ')
    else:
        print('[HX711] disabled by config (HX711_ENABLE=0)')


# サイクル内のトルクとパワー履歴（ゲージ集計用）
keys_for_hist = (gauge.part_keys if ('gauge' in globals() and gauge is not None) else part_keys)
current_power_history = {k: [] for k in keys_for_hist}
# 新: トルク成分ベースのエネルギー集計履歴（上腕=肘トルクy負, 前腕=手首トルクy正）
current_energy_component_history = {k: [] for k in keys_for_hist}
cycle_energy_debug_rows = []

# 上腕(=肘トルク)と前腕(=手首トルク)のキー集合（肩・体幹は未定のため除外）
ELBOW_KEYS = {"elbow_R", "elbow_L"}
WRIST_KEYS = {"wrist_R", "wrist_L"}
# 連続表示用 肘エネルギー(J)積算バッファ (Σ τ·dθ)
_continuous_last_theta = {"elbow_R": None, "elbow_L": None}
_continuous_energy_J = {"elbow_R": 0.0, "elbow_L": 0.0}
_demo_shoulder_base_y = {"R": None, "L": None}
_demo_elbow_base_deg = {"R": None, "L": None}
_demo_gauge_ratio = {"R": 0.0, "L": 0.0}
hx_csv_collected = False  # 途中停止時などで二重取得を防ぐ（シリアルCSV取得済みフラグ）
print("entering main loop")

def _hx711_csv_quick_diag_from_bytes(csv_bytes: bytes, max_rows: int = 20000) -> None:
    """HX711 CSV を簡易診断し、ch2(raw2)が0固定かどうか等をプリントする。
    期待ヘッダ: absEpochMs,rel_ms,kg1,kg2,kg_total,raw1,raw2（先頭#付き可）
    """
    try:
        txt = csv_bytes.decode('utf-8', errors='ignore')
    except Exception:
        print('[HX711][DIAG] decode失敗: bytes=', len(csv_bytes))
        return
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        print('[HX711][DIAG] 空CSV')
        return
    header = lines[0]
    if header.startswith('#'):
        header = header.lstrip('#')
    cols = [c.strip() for c in header.split(',')]
    idx = {c.lower(): i for i, c in enumerate(cols)}
    def _find(*names):
        for n in names:
            j = idx.get(n)
            if j is not None:
                return j
        return None
    i_kg1 = _find('kg1')
    i_kg2 = _find('kg2')
    i_kg_total = _find('kg_total','kgtotal','total')
    i_raw1 = _find('raw1')
    i_raw2 = _find('raw2')
    if any(v is None for v in (i_kg1, i_kg2, i_raw1, i_raw2)):
        print(f"[HX711][DIAG] 想定ヘッダが見当たりません cols={cols}")
        return
    vals_kg1 = []
    vals_kg2 = []
    vals_raw1 = []
    vals_raw2 = []
    for ln in lines[1: max_rows+1]:
        try:
            parts = ln.split(',')
            vals_kg1.append(float(parts[i_kg1]))
            vals_kg2.append(float(parts[i_kg2]))
            vals_raw1.append(float(parts[i_raw1]))
            vals_raw2.append(float(parts[i_raw2]))
        except Exception:
            continue
    if not vals_raw1:
        print('[HX711][DIAG] データ行なし')
        return
    import numpy as _np
    a1 = _np.asarray(vals_raw1, dtype=_np.float64)
    a2 = _np.asarray(vals_raw2, dtype=_np.float64)
    k1 = _np.asarray(vals_kg1, dtype=_np.float64)
    k2 = _np.asarray(vals_kg2, dtype=_np.float64)
    n = len(a1)
    z2 = int(_np.sum(a2 == 0))
    nz2 = n - z2
    print(f"[HX711][DIAG] rows={n} raw1[min,max]=({a1.min():.0f},{a1.max():.0f}) raw2[min,max]=({a2.min():.0f},{a2.max():.0f}) nonzero(raw2)={nz2} ({(nz2/max(1,n))*100:.1f}%)")
    if i_kg_total is not None:
        try:
            kt_vals = []
            for ln in lines[1: 1+n]:
                parts = ln.split(',')
                if len(parts) > i_kg_total:
                    kt_vals.append(float(parts[i_kg_total]))
            kt = _np.asarray(kt_vals, dtype=_np.float64)
            m = min(len(kt), len(k1))
            if m >= 3:
                diff = (kt[:m] - (k1[:m] + k2[:m]))
                print(f"[HX711][DIAG] kg_total-(kg1+kg2): mean={diff.mean():.3f} std={diff.std():.3f} min={diff.min():.3f} max={diff.max():.3f}")
        except Exception:
            pass
    def _safe_corr(x, y):
        if len(x) < 3:
            return _np.nan
        xv = x - x.mean()
        yv = y - y.mean()
        denom = _np.sqrt((xv**2).sum() * (yv**2).sum())
        return float((xv*yv).sum()/denom) if denom > 1e-12 else _np.nan
    c_k2_r2 = _safe_corr(k2, a2)
    c_k2_r1 = _safe_corr(k2, a1)
    print(f"[HX711][DIAG] corr(kg2,raw2)={c_k2_r2:.3f}  corr(kg2,raw1)={c_k2_r1:.3f}")
    if nz2 == 0 and (k2.max() - k2.min()) > 1e-6:
        print('[HX711][DIAG][WARN] raw2が全て0ですが、kg2は変動しています。CSV生成側で raw2 列未書込み/誤割当の可能性。')
        print('                   ファームのCSV出力（列順/値の割当）をご確認ください。')
    elif nz2 == 0:
        print('[HX711][DIAG][WARN] raw2が全て0です。CH2未配線、HX711のCHB未使用/無効、読取未実装などの可能性。')
        print('                   配線とファームのCH切替(READ B 32/64/128等)をご確認ください。')
    else:
        print('[HX711][DIAG] raw2に非ゼロ値があります（ハード/ファームは動作）。校正や符号の見直し推奨。')
# --- Optional perf logging (disabled by default) ---
PERF_LOG = bool(int(os.getenv('PERF_LOG', '0')))
PERF_INT = int(os.getenv('PERF_INT', '60')) if os.getenv('PERF_INT') else 15
LOOP_FILE_PLAYBACK = bool(int(os.getenv('LOOP_FILE_PLAYBACK', '0')))
# ステージ切り分け用: このステージ名の直後でループを早期終了（計測用）
# 例: STOP_AFTER=mediapipe / kps2d / triang / calc / run_specs / torques / local_torque / imshow / gauge / retrieve / write / crop / preproc / postproc
STOP_AFTER = os.getenv('STOP_AFTER', '').strip()
# 追加: デバッグ・バッチ用 フレーム上限（0: 無制限）
try:
    MAX_FRAMES = int(os.getenv('MAX_FRAMES', '0'))
except Exception:
    MAX_FRAMES = 0

# 追加: リアルタイム遅延反映スキップ（処理時間に応じて次フレームを間引く）
# 例: 30fps入力で1.0秒かかった場合、約30フレーム先を次の処理対象にする
RT_DELAY_SKIP_ON = os.getenv('RT_DELAY_SKIP_ON', '0') in ('1', 'true', 'True')
RT_DELAY_SKIP_GAIN = float(os.getenv('RT_DELAY_SKIP_GAIN', '1.0'))
RT_DELAY_SKIP_MIN = int(os.getenv('RT_DELAY_SKIP_MIN', '0'))
RT_DELAY_SKIP_MAX = int(os.getenv('RT_DELAY_SKIP_MAX', '300'))
# 立ち上がり区間だけ高密度に処理するバーストモード
RT_RISE_BURST_ON = os.getenv('RT_RISE_BURST_ON', '1') in ('1', 'true', 'True')
RT_RISE_ZVEL_THR = float(os.getenv('RT_RISE_ZVEL_THR', '0.0008'))  # [m/frame] 目安
RT_RISE_BURST_FRAMES = int(os.getenv('RT_RISE_BURST_FRAMES', '12'))
RT_RISE_BURST_COOLDOWN = int(os.getenv('RT_RISE_BURST_COOLDOWN', '18'))
RT_RISE_ZDIST_MARGIN = float(os.getenv('RT_RISE_ZDIST_MARGIN', '0.0035'))  # cycle閾値近傍で先行発火
RT_RISE_BURST_SKIP = int(os.getenv('RT_RISE_BURST_SKIP', '7'))  # burst中skip=7 -> 約4Hz @ 30fps
RT_POSE_FIXED_HZ_ON = os.getenv('RT_POSE_FIXED_HZ_ON', '1') in ('1', 'true', 'True')
RT_POSE_FIXED_HZ = float(os.getenv('RT_POSE_FIXED_HZ', '4.0'))
RT_DYN_ON_RISE_ONLY = os.getenv('RT_DYN_ON_RISE_ONLY', '1') in ('1', 'true', 'True')
RT_DYN_PREV_FRAMES = int(os.getenv('RT_DYN_PREV_FRAMES', '1'))
RT_SIT_ZDIST_MARGIN = float(os.getenv('RT_SIT_ZDIST_MARGIN', '0.0060'))
RT_SIT_CONSEC_FRAMES = int(os.getenv('RT_SIT_CONSEC_FRAMES', '2'))
RT_CYCLE_AXIS = os.getenv('RT_CYCLE_AXIS', 'y').strip().lower()  # x/y/z
RT_CYCLE_NEGATIVE_DOWN = os.getenv('RT_CYCLE_NEGATIVE_DOWN', '1') in ('1', 'true', 'True')
RT_DROP_VEL_THR = float(os.getenv('RT_DROP_VEL_THR', '0.0008'))
RT_DROP_DIST_MARGIN = float(os.getenv('RT_DROP_DIST_MARGIN', '0.0010'))

# デモ向け: トルク解析を使わず、単眼(cam0)の肩上昇/肘角変化でゲージを制御
DEMO_MONO_GAUGE_ON = os.getenv('DEMO_MONO_GAUGE_ON', '1') in ('1', 'true', 'True')
DEMO_SHOULDER_RISE_FULL_M = float(os.getenv('DEMO_SHOULDER_RISE_FULL_M', '0.10'))
DEMO_ELBOW_DELTA_FULL_DEG = float(os.getenv('DEMO_ELBOW_DELTA_FULL_DEG', '45.0'))
DEMO_SHOULDER_RISE_PARTIAL_M = float(os.getenv('DEMO_SHOULDER_RISE_PARTIAL_M', '0.02'))
DEMO_ELBOW_DELTA_PARTIAL_DEG = float(os.getenv('DEMO_ELBOW_DELTA_PARTIAL_DEG', '8.0'))
DEMO_RATIO_FULL = float(os.getenv('DEMO_RATIO_FULL', '0.80'))
DEMO_RATIO_PARTIAL = float(os.getenv('DEMO_RATIO_PARTIAL', '0.30'))
DEMO_RATIO_UP_STEP = float(os.getenv('DEMO_RATIO_UP_STEP', '0.025'))
DEMO_RATIO_DOWN_STEP = float(os.getenv('DEMO_RATIO_DOWN_STEP', '0.035'))
DEMO_BASELINE_EMA = float(os.getenv('DEMO_BASELINE_EMA', '0.01'))
DEMO_MONO_CAM0_ONLY = os.getenv('DEMO_MONO_CAM0_ONLY', '1') in ('1', 'true', 'True')

class _LoopPerf:
    def __init__(self, enabled: bool, interval: int):
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.acc: dict[str, float] = {}
        self.frame: dict[str, float] = {}
        self.n = 0
        # 詳細トレース設定（環境変数で制御）
        self.trace = bool(int(os.getenv('PERF_TRACE', '0')))
        self.trace_every = int(os.getenv('PERF_TRACE_EVERY', '5')) if os.getenv('PERF_TRACE_EVERY') else 5
        self.topk = int(os.getenv('PERF_TOPK', '7')) if os.getenv('PERF_TOPK') else 7

    def begin_loop(self):
        # フレーム内訳を初期化
        self.frame.clear()

    def add(self, key: str, dt: float):
        # 区間集計（有効時のみ）
        if self.enabled:
            self.acc[key] = self.acc.get(key, 0.0) + dt
        # フレーム内訳（常時蓄積: トレース時のみ出力）
        self.frame[key] = self.frame.get(key, 0.0) + dt

    def next(self):
        if not self.enabled:
            return
        self.n += 1
        if (self.n % self.interval) == 0:
            total = sum(self.acc.values())
            print(f"[PERF] avg over {self.interval} loops (ms):")
            for k, v in sorted(self.acc.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {k:18s}: {v * 1000.0 / self.interval:7.2f}")
            if total > 0:
                print(f"  {'TOTAL':18s}: {total * 1000.0 / self.interval:7.2f}")
            self.acc.clear()

    def end_loop(self, loop_idx: int, frame_dt: float):
        if not self.trace:
            return
        if self.trace_every <= 0:
            return
        if (loop_idx % self.trace_every) != 0:
            return
        total = frame_dt if frame_dt and frame_dt > 0 else sum(self.frame.values())
        print(f"[PERF][frame #{loop_idx}] breakdown (Top{self.topk})")
        for k, v in list(sorted(self.frame.items(), key=lambda kv: kv[1], reverse=True))[: self.topk]:
            pct = (v / total * 100.0) if total and total > 0 else 0.0
            print(f"  {k:18s}: {v*1000.0:7.2f} ms  ({pct:5.1f}%)")

_perf = _LoopPerf(PERF_LOG, PERF_INT)
LOOP_TRACE = os.getenv('LOOP_TRACE', '1') not in ('0','false','False')
# 診断用: 書き込み/表示の無効化トグル
DISABLE_WRITE = os.getenv('DISABLE_WRITE', '0') in ('1','true','True')
DISABLE_IMSHOW = os.getenv('DISABLE_IMSHOW', '0') in ('1','true','True')
USE_NATIVE_DRAW = os.getenv('USE_NATIVE_DRAW', '1') in ('1','true','True')
try:
    VIDEO_TRACE_EVERY = int(os.getenv('VIDEO_TRACE_EVERY', '1'))
except Exception:
    VIDEO_TRACE_EVERY = 1
if LOOP_TRACE:
    print(f"[CFG] LOOP_TRACE=1 VIDEO_TRACE_EVERY={VIDEO_TRACE_EVERY} DISABLE_WRITE={int(DISABLE_WRITE)} DISABLE_IMSHOW={int(DISABLE_IMSHOW)} PERF_TRACE={int(_perf.trace)} PERF_TRACE_EVERY={_perf.trace_every} PERF_TOPK={_perf.topk} STOP_AFTER='{STOP_AFTER or '-'}'")
    print(f"[CFG] USE_NATIVE_DRAW={int(USE_NATIVE_DRAW)} NATIVE_DLL={'ok' if (_native_overlay is not None and getattr(_native_overlay, '_dll', None)) else 'none'}")

# 追加: カメラ診断（実カメラで遅くなる要因の切り分け用）
CAMERA_DIAG = os.getenv('CAMERA_DIAG', '1') in ('1','true','True')
try:
    CAMERA_TRACE_EVERY = max(1, int(os.getenv('CAMERA_TRACE_EVERY', '30')))
except Exception:
    CAMERA_TRACE_EVERY = 30

# ライブカメラ時は既定でリアルタイムimshowを止める（環境変数で明示指定があればそれを優先）
if ('DISABLE_IMSHOW' not in os.environ) and (not file_mode):
    DISABLE_IMSHOW = True
    if LOOP_TRACE:
        print("[CFG] live camera detected -> default DISABLE_IMSHOW=1 (set env DISABLE_IMSHOW=0 to override)")

def _fourcc_to_str(v: float | int | None) -> str:
    try:
        iv = int(v if v is not None else 0)
        return ''.join([chr((iv >> (8*i)) & 0xFF) for i in range(4)])
    except Exception:
        return '----'

def _print_cap_props(name: str, cap: 'cv.VideoCapture') -> None:
    if not CAMERA_DIAG:
        return
    try:
        be = cap.getBackendName() if hasattr(cap, 'getBackendName') else 'unknown'
    except Exception:
        be = 'unknown'
    try:
        w0 = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)); h0 = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps0 = cap.get(cv.CAP_PROP_FPS)
        fourcc = _fourcc_to_str(cap.get(cv.CAP_PROP_FOURCC))
        buff = cap.get(cv.CAP_PROP_BUFFERSIZE) if hasattr(cv, 'CAP_PROP_BUFFERSIZE') else -1
        conv_rgb = cap.get(cv.CAP_PROP_CONVERT_RGB) if hasattr(cv, 'CAP_PROP_CONVERT_RGB') else -1
        # 追加: 自動系の状態も出力（取得できない環境もある）
        try:
            ae = cap.get(cv.CAP_PROP_AUTO_EXPOSURE)
        except Exception:
            ae = None
        try:
            awb = cap.get(cv.CAP_PROP_AUTO_WB) if hasattr(cv, 'CAP_PROP_AUTO_WB') else None
        except Exception:
            awb = None
        try:
            af = cap.get(cv.CAP_PROP_AUTOFOCUS) if hasattr(cv, 'CAP_PROP_AUTOFOCUS') else None
        except Exception:
            af = None
        fmt = (
            f"[CAMDIAG] {name}: backend={be} size={w0}x{h0} fps={fps0:.3f} fourcc={fourcc} "
            f"buffersize={buff} convertRGB={conv_rgb} AE={ae} AWB={awb} AF={af}"
        )
        print(fmt)
    except Exception as _e:
        print(f"[CAMDIAG] {name}: prop read failed: {_e}")

def _set_prop(cap: 'cv.VideoCapture', prop: int, value: float) -> bool:
    try:
        ok = cap.set(prop, value)
        # 反映確認（一部backendでは取得できない）
        _ = cap.get(prop)
        return bool(ok)
    except Exception:
        return False

def _apply_camera_controls(name: str, cap: 'cv.VideoCapture') -> None:
    """実カメラ時のカメラ設定を固定。オート系OFF＋任意の固定値を環境変数から適用。

    優先度: CAM{idx}_* > CAM_* > 既定
    - idx は name が 'cam0'/'cam1' の末尾数字を使用
    - 例: CAM0_EXPOSURE, CAM_EXPOSURE, CAM0_AUTOFOCUS=0, CAM_AUTO_WB=0
    """
    if file_mode:
        return
    idx = None
    try:
        if name.lower().startswith('cam'):
            idx = int(''.join(ch for ch in name if ch.isdigit()))
    except Exception:
        idx = None

    def _env(k: str, default: str | None = None) -> str | None:
        if idx is not None and (f"CAM{idx}_{k}" in os.environ):
            return os.getenv(f"CAM{idx}_{k}")
        return os.getenv(f"CAM_{k}", default)

    # 1) 自動系OFF（既定でOFFを試みる）
    try:
        # Auto Exposure（backend差異に配慮して複数パターンを試す）
        ae_env = _env('AUTO_EXPOSURE', 'off')
        if ae_env and ae_env.lower() in ('0','off','false'):
            for v in (0.0, 0.0, 0.25, 0.75):  # MSMF/DSHOW の差へ便宜上複数トライ
                if _set_prop(cap, cv.CAP_PROP_AUTO_EXPOSURE, v):
                    break
        elif ae_env and ae_env.lower() in ('1','on','true'):
            _set_prop(cap, cv.CAP_PROP_AUTO_EXPOSURE, 1.0)
    except Exception:
        pass
    try:
        awb_env = _env('AUTO_WB', 'off')
        if hasattr(cv, 'CAP_PROP_AUTO_WB'):
            if awb_env and awb_env.lower() in ('0','off','false'):
                _set_prop(cap, cv.CAP_PROP_AUTO_WB, 0.0)
            elif awb_env and awb_env.lower() in ('1','on','true'):
                _set_prop(cap, cv.CAP_PROP_AUTO_WB, 1.0)
    except Exception:
        pass
    try:
        af_env = _env('AUTOFOCUS', 'off')
        if hasattr(cv, 'CAP_PROP_AUTOFOCUS'):
            if af_env and af_env.lower() in ('0','off','false'):
                _set_prop(cap, cv.CAP_PROP_AUTOFOCUS, 0.0)
            elif af_env and af_env.lower() in ('1','on','true'):
                _set_prop(cap, cv.CAP_PROP_AUTOFOCUS, 1.0)
    except Exception:
        pass

    # 2) 固定値の適用（指定がある場合）
    def _env_float(k: str) -> float | None:
        v = _env(k)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    # 解像度・FPS・FOURCC（設定 → 実値の確認の順で行う）
    w_set = _env_float('WIDTH'); h_set = _env_float('HEIGHT'); fps_set = _env_float('FPS')
    if w_set:
        _set_prop(cap, cv.CAP_PROP_FRAME_WIDTH, w_set)
    if h_set:
        _set_prop(cap, cv.CAP_PROP_FRAME_HEIGHT, h_set)
    if fps_set:
        _set_prop(cap, cv.CAP_PROP_FPS, fps_set)
    fourcc_env = _env('FOURCC')
    if fourcc_env and len(fourcc_env) >= 4:
        try:
            cc = cv.VideoWriter_fourcc(*fourcc_env[:4])
            _set_prop(cap, cv.CAP_PROP_FOURCC, float(cc))
        except Exception:
            pass

    exp_set = _env_float('EXPOSURE')
    if exp_set is not None:
        _set_prop(cap, cv.CAP_PROP_EXPOSURE, exp_set)
    gain_set = _env_float('GAIN')
    if gain_set is not None:
        _set_prop(cap, cv.CAP_PROP_GAIN, gain_set)
    wb_set = _env_float('WB_TEMPERATURE')
    if wb_set is not None and hasattr(cv, 'CAP_PROP_WB_TEMPERATURE'):
        _set_prop(cap, cv.CAP_PROP_WB_TEMPERATURE, wb_set)
    focus_set = _env_float('FOCUS')
    if focus_set is not None and hasattr(cv, 'CAP_PROP_FOCUS'):
        _set_prop(cap, cv.CAP_PROP_FOCUS, focus_set)

# ライブカメラの設定固定（AE/AWB/AFオフ＋任意設定）→ 反映後のプロパティ出力
_apply_camera_controls('cam0', cap0)
if cap1 is not None:
    _apply_camera_controls('cam1', cap1)
# オープン済みキャプチャのプロパティ出力（最終状態）
_print_cap_props('cam0', cap0)
if cap1 is not None:
    _print_cap_props('cam1', cap1)

# 入力ソースfps（遅延反映スキップで使用）
try:
    _src_fps0 = float(cap0.get(cv.CAP_PROP_FPS))
except Exception:
    _src_fps0 = 0.0
try:
    _src_fps1 = float(cap1.get(cv.CAP_PROP_FPS))
except Exception:
    _src_fps1 = _src_fps0
_src_fps_candidates = [v for v in (_src_fps0, _src_fps1, float(fps)) if v and np.isfinite(v) and v > 1e-6]
_src_fps = float(_src_fps_candidates[0]) if _src_fps_candidates else 30.0
_rt_fixed_skip = int(max(0, round(_src_fps / max(RT_POSE_FIXED_HZ, 1e-6)) - 1))
_lpf_fps_ema = float(np.clip(_src_fps, E_FPS_MIN, E_FPS_MAX))
_e_dt_sec_current = 1.0 / max(_lpf_fps_ema, 1e-6)
if E_DEBUG:
    print(f"[RTSKIP] src_fps={_src_fps:.3f} (cam0={_src_fps0:.3f}, cam1={_src_fps1:.3f})")
    if RT_POSE_FIXED_HZ_ON:
        print(f"[RTFIX] target_hz={RT_POSE_FIXED_HZ:.3f}, fixed_skip={_rt_fixed_skip}")

# カメラごとの時刻・間隔・直近の取得時間（効果測定用）
_cam0_last_ts = None
_cam1_last_ts = None
_cam0_last_dt = None
_cam1_last_dt = None
_cam0_last_shape = None
_cam1_last_shape = None
_rt_skip_remaining = 0
_rt_burst_remaining = 0
_rt_burst_cooldown = 0
_rt_prev_z_cycle = None
_dyn_active = (not RT_DYN_ON_RISE_ONLY)
_dyn_sit_consec = 0
_dyn_start_frame = None

while True:
    start_time = time.perf_counter()
    _perf.begin_loop()
    # 総ループ数（grabベースで進める）
    skip_counter += 1
    if DEBUG_LOGS:
        print(f"[DBG] while-loop count={skip_counter}")
    # まずは grab でフレームを進める（軽量・スキップ時にデコードしない）
    t_seg = time.perf_counter()
    okg0 = cap0.grab()
    okg1 = (cap1.grab() if cap1 is not None else okg0)
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        #print(f"[TRACE]] grab ok0={okg0} ok1={okg1}")
        pass
    _perf.add('grab', time.perf_counter() - t_seg)
    if IO_DEBUG:
        try:
            pos0 = cap0.get(cv.CAP_PROP_POS_FRAMES)
            pos1 = cap1.get(cv.CAP_PROP_POS_FRAMES)
            _dbg("grab ok:", okg0, okg1, "pos:", pos0, pos1)
        except Exception:
            pass
    if not okg0 or not okg1:
        # いずれかが終端
        try:
            p0 = cap0.get(cv.CAP_PROP_POS_FRAMES); t0 = cap0.get(cv.CAP_PROP_FRAME_COUNT)
            p1 = (cap1.get(cv.CAP_PROP_POS_FRAMES) if cap1 is not None else p0)
            t1 = (cap1.get(cv.CAP_PROP_FRAME_COUNT) if cap1 is not None else t0)
            print(f"Video ended (grab fail) pos0/total0={p0}/{t0}, pos1/total1={p1}/{t1}")
        except Exception:
            print("Video ended")
        # ファイル再生時にループさせるオプション
        if LOOP_FILE_PLAYBACK and file_mode:
            try:
                cap0.set(cv.CAP_PROP_POS_FRAMES, 0)
                if cap1 is not None:
                    cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
                print("[Playback] rewind to frame 0 and continue")
                _perf.next()
                continue
            except Exception as e:
                print(f"[Playback] rewind failed: {e}")
        break
    # 固定Hz/遅延反映のスキップ
    if _rt_skip_remaining > 0:
        _rt_skip_remaining -= 1
        if E_DEBUG and (WHILE_COUNT % 30 == 0):
            print(f"[RTSKIP] skipping frame, remaining={_rt_skip_remaining}")
        _perf.next()
        continue
    # スキップ判定（デコード不要なフレームはここで続行）
    if (skip_counter % skip_mod != 0) and (_rt_burst_remaining <= 0):
        _perf.next()
        continue
    # このタイミングのフレームのみ retrieve してデコード
    # 各カメラのretrieve時間を個別に計測（ブロッキング源の特定）
    t_seg0 = time.perf_counter()
    ret0, frame0 = cap0.retrieve()
    dt_ret0 = time.perf_counter() - t_seg0
    if cap1 is not None:
        t_seg1 = time.perf_counter()
        ret1, frame1 = cap1.retrieve()
        dt_ret1 = time.perf_counter() - t_seg1
    else:
        ret1 = ret0
        frame1 = (None if frame0 is None else frame0.copy())
        dt_ret1 = 0.0
    if DEBUG_LOGS:
        print("frame0", type(frame0), "frame1", type(frame1))
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        s0 = None if frame0 is None else frame0.shape
        s1 = None if frame1 is None else frame1.shape
        #print(f"[TRACE] retrieve ret0={ret0} ret1={ret1} shapes={s0},{s1}")
    # 元々のまとめてのretrieve時間の代わりに、合計で近似反映
    _perf.add('retrieve', dt_ret0 + dt_ret1)

    # カメラ診断ログ（一定間隔）
    if CAMERA_DIAG and (WHILE_COUNT % CAMERA_TRACE_EVERY == 0):
        now = time.perf_counter()
        # cam0: フレーム間隔・ブロック時間・サイズ
        try:
            s0 = None if frame0 is None else tuple(int(x) for x in frame0.shape)
            dt0 = (now - _cam0_last_ts) if (_cam0_last_ts is not None) else None
            _cam0_last_ts = now
            _cam0_last_dt = dt0
            _cam0_last_shape = s0
            eff_fps0 = (1.0 / dt0) if (dt0 and dt0 > 1e-6) else 0.0
            print(f"[CAMDIAG] cam0: retrieve_ms={dt_ret0*1000:.2f} gap_ms={(dt0*1000 if dt0 else -1):.2f} eff_fps={eff_fps0:.1f} shape={s0}")
        except Exception:
            pass
        # cam1: フレーム間隔・ブロック時間・サイズ
        try:
            s1 = None if frame1 is None else tuple(int(x) for x in frame1.shape)
            dt1 = (now - _cam1_last_ts) if (_cam1_last_ts is not None) else None
            _cam1_last_ts = now
            _cam1_last_dt = dt1
            _cam1_last_shape = s1
            eff_fps1 = (1.0 / dt1) if (dt1 and dt1 > 1e-6) else 0.0
            print(f"[CAMDIAG] cam1: retrieve_ms={dt_ret1*1000:.2f} gap_ms={(dt1*1000 if dt1 else -1):.2f} eff_fps={eff_fps1:.1f} shape={s1}")
        except Exception:
            pass
    # 早期プレビュー（デバッグ用）：環境変数 EARLY_PREVIEW=1 で有効
    if os.getenv('EARLY_PREVIEW', '0') in ('1','true','True') and (not HEADLESS) and (not DISABLE_IMSHOW):
        try:
            cv.imshow('EarlyPreview0', frame0)
            cv.imshow('EarlyPreview1', frame1)
            if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
                print("[TRACE] early preview imshow done")
        except Exception as _ep_e:
            if DEBUG_LOGS:
                print(f"[WARN] early preview failed: {_ep_e}")
    if not ret0 or not ret1 or frame0 is None or frame1 is None:
        try:
            p0 = cap0.get(cv.CAP_PROP_POS_FRAMES); t0 = cap0.get(cv.CAP_PROP_FRAME_COUNT)
            p1 = cap1.get(cv.CAP_PROP_POS_FRAMES); t1 = cap1.get(cv.CAP_PROP_FRAME_COUNT)
            print(f"Video ended (retrieve fail) pos0/total0={p0}/{t0}, pos1/total1={p1}/{t1}")
        except Exception:
            print("Video ended")
        # ファイル再生時にループさせるオプション
        if LOOP_FILE_PLAYBACK and file_mode:
            try:
                cap0.set(cv.CAP_PROP_POS_FRAMES, 0)
                cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
                print("[Playback] rewind to frame 0 and continue")
                _perf.next()
                continue
            except Exception as e:
                print(f"[Playback] rewind failed: {e}")
        break
    # retrieve 直後の切り分け
    if STOP_AFTER.lower() in ("retrieve",):
        _perf.next()
        break
    # 保存処理（BGRのままでOK）
    t_seg = time.perf_counter()
    if not DISABLE_WRITE:
        writer0.write(frame0)
        writer1.write(frame1)
        if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
            print("[TRACE] write done")
    else:
        if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
            print("[TRACE] write skipped (DISABLE_WRITE=1)")
    _perf.add('write', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("write",):
        # 以降の処理を切り分けるために早期終了
        _perf.next()
        break

    # トリミング処理の置き換え
    t_seg = time.perf_counter()
    frame0 = frame0[:, x_start:x_end]
    frame1 = frame1[:, x_start:x_end]
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        print(f"[TRACE] crop [{x_start}:{x_end}] -> shapes={frame0.shape},{frame1.shape}")
    _perf.add('crop', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("crop",):
        _perf.next()
        break

    t_seg = time.perf_counter()
    frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
    frame0.flags.writeable = False
    frame1.flags.writeable = False
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        print("[TRACE] preproc RGB -> writeable False")
    _perf.add('preproc', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("preproc",):
        _perf.next()
        break
    t_seg = time.perf_counter()
    roi0 = _pose_roi0 if (POSE_ROI_ON and _pose_roi0_miss <= POSE_ROI_MAX_MISS) else None
    if roi0 is not None and _pose_roi0_miss > 0:
        for _ in range(_pose_roi0_miss):
            roi0 = _expand_roi(roi0, frame0.shape, POSE_ROI_MISS_GROW_RATIO)
            if roi0 is None:
                break
    results0, _roi0_used = _pose_process_with_roi(pose0, frame0, roi0)
    if os.getenv('POSE_DEBUG', '0') in ('1','true','True') and (WHILE_COUNT % max(1, int(os.getenv('POSE_TRACE_EVERY','30'))) == 0):
        try:
            if _roi0_used and roi0 is not None:
                _rw0 = roi0[2] - roi0[0]
                _rh0 = roi0[3] - roi0[1]
                print(f"[Pose] cam0 ROI used scale={MP_INPUT_SCALE} roi={roi0} roi_shape=({_rh0},{_rw0},3)")
            else:
                print(f"[Pose] cam0 fullframe scale={MP_INPUT_SCALE} shape={tuple(frame0.shape)}")
        except Exception:
            pass
    _perf.add('mediapipe0', time.perf_counter() - t_seg)
    t_seg = time.perf_counter()
    roi1 = _pose_roi1 if (POSE_ROI_ON and _pose_roi1_miss <= POSE_ROI_MAX_MISS) else None
    if roi1 is not None and _pose_roi1_miss > 0:
        for _ in range(_pose_roi1_miss):
            roi1 = _expand_roi(roi1, frame1.shape, POSE_ROI_MISS_GROW_RATIO)
            if roi1 is None:
                break
    results1, _roi1_used = _pose_process_with_roi(pose1, frame1, roi1)
    if os.getenv('POSE_DEBUG', '0') in ('1','true','True') and (WHILE_COUNT % max(1, int(os.getenv('POSE_TRACE_EVERY','30'))) == 0):
        try:
            if _roi1_used and roi1 is not None:
                _rw1 = roi1[2] - roi1[0]
                _rh1 = roi1[3] - roi1[1]
                print(f"[Pose] cam1 ROI used scale={MP_INPUT_SCALE} roi={roi1} roi_shape=({_rh1},{_rw1},3)")
            else:
                print(f"[Pose] cam1 fullframe scale={MP_INPUT_SCALE} shape={tuple(frame1.shape)}")
        except Exception:
            pass
    _perf.add('mediapipe1', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("mediapipe", "mediapipe1",):
        _perf.next()
        break
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        try:
            lm0 = 'y' if (getattr(results0, 'pose_landmarks', None) is not None) else 'n'
            lm1 = 'y' if (getattr(results1, 'pose_landmarks', None) is not None) else 'n'
        except Exception:
            lm0 = lm1 = '?'
        #print(f"[TRACE] mediapipe processed lm0={lm0} lm1={lm1}")
    t_seg = time.perf_counter()
    frame0.flags.writeable = True
    frame1.flags.writeable = True
    frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
    frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        #print(f"[TRACE] postproc back to BGR")
        pass
    _perf.add('postproc', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("postproc",):
        _perf.next()
        break
    t_seg = time.perf_counter()
    frame0_keypoints, frame1_keypoints = _extract_keypoints_pair(
        results0, results1, pose_keypoints, frame0, frame1
    )
    if POSE_ROI_ON:
        _next_roi0 = _roi_from_keypoints(frame0_keypoints, frame0.shape)
        if _next_roi0 is None:
            _pose_roi0_miss += 1
            if _pose_roi0_miss > POSE_ROI_MAX_MISS:
                _pose_roi0 = None
        else:
            _pose_roi0 = _next_roi0
            _pose_roi0_miss = 0

        _next_roi1 = _roi_from_keypoints(frame1_keypoints, frame1.shape)
        if _next_roi1 is None:
            _pose_roi1_miss += 1
            if _pose_roi1_miss > POSE_ROI_MAX_MISS:
                _pose_roi1 = None
        else:
            _pose_roi1 = _next_roi1
            _pose_roi1_miss = 0
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        v0 = sum(1 for x, y in frame0_keypoints if x >= 0 and y >= 0)
        v1 = sum(1 for x, y in frame1_keypoints if x >= 0 and y >= 0)
        #print(f"[TRACE] kps2d valid counts: cam0={v0} cam1={v1}")
    _perf.add('kps2d', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("kps2d", "keypoints",):
        _perf.next()
        break
    # --- Debug: 2Dキーポイントの有効数をチェック ---
    if DEBUG_LOGS and WHILE_COUNT % 30 == 0:
        valid0 = sum(1 for x, y in frame0_keypoints if x >= 0 and y >= 0)
        valid1 = sum(1 for x, y in frame1_keypoints if x >= 0 and y >= 0)
        print(f"[DBG] frame {WHILE_COUNT}: valid2D(cam0,cam1)=({valid0},{valid1})")

    # ---- 単眼3D(world) ランドマーク取得は使用しない（無効化） ----
    # if results0.pose_world_landmarks:
    #     world_pts = []
    #     lm_list = results0.pose_world_landmarks.landmark
    #     for idx in pose_keypoints:
    #         if idx < len(lm_list):
    #             lm = lm_list[idx]
    #             world_pts.append([lm.x, lm.y, lm.z])  # そのまま保持 (右手系/単位m)
    #         else:
    #             world_pts.append([float('nan'), float('nan'), float('nan')])
    #     mono3d_records.append(world_pts)
    # else:
    #     # フレーム欠損でも行数合わせのため NaN 行を積む
    #     mono3d_records.append([[float('nan')]*3 for _ in pose_keypoints])

    t_seg = time.perf_counter()
    transformed_p3ds = _triangulate_transform_batch(P0_F64, P1_F64, frame0_keypoints, frame1_keypoints)
    nan_3d = int(np.sum(~np.all(np.isfinite(transformed_p3ds), axis=1)))
    if DEBUG_LOGS and WHILE_COUNT % 30 == 0 and nan_3d:
        print(f"[DBG] frame {WHILE_COUNT}: non-finite 3D points={nan_3d}")
    if landmark_ekf is not None:
        try:
            transformed_p3ds, _vel_filt, _acc_filt = landmark_ekf.step(transformed_p3ds, dt)
        except Exception as _ekf_step_e:  # noqa: BLE001
            if WHILE_COUNT % 120 == 0:
                print(f"[EKF] step failed (frame={WHILE_COUNT}): {_ekf_step_e}")
    kpts_3d.append(transformed_p3ds)
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        #print(f"[TRACE] triang+transform done")
        pass
    _perf.add('triang+transform', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("triang", "triangulate", "triang+transform", "transform"):
        _perf.next()
        break
    _cycle_axis_idx = 1 if RT_CYCLE_AXIS == 'y' else (2 if RT_CYCLE_AXIS == 'z' else 0)
    _cycle_value_now = transformed_p3ds[0][_cycle_axis_idx]
    if 4 < WHILE_COUNT < 15:
        z_value += _cycle_value_now / 10
    elif WHILE_COUNT == 15:
        detector = PushCycleDetector(
            z_value,
            threshold=0.015,
            velocity_epsilon=0.01,
            min_interval=10,
            mode='rise_to_rise',
            negative_down=RT_CYCLE_NEGATIVE_DOWN,
        )

    # 立ち上がり/着座イベントに応じた逆動力学ゲート制御
    _z_cycle = _cycle_value_now
    if WHILE_COUNT > 15 and np.isfinite(_z_cycle):
        if RT_RISE_BURST_ON:
            _burst_reason = None
            if _rt_prev_z_cycle is not None and np.isfinite(_rt_prev_z_cycle):
                _dz_cycle = float(_z_cycle - _rt_prev_z_cycle)
                if RT_CYCLE_NEGATIVE_DOWN:
                    _drop_vel_hit = (_dz_cycle <= -abs(RT_DROP_VEL_THR))
                else:
                    _drop_vel_hit = (_dz_cycle >= abs(RT_DROP_VEL_THR))
                if _drop_vel_hit and _rt_burst_cooldown <= 0:
                    _burst_reason = f"drop_d={_dz_cycle:.6f} thr={RT_DROP_VEL_THR:.6f}"
            try:
                if RT_CYCLE_NEGATIVE_DOWN:
                    _z_th = float(detector.initial_z - RT_DROP_DIST_MARGIN)
                    _drop_dist_hit = (_z_cycle < _z_th)
                else:
                    _z_th = float(detector.initial_z + RT_DROP_DIST_MARGIN)
                    _drop_dist_hit = (_z_cycle > _z_th)
                if (_burst_reason is None) and (_rt_burst_cooldown <= 0) and _drop_dist_hit:
                    _burst_reason = f"drop_v={float(_z_cycle):.6f} th={_z_th:.6f}"
            except Exception:
                _z_th = None
            if _burst_reason is not None:
                _rt_burst_remaining = max(_rt_burst_remaining, RT_RISE_BURST_FRAMES)
                _rt_burst_cooldown = max(_rt_burst_cooldown, RT_RISE_BURST_COOLDOWN)
                _rt_skip_remaining = 0
                if RT_DYN_ON_RISE_ONLY and (not _dyn_active):
                    _dyn_active = True
                    _dyn_sit_consec = 0
                    _dyn_start_frame = max(0, WHILE_COUNT - max(0, RT_DYN_PREV_FRAMES))
                    print(f"[DYNGATE] start frame={WHILE_COUNT} (analyze_from={_dyn_start_frame}) reason={_burst_reason} axis={RT_CYCLE_AXIS}")
                if E_DEBUG:
                    print(
                        f"[RTBURST] frame={WHILE_COUNT} trigger {_burst_reason}, "
                        f"burst={_rt_burst_remaining}, cooldown={_rt_burst_cooldown}"
                    )
        if RT_DYN_ON_RISE_ONLY and _dyn_active:
            try:
                if RT_CYCLE_NEGATIVE_DOWN:
                    _z_sit_th = float(detector.initial_z + RT_SIT_ZDIST_MARGIN)
                    _sit_hit = (_z_cycle > _z_sit_th)
                else:
                    _z_sit_th = float(detector.initial_z - RT_SIT_ZDIST_MARGIN)
                    _sit_hit = (_z_cycle < _z_sit_th)
                if _sit_hit:
                    _dyn_sit_consec += 1
                else:
                    _dyn_sit_consec = 0
                if _dyn_sit_consec >= max(1, RT_SIT_CONSEC_FRAMES):
                    _dyn_active = False
                    _dyn_sit_consec = 0
                    print(f"[DYNGATE] stop frame={WHILE_COUNT} (sit axis={RT_CYCLE_AXIS} v={float(_z_cycle):.6f} th={_z_sit_th:.6f})")
            except Exception:
                pass
        _rt_prev_z_cycle = float(_z_cycle)

    t_seg = time.perf_counter()
    for part_name, calculator in calculators.items():
        # データ計算
        # print(part_name)
        if file_mode:
            i = len(kpts_3d) - 1
        else:
            i = 0
        result = calculator.calculate_link_vectors(kpts_3d, file_mode, i, dt)

        # 結果が不完全ならスキップ
        if result[0] is None:
            continue

        # 結果を7つに展開
        r_vec, vel_vec, omega, cent, p1, acc, ang_acc = result

        # print("calculated")
        # データ格納
        storage.add_data(part_name, r_vec, vel_vec, omega, cent, p1, ang_acc, acc)
    _perf.add('calc+store', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("calc", "calc+store", "store"):
        _perf.next()
        break

    part_names_internal = [
        "upper_arm_R",
        "forearm_R",
        "both_shoulder",
        "both_hip",
        "up_arm_l",
        "forearm_L",
        "upper_Leg_R",
        "upper_Leg_L",
    ]
    part_data = {name: storage.get_data(name) for name in part_names_internal}
    # --- Debug: 各部位データの蓄積長を定期表示 ---
    if DEBUG_LOGS and WHILE_COUNT % 30 == 0:
        lens = {k: (len(v) if isinstance(v, list) else (0 if v is None else 'n/a')) for k, v in part_data.items()}
        print(f"[DBG] frame {WHILE_COUNT}: part_data lengths {lens}")

    if len(kpts_3d) < 4:
        if len(kpts_3d) == 3:
            I1 = calculate_inertia_tensor(
                3, w, np.linalg.norm(transformed_p3ds[0] - transformed_p3ds[2])
            )  # 上腕
            I2 = calculate_inertia_tensor(
                4, w, np.linalg.norm(transformed_p3ds[2] - transformed_p3ds[4])
            )  # 前腕
            len_half_body = 0.25 * np.linalg.norm(
                transformed_p3ds[0]
                + transformed_p3ds[1]
                - transformed_p3ds[7]
                - transformed_p3ds[6]
            )
            I3 = calculate_inertia_tensor(1, w, len_half_body)  # 上胴体
            I4 = calculate_inertia_tensor(0, w, len_half_body)  # 下胴体
            I5 = calculate_inertia_tensor(
                6, w, np.linalg.norm(transformed_p3ds[9] - transformed_p3ds[7])
            )  # 太もも
            I6 = calculate_inertia_tensor(
                7, w, np.linalg.norm(transformed_p3ds[11] - transformed_p3ds[9])
            )  # 前足
            I7 = calculate_inertia_tensor(2, w, 0.25)  # 頭
        continue

    # 計算とデータの格納をループで行う
    if len(kpts_3d) < 7:
        continue
    # 依存データが未蓄積の部位があれば次フレームへ
    required_lists = [
        part_data["upper_arm_R"],
        part_data["forearm_R"],
        part_data["both_shoulder"],
        part_data["both_hip"],
        part_data["up_arm_l"],
        part_data["forearm_L"],
        part_data["upper_Leg_R"],
        part_data["upper_Leg_L"],
    ]

    # ======== 重力向きの自動検出（初期フレーム） ========
    if GRAVITY_AUTO_DETECT and not _gravity_set:
        try:
            if part_data["both_shoulder"] and part_data["both_hip"]:
                c_sh = np.array(part_data["both_shoulder"][-1]["centroid"], dtype=float)
                c_hp = np.array(part_data["both_hip"][-1]["centroid"], dtype=float)
                v_up = c_sh - c_hp  # 上方向推定
                if np.all(np.isfinite(v_up)):
                    _grav_up_samples.append(v_up)
            if len(_grav_up_samples) >= GRAVITY_DETECT_FRAMES:
                v_med = np.median(np.stack(_grav_up_samples, axis=0), axis=0)
                up_label, axis_unit, cosabs = _pick_axis_from_vector(v_med)
                g_label = _opposite_axis_label(up_label)
                # gの大きさは既存gのノルムを保持
                g_mag = float(np.linalg.norm(g)) if np.all(np.isfinite(g)) else 9.80665
                # 上方向axis_unitに対して、重力は下向き
                new_g = -axis_unit * g_mag
                # ランタイムの g を更新
                globals()['g'] = np.array(new_g, dtype=float)
                globals()['_gravity_label'] = g_label
                globals()['_gravity_set'] = True
                if E_DEBUG:
                    print(f"[GRAVITY] up={up_label} g={g_label} vec={new_g.tolist()} cos={cosabs:.3f} plane={GRAVITY_LEVEL_PLANE if GRAVITY_LEVEL_PLANE_ON else 'ANY'}")
        except Exception as _ge:
            if E_DEBUG:
                print(f"[GRAVITY] detect failed: {_ge}")
    if any((lst is None) or (len(lst) == 0) for lst in required_lists):
        continue
    # トルクと力の計算（右/左）をループで簡潔に構築
    right_specs = [
        (I1, m1, part_data["upper_arm_R"], {}),
        (I2, m2, part_data["forearm_R"], {}),
        (I3, w, part_data["both_shoulder"], {"add_part_data": part_data["both_hip"], "condition": 1, "Imode": 3, "Info_I3": transformed_p3ds}),
        (I4, w, part_data["both_hip"], {"add_part_data": part_data["both_shoulder"], "Imode": 4}),
        (I5, m4, part_data["upper_Leg_R"], {}),
    ]
    left_specs = [
        (I1, m1, part_data["up_arm_l"], {}),
        (I2, m2, part_data["forearm_L"], {}),
        (I3, w, part_data["both_shoulder"], {"add_part_data": part_data["both_hip"], "condition": 0, "Imode": 3, "Info_I3": transformed_p3ds}),
        (I4, w, part_data["both_hip"], {"add_part_data": part_data["both_shoulder"], "Imode": 4}),
        (I5, m4, part_data["upper_Leg_L"], {}),
    ]
    _dyn_should_run = ((not RT_DYN_ON_RISE_ONLY) or _dyn_active) and (not DEMO_MONO_GAUGE_ON)
    t_seg = time.perf_counter()
    if _dyn_should_run:
        MsR, FsR, partsR = run_specs(right_specs)
        MsL, FsL, partsL = run_specs(left_specs)
        if TRACE_DYN and (WHILE_COUNT % TRACE_EVERY == 0):
            def _safe_norm_list(lst):
                try:
                    return float(np.linalg.norm(np.array(lst, dtype=np.float64)))
                except Exception:
                    return float('nan')
            print(f"[DYN:run_specs] R | M_norm={_safe_norm_list(MsR):.6f} F_norm={_safe_norm_list(FsR):.6f} parts={len(partsR)}")
            print(f"[DYN:run_specs] L | M_norm={_safe_norm_list(MsL):.6f} F_norm={_safe_norm_list(FsL):.6f} parts={len(partsL)}")
    else:
        MsR, FsR, partsR = [], [], []
        MsL, FsL, partsL = [], [], []
        if E_DEBUG and (WHILE_COUNT % 30 == 0):
            print(f"[DYNGATE] paused frame={WHILE_COUNT} (waiting rise trigger)")
    _perf.add('run_specs', time.perf_counter() - t_seg)
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        #print(f"[TRACE] run_specs done R/L")
        pass
    if STOP_AFTER.lower() in ("run_specs",):
        _perf.next()
        break
    # --- Debug: 各トルク入力の有効性チェック ---
    if DEBUG_LOGS and WHILE_COUNT % 30 == 0:
        def _safe_len(x):
            try:
                return len(x)
            except Exception:
                return 'n/a'
        print(f"[DBG] frame {WHILE_COUNT}: MsR={_safe_len(MsR)} FsR={_safe_len(FsR)} MsL={_safe_len(MsL)} FsL={_safe_len(FsL)}")

    vector = transformed_p3ds[8] - transformed_p3ds[6]
    norm_vec = np.linalg.norm(vector)
    angle_degrees = 0.0
    if norm_vec > 1e-8:
        # Z成分とベクトル長からXY平面との角度を計算（安全にクリップ）
        cos_val = np.clip(vector[2] / norm_vec, -1.0, 1.0)
        angle_with_xy_plane = np.pi / 2 - np.arccos(cos_val)
        angle_degrees = float(np.degrees(angle_with_xy_plane))

    if DEBUG_LOGS:
        print("angle=", angle_degrees)
    # 地面反力の簡易モデル: ファイルモードでない、かつ角度が±20°以内のときに付与
    if (not file_mode) and (-20.0 <= angle_degrees <= 20.0):
        f_E = np.array([0, 0, w * 0.66 * np.linalg.norm(g) / 2])
    else:
        f_E = np.array([0, 0, 0])
    r_x = part_data["both_hip"][-1]["centroid"]
    # r_x = .5*(both_shoulder_data[-1]['p1']+both_hip_data[-1]['p1'])+.25*(both_shoulder_data[-1]['relative_position_vector']+both_hip_data[-1]['relative_position_vector'])  # この例での r_x

    tau_E = np.array([0, 0, 0])

    # r_g は各チェーンの部位順に合わせる（右/左で別個に構築）
    r_g_R = [
        part_data["upper_arm_R"][-1]["centroid"],
        part_data["forearm_R"][-1]["centroid"],
        (part_data["both_shoulder"][-1]["centroid"] * 3 + part_data["both_hip"][-1]["centroid"]) / 4,
        (part_data["both_shoulder"][-1]["centroid"] + part_data["both_hip"][-1]["centroid"] * 3) / 4,
        part_data["upper_Leg_R"][-1]["centroid"],
    ]
    r_g_L = [
        part_data["up_arm_l"][-1]["centroid"],
        part_data["forearm_L"][-1]["centroid"],
        (part_data["both_shoulder"][-1]["centroid"] * 3 + part_data["both_hip"][-1]["centroid"]) / 4,
        (part_data["both_shoulder"][-1]["centroid"] + part_data["both_hip"][-1]["centroid"] * 3) / 4,
        part_data["upper_Leg_L"][-1]["centroid"],
    ]

    t_seg = time.perf_counter()
    if _dyn_should_run:
        USE_NATIVE_DYNAMICS = os.getenv('USE_NATIVE_DYNAMICS', '1') in ('1','true','True')
        if USE_NATIVE_DYNAMICS:
            try:
                # p1sは各部位の関節位置を並べたものが必要。storageから取得。
                def _collect_p1s(parts: list[str]) -> np.ndarray:
                    p1s = []
                    for part in parts:
                        data_list = storage.get_data(part)
                        if data_list:
                            p1s.append(data_list[-1]['p1'])
                        else:
                            p1s.append(np.zeros(3))
                    return np.array(p1s, dtype=np.float64)

                p1sR = _collect_p1s(partsR)
                p1sL = _collect_p1s(partsL)
                r_g_R_arr = np.array(r_g_R, dtype=np.float64)
                r_g_L_arr = np.array(r_g_L, dtype=np.float64)

                tauR = compute_tau_chain_native(np.array(MsR, dtype=np.float64), np.array(FsR, dtype=np.float64), r_g_R_arr, p1sR, np.array(tau_E, dtype=np.float64), np.array(f_E, dtype=np.float64), np.array(r_x, dtype=np.float64))
                tauL = compute_tau_chain_native(np.array(MsL, dtype=np.float64), np.array(FsL, dtype=np.float64), r_g_L_arr, p1sL, np.array(tau_E, dtype=np.float64), np.array(f_E, dtype=np.float64), np.array(r_x, dtype=np.float64))
                if TRACE_DYN and (WHILE_COUNT % TRACE_EVERY == 0):
                    # レバーアーム |r_g - p1| のノルム統計でゼロ・ミスアラインを検知
                    try:
                        rg_minus_p1_R = np.linalg.norm(r_g_R_arr - p1sR, axis=1)
                        rg_minus_p1_L = np.linalg.norm(r_g_L_arr - p1sL, axis=1)
                        print(f"[DYN:TAU] lever R | min={float(np.min(rg_minus_p1_R)):.4f} max={float(np.max(rg_minus_p1_R)):.4f} L | min={float(np.min(rg_minus_p1_L)):.4f} max={float(np.max(rg_minus_p1_L)):.4f}")
                    except Exception:
                        pass
                if TRACE_DYN and (WHILE_COUNT % TRACE_EVERY == 0):
                    print(f"[DYN:TAU] R | tau_norm={float(np.linalg.norm(tauR)):.6f}  L | tau_norm={float(np.linalg.norm(tauL)):.6f}")
                    # 詳細を見たい場合は下記を一時的に解除
                    # print('[DYN:TAU] tauR=', tauR)
                    # print('[DYN:TAU] tauL=', tauL)

                # 既存構造 (値, 部位名) の形に合わせる
                torquesR = [(tauR[i], partsR[i]) for i in range(len(partsR))]
                torquesL = [(tauL[i], partsL[i]) for i in range(len(partsL))]
            except (RuntimeError, ValueError, TypeError, AttributeError) as _nd_e:
                if os.getenv('POSE_DEBUG','0') in ('1','true','True'):
                    print(f"[Dyn] native tau fallback due to: {_nd_e}")
                # フォールバック
                torquesR = calculate_individual_torques(MsR, FsR, np.array(r_g_R), tau_E, f_E, r_x, partsR, storage)
                torquesL = calculate_individual_torques(MsL, FsL, np.array(r_g_L), tau_E, f_E, r_x, partsL, storage)
        else:
            torquesR = calculate_individual_torques(MsR, FsR, np.array(r_g_R), tau_E, f_E, r_x, partsR, storage)
            torquesL = calculate_individual_torques(MsL, FsL, np.array(r_g_L), tau_E, f_E, r_x, partsL, storage)
    else:
        _zero = np.zeros(3, dtype=np.float64)
        torquesR = [(_zero.copy(), "wrist_R"), (_zero.copy(), "elbow_R"), (_zero.copy(), "shoulder_R")]
        torquesL = [(_zero.copy(), "wrist_L"), (_zero.copy(), "elbow_L"), (_zero.copy(), "shoulder_L")]
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        print(f"[TRACE] torques computed R/L")
    _perf.add('torques', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("torques", "torque"):
        _perf.next()
        break

    # 各リンクベクトルとグローバルトルクを辞書に集約
    links = {
        "wrist_R": transformed_p3ds[4] - transformed_p3ds[2],
        "elbow_R": transformed_p3ds[2] - transformed_p3ds[0],
        "shoulder_R": -(transformed_p3ds[1] - transformed_p3ds[0]),
        "wrist_L": transformed_p3ds[5] - transformed_p3ds[3],
        "elbow_L": transformed_p3ds[3] - transformed_p3ds[1],
        "shoulder_L": (transformed_p3ds[1] - transformed_p3ds[0]),
    }
    if DEBUG_LOGS and WHILE_COUNT % TRACE_EVERY == 0:
        link_norms = {k: float(np.linalg.norm(v)) if v is not None and np.all(np.isfinite(v)) else None for k, v in links.items()}
        print(f"[DBG] frame {WHILE_COUNT}: link norms {link_norms}")
    globals_map = {
        "wrist_R": torquesR[0][0],
        "elbow_R": torquesR[1][0],
        "shoulder_R": torquesR[2][0],
        "wrist_L": torquesL[0][0],
        "elbow_L": torquesL[1][0],
        "shoulder_L": torquesL[2][0],
    }
    if TRACE_DYN and (WHILE_COUNT % 30 == 0):
        try:
            gn = {k: float(np.linalg.norm(v)) for k, v in globals_map.items()}
        except Exception:
            gn = {k: float('nan') for k in globals_map.keys()}
        print(f"[DYN:GLOBAL TAU] norms={gn}")
    # parent linkを定義（ローカル軸y安定化: 肘面基準）
    parent_links = {
        "wrist_R": links["elbow_R"],
        "elbow_R": links["shoulder_R"],
        "shoulder_R": None,
        "wrist_L": links["elbow_L"],
        "elbow_L": links["shoulder_L"],
        "shoulder_L": None,
    }
    locals_map = {}
    t_seg = time.perf_counter()
    for _k in globals_map.keys():
        try:
            locals_map[_k] = compute_local_torque(globals_map[_k], links[_k], parent_vec=parent_links[_k])
        except Exception as e:
            if DEBUG_LOGS:
                print(f"[DBG] compute_local_torque failed for {_k}: {e}")
            locals_map[_k] = np.array([0.0, 0.0, 0.0])
    _perf.add('local_torque', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("local_torque", "local"):
        _perf.next()
        break
    if DEBUG_LOGS and WHILE_COUNT % TRACE_EVERY == 0:
        y_comp = {k: float(v[1]) if v is not None and np.all(np.isfinite(v)) else None for k, v in locals_map.items()}
        print(f"[DBG] frame {WHILE_COUNT}: local y {{elbow±/wrist±}} {y_comp}")

    # ---- Offline wrist energy capture (optional, bilateral) ----
    if os.getenv('OFFLINE_WRIST_CAPTURE','0') in ('1','true','True'):
        try:
            # Right side
            _fw_vec_R = links.get('wrist_R')
            _tau_wrist_R = locals_map.get('wrist_R')
            if '_offline_wrist_vectors' not in globals():
                globals()['_offline_wrist_vectors'] = []  # type: ignore
                globals()['_offline_wrist_tau_y'] = []     # type: ignore
            # Left side (store in separate globals)
            if '_offline_wrist_vectors_L' not in globals():
                globals()['_offline_wrist_vectors_L'] = []  # type: ignore
                globals()['_offline_wrist_tau_y_L'] = []     # type: ignore
            if _fw_vec_R is not None and np.all(np.isfinite(_fw_vec_R)):
                globals()['_offline_wrist_vectors'].append(np.asarray(_fw_vec_R, dtype=float))  # type: ignore
                tyR = float(_tau_wrist_R[1]) if (_tau_wrist_R is not None and np.all(np.isfinite(_tau_wrist_R))) else 0.0
                globals()['_offline_wrist_tau_y'].append(tyR)  # type: ignore
            # Left forearm vector (elbow_L->wrist_L)
            _fw_vec_L = links.get('wrist_L')
            _tau_wrist_L = locals_map.get('wrist_L')
            if _fw_vec_L is not None and np.all(np.isfinite(_fw_vec_L)):
                globals()['_offline_wrist_vectors_L'].append(np.asarray(_fw_vec_L, dtype=float))  # type: ignore
                tyL = float(_tau_wrist_L[1]) if (_tau_wrist_L is not None and np.all(np.isfinite(_tau_wrist_L))) else 0.0
                globals()['_offline_wrist_tau_y_L'].append(tyL)  # type: ignore
        except Exception as _cap_e:  # noqa: BLE001
            if WHILE_COUNT % 120 == 0:
                print(f"[OFFLINE_WRIST_CAPTURE] capture fail (bilateral): {_cap_e}")

    # ==== サイクルE用の角度・トルク蓄積（肘） ====
    try:
        v_ua_R = links["elbow_R"]          # 肘関節に近い上腕方向（肩->肘）
        v_fa_R = links["wrist_R"]          # 前腕方向（肘->手首）
        v_ua_L = links["elbow_L"]
        v_fa_L = links["wrist_L"]
        th_R = angle_between(v_ua_R, v_fa_R)
        th_L = angle_between(v_ua_L, v_fa_L)
        tau_R = float(locals_map.get("elbow_R", np.zeros(3))[1])
        tau_L = float(locals_map.get("elbow_L", np.zeros(3))[1])
        _E_buffers['elbow_R']['theta'].append(th_R)
        _E_buffers['elbow_R']['tau'].append(tau_R)
        _E_buffers['elbow_L']['theta'].append(th_L)
        _E_buffers['elbow_L']['tau'].append(tau_L)
        
        # ======== 適応的fc: f0推定ステップ ========
        if E_FC_ADAPTIVE_ON:
            fps_for_lpf = float(_lpf_fps_ema) if (_lpf_fps_ema is not None and np.isfinite(_lpf_fps_ema)) else 30.0
            fps_for_lpf = float(np.clip(fps_for_lpf, E_FPS_MIN, E_FPS_MAX))
            # lazy init: 初回フレームのみ実効fpsで初期化
            if _f0_estimator is None:
                _f0_estimator = OnlineF0Estimator(fps=fps_for_lpf, win_sec=E_F0_WIN_SEC, fmin=E_F0_FMIN)
            else:
                _f0_estimator.set_fps(fps_for_lpf)

            _e_dt_sec_current = 1.0 / max(fps_for_lpf, 1e-6)
            _fc_update_interval = max(1, int(round(fps_for_lpf / max(E_FC_UPDATE_HZ, 1e-6))))

            # 各フレームでtheta値をestimatorに供給（左右平均を1サンプルとして使用）
            if np.isfinite(th_R) and np.isfinite(th_L):
                _theta_for_f0 = 0.5 * (th_R + th_L)
            elif np.isfinite(th_R):
                _theta_for_f0 = th_R
            else:
                _theta_for_f0 = th_L
            _f0_estimator.step(_theta_for_f0)
            _fc_update_counter += 1
            
            # E_FC_UPDATE_HZ頻度で f0推定とfc更新
            if _fc_update_counter >= _fc_update_interval:
                f0_hat, conf_db = _f0_estimator.estimate()
                _fc_current = _fc_scheduler(
                    f0_hat, conf_db, _fc_current,
                    E_FC_MIN, E_FC_MAX, E_FC_K,
                    E_FC_EMA_BETA, E_F0_SNR_THRESHOLD
                )
                if E_DEBUG:
                    print(f"[ADAPTIVE_FC] fps={fps_for_lpf:.2f} f0={f0_hat:.2f}Hz conf={conf_db:.1f}dB fc={_fc_current:.3f}Hz")
                _fc_update_counter = 0
        
        # 連続表示用: 正仕事のみ Σ τ·dθ を積算
        for _side, _th_now, _tau_now in (("elbow_R", th_R, tau_R), ("elbow_L", th_L, tau_L)):
            th_prev = _continuous_last_theta[_side]
            if th_prev is not None and np.isfinite(th_prev) and np.isfinite(_th_now):
                dth = _th_now - th_prev  # rad
                work_inc = _tau_now * dth
                if work_inc > 0:
                    _continuous_energy_J[_side] += work_inc
            _continuous_last_theta[_side] = _th_now
    except Exception as _e_acc:
        if E_DEBUG and (WHILE_COUNT % 60 == 0):
            print(f"[EPIPE] accumulate failed: {_e_acc}")

    # ================= フォールバック: サイクル未検出時の暫定エネルギー更新 =================
    # 目的: detector.update が発火しない環境でもゲージ針が全く動かない状況を回避し、
    # デバッグ観察を容易にする。一定フレーム経過後、瞬時トルクノルムを擬似エネルギーとして蓄積。
    try:
        FALLBACK_ENABLE = True
        NO_CYCLE_FALLBACK_FRAMES = 120  # これを超えても impulse_records が空なら発動開始
        visible_keys = (gauge.part_keys if (gauge is not None) else part_keys)
        if (
            FALLBACK_ENABLE
            and gauge is not None
            and WHILE_COUNT > NO_CYCLE_FALLBACK_FRAMES
            and all(len(impulse_records.get(k, [])) == 0 for k in visible_keys)
            and WHILE_COUNT % 10 == 0  # 頻度制御
        ):
            pseudo_impulses = {}
            for k in visible_keys:
                # ローカルトルクベクトル z 成分かノルムを簡易代理指標に
                lt = locals_map.get(k)
                if lt is None or not np.all(np.isfinite(lt)):
                    continue
                pseudo_E = float(np.linalg.norm(lt)) * dt  # 単純スケール
                impulse_records[k].append(pseudo_E)
                current_impulses[k] = pseudo_E
                pseudo_impulses[k] = pseudo_E
            if pseudo_impulses:
                gauge.update_impulses(current_impulses)
                try:
                    gauge.update()
                except Exception as _ge:  # noqa: BLE001
                    if DEBUG_LOGS:
                        print(f"[GaugeFallback] update error: {_ge}")
                if DEBUG_LOGS:
                    print(f"[GaugeFallback] pseudo energies frame={WHILE_COUNT}: {pseudo_impulses}")
    except Exception as _fb_e:  # noqa: BLE001
        if WHILE_COUNT % 120 == 0:
            print(f"[GaugeFallback] failed: {_fb_e}")

    # ==== ローカル座標軸デバッグ描画（任意） ====
    # 低頻度でのキー受付: 'a' キーでトグル (OpenCV window フォーカス時)
    if ENABLE_AXES_DEBUG:
        try:
            draw_all_local_axes(frame0, frame1, P0, P1, transformed_p3ds, links, x_offset=x_start)
        except Exception as e:
            if WHILE_COUNT % 100 == 0:
                print(f"[axes_debug] 描画失敗: {e}")
    # キートグル処理
    k = cv.waitKey(1) & 0xFF
    if k == ord('a'):
        ENABLE_AXES_DEBUG = not ENABLE_AXES_DEBUG
        print(f"[axes_debug] ENABLE_AXES_DEBUG -> {ENABLE_AXES_DEBUG}")

    # ディクショナリにトルク値を格納
    for key in [
        "wrist_R",
        "elbow_R",
        "shoulder_R",
        "wrist_L",
        "elbow_L",
        "shoulder_L",
    ]:
        storage.add_torque(key, locals_map[key])
    # torque_sssへのトルク値の追加（左右腕 6 要素）
    temp_local = [locals_map[k] for k in part_keys]

    # --- 新: エネルギー用トルク成分履歴収集 ---
    for k in current_energy_component_history.keys():  # 表示対象 (フィルタ後) のみに限定
        ty = locals_map[k][1] if k in locals_map else 0.0
        contrib = 0.0
        if k in ELBOW_KEYS:
            # 上腕: ローカル y 成分のマイナスのみ（屈曲 or 伸展側想定）
            if ty < 0:
                contrib = -ty  # 正値として蓄積
        elif k in WRIST_KEYS:
            # 前腕: ローカル y 成分のプラスのみ
            if ty > 0:
                contrib = ty
        # 肩・体幹は未定: 0 のまま
        current_energy_component_history[k].append(contrib)

    # パワー（仕事率）を算出: P = tau · omega（グローバル同士の内積）
    omega_map = {
        "wrist_R": part_data["forearm_R"][-1]["omega"],
        "elbow_R": part_data["upper_arm_R"][-1]["omega"],
        "shoulder_R": part_data["both_shoulder"][-1]["omega"],
        "wrist_L": part_data["forearm_L"][-1]["omega"],
        "elbow_L": part_data["up_arm_l"][-1]["omega"],
        "shoulder_L": part_data["both_shoulder"][-1]["omega"],
    }
    for key in current_power_history.keys():  # 表示対象に合わせて計算
        tau_g = globals_map.get(key)
        omg = omega_map.get(key)
        if tau_g is None or omg is None or not (np.all(np.isfinite(tau_g)) and np.all(np.isfinite(omg))):
            p_val = 0.0
        else:
            p_val = float(np.dot(tau_g, omg))
        current_power_history[key].append(p_val)
    if WHILE_COUNT > 15:
        _z_cycle = transformed_p3ds[0][_cycle_axis_idx]
        if np.isfinite(_z_cycle) and detector.update(_z_cycle, WHILE_COUNT):
            hist_len = len(current_torque_history[part_keys[0]])
            print("detected")
            if prev_cycle_frame is not None and hist_len >= min_history_len:
                # 対象キー集合: 実際に集計している辞書のキーに合わせる（ゲージの有無でズレないように）
                keys_now = list(current_power_history.keys())
                # サイクル総エネルギー [J] を各部位で計算
                for pk in keys_now:
                    if pk in ELBOW_KEYS:
                        buf = _E_buffers.get(pk, {'theta': [], 'tau': []})
                        e_pos, e_neg, info = compute_cycle_energy_filtered(
                            np.array(buf['theta']), np.array(buf['tau']), _e_dt_sec_current,
                            fc_override=_fc_current if E_FC_ADAPTIVE_ON else None
                        )
                        energy = e_pos  # ゲージ用途: 正仕事
                        cycle_energy_debug_rows.append({
                            'frame': int(WHILE_COUNT),
                            'part': str(pk),
                            'e_pos': float(e_pos),
                            'e_neg': float(e_neg),
                            'fc_current': float(_fc_current),
                            'dt_sec': float(_e_dt_sec_current),
                            'n_u': int(info.get('n_u', 0)) if isinstance(info, dict) else 0,
                        })
                        if E_DEBUG:
                            print(f"[EPIPE] {pk} E+= {energy:.4f} info={info}")
                    elif pk in WRIST_KEYS:
                        comp_series = pd.Series(current_energy_component_history[pk])
                        energy = float(comp_series.sum() * dt)
                    else:
                        p_series = pd.Series(current_power_history[pk])
                        energy = float(p_series.sum() * dt)
                    impulse_records[pk].append(energy)
                    if gauge is not None:
                        current_impulses[pk] = energy

                # 非監修モードではゲージ更新
                if gauge is not None:
                    gauge.update_impulses(current_impulses)
                    # サイクル確定時は強制更新
                    try:
                        gauge.update()
                    except Exception as e:  # noqa: BLE001
                        print(f"[Gauge] cycle update error: {e}")
            # 次サイクル準備
            prev_cycle_frame = WHILE_COUNT
            # 実際に保持しているキー集合でクリア（ズレ防止）
            for key in list(current_power_history.keys()):
                if key in current_torque_history:
                    current_torque_history[key].clear()
                current_power_history[key].clear()
                if key in current_energy_component_history:
                    current_energy_component_history[key].clear()
            # Eバッファもクリア
            for _kE in _E_buffers.keys():
                _E_buffers[_kE]['theta'].clear()
                _E_buffers[_kE]['tau'].clear()

            # （任意）検出ログ出力
            print(f"Cycle impulse appended at frame {WHILE_COUNT}")
        for key, vec in zip(part_keys, temp_local):
            current_torque_history[key].append(vec[2])
    else:
        pass
    # 連続エネルギー値/単眼デモ判定をゲージへ毎フレーム反映
    if gauge is not None:
        if DEMO_MONO_GAUGE_ON:
            ratio_map = {}
            for _side in ("R", "L"):
                y_now = _shoulder_world_y_m(results0, _side)
                th_now = _elbow_angle_deg_cam0(frame0_keypoints, _side)

                y_base = _demo_shoulder_base_y[_side]
                th_base = _demo_elbow_base_deg[_side]
                if y_now is not None:
                    if y_base is None:
                        y_base = y_now
                    else:
                        y_base = float((1.0 - DEMO_BASELINE_EMA) * y_base + DEMO_BASELINE_EMA * y_now)
                    _demo_shoulder_base_y[_side] = y_base
                if th_now is not None:
                    if th_base is None:
                        th_base = th_now
                    else:
                        th_base = float((1.0 - DEMO_BASELINE_EMA) * th_base + DEMO_BASELINE_EMA * th_now)
                    _demo_elbow_base_deg[_side] = th_base

                shoulder_rise_m = 0.0
                elbow_delta_deg = 0.0
                if (y_now is not None) and (_demo_shoulder_base_y[_side] is not None):
                    shoulder_rise_m = float(_demo_shoulder_base_y[_side] - y_now)
                if (th_now is not None) and (_demo_elbow_base_deg[_side] is not None):
                    elbow_delta_deg = float(abs(th_now - _demo_elbow_base_deg[_side]))

                full_ok = (shoulder_rise_m >= DEMO_SHOULDER_RISE_FULL_M) and (elbow_delta_deg >= DEMO_ELBOW_DELTA_FULL_DEG)
                partial_ok = (shoulder_rise_m >= DEMO_SHOULDER_RISE_PARTIAL_M) and (elbow_delta_deg >= DEMO_ELBOW_DELTA_PARTIAL_DEG)

                if full_ok:
                    target_ratio = DEMO_RATIO_FULL
                elif partial_ok:
                    target_ratio = DEMO_RATIO_PARTIAL
                else:
                    target_ratio = 0.0

                cur_ratio = float(_demo_gauge_ratio[_side])
                if target_ratio > cur_ratio:
                    cur_ratio = min(target_ratio, cur_ratio + DEMO_RATIO_UP_STEP)
                else:
                    cur_ratio = max(target_ratio, cur_ratio - DEMO_RATIO_DOWN_STEP)
                cur_ratio = float(np.clip(cur_ratio, 0.0, 1.0))
                _demo_gauge_ratio[_side] = cur_ratio

                ratio_map[f"wrist_{_side}"] = cur_ratio
                ratio_map[f"elbow_{_side}"] = cur_ratio

            try:
                gauge.set_direct_ratios(ratio_map, fill_rgba=(0.20, 0.80, 0.20, 0.90))
            except Exception as _gdir_e:
                if DEBUG_LOGS and (WHILE_COUNT % 30 == 0):
                    print(f"[GaugeDemo] set_direct_ratios failed: {_gdir_e}")
        else:
            keys_now = list(current_power_history.keys())
            for pk in keys_now:
                if pk in ELBOW_KEYS:
                    # 正しいJ単位の連続エネルギー
                    energy_cont = float(_continuous_energy_J.get(pk, 0.0))
                elif pk in WRIST_KEYS:
                    # 暫定: 旧トルク成分の時間積分をスケールダウン
                    raw_sum = float(sum(current_energy_component_history.get(pk, [])) * dt)
                    energy_cont = raw_sum / 50.0
                    if energy_cont == 0.0:
                        # 微小代替: ローカルトルクy絶対値で最初の僅かな動きを可視化
                        try:
                            lt = locals_map.get(pk)
                            ty_abs = abs(float(lt[1])) if lt is not None and np.all(np.isfinite(lt)) else 0.0
                        except Exception:
                            ty_abs = 0.0
                        energy_cont = float(ty_abs * dt)
                else:
                    energy_cont = float(sum(current_power_history.get(pk, [])) * dt)
                current_impulses[pk] = energy_cont
            gauge.update_impulses(current_impulses)
            try:
                _gauge_log_state(tag="after_update_impulses")
            except Exception:
                pass
    # ノルムだけ取り出してプロット用に
    temp_norms = [np.linalg.norm(v) for v in temp_local]
    aim_torque.append(temp_local)

    # 新しい画像サイズを計算（余白分を加える）
    new_height = frame0.shape[0]
    new_width = frame0.shape[1] + PADDING

    # 描画フレーム生成（合成とテキスト描画を分割して計測）
    t_draw_total = time.perf_counter()
    # 1) 合成（new_frame確保 + 左側にframe0貼り付け）
    t_alloc = time.perf_counter()
    new_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_frame[: frame0.shape[0], : frame0.shape[1]] = frame0
    _perf.add('draw_alloc', time.perf_counter() - t_alloc)

    # 余白部分に左右それぞれの局所トルクを表示
    jp_labels = {
        "wrist_R": "右手首",
        "elbow_R": "右肘",
        "shoulder_R": "右肩",
        "wrist_L": "左手首",
        "elbow_L": "左肘",
        "shoulder_L": "左肩",
    }
    # 表示対象キー: 非監修モードではフィルタ済み gauge.part_keys を使う
    display_keys = (gauge.part_keys if (gauge is not None) else part_keys)
    # 2) テキスト描画
    draw_put_total = 0.0
    if USE_NATIVE_DRAW and (_native_overlay is not None) and getattr(_native_overlay, '_dll', None):
        # ネイティブ: 一括描画（BGRAで作業）
        t_lbl_all = time.perf_counter()
        try:
            bgra = cv.cvtColor(new_frame, cv.COLOR_BGR2BGRA)
        except Exception:
            # 予防的フォールバック
            bgra = np.concatenate([new_frame, np.full((*new_frame.shape[:2],1), 255, dtype=new_frame.dtype)], axis=2)
        items = []
        for i, key in enumerate(display_keys):
            lbl = jp_labels.get(key, key)
            y = 40 + 30 * i
            cur_E = float(current_impulses.get(key, 0.0))
            text = f"{lbl} E:{cur_E:.1f}"
            items.append({
                'x': new_width - 350,
                'y': y,
                'font': 24,
                'color': (255, 255, 255, 255),
                'text': text,
            })
        rc = _native_overlay.draw_texts_bgra(bgra, items)
        if rc < 0 and LOOP_TRACE:
            print(f"[TRACE] native draw failed rc={rc}")
            pass
        new_frame = cv.cvtColor(bgra, cv.COLOR_BGRA2BGR)
        draw_put_total = time.perf_counter() - t_lbl_all
        _perf.add('draw_put_total', draw_put_total)
    else:
        for i, key in enumerate(display_keys):
            lbl = jp_labels.get(key, key)
            y = 40 + 30 * i
            cur_E = float(current_impulses.get(key, 0.0))
            text = f"{lbl} E:{cur_E:.1f}"
            t_lbl = time.perf_counter()
            new_frame = put_text_jp(
                new_frame,
                text,
                (new_width - 350, y),
                24,
                (255, 255, 255),
                20,
            )
            dt_lbl = time.perf_counter() - t_lbl
            draw_put_total += dt_lbl
            try:
                _perf.add(f'draw_put:{key}', dt_lbl)
            except Exception:
                _perf.add('draw_put:unknown', dt_lbl)
            if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
                #print(f"[TRACE] draw_put key={key} dt={dt_lbl*1000.0:.2f} ms")
                pass
        _perf.add('draw_put_total', draw_put_total)
    _perf.add('draw_text', time.perf_counter() - t_draw_total)

    # グラフ用には10倍して丸め
    # グラフ更新は未使用のため割愛
    # --- 安全リサイズ: ウィンドウが閉じられていたら再生成する ---
    if not HEADLESS and not DISABLE_IMSHOW:
        t_seg = time.perf_counter()
        try:
            vis_prop = cv.getWindowProperty("MyWindow", cv.WND_PROP_VISIBLE)
        except Exception:
            vis_prop = -1
        if vis_prop < 1:  # -1 / 0 の場合再生成
            if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
                #print(f"[TRACE] namedWindow create: vis_prop={vis_prop}")
                pass
            cv.namedWindow("MyWindow", cv.WINDOW_NORMAL)
        cv.resizeWindow("MyWindow", new_width, new_height)
        cv.imshow("MyWindow", new_frame)
        if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
            #print(f"[TRACE] imshow done size=({new_width}x{new_height})")
            pass
        _perf.add('imshow+resize', time.perf_counter() - t_seg)
    else:
        if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
            reason = "HEADLESS=1" if HEADLESS else "DISABLE_IMSHOW=1"
            #print(f"[TRACE] {reason} -> skip imshow")
        # 表示しない場合でも、揃えるためにゼロコストとして計上
        _perf.add('imshow+resize', 0.0)
    if STOP_AFTER.lower() in ("imshow", "display",):
        _perf.next()
        break
    # Gauge 内部フレームカウンタ前進（ウォームアップ解除のため）
    if gauge is not None:
        try:
            gauge.tick()
        except Exception:
            pass
    # ユーザーがウィンドウを閉じた場合も安全に脱出
    if not HEADLESS and not DISABLE_IMSHOW:
        if cv.getWindowProperty("MyWindow", cv.WND_PROP_VISIBLE) < 1:
            if DEBUG_LOGS:
                print("[Loop] Window closed by user -> break")
            break

    # フレーム毎更新を間引き (負荷/競合低減) + 角度健全性チェック
    if gauge is not None and (WHILE_COUNT % _GAUGE_UPDATE_INTERVAL == 0):
        try:
            t_seg = time.perf_counter()
            try:
                _gauge_log_state(tag="before_draw")
            except Exception:
                pass
            angles_dbg = gauge.get_angles()
            if DEBUG_LOGS and any((a < 0 or a > 180 or not np.isfinite(a)) for a in angles_dbg):
                print(f"[WARN] angle out of range at frame {WHILE_COUNT}: {angles_dbg}")
            gauge.update()
            try:
                _gauge_log_state(tag="after_draw")
            except Exception:
                pass
            _perf.add('gauge_update', time.perf_counter() - t_seg)
        except Exception as _g_err:  # 予期しない例外でメインループを止めない
            if DEBUG_LOGS:
                print(f"[Gauge] periodic update() 失敗: {_g_err}")
    if STOP_AFTER.lower() in ("gauge", "gauge_update"):
        _perf.next()
        break

    # ヘルススナップショット
    if WHILE_COUNT % _HEALTH_LOG_INTERVAL == 0:
        _health_snapshot(WHILE_COUNT)
    # cv.imshow("cam1", frame1)
    WHILE_COUNT += 1
    # --- Debug: 実処理済みフレーム数（間引き後の処理回数）---
    if DEBUG_LOGS:
        print(f"[DBG] processed-loop count={WHILE_COUNT}")
        end_time = time.perf_counter()
        frame_dt = end_time - start_time
        print("dt=", f"{frame_dt:.3f}")
        if frame_dt > 0:
            print("FPS:", f"{1 / frame_dt:.1f}")
    else:
        end_time = time.perf_counter()
        frame_dt = end_time - start_time
    _perf.add('loop_total', frame_dt)
    _perf.next()
    _perf.end_loop(WHILE_COUNT, frame_dt)
    if _rt_burst_remaining > 0:
        _rt_burst_remaining -= 1
    if _rt_burst_cooldown > 0:
        _rt_burst_cooldown -= 1
    if frame_dt > 1e-6:
        _fps_inst = float(np.clip(1.0 / frame_dt, E_FPS_MIN, E_FPS_MAX))
        if _lpf_fps_ema is None or (not np.isfinite(_lpf_fps_ema)):
            _lpf_fps_ema = _fps_inst
        else:
            _lpf_fps_ema = (1.0 - E_FPS_EMA_BETA) * float(_lpf_fps_ema) + E_FPS_EMA_BETA * _fps_inst
    # 5ループに1回の実行時間表示
    try:
        if WHILE_COUNT % 5 == 0:
            print(f"[LOOP] #{WHILE_COUNT} dt={frame_dt:.4f}s fps={(1.0/frame_dt if frame_dt>0 else 0):.1f}")
    except Exception:
        pass
    # 次回処理までのスキップ量を、直近処理時間から算出
    if RT_POSE_FIXED_HZ_ON:
        _rt_skip_remaining = _rt_fixed_skip
        if E_DEBUG and (WHILE_COUNT % 5 == 0):
            print(f"[RTFIX] fixed_hz={RT_POSE_FIXED_HZ:.3f} -> set skip_next={_rt_skip_remaining}")
    elif RT_DELAY_SKIP_ON:
        if _rt_burst_remaining > 0:
            _rt_skip_remaining = RT_RISE_BURST_SKIP
            if E_DEBUG and (WHILE_COUNT % 5 == 0):
                print(f"[RTBURST] active={_rt_burst_remaining} -> force skip_next={RT_RISE_BURST_SKIP} (burst mode)")
        else:
            raw_interval_frames = int(round(frame_dt * _src_fps * max(0.0, RT_DELAY_SKIP_GAIN)))
            # 例: interval=30フレーム相当 -> 既に1フレーム進んでいるため残り29をskip
            skip_next = max(0, raw_interval_frames - 1)
            skip_next = int(np.clip(skip_next, RT_DELAY_SKIP_MIN, RT_DELAY_SKIP_MAX))
            _rt_skip_remaining = skip_next
            if E_DEBUG and (WHILE_COUNT % 5 == 0):
                print(f"[RTSKIP] frame_dt={frame_dt:.4f}s -> interval={raw_interval_frames}f, set skip_next={_rt_skip_remaining}")
    # 早期終了: フレーム上限
    if MAX_FRAMES and WHILE_COUNT >= MAX_FRAMES:
        print(f"[STOP] Reached MAX_FRAMES={MAX_FRAMES} -> exiting loop")
        break
    # キー入力で終了（HEADLESS ではスキップ）
    if not HEADLESS:
        key = cv.waitKey(1) & 0xFF
        # 早期終了の原因調査用ログ（まれに OS レベルで ESC=27 が発生するケースを検知）
        if key not in (255, ):  # 255 は no-key のことが多い
            print(f"[KEY] code={key}")
        # 即時終了キー: 'q' or 'Q' または（設定で有効なら）ESC
        if key in (ord('q'), ord('Q')) or (key == 27 and IMMEDIATE_ESC_BREAK):
            reason = 'q/Q' if key in (ord('q'), ord('Q')) else 'ESC'
            print(f"← 終了キー検出（{reason}）：ループを抜けます")
            # ==== 途中終了時 Serial 終了 & 即時 CSV 収集 ====
            if hx_recording_started and not hx_csv_collected and (hx_serial_client is not None):
                try:
                    print('[HX711] 途中停止: Serial STOP + DUMP 実行')
                    csv_bytes = hx_serial_client.stop_and_dump_serial()
                    if HX_DIAG:
                        print('[HX711] 部分CSVクイック診断:')
                        _hx711_csv_quick_diag_from_bytes(csv_bytes)
                    safe_ts = (hx_start_iso or _iso8601_utc_ms()).replace(':','').replace('-','')
                    interim_path = os.path.join(save_dir, f"hx711_log_{safe_ts}_partial.csv")
                    with open(interim_path, 'wb') as f:
                        f.write(csv_bytes)
                    print(f"[HX711] 部分 CSV 保存: {interim_path} (len={len(csv_bytes)} bytes)")
                    hx_csv_collected = True
                except Exception as e:  # noqa: BLE001
                    print(f"[HX711] 途中 Serial 取得失敗: {e}")
            break
        # 旧: ESC 長押しにも対応（設定で有効時）
        if key == 27:
            esc_count += 1
        else:
            esc_count = 0
        # 'o' で ON(DTR True), 'p' で OFF(DTR False)
        if key == ord('o') and serial_ctrl.enabled:
            serial_ctrl.set_dtr(True)
            print("[Serial] DTR -> True (ON)")
        elif key == ord('p') and serial_ctrl.enabled:
            serial_ctrl.set_dtr(False)
            print("[Serial] DTR -> False (OFF)")
        if esc_count > ESC_HOLD_FRAMES:
            print("← ESC長押し検出：ループを抜けます")
            # ==== 途中終了時 Serial 終了 & 即時 CSV 収集 ====
            if hx_recording_started and not hx_csv_collected and (hx_serial_client is not None):
                try:
                    print('[HX711] 途中停止: Serial STOP + DUMP 実行')
                    csv_bytes = hx_serial_client.stop_and_dump_serial()
                    if HX_DIAG:
                        print('[HX711] 部分CSVクイック診断:')
                        _hx711_csv_quick_diag_from_bytes(csv_bytes)
                    safe_ts = (hx_start_iso or _iso8601_utc_ms()).replace(':','').replace('-','')
                    interim_path = os.path.join(save_dir, f"hx711_log_{safe_ts}_partial.csv")
                    with open(interim_path, 'wb') as f:
                        f.write(csv_bytes)
                    print(f"[HX711] 部分 CSV 保存: {interim_path} (len={len(csv_bytes)} bytes)")
                    hx_csv_collected = True
                except Exception as e:  # noqa: BLE001
                    print(f"[HX711] 途中 Serial 取得失敗: {e}")
            break  # ← ESC 長押し時のみループ離脱

## 手動クリーンアップ削除: atexit._cleanup_resources に一本化済み
## (重複 close によるネイティブクラッシュ防止)

# ================= HX711 ロガー停止 & 取得 (Serial 経由) =================
hx_csv_path = None
# ループ内で既に取得済みでなければ最終取得（Serial STOP + DUMP）
if hx_recording_started and hx_start_iso and not hx_csv_collected:
    if hx_serial_client is not None:
        try:
            csv_bytes = hx_serial_client.stop_and_dump_serial()
            if HX_DIAG:
                print('[HX711] 最終CSVクイック診断:')
                _hx711_csv_quick_diag_from_bytes(csv_bytes)
            # 保存ファイル名に開始 ISO の日付時刻を含める
            safe_ts = hx_start_iso.replace(':', '').replace('-', '')
            hx_csv_path = os.path.join(save_dir, f"hx711_log_{safe_ts}.csv")
            with open(hx_csv_path, 'wb') as f:
                f.write(csv_bytes)
            print(f"[HX711] Serial CSV 保存: {hx_csv_path} (len={len(csv_bytes)} bytes)")
        except Exception as e:  # noqa: BLE001
            print(f"[HX711] Serial 取得失敗: {e}")
else:
    print('[HX711] 取得スキップ (開始未成功)')

# ファイルに書き込む
# with open(folder_path + "\\max_value.txt", "w", encoding="utf-8") as file:
#    file.write(str(max(aim_torque)))

# -------------------------------
# ① kpts_3d：3D座標データを保存
# -------------------------------
# kpts_3d: List[np.ndarray]  → (12, 3)の各フレームごとのリストと想定
flattened_rows = []

for frame_idx, frame in enumerate(kpts_3d):
    row = {"frame": frame_idx}
    for joint_idx in range(frame.shape[0]):
        x, y, z = frame[joint_idx]
        row[f"joint_{joint_idx}_x"] = round(float(x), 4)
        row[f"joint_{joint_idx}_y"] = round(float(y), 4)
        row[f"joint_{joint_idx}_z"] = round(float(z), 4)
    flattened_rows.append(row)

df_coords = pd.DataFrame(flattened_rows)
_grav_tag = f"_g{_gravity_label}" if GRAVITY_TAG_IN_CSV else ""
df_coords.to_csv(os.path.join(save_dir, f"kpts3d_{timestamp}{_grav_tag}.csv"), index=False)
print("✅ 3D座標データを保存しました")

# -------------------------------
# ③ 単眼3D (world) 座標データ保存
# -------------------------------
# 単眼3D(world) 座標の保存は廃止
print(impulse_records)
# ── 4) ループ後処理（簡素化：統計更新は行わず既存値利用） ──────────────────
print("終了: インパルス記録数 summary => " + ", ".join(f"{k}:{len(v)}" for k,v in impulse_records.items()))


# -------------------------------
# ② トルクデータを保存
# -------------------------------
# torques: [(値, 部位名)] × 各フレーム分

# 部位名の順番（torques[i][1] の順に合わせておくこと）
part_names = [
    "wrist_R",
    "elbow_R",
    "shoulder_R",
    "wrist_L",
    "elbow_L",
    "shoulder_L",
]

# CSVに出力するデータを整形
csv_rows = []
for frame_idx, frame_data in enumerate(aim_torque):
    row = {"frame": frame_idx}
    for part_idx, vec in enumerate(frame_data):
        part = part_names[part_idx]
        row[f"{part}_x"] = round(float(vec[0]), 4)
        row[f"{part}_y"] = round(float(vec[1]), 4)
        row[f"{part}_z"] = round(float(vec[2]), 4)
    csv_rows.append(row)

# DataFrameにして保存
df = pd.DataFrame(csv_rows)
save_path = os.path.join(save_dir, f"aim_torque_vec_{timestamp}{_grav_tag}.csv")
df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"✅ aim_torque（ベクトル形式）を保存しました: {save_path}")

# -------------------------------
# ②-2 サイクルエネルギー診断データを保存（LPFチューニング用）
# -------------------------------
if cycle_energy_debug_rows:
    df_cycle_dbg = pd.DataFrame(cycle_energy_debug_rows)
    cycle_dbg_path = os.path.join(save_dir, f"cycle_energy_debug_{timestamp}{_grav_tag}.csv")
    df_cycle_dbg.to_csv(cycle_dbg_path, index=False, encoding="utf-8-sig")
    print(f"✅ cycle_energy_debug を保存しました: {cycle_dbg_path}")
else:
    print("[INFO] cycle_energy_debug: 有効サイクルが無いため出力なし")

# -------------------------------
# ⑤ Offline wrist capture NPY 保存 (任意)
# -------------------------------
if os.getenv('OFFLINE_WRIST_CAPTURE','0') in ('1','true','True'):
    try:
        _wv = globals().get('_offline_wrist_vectors')
        _wt = globals().get('_offline_wrist_tau_y')
        if _wv and _wt and len(_wv) == len(_wt):
            wv_arr = np.asarray(_wv, dtype=float)
            wt_arr = np.asarray(_wt, dtype=float)
            np.save(os.path.join(save_dir, f"forearm_R_{timestamp}.npy"), wv_arr)
            np.save(os.path.join(save_dir, f"tau_wrist_R_{timestamp}.npy"), wt_arr)
            print(f"✅ OFFLINE_WRIST_CAPTURE: 保存 forearm_R_{timestamp}.npy / tau_wrist_R_{timestamp}.npy (N={len(wv_arr)})")
        else:
            print('[OFFLINE_WRIST_CAPTURE] データ不足のため保存スキップ')
        # Left side save
        _wvL = globals().get('_offline_wrist_vectors_L')
        _wtL = globals().get('_offline_wrist_tau_y_L')
        if _wvL and _wtL and len(_wvL) == len(_wtL):
            wv_arr_L = np.asarray(_wvL, dtype=float)
            wt_arr_L = np.asarray(_wtL, dtype=float)
            np.save(os.path.join(save_dir, f"forearm_L_{timestamp}.npy"), wv_arr_L)
            np.save(os.path.join(save_dir, f"tau_wrist_L_{timestamp}.npy"), wt_arr_L)
            print(f"✅ OFFLINE_WRIST_CAPTURE: 保存 forearm_L_{timestamp}.npy / tau_wrist_L_{timestamp}.npy (N={len(wv_arr_L)})")
    except Exception as _sv_e:  # noqa: BLE001
        print(f"[OFFLINE_WRIST_CAPTURE] 保存失敗: {_sv_e}")
# %%
# Matplotlib 図のクローズは不要

# m_max_part 用テンプレ/参照場所のメモ
try:
    if 'SUBJECT_ID' in globals() and SUBJECT_ID:
        print(f"[INFO] m_max_part を編集したい場合は ./m_max_part_{SUBJECT_ID}.json を作成・編集してください。")
    else:
        print('[INFO] m_max_part を編集したい場合は ./m_max_part_SXXX.json の形式で作成してください (例: m_max_part_S001.json)。')
except Exception:
    pass
