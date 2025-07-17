# %%
import logging
import threading
import tracemalloc
import atexit
import traceback
import os
import time
import importlib  # Pylint E0601 対策: 後段で importlib.util を参照するため先行 import
import serial
import sys
import datetime

# pylint: disable=no-member
import cv2 as cv
import japanize_matplotlib  # pylint: disable=unused-import # 日本語表示のサポート
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import csv
import json

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
)
from Gauge_display import GaugeDisplay
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
    from scipy.signal import butter, filtfilt
    from scipy.interpolate import PchipInterpolator
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# ===================== サイクルE（ゲージ用）フィルタパイプライン設定 =====================
# 5 Hz 前提: fc は 1.0〜1.5 Hz 推奨。env で微調整可。
E_FC = float(os.getenv('E_FC', '1.2'))
E_LPF_ORDER = int(os.getenv('E_LPF_ORDER', '2'))  # 2〜4
E_RESAMPLE_N = int(os.getenv('E_RESAMPLE_N', '80'))  # 50〜100
E_MAX_DTH = float(os.getenv('E_MAX_DTH', '0.25'))  # 角度ステップの上限 [rad]
E_WINSOR_PCTL_LOW = float(os.getenv('E_WLOW', '5'))  # トルクの下側ウィンズライジング
E_WINSOR_PCTL_HIGH = float(os.getenv('E_WHIGH', '95'))
E_DEBUG = os.getenv('E_DEBUG', '0') in ('1','true','True')

# 肘の角度・トルクのフレーム蓄積バッファ（1サイクル分）
_E_buffers = {
    'elbow_R': {'theta': [], 'tau': []},
    'elbow_L': {'theta': [], 'tau': []},
}

def _butter_lowpass_filtfilt(x: np.ndarray, fs: float, fc: float, order: int) -> np.ndarray:
    if len(x) < max(8, 3*order+1):
        return x.copy()
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

def _winsorize(y: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    if len(y) < 4:
        return y.copy()
    lo = np.percentile(y, p_low)
    hi = np.percentile(y, p_high)
    return np.clip(y, lo, hi)

def compute_cycle_energy_filtered(theta: np.ndarray, tau: np.ndarray, dt_sec: float) -> tuple[float, float, dict]:
    """推奨パイプラインで E⁺/E⁻ を返す。
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
    info = {'status': 'ok', 'n_u': int(len(th_u))}
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
                print(f"[Pose] Auto-detected model: {POSE_TASK_MODEL}")
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
    print(f"[CPU] MP_THREADS={MP_THREADS}  physical={_phy}  logical={_log}  os.cpu_count()={os.cpu_count()}")
    print(f"[CPU] OpenCV: numCPUs={_cv_cpus}  numThreads={_cv_threads}")
    print(f"[Pose] USE_POSE_LANDMARKER={USE_POSE_LANDMARKER}  model={POSE_TASK_MODEL}  exists={os.path.exists(POSE_TASK_MODEL)}")
    print(f"[Pose] MP_INPUT_SCALE={MP_INPUT_SCALE}")
    # 任意: OpenCV スレッド数を明示設定（OPENCV_THREADS）
    _ocv_thr_env = os.getenv('OPENCV_THREADS', '').strip()
    if _ocv_thr_env:
        try:
            _ocv_thr = max(1, int(_ocv_thr_env))
            if hasattr(cv, 'setNumThreads'):
                cv.setNumThreads(_ocv_thr)
                print(f"[CPU] OpenCV setNumThreads -> {cv.getNumThreads() if hasattr(cv, 'getNumThreads') else _ocv_thr}")
        except Exception as _e:
            print(f"[CPU] OpenCV setNumThreads failed: {_e}")

def _tasks_imports():
    try:
        # 互換性のため段階的に import を試す
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions  # type: ignore
        try:
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode  # type: ignore
        except Exception:
            # 古いAPI名
            from mediapipe.tasks.python.vision import VisionRunningMode as VisionTaskRunningMode  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
        return PoseLandmarker, PoseLandmarkerOptions, VisionTaskRunningMode, BaseOptions
    except Exception as _e:
        if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
            print(f"[Pose] Tasks API import failed: {_e}")
        return None, None, None, None

class _LM:
    __slots__ = ('x','y','z','visibility')
    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.0):
        self.x = float(x); self.y = float(y); self.z = float(z); self.visibility = float(visibility)

class _NLList:
    __slots__ = ('landmark',)
    def __init__(self, landmark_list):
        self.landmark = landmark_list

class _PoseResult:
    __slots__ = ('pose_landmarks',)
    def __init__(self, lm_list_or_none):
        # Solutions互換: None または _NLList
        self.pose_landmarks = lm_list_or_none

class PoseEstimator:
    def __init__(self, use_tasks: bool, model_path: str, min_det: float = 0.5, min_track: float = 0.5, num_threads: int | None = None):
        self._mode = 'solutions'
        self._pose = None
        self._landmarker = None
        # ネイティブ優先（指定時・DLL有効時）
        self._native = None
        if USE_NATIVE_POSE and (_native_pose is not None):
            try:
                self._native = _native_pose.NativePoseEstimator(model_path, num_threads=num_threads or MP_THREADS)
                self._mode = 'native'
                if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                    print("[Pose] Using NativePoseEstimator (DLL)")
            except Exception as _ne:
                if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                    print(f"[Pose] Native pose not available -> { _ne }")
        if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
            print(f"[Pose] init: use_tasks={use_tasks} model_path={model_path} exists={os.path.exists(model_path)}")

        # Tasks API を試す（ネイティブが無効の場合）
        if (self._mode != 'native') and use_tasks and os.path.exists(model_path):
            PL, PLOpt, VRM, BO = _tasks_imports()
            if PL and PLOpt and VRM and BO:
                # XNNPACK スレッド設定（Tasks API BaseOptions）
                nt = int(num_threads) if num_threads and num_threads > 0 else MP_THREADS
                base = None
                try:
                    base = BO(model_asset_path=model_path, num_threads=nt)
                except TypeError as _te:
                    # 古い MediaPipe では BaseOptions に num_threads が無い
                    if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                        print(f"[Pose] BaseOptions num_threads unsupported -> retry without it: {_te}")
                    base = BO(model_asset_path=model_path)
                    nt = None  # 表示用（未設定）
                except Exception as _pe:
                    if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                        print(f"[Pose] BaseOptions init failed: {_pe}")
                    base = None
                if base is not None:
                    try:
                        opts = PLOpt(base_options=base, running_mode=VRM.IMAGE, num_poses=1)
                        # 環境変数からTasksの閾値を反映（あれば）
                        try:
                            _t_min_det = float(os.getenv('POSE_TASK_MIN_DET', os.getenv('POSE_MIN_DET', '0.5')))
                        except Exception:
                            _t_min_det = 0.5
                        try:
                            _t_min_track = float(os.getenv('POSE_TASK_MIN_TRACK', os.getenv('POSE_MIN_TRACK', '0.5')))
                        except Exception:
                            _t_min_track = 0.5
                        try:
                            _t_min_presence = float(os.getenv('POSE_TASK_MIN_PRESENCE', '0.5'))
                        except Exception:
                            _t_min_presence = 0.5
                        # オプションに存在する項目のみ設定（互換維持）
                        try:
                            if hasattr(opts, 'min_pose_detection_confidence'):
                                setattr(opts, 'min_pose_detection_confidence', _t_min_det)
                            if hasattr(opts, 'min_tracking_confidence'):
                                setattr(opts, 'min_tracking_confidence', _t_min_track)
                            if hasattr(opts, 'min_pose_presence_confidence'):
                                setattr(opts, 'min_pose_presence_confidence', _t_min_presence)
                            if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                                print(f"[Pose] Tasks thresholds: det={_t_min_det} track={_t_min_track} presence={_t_min_presence}")
                        except Exception as _te2:
                            if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                                print(f"[Pose] Tasks threshold apply failed (ignored): {_te2}")
                        self._landmarker = PL.create_from_options(opts)
                        self._mode = 'tasks'
                        if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                            th_str = str(nt) if nt is not None else 'n/a'
                            print(f"[Pose] Using Tasks PoseLandmarker (threads={th_str}, model={model_path})")
                    except Exception as _pe:
                        if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                            print(f"[Pose] Tasks create_from_options failed: {_pe}")
                        # 後段で Solutions へ
            else:
                if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                    print("[Pose] Mediapipe Tasks API not available -> using Solutions Pose")
        else:
            if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                reason = []
                if self._mode == 'native':
                    reason.append('native DLL active')
                if not use_tasks:
                    reason.append('USE_POSE_LANDMARKER=0')
                if not os.path.exists(model_path):
                    reason.append('model not found')
                print(f"[Pose] Not using Tasks: {'; '.join(reason) if reason else 'unknown reason'}")

        if (self._mode != 'tasks') and (self._mode != 'native'):
            # 従来の Solutions Pose
            self._pose = mp.solutions.pose.Pose(min_detection_confidence=min_det, min_tracking_confidence=min_track)
            if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                print("[Pose] Using Solutions Pose")

    def process(self, frame_rgb: np.ndarray):
        if self._mode == 'native' and self._native is not None:
            try:
                return self._native.process(frame_rgb)
            except Exception as _pe:
                if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                    print(f"[Pose] native detect failed -> fallback: {_pe}")
                # ネイティブ失敗時は後段へ
        if self._mode == 'tasks' and self._landmarker is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = self._landmarker.detect(mp_image)
                # デバッグ: Tasksの検出数を表示
                if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                    try:
                        _cnt = len(res.pose_landmarks) if res and getattr(res, 'pose_landmarks', None) is not None else 0
                        print(f"[Pose] Tasks result: count={_cnt}")
                    except Exception:
                        pass
                if res and getattr(res, 'pose_landmarks', None) and len(res.pose_landmarks) > 0:
                    # 先頭人物のみ採用
                    pts = res.pose_landmarks[0]
                    lm_list = [_LM(x=p.x, y=p.y, z=getattr(p, 'z', 0.0), visibility=getattr(p, 'visibility', 0.0)) for p in pts]
                    return _PoseResult(_NLList(lm_list))
                return _PoseResult(None)
            except Exception as _pe:
                # 失敗時は空検出にする（ループ継続優先）
                if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                    print(f"[Pose] detect failed: {_pe} -> fallback to Solutions this frame")
                # 遅延フォールバック: Solutions を作成して処理
                if self._pose is None:
                    try:
                        self._pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                        self._mode = 'solutions'
                    except Exception:
                        return _PoseResult(None)
                # Solutions にフォールバック実行
                sol_res = self._pose.process(frame_rgb)  # type: ignore
                # ここでは frame_rgb は縮小後（mp0/mp1）なので、必要であれば上位で補正してください
                return sol_res
        else:
            # Solutions Pose と互換の results を返す
            return self._pose.process(frame_rgb)  # type: ignore

    def close(self):
        try:
            if self._mode == 'tasks' and self._landmarker is not None:
                try:
                    self._landmarker.close()
                except Exception:
                    pass
            elif self._pose is not None:
                try:
                    self._pose.close()
                except Exception:
                    pass
        except Exception:
            pass

# ================= HX711 Recorder (M5StampS3) 連携 追加インポート (オプション) =================
HX_RECORDER_AVAILABLE = False
BLE_RECORDER_AVAILABLE = False
RecorderClient = None  # type: ignore
BLERecorderClientSync = None  # type: ignore
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
SUBJECT_ID = 8#被験者番号
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
        self.ser: serial.Serial | None = None
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
        except (serial.SerialException, OSError, ValueError) as e:
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

print("Video loaded")
# 入力ストリーム解決ロジック

def _find_latest_recordings(base_dir: str) -> Tuple[str | None, str | None]:
    """最新の cam0_output_*.mp4 / cam1_output_*.mp4 を探して返す。なければ None。
    時刻降順で最初のものを採用。
    """
    cam0_list = sorted(glob.glob(os.path.join(base_dir, "cam0_output_*.mp4")), reverse=True)
    cam1_list = sorted(glob.glob(os.path.join(base_dir, "cam1_output_*.mp4")), reverse=True)
    cam0 = cam0_list[0] if cam0_list else None
    cam1 = cam1_list[0] if cam1_list else None
    return cam0, cam1

def _find_recording_pairs(base_dir: str) -> list[tuple[str, str]]:
    """cam0_output_*.mp4 と cam1_output_*.mp4 の『共通サフィックス』でペアを作り、新しい順に返す。
    例: cam0_output_0924_084204.mp4 と cam1_output_0924_084204.mp4
    """
    cam0_list = glob.glob(os.path.join(base_dir, "cam0_output_*.mp4"))
    cam1_list = glob.glob(os.path.join(base_dir, "cam1_output_*.mp4"))
    def suffix(p: str) -> str:
        return os.path.basename(p).replace("cam0_output_", "").replace("cam1_output_", "")
    m0 = {suffix(p): p for p in cam0_list}
    m1 = {suffix(p): p for p in cam1_list}
    common = sorted(set(m0.keys()) & set(m1.keys()), reverse=True)
    return [(m0[s], m1[s]) for s in common]


def _resolve_input_streams() -> Tuple[object, object, bool, str]:
    """
    入力ストリーム1/2 を決定する。
    優先度: USE_SAMPLE_VIDEOS → 実カメラ → (失敗時) AUTO_FALLBACK_TO_FILES
    戻り値: (s1, s2, file_mode, reason)
    """
    # 1) 強制サンプル
    if USE_SAMPLE_VIDEOS:
        if PREFER_RECORDING_PAIRS:
            # まずは cam0/cam1 の『共通サフィックス』で一致する最新ペアを使用（長さ不一致による早期終了を避ける）
            pairs = _find_recording_pairs(folder_path)
            if pairs:
                cam0_file, cam1_file = pairs[0]
                return cam0_file, cam1_file, True, "USE_SAMPLE_VIDEOS=1 (pair)"
        # 既知のメディアサンプルへフォールバック
        cam0_file = os.path.join(folder_path, "media", "cam000_test.mp4")
        cam1_file = os.path.join(folder_path, "media", "cam111_test.mp4")
        print(f"[Info] Using sample videos: {cam0_file}, {cam1_file}")
        return cam0_file, cam1_file, True, "USE_SAMPLE_VIDEOS=1 (media)"

    # 2) 既定の設定（カメラ 0/1 か、configがパス指定ならそのまま）
    s1, s2 = input_stream1, input_stream2

    # s1, s2 が整数（カメラID）であれば一旦試す
    def _try_open(pair: Tuple[object, object]) -> bool:
        tmp0 = cv.VideoCapture(pair[0])
        tmp1 = cv.VideoCapture(pair[1])
        ok = tmp0.isOpened() and tmp1.isOpened()
        tmp0.release()
        tmp1.release()
        return ok

    # 2a) パス指定ならそのまま file_mode=True
    if not (isinstance(s1, int) and isinstance(s2, int)):
        return s1, s2, True, "config: file paths"

    # 2b) カメラを試す
    if _try_open((s1, s2)):
        return s1, s2, False, "config: cameras 0/1"

    # 3) カメラNGで自動フォールバック
    if AUTO_FALLBACK_TO_FILES:
        cam0_file, cam1_file = _find_latest_recordings(folder_path)
        if cam0_file is None:
            cam0_file = os.path.join(folder_path, "media", "cam000_test.mp4")
        if cam1_file is None:
            cam1_file = os.path.join(folder_path, "media", "cam111_test.mp4")
        return cam0_file, cam1_file, True, "AUTO_FALLBACK_TO_FILES=1 (camera open failed)"

    # 4) それでもダメなら元の設定のまま（失敗する可能性あり）
    return s1, s2, not (s1 == 0 and s2 == 1), "no fallback"


local_input_stream1, local_input_stream2, file_mode, resolve_reason = _resolve_input_streams()
print(f"Input streams resolved: file_mode={file_mode}, reason={resolve_reason}")

def _dbg(*args, **kwargs):
    if IO_DEBUG:
        print("[IODBG]", *args, **kwargs)

def _open_capture_and_read_first(src: object) -> tuple[cv.VideoCapture, bool, object | None]:
    """与えられた入力（カメラID or ファイルパス）について、複数バックエンドで VideoCapture を試し、
    最初のフレームを読み込んで返す。戻り値: (cap, ret, frame)
    """
    # Backend 優先順位の決定
    # ファイルパスの場合は FFMPEG 優先（MSMF がフレーム数を誤検出/早期終了する事例対策）
    if isinstance(src, str):
        backends = []
        for b in (getattr(cv, 'CAP_FFMPEG', None), getattr(cv, 'CAP_MSMF', None), getattr(cv, 'CAP_DSHOW', None), cv.CAP_ANY):
            if isinstance(b, int) and b not in backends:
                backends.append(b)
    else:
        backends = [cv.CAP_ANY]
        for b in (getattr(cv, 'CAP_FFMPEG', None), getattr(cv, 'CAP_MSMF', None), getattr(cv, 'CAP_DSHOW', None)):
            if isinstance(b, int) and b not in backends:
                backends.append(b)
    # 文字列パスは正規化
    src_to_use = src
    if isinstance(src, str):
        src_to_use = os.path.normpath(src)
    last_err = None
    for be in backends:
        try:
            cap = cv.VideoCapture(src_to_use, be) if be != cv.CAP_ANY else cv.VideoCapture(src_to_use)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    try:
                        if isinstance(src, str):
                            total = cap.get(cv.CAP_PROP_FRAME_COUNT)
                        else:
                            total = -1
                        _dbg("opened backend=", be, "src=", src_to_use, "frames=", total)
                    except Exception:
                        pass
                    return cap, True, frame
                cap.release()
        except Exception as e:  # noqa: BLE001
            last_err = e
            try:
                cap.release()
            except Exception:
                pass
    # 失敗時はダミー cap を返す
    cap = cv.VideoCapture()
    return cap, False, None

reps = 10
one_rm_percentage = df_rm.loc[df_rm["反復回数"] == reps, "1RM%"].values[0] / 100
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

P0 = get_projection_matrix(0, file_mode)
P1 = get_projection_matrix(1, file_mode)
print("Projection matrices loaded")
# %%
save_path0 = f"cam0_output_{timestamp}.mp4"
save_path1 = f"cam1_output_{timestamp}.mp4"

cap0, ok0, frame0 = _open_capture_and_read_first(local_input_stream1)
cap1, ok1, frame1 = _open_capture_and_read_first(local_input_stream2)

if not (ok0 and ok1) or frame0 is None or frame1 is None:
    print("❌ 入力の読み込みに失敗しました。解決理由:", resolve_reason)
    opened = False
    # 優先順に従って試す
    if PREFER_RECORDING_PAIRS:
        # 1) 録画ペア
        pairs = _find_recording_pairs(folder_path)
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
            cap0, ok0, frame0 = _open_capture_and_read_first(p0)
            cap1, ok1, frame1 = _open_capture_and_read_first(p1)
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
            cap0, ok0, frame0 = _open_capture_and_read_first(cam0_file)
            cap1, ok1, frame1 = _open_capture_and_read_first(cam1_file)
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
        cap0, ok0, frame0 = _open_capture_and_read_first(cam0_file)
        cap1, ok1, frame1 = _open_capture_and_read_first(cam1_file)
        opened = ok0 and ok1 and frame0 is not None and frame1 is not None
        # 2) 録画ペア
        if not opened:
            pairs = _find_recording_pairs(folder_path)
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
                cap0, ok0, frame0 = _open_capture_and_read_first(p0)
                cap1, ok1, frame1 = _open_capture_and_read_first(p1)
                if ok0 and ok1 and frame0 is not None and frame1 is not None:
                    print("OK: opened recording pair")
                    opened = True
                    break
    if not opened:
        # 見つかったペア一覧を提示
        pairs = _find_recording_pairs(folder_path)
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
    total0 = cap0.get(cv.CAP_PROP_FRAME_COUNT); total1 = cap1.get(cv.CAP_PROP_FRAME_COUNT)
    fps0 = cap0.get(cv.CAP_PROP_FPS); fps1 = cap1.get(cv.CAP_PROP_FPS)
    pos0 = cap0.get(cv.CAP_PROP_POS_FRAMES); pos1 = cap1.get(cv.CAP_PROP_POS_FRAMES)
    print(f"[Input] stats: frames0={total0}, fps0={fps0}, pos0={pos0} | frames1={total1}, fps1={fps1}, pos1={pos1}")
except Exception:
    pass

# ファイル入力の場合は、初期フレーム消費後に先頭へ巻き戻してからメインループに入る
if file_mode:
    try:
        cap0.set(cv.CAP_PROP_POS_FRAMES, 0)
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
ret1, frame1 = cap1.read()

# Mediapipe用にRGB化 & 推定
if frame0 is None or frame1 is None:
    print("⚠️ 初期フレームが None です。再試行します。")
    # 1) 数回リトライ
    recovered = False
    for _i in range(10):
        ret0, frame0 = cap0.read(); ret1, frame1 = cap1.read()
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
        cap0, ok0, frame0 = _open_capture_and_read_first(local_input_stream1)
        cap1, ok1, frame1 = _open_capture_and_read_first(local_input_stream2)
        recovered = ok0 and ok1 and frame0 is not None and frame1 is not None
    # 3) サンプル／録画ペアへフォールバック
    if not recovered:
        print("[Input] Fallback: try sample media or recording pairs")
        opened = False
        if PREFER_RECORDING_PAIRS:
            pairs = _find_recording_pairs(folder_path)
            for p0, p1 in pairs:
                try:
                    cap0.release(); cap1.release()
                except Exception:
                    pass
                cap0, ok0, frame0 = _open_capture_and_read_first(p0)
                cap1, ok1, frame1 = _open_capture_and_read_first(p1)
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
            cap0, ok0, frame0 = _open_capture_and_read_first(cam0_file)
            cap1, ok1, frame1 = _open_capture_and_read_first(cam1_file)
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


x0_min, x0_max = get_valid_x_range(frame0_kpts, frame0.shape[1])
x1_min, x1_max = get_valid_x_range(frame1_kpts, frame1.shape[1])

x_min = min(x0_min, x1_min)
x_max = max(x0_max, x1_max)
x_margin = 50

# 2カメラで共通に安全なトリミング範囲を設定
width_min = min(frame0.shape[1], frame1.shape[1])
x_start = max(0, x_min - x_margin)
x_end = min(width_min, x_max + x_margin)
if x_end <= x_start:
    # フォールバック: 全幅
    x_start, x_end = 0, width_min
print(f"✅ トリミング範囲: x = {x_start} ～ {x_end}")


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

if RecorderClient is not None:
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
    print('[HX711] モジュール未インポートのため開始スキップ')


# サイクル内のトルクとパワー履歴（ゲージ集計用）
keys_for_hist = (gauge.part_keys if ('gauge' in globals() and gauge is not None) else part_keys)
current_power_history = {k: [] for k in keys_for_hist}
# 新: トルク成分ベースのエネルギー集計履歴（上腕=肘トルクy負, 前腕=手首トルクy正）
current_energy_component_history = {k: [] for k in keys_for_hist}

# 上腕(=肘トルク)と前腕(=手首トルク)のキー集合（肩・体幹は未定のため除外）
ELBOW_KEYS = {"elbow_R", "elbow_L"}
WRIST_KEYS = {"wrist_R", "wrist_L"}
# 連続表示用 肘エネルギー(J)積算バッファ (Σ τ·dθ)
_continuous_last_theta = {"elbow_R": None, "elbow_L": None}
_continuous_energy_J = {"elbow_R": 0.0, "elbow_L": 0.0}
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
_apply_camera_controls('cam1', cap1)
# オープン済みキャプチャのプロパティ出力（最終状態）
_print_cap_props('cam0', cap0)
_print_cap_props('cam1', cap1)

# カメラごとの時刻・間隔・直近の取得時間（効果測定用）
_cam0_last_ts = None
_cam1_last_ts = None
_cam0_last_dt = None
_cam1_last_dt = None
_cam0_last_shape = None
_cam1_last_shape = None

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
    okg1 = cap1.grab()
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
            p1 = cap1.get(cv.CAP_PROP_POS_FRAMES); t1 = cap1.get(cv.CAP_PROP_FRAME_COUNT)
            print(f"Video ended (grab fail) pos0/total0={p0}/{t0}, pos1/total1={p1}/{t1}")
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
    # スキップ判定（デコード不要なフレームはここで続行）
    if skip_counter % skip_mod != 0:
        _perf.next()
        continue
    # このタイミングのフレームのみ retrieve してデコード
    # 各カメラのretrieve時間を個別に計測（ブロッキング源の特定）
    t_seg0 = time.perf_counter()
    ret0, frame0 = cap0.retrieve()
    dt_ret0 = time.perf_counter() - t_seg0
    t_seg1 = time.perf_counter()
    ret1, frame1 = cap1.retrieve()
    dt_ret1 = time.perf_counter() - t_seg1
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
    # MediaPipe 入力を縮小（推論専用バッファ）。元の frame0/1 は BGR 戻しや描画のため保持
    if MP_INPUT_SCALE < 1.0:
        mp0 = cv.resize(frame0, None, fx=MP_INPUT_SCALE, fy=MP_INPUT_SCALE, interpolation=cv.INTER_AREA)
    else:
        mp0 = frame0
    results0 = pose0.process(mp0)
    if os.getenv('POSE_DEBUG', '0') in ('1','true','True') and (WHILE_COUNT % max(1, int(os.getenv('POSE_TRACE_EVERY','30'))) == 0):
        try:
            print(f"[Pose] input scale={MP_INPUT_SCALE} mp0_shape={tuple(mp0.shape)}")
        except Exception:
            pass
    _perf.add('mediapipe0', time.perf_counter() - t_seg)
    t_seg = time.perf_counter()
    if MP_INPUT_SCALE < 1.0:
        mp1 = cv.resize(frame1, None, fx=MP_INPUT_SCALE, fy=MP_INPUT_SCALE, interpolation=cv.INTER_AREA)
    else:
        mp1 = frame1
    results1 = pose1.process(mp1)
    if os.getenv('POSE_DEBUG', '0') in ('1','true','True') and (WHILE_COUNT % max(1, int(os.getenv('POSE_TRACE_EVERY','30'))) == 0):
        try:
            print(f"[Pose] input scale={MP_INPUT_SCALE} mp1_shape={tuple(mp1.shape)}")
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
    frame0_keypoints, frame1_keypoints = extract_keypoints(
        results0, results1, pose_keypoints, frame0, frame1
    )
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

    frame_p3ds = []

    nan_3d = 0
    t_seg = time.perf_counter()
    for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
        if uv1[0] == -1 or uv2[0] == -1:
            _p3d = [-1, -1, -1]
        else:
            _p3d = DLT(P0, P1, uv1, uv2)
        if not np.all(np.isfinite(_p3d)):
            nan_3d += 1
        frame_p3ds.append(_p3d)
    if DEBUG_LOGS and WHILE_COUNT % 30 == 0 and nan_3d:
        print(f"[DBG] frame {WHILE_COUNT}: non-finite 3D points={nan_3d}")
    temp_np = np.array(frame_p3ds).reshape((12, 3)) * 0.01
    transformed_p3ds = np.zeros_like(temp_np)
    # ファイル/カメラの別に係らず同一変換を適用
    transformed_p3ds[:, 0] = -temp_np[:, 0]
    transformed_p3ds[:, 1] = -temp_np[:, 2]
    transformed_p3ds[:, 2] = -temp_np[:, 1]
    kpts_3d.append(transformed_p3ds)
    if LOOP_TRACE and (WHILE_COUNT % VIDEO_TRACE_EVERY == 0):
        #print(f"[TRACE] triang+transform done")
        pass
    _perf.add('triang+transform', time.perf_counter() - t_seg)
    if STOP_AFTER.lower() in ("triang", "triangulate", "triang+transform", "transform"):
        _perf.next()
        break
    if 4 < WHILE_COUNT < 15:
        z_value += (transformed_p3ds[0][2]) / 10
    elif WHILE_COUNT == 15:
        detector = PushCycleDetector(z_value)

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
    t_seg = time.perf_counter()
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
    locals_map = {}
    t_seg = time.perf_counter()
    for _k in globals_map.keys():
        try:
            locals_map[_k] = compute_local_torque(globals_map[_k], links[_k])
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
        th_R = _angle_between(v_ua_R, v_fa_R)
        th_L = _angle_between(v_ua_L, v_fa_L)
        tau_R = float(locals_map.get("elbow_R", np.zeros(3))[1])
        tau_L = float(locals_map.get("elbow_L", np.zeros(3))[1])
        _E_buffers['elbow_R']['theta'].append(th_R)
        _E_buffers['elbow_R']['tau'].append(tau_R)
        _E_buffers['elbow_L']['theta'].append(th_L)
        _E_buffers['elbow_L']['tau'].append(tau_L)
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
        if detector.update(transformed_p3ds[0][2], WHILE_COUNT):
            hist_len = len(current_torque_history[part_keys[0]])
            print("detected")
            if prev_cycle_frame is not None and hist_len >= min_history_len:
                # 対象キー集合: 実際に集計している辞書のキーに合わせる（ゲージの有無でズレないように）
                keys_now = list(current_power_history.keys())
                # サイクル総エネルギー [J] を各部位で計算
                for pk in keys_now:
                    if pk in ELBOW_KEYS:
                        buf = _E_buffers.get(pk, {'theta': [], 'tau': []})
                        e_pos, e_neg, info = compute_cycle_energy_filtered(np.array(buf['theta']), np.array(buf['tau']), dt)
                        energy = e_pos  # ゲージ用途: 正仕事
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
    # 連続エネルギー値をゲージへ毎フレーム反映
    if gauge is not None:
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
    # 5ループに1回の実行時間表示
    try:
        if WHILE_COUNT % 5 == 0:
            print(f"[LOOP] #{WHILE_COUNT} dt={frame_dt:.4f}s fps={(1.0/frame_dt if frame_dt>0 else 0):.1f}")
    except Exception:
        pass
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
df_coords.to_csv(os.path.join(save_dir, f"kpts3d_{timestamp}.csv"), index=False)
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
save_path = os.path.join(save_dir, f"aim_torque_vec_{timestamp}.csv")
df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"✅ aim_torque（ベクトル形式）を保存しました: {save_path}")

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
