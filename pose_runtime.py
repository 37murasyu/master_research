"""
Pose runtime abstraction:
- Unified PoseEstimator with three backends: Native DLL, MediaPipe Tasks, and Solutions Pose.
- Self-contained environment toggles.

Environment variables:
- USE_POSE_LANDMARKER (default 1)
- USE_NATIVE_POSE_MODE (off/on/auto, default auto)
- USE_NATIVE_POSE (legacy fallback toggle)
- POSE_TASK_MODEL (fallback to pose_landmarker_lite.task in this folder)
- MP_THREADS (auto-detect if not set)
- MP_INPUT_SCALE (0.25..1.0; default 0.5)
- POSE_* thresholds for Tasks API: POSE_TASK_MIN_DET, POSE_TASK_MIN_TRACK, POSE_TASK_MIN_PRESENCE
- POSE_DEBUG (debug prints)
"""
from __future__ import annotations

import os
import glob
import math
from typing import Any

import numpy as np
import mediapipe as mp
import cv2 as cv

try:
    import py_native_pose as _native_pose
except Exception:
    _native_pose = None


def _detect_threads(default: int = 8) -> int:
    cores = None
    try:
        import importlib.util
        if importlib.util.find_spec('psutil') is not None:  # type: ignore[attr-defined]
            import importlib
            _psutil = importlib.import_module('psutil')  # type: ignore
            cores = _psutil.cpu_count(logical=False) or _psutil.cpu_count(logical=True)
    except Exception:
        cores = None
    if not cores:
        cores = os.cpu_count() or default
    return max(2, min(int(cores), 32))

# ========== Env toggles ==========
USE_POSE_LANDMARKER = str(os.getenv('USE_POSE_LANDMARKER', '1')).lower() in ('1', 'true', 'yes')
_native_mode = str(os.getenv('USE_NATIVE_POSE_MODE', '')).strip().lower()
if _native_mode not in ('off', 'on', 'auto'):
    if _native_mode:
        USE_NATIVE_POSE = _native_mode in ('1', 'true', 'yes')
        NATIVE_POSE_MODE = 'on' if USE_NATIVE_POSE else 'off'
    else:
        # 後方互換: 既存 USE_NATIVE_POSE を優先（未指定時は auto）
        _legacy = os.getenv('USE_NATIVE_POSE')
        if _legacy is None:
            NATIVE_POSE_MODE = 'auto'
            USE_NATIVE_POSE = (_native_pose is not None) and hasattr(_native_pose, 'NativePoseEstimator')
        else:
            USE_NATIVE_POSE = str(_legacy).lower() in ('1', 'true', 'yes')
            NATIVE_POSE_MODE = 'on' if USE_NATIVE_POSE else 'off'
else:
    NATIVE_POSE_MODE = _native_mode
    if NATIVE_POSE_MODE == 'on':
        USE_NATIVE_POSE = True
    elif NATIVE_POSE_MODE == 'off':
        USE_NATIVE_POSE = False
    else:
        USE_NATIVE_POSE = (_native_pose is not None) and hasattr(_native_pose, 'NativePoseEstimator')
DEFAULT_TASK_MODEL = os.path.join(os.path.dirname(__file__), 'pose_landmarker_lite.task')
POSE_TASK_MODEL = os.getenv('POSE_TASK_MODEL', DEFAULT_TASK_MODEL)
if not os.path.exists(POSE_TASK_MODEL):
    try:
        _dir = os.path.dirname(__file__)
        _candidates = [p for p in glob.glob(os.path.join(_dir, '*.task'))]
        if _candidates:
            _candidates.sort(key=lambda p: (0 if 'pose' in os.path.basename(p).lower() else 1, len(os.path.basename(p))))
            POSE_TASK_MODEL = _candidates[0]
            if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                print(f"[PoseRT] Auto-detected model: {POSE_TASK_MODEL}")
    except Exception:
        pass

try:
    MP_THREADS = int(os.getenv('MP_THREADS', '').strip()) if os.getenv('MP_THREADS') else _detect_threads(16)
except Exception:
    MP_THREADS = _detect_threads(16)

try:
    MP_INPUT_SCALE = float(os.getenv('MP_INPUT_SCALE', '0.5').strip())
except Exception:
    MP_INPUT_SCALE = 0.5
MP_INPUT_SCALE = max(0.25, min(MP_INPUT_SCALE, 1.0))

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
    print(f"[PoseRT] USE_POSE_LANDMARKER={USE_POSE_LANDMARKER}  model={POSE_TASK_MODEL}  exists={os.path.exists(POSE_TASK_MODEL)}")
    print(f"[PoseRT] NATIVE_POSE_MODE={NATIVE_POSE_MODE} use_native={USE_NATIVE_POSE} native_module={'yes' if _native_pose is not None else 'no'}")
    print(f"[PoseRT] MP_INPUT_SCALE={MP_INPUT_SCALE}")
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
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions  # type: ignore
        try:
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode  # type: ignore
        except Exception:
            from mediapipe.tasks.python.vision import VisionRunningMode as VisionTaskRunningMode  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
        return PoseLandmarker, PoseLandmarkerOptions, VisionTaskRunningMode, BaseOptions
    except Exception as _e:
        if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
            print(f"[PoseRT] Tasks API import failed: {_e}")
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
        self.pose_landmarks = lm_list_or_none


class PoseEstimator:
    def __init__(self, use_tasks: bool, model_path: str, min_det: float = 0.5, min_track: float = 0.5, num_threads: int | None = None):
        self._mode = 'solutions'
        self._pose = None
        self._landmarker = None
        self._native = None
        if USE_NATIVE_POSE and (_native_pose is not None):
            try:
                self._native = _native_pose.NativePoseEstimator(model_path, num_threads=num_threads or MP_THREADS)
                self._mode = 'native'
                if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                    print("[PoseRT] Using NativePoseEstimator (DLL)")
            except Exception as _ne:
                if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                    print(f"[PoseRT] Native pose not available -> { _ne }")
        if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
            print(f"[PoseRT] init: use_tasks={use_tasks} model_path={model_path} exists={os.path.exists(model_path)}")

        if (self._mode != 'native') and use_tasks and os.path.exists(model_path):
            PL, PLOpt, VRM, BO = _tasks_imports()
            if PL and PLOpt and VRM and BO:
                nt = int(num_threads) if num_threads and num_threads > 0 else MP_THREADS
                base = None
                try:
                    base = BO(model_asset_path=model_path, num_threads=nt)
                except TypeError as _te:
                    if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                        print(f"[PoseRT] BaseOptions num_threads unsupported -> retry without it: {_te}")
                    base = BO(model_asset_path=model_path)
                    nt = None
                except Exception as _pe:
                    if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                        print(f"[PoseRT] BaseOptions init failed: {_pe}")
                    base = None
                if base is not None:
                    try:
                        opts = PLOpt(base_options=base, running_mode=VRM.IMAGE, num_poses=1)
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
                        try:
                            if hasattr(opts, 'min_pose_detection_confidence'):
                                setattr(opts, 'min_pose_detection_confidence', _t_min_det)
                            if hasattr(opts, 'min_tracking_confidence'):
                                setattr(opts, 'min_tracking_confidence', _t_min_track)
                            if hasattr(opts, 'min_pose_presence_confidence'):
                                setattr(opts, 'min_pose_presence_confidence', _t_min_presence)
                            if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                                print(f"[PoseRT] Tasks thresholds: det={_t_min_det} track={_t_min_track} presence={_t_min_presence}")
                        except Exception as _te2:
                            if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                                print(f"[PoseRT] Tasks threshold apply failed (ignored): {_te2}")
                        self._landmarker = PL.create_from_options(opts)
                        self._mode = 'tasks'
                        if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                            th_str = str(nt) if nt is not None else 'n/a'
                            print(f"[PoseRT] Using Tasks PoseLandmarker (threads={th_str}, model={model_path})")
                    except Exception as _pe:
                        if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                            print(f"[PoseRT] Tasks create_from_options failed: {_pe}")
            else:
                if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                    print("[PoseRT] Mediapipe Tasks API not available -> using Solutions Pose")
        else:
            if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                reason = []
                if self._mode == 'native':
                    reason.append('native DLL active')
                if not use_tasks:
                    reason.append('USE_POSE_LANDMARKER=0')
                if not os.path.exists(model_path):
                    reason.append('model not found')
                print(f"[PoseRT] Not using Tasks: {'; '.join(reason) if reason else 'unknown reason'}")

        if (self._mode != 'tasks') and (self._mode != 'native'):
            self._pose = mp.solutions.pose.Pose(min_detection_confidence=min_det, min_tracking_confidence=min_track)
            if os.getenv('POSE_DEBUG', '0') in ('1', 'true', 'True'):
                print("[PoseRT] Using Solutions Pose")

    def process(self, frame_rgb: np.ndarray):
        if self._mode == 'native' and self._native is not None:
            try:
                return self._native.process(frame_rgb)
            except Exception as _pe:
                if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                    print(f"[PoseRT] native detect failed -> fallback: {_pe}")
        if self._mode == 'tasks' and self._landmarker is not None:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = self._landmarker.detect(mp_image)
                if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                    try:
                        _cnt = len(res.pose_landmarks) if res and getattr(res, 'pose_landmarks', None) is not None else 0
                        print(f"[PoseRT] Tasks result: count={_cnt}")
                    except Exception:
                        pass
                if res and getattr(res, 'pose_landmarks', None) and len(res.pose_landmarks) > 0:
                    pts = res.pose_landmarks[0]
                    lm_list = [_LM(x=p.x, y=p.y, z=getattr(p, 'z', 0.0), visibility=getattr(p, 'visibility', 0.0)) for p in pts]
                    return _PoseResult(_NLList(lm_list))
                return _PoseResult(None)
            except Exception as _pe:
                if os.getenv('POSE_DEBUG', '0') in ('1','true','True'):
                    print(f"[PoseRT] detect failed: {_pe} -> fallback to Solutions this frame")
                if self._pose is None:
                    try:
                        self._pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                        self._mode = 'solutions'
                    except Exception:
                        return _PoseResult(None)
                sol_res = self._pose.process(frame_rgb)  # type: ignore
                return sol_res
        else:
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

__all__ = ['PoseEstimator', 'USE_POSE_LANDMARKER', 'USE_NATIVE_POSE', 'POSE_TASK_MODEL', 'MP_THREADS', 'MP_INPUT_SCALE']
