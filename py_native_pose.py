from __future__ import annotations
import os
import ctypes as C
from ctypes import c_int, c_char_p, c_void_p, c_uint8, POINTER, Structure
from typing import Optional, List
import numpy as np

_here = os.path.dirname(__file__)

# トレース制御（Python ラッパ側）
_TRACE = str(os.getenv('NPOSE_PY_TRACE', '0')).lower() in ('1', 'true', 'yes')
try:
    _TRACE_EVERY = max(1, int(os.getenv('NPOSE_PY_TRACE_EVERY', '10')))
except Exception:
    _TRACE_EVERY = 10

# DLL 探索
_candidates = [
    os.path.join(_here, 'native_pose', 'build', 'Release', 'native_pose.dll'),
    os.path.join(_here, 'native_pose', 'build', 'native_pose.dll'),
    os.path.join(_here, 'native_pose.dll'),
]
_dll_path = next((p for p in _candidates if os.path.exists(p)), None)
_lib = None
if _dll_path:
    try:
        _lib = C.CDLL(_dll_path)
        if _TRACE:
            print(f"[NPOSE:PY] loaded DLL: {_dll_path}")
    except Exception:
        _lib = None
        if _TRACE:
            print(f"[NPOSE:PY] DLL load failed: {_dll_path}")
else:
    if _TRACE:
        print("[NPOSE:PY] DLL not found in candidates:")
        for p in _candidates:
            print(f"  - {p}")

class _NPoseLm(Structure):
    _fields_ = [
        ("x", C.c_float),
        ("y", C.c_float),
        ("z", C.c_float),
        ("visibility", C.c_float),
    ]

class _NPoseResult(Structure):
    _fields_ = [
        ("has_landmarks", c_int),
        ("landmark_count", c_int),
        ("landmarks", POINTER(_NPoseLm)),
    ]

if _lib is not None:
    _lib.npose_create.argtypes = [c_char_p, c_int, C.POINTER(c_void_p)]
    _lib.npose_create.restype = c_int
    _lib.npose_destroy.argtypes = [c_void_p]
    _lib.npose_destroy.restype = None
    _lib.npose_detect.argtypes = [c_void_p, POINTER(c_uint8), c_int, c_int, c_int, C.POINTER(_NPoseResult)]
    _lib.npose_detect.restype = c_int

class NativePoseEstimator:
    def __init__(self, model_path: str, num_threads: int | None = None):
        if _lib is None:
            raise RuntimeError("native_pose DLL not found or failed to load")
        self._h = c_void_p()
        nt = int(num_threads) if (num_threads and num_threads > 0) else 0
        if _TRACE:
            print(f"[NPOSE:PY] npose_create(model='{model_path}', threads={nt})")
        rc = _lib.npose_create(model_path.encode('utf-8'), nt, C.byref(self._h))
        if _TRACE:
            print(f"[NPOSE:PY] npose_create -> rc={rc} handle={(int(self._h.value) if self._h.value else 0)}")
        if rc != 0 or not self._h:
            raise RuntimeError(f"npose_create failed: {rc}")

    def close(self) -> None:
        if self._h and _lib:
            _lib.npose_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def process(self, frame_rgb: np.ndarray):
        # frame_rgb: HxWx3 RGB または HxWx4 BGRA を想定。内部でBGRA化
        if frame_rgb.ndim != 3:
            raise ValueError("frame must be HxWxC")
        h, w, c = frame_rgb.shape
        if c == 3:
            # 変換: RGB -> BGRA
            bgra = np.concatenate([frame_rgb, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
        elif c == 4:
            bgra = frame_rgb
        else:
            raise ValueError("channels must be 3 or 4")
        res = _NPoseResult()
        if _TRACE:
            print(f"[NPOSE:PY] npose_detect(w={w}, h={h}, stride={w*4})")
        rc = _lib.npose_detect(self._h, bgra.ctypes.data_as(POINTER(c_uint8)), w, h, w*4, C.byref(res))
        if _TRACE:
            print(f"[NPOSE:PY] detect rc={rc} has={res.has_landmarks} count={res.landmark_count}")
        if rc != 0:
            # 失敗時は Solutions と互換の None を返す
            return type('PoseResult', (), {'pose_landmarks': None})()
        if res.has_landmarks and res.landmark_count > 0 and res.landmarks:
            # Solutions 互換の形に組み立て
            pts = []
            for i in range(res.landmark_count):
                p = res.landmarks[i]
                pts.append(type('LM', (), {'x': float(p.x), 'y': float(p.y), 'z': float(p.z), 'visibility': float(p.visibility)})())
            lm_list = type('NLList', (), {'landmark': pts})()
            return type('PoseResult', (), {'pose_landmarks': lm_list})()
        else:
            return type('PoseResult', (), {'pose_landmarks': None})()

__all__ = [
    'NativePoseEstimator'
]
