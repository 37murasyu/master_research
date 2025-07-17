import ctypes
import os
import sys
from ctypes import wintypes

# Load DLL (adjust path/name after you build overlay.dll)
def _find_overlay_dll():
    # 1) Explicit env var
    env = os.environ.get('NATIVE_OVERLAY_DLL')
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(__file__)
    # 2) Typical locations: native_overlay/build, native_overlay
    candidates = [
        os.path.join(here, 'native_overlay', 'build', 'overlay.dll'),
        os.path.join(here, 'native_overlay', 'x64', 'Release', 'overlay.dll'),
        os.path.join(here, 'native_overlay', 'x64', 'Debug', 'overlay.dll'),
        os.path.join(here, 'native_overlay', 'overlay.dll'),
        os.path.join(here, 'overlay.dll'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # 3) Fallback: original path
    return os.path.join(here, 'native_overlay', 'overlay.dll')

_DLL_NAME = _find_overlay_dll()

class TextItem(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("fontSize", ctypes.c_int32),
        ("colorARGB", ctypes.c_uint32),
        ("text", ctypes.c_wchar_p),
    ]

try:
    _dll = ctypes.WinDLL(_DLL_NAME)
    _dll.DrawTextOverlay.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(TextItem), ctypes.c_int32
    ]
    _dll.DrawTextOverlay.restype = ctypes.c_int
except OSError:
    _dll = None


def draw_texts_bgra(image_bgra, items):
    """
    image_bgra: numpy.ndarray (H, W, 4) dtype=uint8
    items: list[ {x:int, y:int, font:int, color:(r,g,b,a), text:str} ]
    returns: 0 on success; negative on failure; 1 if DLL unavailable
    """
    import numpy as np

    if _dll is None:
        return 1
    if image_bgra.dtype != np.uint8 or image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        raise ValueError("image_bgra must be HxWx4 uint8")
    h, w, _ = image_bgra.shape
    stride = image_bgra.strides[0]

    texts = []
    for it in items:
        r, g, b, a = it.get('color', (255,255,255,255))
        argb = ((a & 0xFF) << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)
        texts.append(TextItem(
            int(it['x']), int(it['y']), int(it.get('font', 24)), ctypes.c_uint32(argb), it['text']
        ))
    arr_type = TextItem * len(texts)
    c_items = arr_type(*texts)

    buf_ptr = image_bgra.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    return _dll.DrawTextOverlay(buf_ptr, stride, w, h, c_items, len(texts))
