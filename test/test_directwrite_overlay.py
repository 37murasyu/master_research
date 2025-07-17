import os
import sys
import numpy as np
# pylint: disable=no-member
import cv2 as cv

# Ensure repository root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from py_native_overlay import draw_texts_bgra


def _to_bgra(img_bgr):
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2BGRA)


def main():
    img = np.zeros((240, 480, 3), dtype=np.uint8)
    img[:] = (32, 32, 32)
    img_bgra = _to_bgra(img)

    items = [
        {"x": 20, "y": 30, "font": 24, "color": (255, 255, 0, 255), "text": "DirectWrite テスト"},
        {"x": 20, "y": 80, "font": 32, "color": (0, 200, 255, 255), "text": "日本語: 明朝/メイリオ"},
    ]

    rc = draw_texts_bgra(img_bgra, items)
    print("Draw result:", rc)
    out = os.path.join(ROOT, "output_data", "directwrite_overlay_test.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.imwrite(out, cv.cvtColor(img_bgra, cv.COLOR_BGRA2BGR))
    print("Saved:", out)


if __name__ == "__main__":
    main()
