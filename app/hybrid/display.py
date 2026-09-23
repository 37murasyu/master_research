"""Side-by-side display functions; no windows or network side effects."""

from functools import lru_cache
import cv2 as cv
import numpy as np
from config import pose_keypoints
from utils import draw_text_jp

EDGES = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (15, 17),
    (15, 19),
    (16, 18),
    (16, 20),
)


@lru_cache(maxsize=8)
def qr_image(url):
    import segno

    matrix = np.array(
        list(segno.make(url, micro=False, error="m").matrix_iter(scale=1, border=4)),
        dtype=np.uint8,
    )
    scale = max(1, 280 // matrix.shape[0])
    return cv.cvtColor(
        np.repeat(np.repeat((1 - matrix) * 255, scale, 0), scale, 1), cv.COLOR_GRAY2BGR
    )


def panel(image, landmarks=None, corners=None):
    out = np.full((360, 640, 3), 28, np.uint8)
    if image is None:
        return out
    h, w = image.shape[:2]
    scale = min(640 / w, 360 / h)
    rw, rh = round(w * scale), round(h * scale)
    x, y = (640 - rw) // 2, (360 - rh) // 2
    out[y : y + rh, x : x + rw] = cv.resize(image, (rw, rh))
    if landmarks is not None:
        pts = {}
        for i, p in enumerate(landmarks):
            if (
                np.isfinite(p).all()
                and p[3] >= 0.5
                and 0 <= p[0] <= 1
                and 0 <= p[1] <= 1
            ):
                pts[i] = (x + round(p[0] * rw), y + round(p[1] * rh))
        for a, b in EDGES:
            if a in pts and b in pts:
                cv.line(out, pts[a], pts[b], (90, 220, 90), 2, cv.LINE_AA)
        for i, pt in pts.items():
            cv.circle(out, pt, 4 if i in pose_keypoints else 2, (0, 180, 255), -1)
    if corners is not None:
        for px, py in np.asarray(corners).reshape(-1, 2):
            cv.circle(
                out,
                (x + round(px * scale), y + round(py * scale)),
                3,
                (255, 80, 255),
                -1,
            )
    return out


def compose(
    left,
    right,
    *,
    left_landmarks=None,
    right_landmarks=None,
    url="",
    lines=(),
    left_corners=None,
    right_corners=None,
):
    out = np.full((440, 1280, 3), 28, np.uint8)
    out[:360, :640] = panel(left, left_landmarks, left_corners)
    out[:360, 640:] = panel(right, right_landmarks, right_corners)
    if right is None and url:
        qr = qr_image(url)
        h, w = qr.shape[:2]
        out[30 : 30 + h, 960 - w // 2 : 960 - w // 2 + w] = qr
        # ASCII URL fits by choosing scale based on the rendered width.
        scale = min(
            0.42, 600 / max(1, cv.getTextSize(url, cv.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0])
        )
        cv.putText(
            out,
            url,
            (660, 338),
            cv.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
    for i, text in enumerate(lines[:3]):
        out = draw_text_jp(
            out, str(text), (10, 365 + i * 24), 18, (240, 240, 240), line_width=140
        )
    return out
