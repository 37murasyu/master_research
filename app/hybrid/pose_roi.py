"""Mac 側の姿勢推定の ROI（関心領域の切り出し）の純粋な関数（B10）。

USB の経路（``master_research_code.py`` の ``_roi_from_keypoints``・``_expand_roi``・
``_remap_landmarks_to_fullframe``・横の切り出し）と同じ式。Qt・MediaPipe に依存しない。

- 前のフレームの点（画素）から、余白を付けた正方形の ROI を決める（``roi_from_keypoints``）
- 点を見失ったフレームの数だけ ROI を広げる（``expand_roi``）
- 切り出した画像に対する正規化座標を、全体の画像に対する正規化座標へ戻す（``remap_to_fullframe``）
- フレームをまたいだ ROI の追跡（``RoiTracker``。``PoseDetector`` が 1 つ持つ）
- 横の切り出しの範囲（``x_crop_range``）と、切り出した画素座標を全体へ戻す（``uncrop_x_pixels``）

横の切り出しの 2 つは今は試験しか使わないが、混成で使うときに USB と同じ式になるよう一緒に移して残している。
USB の横の切り出しは、切り出した画像の画素座標を全体の画像用の射影行列で三角測量している（x_start を
戻していない、KNOWN_ISSUES.md）。移植では ``uncrop_x_pixels``・``remap_to_fullframe`` で必ず全体へ戻す。
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "MARGIN_RATIO",
    "MIN_SIDE_RATIO",
    "MIN_VALID_KPTS",
    "MAX_MISS",
    "MISS_GROW_RATIO",
    "X_CROP_MARGIN",
    "X_CROP_MIN_WIDTH_RATIO",
    "Roi",
    "roi_from_keypoints",
    "expand_roi",
    "remap_to_fullframe",
    "landmarks_to_pixels",
    "RoiTracker",
    "x_crop_range",
    "uncrop_x_pixels",
]

# 本体の既定値（POSE_ROI_*・POSE_X_CROP_*）
MARGIN_RATIO = 0.25
MIN_SIDE_RATIO = 0.45
MIN_VALID_KPTS = 4
MAX_MISS = 4
MISS_GROW_RATIO = 0.25
X_CROP_MARGIN = 140
X_CROP_MIN_WIDTH_RATIO = 0.85

Roi = tuple[int, int, int, int]  # (x0, y0, x1, y1)。画素、x1・y1 は含まない


def roi_from_keypoints(kpts: Iterable[Sequence[float]], frame_shape, *, margin_ratio: float = MARGIN_RATIO,
                       min_side_ratio: float = MIN_SIDE_RATIO, min_valid: int = MIN_VALID_KPTS) -> Roi | None:
    """画素の点（欠けは負）を囲む正方形の ROI。点が ``min_valid`` 未満、または画像の外なら None（全画面）。

    点の範囲に ``margin_ratio`` の余白を付け、短い辺の ``min_side_ratio`` 倍を最小の辺にした正方形を、
    画像の端で切り詰める。本体の ``_roi_from_keypoints`` と同じ式。
    """
    h, w = frame_shape[:2]
    valid = [(int(x), int(y)) for x, y in kpts if x >= 0 and y >= 0]
    if len(valid) < max(1, min_valid):
        return None

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    margin_x = int(round((x1 - x0 + 1) * margin_ratio))
    margin_y = int(round((y1 - y0 + 1) * margin_ratio))
    x0 -= margin_x
    x1 += margin_x
    y0 -= margin_y
    y1 += margin_y

    min_side = int(round(max(1, min(w, h)) * min_side_ratio))
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


def expand_roi(roi: Roi | None, frame_shape, grow_ratio: float = MISS_GROW_RATIO) -> Roi | None:
    """ROI を幅・高さの ``grow_ratio`` 倍ずつ四方へ広げ、画像の端で切り詰める。本体の ``_expand_roi`` と同じ式。"""
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


def remap_to_fullframe(points: Iterable[Sequence[float]], roi: Roi, full_shape) -> list[tuple]:
    """切り出した画像に対する正規化座標 ``(x, y, z, visibility)`` を、全体の画像に対する正規化座標へ戻す。

    本体の ``_remap_landmarks_to_fullframe`` と同じ式。z と visibility は本体と同じく触らない。
    横の切り出しの範囲 ``(x_start, x_end)`` は ``roi=(x_start, 0, x_end, h)`` として戻せる。
    """
    x0, y0, x1, y1 = roi
    h, w = full_shape[:2]
    rw = max(1, int(x1 - x0))
    rh = max(1, int(y1 - y0))
    return [((x0 + float(p[0]) * rw) / float(w), (y0 + float(p[1]) * rh) / float(h), *p[2:]) for p in points]


def landmarks_to_pixels(points: Sequence[Sequence[float]] | None, frame_shape, ids: Sequence[int]) -> list[list[int]]:
    """正規化座標の点のうち ``ids`` を、``ids`` の順に画素へ直す。点が無ければ全部 ``[-1, -1]``。

    本体の ``_extract_keypoints_fast_single`` と同じく visibility は見ない（画像の外は負の座標になり ROI で除かれる）。
    本体は ID の昇順に並べる。同じ並びにするなら昇順で渡す（``RoiTracker`` は作るときに 1 度だけ並べる）。
    """
    if not points:
        return [[-1, -1] for _ in ids]
    h, w = frame_shape[:2]
    return [[int(round(float(points[i][0]) * w)), int(round(float(points[i][1]) * h))] for i in ids]


class RoiTracker:
    """フレームをまたいだ ROI の追跡。本体の ``_pose_roi0``・``_pose_roi0_miss`` とその更新と同じ手順。

    前のフレームの点から次の ROI を決める（``observe``）。点を見失ったら、見失ったフレームの数だけ ROI を広げて
    切り出し（``crop_for``）、``MAX_MISS`` 回を超えたら全画面に戻す。点が少なすぎて ROI を決められないときも
    見失ったと数える。MediaPipe に依存しない。
    """

    def __init__(self, ids: Iterable[int]):
        self._ids = sorted(ids)  # 使う点のランドマーク ID。本体と同じく昇順（ROI そのものは並びに依らない）
        self._roi: Roi | None = None  # 最後に点から決めた ROI。最初は前の点が無いので全画面
        self._miss = 0  # その ROI で点を見失い続けているフレームの数

    def crop_for(self, frame_shape) -> Roi | None:
        """このフレームで切り出す ROI。見失った回数だけ広げる。None なら全画面。"""
        roi = self._roi
        for _ in range(self._miss):
            roi = expand_roi(roi, frame_shape, MISS_GROW_RATIO)
        return roi

    def observe(self, points: Sequence[Sequence[float]] | None, frame_shape) -> None:
        """このフレームの点（全体の画像に対する正規化座標。人がいなければ None）から次の ROI を決める。"""
        roi = roi_from_keypoints(landmarks_to_pixels(points, frame_shape, self._ids), frame_shape)
        if roi is not None:
            self._roi, self._miss = roi, 0
        elif self._roi is not None:
            self._miss += 1
            if self._miss > MAX_MISS:
                self._roi, self._miss = None, 0


def _valid_x_range(kpts, frame_width: int) -> tuple[int, int]:
    valid = [x for x, y in kpts if x >= 0]
    if not valid:
        return 0, frame_width  # 検出なしなら該当フレームのフル幅
    return max(0, int(min(valid))), min(frame_width, int(max(valid)))


def x_crop_range(kpts0, kpts1, width0: int, width1: int, *, margin: int = X_CROP_MARGIN,
                 min_width_ratio: float = X_CROP_MIN_WIDTH_RATIO) -> tuple[int, int]:
    """2 台で共通の横の切り出しの範囲 ``(x_start, x_end)``。本体の起動時の範囲の決め方と同じ式。

    切り出した画像で得た座標は、``uncrop_x_pixels``（画素）か ``remap_to_fullframe``（正規化）で全体へ戻すこと。
    """
    x0_min, x0_max = _valid_x_range(kpts0, width0)
    x1_min, x1_max = _valid_x_range(kpts1, width1)
    x_min = min(x0_min, x1_min)
    x_max = max(x0_max, x1_max)
    x_margin = max(0, int(margin))

    width_min = min(width0, width1)
    x_start = max(0, x_min - x_margin)
    x_end = min(width_min, x_max + x_margin)

    # 推定範囲が狭すぎると検出ロストを誘発するため、最低幅を確保
    min_crop_w = int(round(width_min * max(0.3, min(1.0, min_width_ratio))))
    if (x_end - x_start) < min_crop_w:
        cx = (x_start + x_end) // 2
        half = min_crop_w // 2
        x_start = max(0, cx - half)
        x_end = min(width_min, cx + half)

    if x_end <= x_start:
        x_start, x_end = 0, width_min
    return x_start, x_end


def uncrop_x_pixels(kpts, x_start: int) -> list[list[int]]:
    """横に切り出した画像の画素座標を全体の画像の画素座標へ戻す（x に ``x_start`` を足す）。欠けた点は触らない。"""
    return [[x + x_start, y] if (x >= 0 and y >= 0) else [x, y] for x, y in kpts]
