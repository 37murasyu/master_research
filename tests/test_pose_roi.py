"""Mac 側の姿勢推定の ROI（関心領域の切り出し）の純粋な関数を固定する（B10）。

**なぜこのテストがあるか。**

USB の経路の ROI は ``master_research_code.py`` に直書きされていて（``_roi_from_keypoints``・``_expand_roi``・
``_remap_landmarks_to_fullframe``・横の切り出し）、混成へは ``app/hybrid/pose_roi.py`` に同じ式で移した。
本体は台本（読み込むとカメラを開く）なので import できない。そこで本体の該当する関数と文を ``ast`` で抜き出して
実行し、同じ入力で移植と数値が一致することを確かめる。本体の式が変われば、このテストが落ちて気づける。

ただし USB の横の切り出しは、切り出した画像の画素座標を全体の画像用の射影行列で三角測量している（x_start を
戻していない、KNOWN_ISSUES.md）。移植では座標を全体へ戻す関数を足し、その振る舞いを別に固定する。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.hybrid import pose_roi

MAIN_SCRIPT = Path(__file__).resolve().parents[1] / "master_research_code.py"

# 本体の既定値（master_research_code.py の POSE_ROI_*・POSE_X_CROP_*）
MAIN_CONSTANTS = {
    "POSE_ROI_MARGIN_RATIO": 0.25,
    "POSE_ROI_MIN_SIDE_RATIO": 0.45,
    "POSE_ROI_MIN_VALID_KPTS": 4,
    "POSE_X_CROP_MARGIN": 140,
    "POSE_X_CROP_MIN_WIDTH_RATIO": 0.85,
}


@pytest.fixture(scope="module")
def main_tree():
    return ast.parse(MAIN_SCRIPT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def main_funcs(main_tree):
    """本体の ROI の関数を抜き出して、本体の既定値の定数とともに実行した名前空間。"""
    names = {"_roi_from_keypoints", "_expand_roi", "_remap_landmarks_to_fullframe", "_pose_has_landmarks",
             "get_valid_x_range"}
    defs = [n for n in main_tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in defs} == names, "本体の ROI の関数が見つからない（名前が変わった？）"
    namespace = dict(MAIN_CONSTANTS)
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(MAIN_SCRIPT), "exec"), namespace)
    return namespace


def _main_x_crop(main_tree, main_funcs, kpts0, kpts1, width0, width1):
    """本体の横の切り出しの範囲を決める文（``x0_min, x0_max = get_valid_x_range(...)`` から全幅への戻しまで）を実行する。"""
    body = main_tree.body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and "x0_min" in ast.unparse(n.targets[0]))
    end = next(i for i, n in enumerate(body[start:], start)
               if isinstance(n, ast.If) and ast.unparse(n.test) == "x_end <= x_start")
    namespace = dict(main_funcs)
    namespace.update(frame0_kpts=kpts0, frame1_kpts=kpts1,
                     frame0=np.zeros((4, width0, 3), np.uint8), frame1=np.zeros((4, width1, 3), np.uint8))
    exec(compile(ast.Module(body=body[start:end + 1], type_ignores=[]), str(MAIN_SCRIPT), "exec"), namespace)
    return namespace["x_start"], namespace["x_end"]


def _random_kpts(rng, w, h, n=16, missing=0.2):
    pts = []
    for _ in range(n):
        if rng.random() < missing:
            pts.append([-1, -1])
        else:
            pts.append([int(rng.integers(-40, w + 40)), int(rng.integers(-40, h + 40))])
    return pts


# --- 本体の式との一致 -------------------------------------------------------------


def test_the_roi_matches_the_main_script(main_funcs):
    rng = np.random.default_rng(0)
    for _ in range(500):
        w, h = int(rng.integers(64, 1920)), int(rng.integers(64, 1080))
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        spread = int(rng.integers(1, max(2, min(w, h))))
        kpts = [[int(cx + rng.integers(-spread, spread + 1)), int(cy + rng.integers(-spread, spread + 1))]
                if rng.random() > 0.2 else [-1, -1] for _ in range(16)]
        shape = (h, w, 3)
        assert pose_roi.roi_from_keypoints(kpts, shape) == main_funcs["_roi_from_keypoints"](kpts, shape)


def test_the_expansion_matches_the_main_script(main_funcs):
    rng = np.random.default_rng(1)
    for _ in range(500):
        w, h = int(rng.integers(64, 1920)), int(rng.integers(64, 1080))
        x0, x1 = sorted(int(v) for v in rng.integers(0, w, 2))
        y0, y1 = sorted(int(v) for v in rng.integers(0, h, 2))
        roi, grow = (x0, y0, x1 + 1, y1 + 1), float(rng.choice([0.0, 0.25, 0.5, -0.1]))
        shape = (h, w, 3)
        assert pose_roi.expand_roi(roi, shape, grow) == main_funcs["_expand_roi"](roi, shape, grow)
    assert pose_roi.expand_roi(None, (10, 10)) is None


def test_the_remap_matches_the_main_script(main_funcs):
    rng = np.random.default_rng(2)
    for _ in range(100):
        w, h = int(rng.integers(64, 1920)), int(rng.integers(64, 1080))
        x0, x1 = sorted(int(v) for v in rng.integers(0, w, 2))
        y0, y1 = sorted(int(v) for v in rng.integers(0, h, 2))
        roi = (x0, y0, x1 + 1, y1 + 1)
        points = [tuple(float(v) for v in rng.uniform(-0.2, 1.2, 4)) for _ in range(33)]
        marks = [SimpleNamespace(x=p[0], y=p[1]) for p in points]
        results = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=marks))
        assert main_funcs["_remap_landmarks_to_fullframe"](results, roi, (h, w, 3))
        mine = pose_roi.remap_to_fullframe(points, roi, (h, w, 3))
        assert np.allclose([(p[0], p[1]) for p in mine], [(m.x, m.y) for m in marks], rtol=0, atol=1e-12)
        assert [(p[2], p[3]) for p in mine] == [(p[2], p[3]) for p in points], "z と visibility は本体と同じく触らない"


def test_the_side_crop_range_matches_the_main_script(main_tree, main_funcs):
    rng = np.random.default_rng(3)
    for _ in range(300):
        w0, w1 = int(rng.integers(200, 1920)), int(rng.integers(200, 1920))
        k0 = _random_kpts(rng, w0, 720, missing=float(rng.choice([0.2, 1.0])))
        k1 = _random_kpts(rng, w1, 720, missing=float(rng.choice([0.2, 1.0])))
        assert pose_roi.x_crop_range(k0, k1, w0, w1) == _main_x_crop(main_tree, main_funcs, k0, k1, w0, w1)


# --- 振る舞い -----------------------------------------------------------------------


def test_the_roi_is_a_square_around_the_points_with_a_margin():
    kpts = [[500, 300], [540, 300], [500, 380], [540, 380]]
    # 幅 41・高さ 81 に 25% の余白 → 中心 (520, 340)、半辺は最小の辺 0.45×720/2=162 が勝つ
    assert pose_roi.roi_from_keypoints(kpts, (720, 1280, 3)) == (358, 178, 682, 502)


def test_too_few_points_give_no_roi():
    kpts = [[500, 300], [540, 300], [500, 380], [-1, -1], [-1, -1]]
    assert pose_roi.roi_from_keypoints(kpts, (720, 1280)) is None
    assert pose_roi.roi_from_keypoints([], (720, 1280)) is None


def test_the_roi_is_clipped_at_the_image_edges():
    kpts = [[0, 0], [20, 0], [0, 20], [20, 20]]
    x0, y0, x1, y1 = pose_roi.roi_from_keypoints(kpts, (720, 1280))
    assert (x0, y0) == (0, 0) and x1 <= 1280 and y1 <= 720
    assert pose_roi.expand_roi((1200, 650, 1280, 720), (720, 1280), 0.5) == (1160, 615, 1280, 720)


def test_landmarks_become_pixels_like_the_main_script():
    points = [(0.5, 0.25, 0.0, 0.9), (-0.1, 0.5, 0.0, 0.9), (1.2, 1.0, 0.0, 0.1)]
    assert pose_roi.landmarks_to_pixels(points, (100, 200), [0, 1, 2]) == [[100, 25], [-20, 50], [240, 100]]
    assert pose_roi.landmarks_to_pixels(None, (100, 200), [0, 1]) == [[-1, -1], [-1, -1]]


def test_the_remap_puts_the_crop_back_into_the_full_frame():
    # 左上 (100, 50)・幅 200・高さ 100 の切り出しの中心は、全体 (400×200) の (200, 100) = 正規化 (0.5, 0.5)
    (x, y, z, v), = pose_roi.remap_to_fullframe([(0.5, 0.5, -0.3, 0.8)], (100, 50, 300, 150), (200, 400, 3))
    assert (x, y, z, v) == pytest.approx((0.5, 0.5, -0.3, 0.8))


def test_the_side_crop_keeps_a_minimum_width_and_falls_back_to_the_full_width():
    k = [[600, 100], [640, 200]]
    # 余白 140 で [460, 780]、最低幅 0.85×1280=1088 に広げて中心 620 の ±544 → [76, 1164]
    assert pose_roi.x_crop_range(k, k, 1280, 1280) == (76, 1164)
    none = [[-1, -1]] * 4
    assert pose_roi.x_crop_range(none, none, 1280, 1000) == (0, 1000)


def test_the_side_crop_coordinates_go_back_to_the_full_frame():
    """USB は x_start を戻さずに三角測量している。移植では戻す（欠けた点は -1 のまま）。"""
    assert pose_roi.uncrop_x_pixels([[10, 20], [-1, -1], [0, 5]], 76) == [[86, 20], [-1, -1], [76, 5]]
    # 正規化座標は横だけの ROI として戻す
    (x, y, _, _), = pose_roi.remap_to_fullframe([(0.5, 0.25, 0.0, 1.0)], (76, 0, 1164, 720), (720, 1280))
    assert (x * 1280, y * 720) == pytest.approx((620, 180))
