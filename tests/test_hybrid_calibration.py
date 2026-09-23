import cv2 as cv
import numpy as np
import pytest
from app.hybrid.checkerboard import (
    Board,
    calibrate_intrinsics,
    calibrate_stereo,
    square_lengths,
    detect_board,
)
from app.hybrid.calibration_io import (
    save_calibration,
    load_calibration,
    cache_key,
    save_intrinsics,
    load_intrinsics,
)
from utils import get_projection_matrix


def synthetic_views(board, size0=(1280, 720), size1=(1920, 1080)):
    k0 = np.array([[950.0, 0, size0[0] / 2], [0, 960.0, size0[1] / 2], [0, 0, 1]])
    k1 = np.array([[1400.0, 0, size1[0] / 2], [0, 1420.0, size1[1] / 2], [0, 0, 1]])
    r = cv.Rodrigues(np.array([0.02, 0.10, -0.01]))[0]
    t = np.array([[-24.0], [0.4], [0.3]])
    a, b = [], []
    rng = np.random.default_rng(2026)
    for _ in range(20):
        rv = rng.uniform(-0.45, 0.45, 3)
        tv = np.array([rng.uniform(-13, 2), rng.uniform(-8, 2), rng.uniform(65, 110)])
        a.append(cv.projectPoints(board.object_points, rv, tv, k0, np.zeros(5))[0])
        b.append(
            cv.projectPoints(
                board.object_points,
                cv.Rodrigues(r @ cv.Rodrigues(rv)[0])[0],
                r @ tv.reshape(3, 1) + t,
                k1,
                np.zeros(5),
            )[0]
        )
    return a, b, k0, k1, r, t


def test_board_parity():
    with pytest.raises(ValueError):
        Board(rows=5, cols=7)
    with pytest.raises(ValueError):
        Board(rows=4, cols=6)
    with pytest.raises(ValueError):
        Board(square_cm=0)


def test_synthetic_calibration_and_io(tmp_path):
    board = Board()
    a, b, k0, k1, r, t = synthetic_views(board)
    i0 = calibrate_intrinsics(board, a, (1280, 720))
    i1 = calibrate_intrinsics(board, b, (1920, 1080))
    assert np.allclose(np.diag(i0.K)[:2], np.diag(k0)[:2], rtol=0.02)
    stereo = calibrate_stereo(board, a, b, i0, i1)
    assert np.linalg.norm(stereo.T) / np.linalg.norm(t) == pytest.approx(1, rel=0.02)
    assert np.linalg.norm(cv.Rodrigues(stereo.R @ r.T)[0]) * 180 / np.pi < 1
    lengths = square_lengths(board, a[0], b[0], i0, i1, stereo)
    assert np.max(np.abs(lengths - 3)) < 0.05
    directory = save_calibration(
        i0,
        i1,
        stereo,
        board,
        cameras=[
            {"kind": "mac", "device_id": "mac-0"},
            {"kind": "pixel", "device_id": "pixel-1"},
        ],
        root=tmp_path,
    )
    loaded = load_calibration(directory)
    assert loaded.meta["units"] == "cm"
    assert np.allclose(
        get_projection_matrix(0, False, base_dir=directory),
        i0.K @ np.c_[np.eye(3), np.zeros(3)],
    )
    assert np.allclose(
        get_projection_matrix(1, False, base_dir=directory),
        i1.K @ np.c_[stereo.R, stereo.T],
    )
    assert load_calibration("latest", root=tmp_path).directory == directory
    key = cache_key("pixel", "pixel-1", (1920, 1080))
    save_intrinsics(key, i1, root=tmp_path)
    assert np.allclose(load_intrinsics(key, root=tmp_path).K, i1.K)
    assert (
        load_intrinsics(cache_key("pixel", "pixel-2", (1920, 1080)), root=tmp_path)
        is None
    )


def test_outlier_refit():
    board = Board()
    a, _, _, _, _, _ = synthetic_views(board)
    rng = np.random.default_rng(1)
    a[-1] = a[-1] + rng.normal(0, 10, a[-1].shape).astype(np.float32)
    fit = calibrate_intrinsics(board, a, (1280, 720))
    assert 19 not in fit.used_indices
    assert fit.rms < 0.1


def render_board(board, cell=60):
    image = np.full(((board.rows + 3) * cell, (board.cols + 3) * cell), 180, np.uint8)
    for y in range(board.rows + 1):
        for x in range(board.cols + 1):
            image[(y + 1) * cell : (y + 2) * cell, (x + 1) * cell : (x + 2) * cell] = (
                255 if (x + y) % 2 else 0
            )
    return image


def test_detect_rotated_board_order():
    board = Board()
    image = render_board(board)
    a = detect_board(image, board)
    b = detect_board(cv.rotate(image, cv.ROTATE_180), board)
    assert a is not None and b is not None
    mapped = np.array([image.shape[1] - 1, image.shape[0] - 1]) - b.reshape(-1, 2)
    assert np.max(np.abs(a.reshape(-1, 2) - mapped)) < 0.2


def test_equal_resolution_and_stereo_outlier():
    board = Board()
    a, b, k0, k1, r, t = synthetic_views(board, size1=(1280, 720))
    i0 = calibrate_intrinsics(board, a, (1280, 720))
    i1 = calibrate_intrinsics(board, b, (1280, 720))
    b[-1] = b[-1] + np.array([[[35.0, -20.0]]], np.float32)
    fit = calibrate_stereo(board, a, b, i0, i1)
    assert 19 not in fit.used_indices
    assert fit.rms < 0.1
    assert np.linalg.norm(fit.T) / np.linalg.norm(t) == pytest.approx(1, rel=0.02)
