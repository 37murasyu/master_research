"""盤を立てて静止させた向き（重力の候補）を求める純粋な関数を固定する（B2）。

**なぜこのテストがあるか。**

USB の経路は校正の全ビューの盤の短辺の中央値を重力の候補にしている（``calib.py`` の
``_save_checkerboard_short_axis``）。混成の校正は盤を自由に傾けて集めるので同じ方法は使えず、校正の最後に
「盤を立てて静止」させた Mac の画像から短辺の向きを別に求める。ここを間違えると、計測の重力が 90° ずれた軸や
逆向きに吸着し、トルクの重力の項がすべて狂う。特に次の 3 点を固定する。

- 混成の盤（``app.hybrid.checkerboard.Board``）は x が cols（長辺）、y が rows（短辺）で、USB の ``calib.py`` と逆。
  短辺は rows<cols なら y
- ``findChessboardCorners`` は角点の並びを 180° 反転して返しうるので、標本ごとに上向き（カメラの −y 側）へ
  符号を揃えてから中央値を取る
- 静止していない（2 px 超）・大きく傾いた（30° 超）標本は捨てる
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
import pytest

from app.hybrid import gravity
from app.hybrid import gravity_board as gb
from app.hybrid.checkerboard import Board

K = np.array([[950.0, 0, 640.0], [0, 960.0, 360.0], [0, 0, 1]])
DIST = np.zeros(5)


def _rz(deg):
    return cv.Rodrigues(np.array([0.0, 0.0, np.radians(deg)]))[0]


def _rx(deg):
    return cv.Rodrigues(np.array([np.radians(deg), 0.0, 0.0]))[0]


def _corners(board, R, t=(-9.0, -4.5, 80.0)):
    """盤の中心付近がカメラの正面に来る位置へ置いて、Mac の画像に投影した角点。"""
    rvec = cv.Rodrigues(np.asarray(R, dtype=float))[0]
    return cv.projectPoints(board.object_points, rvec, np.asarray(t, dtype=float), K, DIST)[0].astype(np.float32)


def test_the_short_side_of_the_hybrid_board_is_y():
    assert gb.short_axis_board(Board()) == "y"
    assert gb.short_axis_board(Board(rows=7, cols=4)) == "x"


def test_board_axes_recover_the_pose():
    board = Board()
    R = _rz(12) @ _rx(-8)
    axes = gb.board_axes_cam0(_corners(board, R), board, K, DIST)
    assert np.allclose(axes, R, atol=1e-3)


def test_an_upright_board_points_up_in_both_corner_orders():
    """角点の並びが 180° 反転しても（検出器はどちらも返しうる）、上向きの向きは同じ。"""
    board = Board()
    corners = _corners(board, np.eye(3))
    for order in (corners, corners[::-1].copy()):
        up = gb.short_axis_up_cam0(gb.board_axes_cam0(order, board, K, DIST), board)
        assert np.allclose(up, [0.0, -1.0, 0.0], atol=1e-3)
        assert np.allclose(gb.to_runtime(up), [0.0, 0.0, 1.0], atol=1e-3)


def test_the_tilt_is_measured_from_the_camera_up():
    board = Board()
    for R, expected in ((_rz(20), 20.0), (_rx(15), 15.0)):
        up = gb.short_axis_up_cam0(gb.board_axes_cam0(_corners(board, R), board, K, DIST), board)
        assert gb.angle_deg(up, gb.CAMERA_UP) == pytest.approx(expected, abs=0.1)


def test_the_collector_needs_still_samples():
    board = Board()
    collector = gb.UprightCollector(board, K, DIST)
    corners = _corners(board, _rz(3))
    statuses = []
    while not collector.done and len(statuses) < 20:
        statuses.append(collector.add(corners))
    assert collector.done
    assert statuses[0] == "waiting", "直前の画像が無いと静止を判定できない"
    assert statuses.count("accepted") == gb.NEEDED_SAMPLES == 10


def test_a_moving_board_is_not_sampled():
    board = Board()
    collector = gb.UprightCollector(board, K, DIST)
    corners = _corners(board, np.eye(3))
    statuses = [collector.add(corners + np.float32(5.0 * k)) for k in range(15)]
    assert "accepted" not in statuses
    assert statuses.count("moving") == 14
    assert collector.last_motion_px == pytest.approx(5.0 * np.sqrt(2), rel=1e-3)


def test_a_steeply_tilted_board_is_not_sampled():
    board = Board()
    collector = gb.UprightCollector(board, K, DIST)
    corners = _corners(board, _rz(40))
    statuses = [collector.add(corners) for _ in range(15)]
    assert "accepted" not in statuses
    assert statuses.count("tilted") == 14
    assert collector.last_tilt_deg == pytest.approx(40.0, abs=0.2)


def test_a_lost_board_resets_the_stillness():
    board = Board()
    collector = gb.UprightCollector(board, K, DIST)
    corners = _corners(board, np.eye(3))
    assert [collector.add(corners), collector.add(corners), collector.add(None), collector.add(corners)] == [
        "waiting", "accepted", "no_board", "waiting"]


def test_the_result_has_the_contract_form():
    """計測側（``gravity.read_board_up``）が読む鍵と、盤の傾き・ばらつき・標本数を残す。"""
    board = Board()
    collector = gb.UprightCollector(board, K, DIST)
    assert collector.result() is None
    corners = _corners(board, _rz(6))
    flipped = corners[::-1].copy()
    k = 0
    while not collector.done:
        collector.add(corners if (k // 2) % 2 == 0 else flipped)
        k += 1
    entry = collector.result()
    assert set(entry) >= {"rows", "columns", "short_side_axis_board", "vector_cam0", "vector_runtime",
                          "up_label_runtime", "tilt_deg", "spread_deg", "samples", "method", "created_at"}
    assert (entry["rows"], entry["columns"], entry["short_side_axis_board"]) == (4, 7, "y")
    assert entry["samples"] == 10
    assert entry["up_label_runtime"] == "Z+"
    assert entry["tilt_deg"] == pytest.approx(6.0, abs=0.2)
    assert entry["spread_deg"] < 0.5, "並びが反転した標本も上向きへ揃えてから中央値を取る"
    assert np.linalg.norm(entry["vector_cam0"]) == pytest.approx(1.0)
    assert np.allclose(gb.to_runtime(entry["vector_cam0"]), entry["vector_runtime"])


def test_read_board_up_reads_what_the_collector_wrote():
    board = Board()
    collector = gb.UprightCollector(board, K, DIST)
    corners = _corners(board, _rz(4))
    while not collector.done:
        collector.add(corners)
    meta = {"checkerboard_short_axis": collector.result()}
    up = gravity.read_board_up(meta)
    assert np.allclose(up, meta["checkerboard_short_axis"]["vector_runtime"])
    assert gravity.read_board_up({}) is None
    assert gravity.read_board_up(None) is None
