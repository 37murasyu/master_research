"""盤を立てて静止させた短辺の向き（重力の候補）を求める。カメラにも UI にも依らない純粋な関数。

混成の校正の最後に、盤を鉛直に立てて（短辺を上下に）Mac のカメラの前で静止させ、その短辺の向きを
校正フォルダの ``meta.json`` の ``checkerboard_short_axis`` に残す（USB の ``calib.py`` の
``_save_checkerboard_short_axis`` と同じ形に、傾き・ばらつき・標本数などを足したもの）。計測側は
``app.hybrid.gravity.read_board_up`` で読み、最寄りの軸に吸着させ、符号は体幹で確かめる。

- 混成の盤（``app.hybrid.checkerboard.Board.object_points``）は x が cols（長辺）、y が rows（短辺）で、
  USB の ``calib.py`` と逆。短辺は rows<cols なら y
- ``findChessboardCorners`` は角点の並びを 180° 反転して返しうるので、標本ごとに上向き（画像の上＝カメラの
  −y 側）へ符号を揃えてから中央値を取る
- 実行時の座標は cam0 の (−x, −z, −y)（``app.runners.network_measure`` と同じ）。上向きなら概ね Z+
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2 as cv
import numpy as np

from app.hybrid.gravity import _unit, cam0_to_runtime as to_runtime

__all__ = [
    "CAMERA_UP", "NEEDED_SAMPLES", "STILL_PX", "MAX_TILT_DEG", "WARN_TILT_DEG", "METHOD",
    "short_axis_board", "board_axes_cam0", "short_axis_up_cam0", "to_runtime", "angle_deg", "up_label",
    "short_axis_entry", "UprightCollector",
]

# カメラの座標で画像の上向き（OpenCV のカメラは y が下向き）
CAMERA_UP = np.array([0.0, -1.0, 0.0])
# 採用する標本の数・静止の判定（盤集めの ``collector.STILL_PX`` と同じ 2 px）・捨てる傾き・警告する傾き
NEEDED_SAMPLES = 10
STILL_PX = 2.0
MAX_TILT_DEG = 30.0
WARN_TILT_DEG = 10.0
METHOD = "upright_board_solvepnp_median"
_LABELS = ("X", "Y", "Z")


def short_axis_board(board) -> str:
    """盤の座標で短辺が沿う軸。混成の盤は x が cols、y が rows に沿うので、rows<cols なら "y"。"""
    return "y" if board.rows < board.cols else "x"


def board_axes_cam0(corners, board, K, distortion) -> np.ndarray | None:
    """盤の座標軸を cam0 のカメラ座標で表した回転行列（列が盤の x・y・法線）。解けなければ None。"""
    if corners is None:
        return None
    ok, rvec, _ = cv.solvePnP(board.object_points, np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
                              np.asarray(K, dtype=np.float64), np.asarray(distortion, dtype=np.float64))
    if not ok:
        return None
    return cv.Rodrigues(rvec)[0]


def short_axis_up_cam0(axes, board) -> np.ndarray:
    """回転行列から短辺の向きを取り出し、上向き（カメラの −y 側）へ符号を揃えた単位ベクトル。"""
    column = 1 if short_axis_board(board) == "y" else 0
    v = _unit(np.asarray(axes, dtype=np.float64)[:, column])
    return v if float(v @ CAMERA_UP) >= 0.0 else -v


def angle_deg(a, b) -> float:
    """2 つのベクトルのなす角 [°]。"""
    return float(np.degrees(np.arccos(np.clip(float(_unit(a) @ _unit(b)), -1.0, 1.0))))


def up_label(vector_runtime) -> str:
    """実行時の座標で最寄りの軸の名前（"Z+" など）。"""
    v = _unit(vector_runtime)
    i = int(np.argmax(np.abs(v)))
    return _LABELS[i] + ("+" if v[i] >= 0 else "-")


def short_axis_entry(vector_cam0, board, *, tilt_deg: float, spread_deg: float, samples: int,
                     method: str = METHOD) -> dict:
    """校正の ``meta.json`` の ``checkerboard_short_axis`` に入れる辞書（USB の形＋傾き・ばらつきなど）。"""
    v = _unit(vector_cam0)
    runtime = to_runtime(v)
    return {
        "rows": int(board.rows),
        "columns": int(board.cols),
        "short_side_axis_board": short_axis_board(board),
        "vector_cam0": [float(x) for x in v],
        "vector_runtime": [float(x) for x in runtime],
        "up_label_runtime": up_label(runtime),
        "tilt_deg": float(tilt_deg),
        "spread_deg": float(spread_deg),
        "samples": int(samples),
        "method": method,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class UprightCollector:
    """盤を立てて静止させた Mac の画像から、短辺の上向きの標本を集める。

    ``add(corners)`` は 1 枚ごとの判定を返す: ``"no_board"``（盤が無い・解けない）、``"waiting"``（直前の画像が
    無く静止を判定できない）、``"moving"``（直前から角点が ``still_px`` を超えて動いた）、``"tilted"``
    （カメラの上向きから ``max_tilt_deg`` を超えて傾いた）、``"accepted"``（標本に採った）。
    """

    def __init__(self, board, K, distortion, *, needed: int = NEEDED_SAMPLES, still_px: float = STILL_PX,
                 max_tilt_deg: float = MAX_TILT_DEG):
        self.board = board
        self.K = np.asarray(K, dtype=np.float64)
        self.distortion = np.asarray(distortion, dtype=np.float64)
        self.needed = int(needed)
        self.still_px = float(still_px)
        self.max_tilt_deg = float(max_tilt_deg)
        self.samples: list[np.ndarray] = []
        self.last_motion_px: float | None = None
        self.last_tilt_deg: float | None = None
        self._previous: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return len(self.samples) >= self.needed

    def add(self, corners) -> str:
        if corners is None:
            self._previous = None
            return "no_board"
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        previous, self._previous = self._previous, points
        if previous is None or previous.shape != points.shape:
            return "waiting"
        self.last_motion_px = float(np.max(np.linalg.norm(points - previous, axis=1)))
        if self.last_motion_px > self.still_px:
            return "moving"
        axes = board_axes_cam0(points, self.board, self.K, self.distortion)
        if axes is None:
            return "no_board"
        up = short_axis_up_cam0(axes, self.board)
        self.last_tilt_deg = angle_deg(up, CAMERA_UP)
        if self.last_tilt_deg > self.max_tilt_deg:
            return "tilted"
        if not self.done:
            self.samples.append(up)
        return "accepted"

    def median_up_cam0(self) -> np.ndarray | None:
        """標本（どれも上向きに揃えてある）の成分ごとの中央値を単位ベクトルにしたもの。標本が無ければ None。"""
        if not self.samples:
            return None
        return _unit(np.median(np.stack(self.samples), axis=0))

    def result(self) -> dict | None:
        """``checkerboard_short_axis`` の辞書。標本が足りなければ None。"""
        if not self.done:
            return None
        up = self.median_up_cam0()
        spread = max(angle_deg(sample, up) for sample in self.samples)
        return short_axis_entry(up, self.board, tilt_deg=angle_deg(up, CAMERA_UP), spread_deg=spread,
                                samples=len(self.samples))
