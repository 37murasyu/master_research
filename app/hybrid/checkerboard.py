"""Checkerboard calibration in centimetres, independent of cameras and UI."""

from dataclasses import dataclass, field
import cv2 as cv
import numpy as np

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)


@dataclass(frozen=True)
class Board:
    rows: int = 4
    cols: int = 7
    square_cm: float = 3.0

    def __post_init__(self):
        if self.rows < 2 or self.cols < 2 or self.rows % 2 == self.cols % 2:
            raise ValueError("盤の内側交点数は片方偶数・片方奇数にしてください")
        if not np.isfinite(self.square_cm) or self.square_cm <= 0:
            raise ValueError("マス寸法は正の cm 値を指定してください")

    @property
    def pattern(self):
        return self.cols, self.rows

    @property
    def object_points(self):
        points = np.zeros((self.rows * self.cols, 3), np.float32)
        points[:, :2] = (
            np.mgrid[: self.cols, : self.rows].T.reshape(-1, 2) * self.square_cm
        )
        return points


@dataclass
class Intrinsics:
    K: np.ndarray
    distortion: np.ndarray
    size: tuple[int, int]
    rms: float
    errors: list[float] = field(default_factory=list)
    used_indices: list[int] = field(default_factory=list)


@dataclass
class Stereo:
    R: np.ndarray
    T: np.ndarray
    rms: float
    errors: list[float]
    used_indices: list[int]
    square_error_mm: float = 0.0


def detect_board(image, board):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    factor = min(1.0, 640 / gray.shape[1])
    small = cv.resize(gray, None, fx=factor, fy=factor) if factor < 1 else gray
    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    found, _ = cv.findChessboardCorners(
        small, board.pattern, flags | cv.CALIB_CB_FAST_CHECK
    )
    if not found:
        return None
    found, corners = cv.findChessboardCorners(gray, board.pattern, flags)
    if not found:
        return None
    grid = corners.reshape(board.rows, board.cols, 2)
    spacing = np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2))
    window = int(np.clip(round(spacing * 0.35), 3, 11))
    return cv.cornerSubPix(gray, corners, (window, window), (-1, -1), CRITERIA)


def _views(board, views):
    result = [np.asarray(v, np.float32).reshape(-1, 1, 2) for v in views]
    if len(result) < 3 or any(
        len(v) != board.rows * board.cols or not np.isfinite(v).all() for v in result
    ):
        raise ValueError("校正には全角点を含む有限なビューが3枚以上必要です")
    return result


def _keep(errors):
    return [i for i, e in enumerate(errors) if e <= max(3 * np.median(errors), 1e-4)]


def calibrate_intrinsics(board, views, size):
    views = _views(board, views)

    def fit(indices):
        rms, k, d, rv, tv = cv.calibrateCamera(
            [board.object_points] * len(indices),
            [views[i] for i in indices],
            tuple(size),
            None,
            None,
            flags=cv.CALIB_FIX_K3,
            criteria=CRITERIA,
        )
        errors = [
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (
                                cv.projectPoints(board.object_points, r, t, k, d)[0]
                                - views[i]
                            )
                            ** 2,
                            axis=2,
                        )
                    )
                )
            )
            for i, r, t in zip(indices, rv, tv)
        ]
        return Intrinsics(k, d, tuple(size), float(rms), errors, indices)

    result = fit(list(range(len(views))))
    keep = _keep(result.errors)
    if 3 <= len(keep) < len(views):
        result = fit(keep)
    return result


def reprojections(board, left, right, i0, i1, stereo):
    ok, r, t = cv.solvePnP(board.object_points, np.asarray(left), i0.K, i0.distortion)
    if not ok:
        raise ValueError("盤の姿勢を推定できません")
    p0 = cv.projectPoints(board.object_points, r, t, i0.K, i0.distortion)[0]
    r1 = cv.Rodrigues(stereo.R @ cv.Rodrigues(r)[0])[0]
    p1 = cv.projectPoints(
        board.object_points, r1, stereo.R @ t + stereo.T, i1.K, i1.distortion
    )[0]
    return p0, p1


def max_view_error(board, intrinsics, views):
    """既知の内部パラメータで各ビューの盤を当てはめたときの、最大の再投影誤差 [px]。

    保存済みの内部パラメータ（キャッシュ）が今のカメラのものかを確かめるのに使う。
    ビューごとに盤の姿勢だけを解くので、K と歪みが合っていれば誤差は検出の精度程度に収まる。
    """
    errors = []
    for points in views:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        ok, r, t = cv.solvePnP(board.object_points, points, intrinsics.K, intrinsics.distortion)
        if not ok:
            return float("inf")
        projected = cv.projectPoints(board.object_points, r, t, intrinsics.K, intrinsics.distortion)[0]
        errors.append(float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=2)))))
    return max(errors, default=float("inf"))


def calibrate_stereo(board, left, right, i0, i1):
    left, right = _views(board, left), _views(board, right)
    if len(left) != len(right):
        raise ValueError("左右のペア数が一致しません")

    def fit(indices):
        rms, _, _, _, _, r, t, _, _ = cv.stereoCalibrate(
            [board.object_points] * len(indices),
            [left[i] for i in indices],
            [right[i] for i in indices],
            i0.K.copy(),
            i0.distortion.copy(),
            i1.K.copy(),
            i1.distortion.copy(),
            i0.size,
            flags=cv.CALIB_FIX_INTRINSIC,
            criteria=CRITERIA,
        )
        result = Stereo(r, t, float(rms), [], indices)
        for i in indices:
            a, b = reprojections(board, left[i], right[i], i0, i1, result)
            result.errors.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.concatenate(
                                (
                                    (a - left[i]).reshape(-1, 2),
                                    (b - right[i]).reshape(-1, 2),
                                )
                            )
                            ** 2
                        )
                        * 2
                    )
                )
            )
        return result

    result = fit(list(range(len(left))))
    keep = _keep(result.errors)
    if 3 <= len(keep) < len(left):
        result = fit(keep)
    lengths = np.concatenate(
        [
            square_lengths(board, left[i], right[i], i0, i1, result)
            for i in result.used_indices
        ]
    )
    result.square_error_mm = float(np.max(np.abs(lengths - board.square_cm)) * 10)
    return result


def square_lengths(board, left, right, i0, i1, stereo):
    a = cv.undistortPoints(np.asarray(left), i0.K, i0.distortion, P=i0.K).reshape(-1, 2)
    b = cv.undistortPoints(np.asarray(right), i1.K, i1.distortion, P=i1.K).reshape(
        -1, 2
    )
    homogeneous = cv.triangulatePoints(
        i0.K @ np.c_[np.eye(3), np.zeros(3)], i1.K @ np.c_[stereo.R, stereo.T], a.T, b.T
    )
    points = (homogeneous[:3] / homogeneous[3]).T.reshape(board.rows, board.cols, 3)
    return np.concatenate(
        [np.linalg.norm(np.diff(points, axis=axis), axis=2).ravel() for axis in (0, 1)]
    )
