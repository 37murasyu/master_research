"""混成ステレオの記録（``landmarks2d_*``）から、遅い方のカメラの実際の撮影時刻で三角測量し直す。

計測中は同期バッファ（``app.net.sync_buffer``）が 2 台を 30 Hz の格子へ線形補間で並べ直してから三角測量する。
実機の Pixel 7a は 10〜15 Hz しか出ないので、記録された 3D（``kpts3d_*``）の Pixel 側は大半が補間で作った点で、
補間の区間は直線になる。これを EKF の雑音の推定（S6）にかけると「なめらかで雑音が小さい」系列に見えて推定が狂う。

そこで、遅い方のカメラ（点の数が少ない方）の実際の撮影時刻ごとに 1 組を作り、速い方だけを線形補間する（間隔が
短いので補間の影響が小さい）。補間の規則（線形、visibility は低い方、``max_gap_ns``（既定は同期バッファの既定の
格子の ``GridSpec.max_gap_ns``、100 ms）を超える穴は埋めない）は同期バッファと同じ。三角測量は計測と同じ
``NetworkMeasurement.points_3d``（歪み補正を含む）。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from app.hybrid.calibration_io import load_session_calibration
from app.hybrid.ekf import EkfSettings
from app.net.protocol import LANDMARK_COUNT, LandmarkFrame
from app.net.sync_buffer import DEFAULT_GRID, InterpolatedFrame, PairedSample
from app.runners.network_measure import MeasurementConfig, NetworkMeasurement
from config import pose_keypoints

__all__ = ["Retriangulated", "read_landmarks", "pairs_at_real_times", "retriangulate"]

ROLES = ("cam0", "cam1")


@dataclass(frozen=True)
class Retriangulated:
    """三角測量し直した 3D。``points`` は (組, 点, 3)、並びはランドマーク ID の昇順（m）。"""

    t_ns: np.ndarray
    points: np.ndarray
    reference: str          # 時刻を決めたカメラ（遅い方）
    skipped: int            # 速い方に長い穴があって組を作れなかった数
    landmark_ids: tuple[int, ...]


def read_landmarks(session: str | Path) -> dict[str, list[LandmarkFrame]]:
    """``landmarks2d_*.csv`` をロールごとの ``LandmarkFrame``（撮影時刻の昇順）に戻す。

    計測を kill で止めると、最後のフレームが途中までしか書かれていないことがある（記録器は 1 秒ごとにしか
    書き出さず、書き込みの区切りがフレームの途中に来うる）。改行の無い最終行と、点が ``LANDMARK_COUNT`` 個で
    ないフレームは捨て、捨てた数を標準エラーに出す（点の足りないフレームを流すと三角測量が IndexError で落ちる）。
    """
    path = next(Path(session).glob("landmarks2d_*.csv"))
    data = path.read_bytes()
    # 途中の行は、数の列が欠けると列の型（t_ns の整数）まで崩すので、表に読む前に落とす
    complete = data[:data.rfind(b"\n") + 1]
    partial = len(complete) < len(data)
    frames: dict[str, list[LandmarkFrame]] = {role: [] for role in ROLES}
    if not complete:
        _report_dropped(path, 0, partial)
        return frames  # 見出しの行も書き終えていない
    table = pd.read_csv(io.BytesIO(complete)).sort_values(["role", "t_ns", "seq", "landmark"])
    dropped = 0
    for (role, seq, t_ns), rows in table.groupby(["role", "seq", "t_ns"], sort=False):
        if len(rows) != LANDMARK_COUNT:
            dropped += 1
            continue
        marks = [tuple(float(v) for v in row) for row in rows[["x", "y", "z", "visibility"]].to_numpy()]
        first = rows.iloc[0]
        frames.setdefault(role, []).append(
            LandmarkFrame(role, int(seq), int(t_ns), int(first["width"]), int(first["height"]), marks))
    for role in frames:
        frames[role].sort(key=lambda f: f.t_capture_ns)
    _report_dropped(path, dropped, partial)
    return frames


def _report_dropped(path: Path, dropped: int, partial: bool) -> None:
    if dropped or partial:
        line = "と改行の無い最終行" if partial else ""
        print(f"[landmarks2d] {path.name} は途中で切れている（kill など）。点が {LANDMARK_COUNT} 個でないフレーム "
              f"{dropped} 個{line}を捨てました", file=sys.stderr)


def _interpolate(frames: list[LandmarkFrame], times: list[int], t: int, max_gap_ns: int) -> InterpolatedFrame | None:
    index = bisect.bisect_left(times, t)
    if index < len(frames) and times[index] == t:
        exact = frames[index]
        return InterpolatedFrame(exact.role, t, exact.width, exact.height, list(exact.landmarks))
    if index == 0 or index >= len(frames):
        return None
    before, after = frames[index - 1], frames[index]
    span = after.t_capture_ns - before.t_capture_ns
    if span <= 0 or span > max_gap_ns:
        return None
    ratio = (t - before.t_capture_ns) / span
    marks = [(b[0] + (a[0] - b[0]) * ratio, b[1] + (a[1] - b[1]) * ratio, b[2] + (a[2] - b[2]) * ratio,
              min(b[3], a[3])) for b, a in zip(before.landmarks, after.landmarks)]
    return InterpolatedFrame(before.role, t, before.width, before.height, marks)


def pairs_at_real_times(frames: dict[str, list[LandmarkFrame]], *, reference: str | None = None,
                        max_gap_ns: int = DEFAULT_GRID.max_gap_ns) -> tuple[str, list[PairedSample], int]:
    """遅い方（``reference``、既定は点の数が少ない方）の撮影時刻ごとに組を作る。戻り値は (reference, 組, 作れなかった数)。"""
    reference = reference or min(ROLES, key=lambda role: len(frames.get(role, [])))
    other = ROLES[1] if reference == ROLES[0] else ROLES[0]
    others = frames.get(other, [])
    times = [f.t_capture_ns for f in others]
    pairs, skipped = [], 0
    for frame in frames.get(reference, []):
        partner = _interpolate(others, times, frame.t_capture_ns, max_gap_ns)
        if partner is None:
            skipped += 1
            continue
        own = InterpolatedFrame(frame.role, frame.t_capture_ns, frame.width, frame.height, list(frame.landmarks))
        pairs.append(PairedSample(t_ns=frame.t_capture_ns, frames={reference: own, other: partner}))
    return reference, pairs, skipped


def retriangulate(session: str | Path, *, reference: str | None = None,
                  max_gap_ns: int = DEFAULT_GRID.max_gap_ns) -> Retriangulated:
    """計測フォルダの 2D から、遅い方のカメラの撮影時刻で 3D を作り直す（校正は計測フォルダに写したもの）。"""
    calibration = load_session_calibration(session)
    # 使うのは状態を持たない points_3d だけ。既定の EKF（雑音の解決と LandmarkEKF の用意）は要らない
    measurement = NetworkMeasurement(*calibration.projections, pose_keypoints,
                                     MeasurementConfig(ekf=EkfSettings(enabled=False)),
                                     lens=dict(zip(ROLES, calibration.intrinsics)))
    reference, pairs, skipped = pairs_at_real_times(read_landmarks(session), reference=reference,
                                                    max_gap_ns=max_gap_ns)
    points = [measurement.points_3d(pair) for pair in pairs]
    ids = tuple(sorted(pose_keypoints))
    return Retriangulated(
        t_ns=np.asarray([pair.t_ns for pair in pairs], dtype=np.int64),
        points=np.asarray(points, dtype=float).reshape(len(points), len(ids), 3),
        reference=reference,
        skipped=skipped,
        landmark_ids=ids,
    )
