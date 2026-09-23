"""混成の重力の決め方（チェッカーボードの短辺か、先頭のフレームの体幹か）。

USB 経路（``master_research_code.py`` の ``_pick_axis_from_vector``・``_load_checkerboard_short_axis_runtime``・
1534 行付近）と同じ選択肢を、混成の実行時の座標（cam0 基準の (−x, −z, −y)、z が上）で使えるようにする。

1. 校正の meta に盤を立てた短辺の向き（``checkerboard_short_axis``）があれば、最寄りの座標軸に吸着させる
   （``push_up_model.nearest_axis``。水平面の制約・優先軸・曖昧さの幅は USB と同じ意味）。
   短辺の符号は角点の並びで逆になりうるので、**符号は体幹で決める**。盤と体幹がほぼ直交なら体幹に戻す。
2. 盤が無ければ ``push_up_model.estimate_gravity``（今の混成と同じ）。
3. 体幹も取れなければ ``config.g``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

import config
from push_up_model import estimate_gravity, nearest_axis

__all__ = ["GravityChoice", "candidate_axes", "choose_gravity", "read_board_up", "AXIS_LABELS"]

AXIS_LABELS = ("X+", "X-", "Y+", "Y-", "Z+", "Z-")
_PLANE_AXES = {"YZ": ["Y+", "Y-", "Z+", "Z-"], "XZ": ["X+", "X-", "Z+", "Z-"], "XY": ["X+", "X-", "Y+", "Y-"]}
# 盤と体幹の向きの |cos| がこれ未満（60° より開く）なら、盤の保存が壊れているとみなす
MIN_BOARD_TRUNK_COS = 0.5


@dataclass(frozen=True)
class GravityChoice:
    """採用した重力。``label`` は重力の向き（"Z-" など）、``up_label`` はその逆。"""

    vector: np.ndarray
    label: str
    up_label: str
    source: str  # "checkerboard" | "trunk" | "default"
    detail: str


def _unit(vector) -> np.ndarray | None:
    try:
        v = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        return None
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return None
    return v / norm


def _opposite(label: str) -> str:
    return label[0] + ("-" if label[1] == "+" else "+")


def read_board_up(meta: Mapping | None) -> np.ndarray | None:
    """校正の meta から盤の短辺の上向き（実行時の座標の単位ベクトル）を読む。無い・壊れていれば None。

    ``meta["checkerboard_short_axis"]["vector_runtime"]`` を使い、無ければ ``vector_cam0`` を
    (−x, −z, −y) に直す（``calib.py`` の ``_save_checkerboard_short_axis`` と同じ変換）。
    """
    if not isinstance(meta, Mapping):
        return None
    entry = meta.get("checkerboard_short_axis")
    if not isinstance(entry, Mapping):
        return None
    runtime = _unit(entry.get("vector_runtime"))
    if runtime is not None:
        return runtime
    cam0 = _unit(entry.get("vector_cam0"))
    if cam0 is None:
        return None
    return _unit([-cam0[0], -cam0[2], -cam0[1]])


def candidate_axes(level_plane_on: bool, plane: str) -> list[str]:
    """吸着先の候補の軸（本体の ``_candidate_axis_labels`` と同じ。知らない平面は 6 つ）。"""
    if not level_plane_on:
        return list(AXIS_LABELS)
    return list(_PLANE_AXES.get(str(plane).upper(), AXIS_LABELS))


def _default(magnitude: float, why: str) -> GravityChoice:
    vector = np.asarray(config.g, dtype=np.float64).copy()
    label, _, _ = nearest_axis(vector)
    return GravityChoice(vector, label, _opposite(label), "default",
                         f"{why}。既定の {vector.tolist()} を使う")


def _from_trunk(estimate, detail: str = "") -> GravityChoice:
    up_label, _, _ = nearest_axis(estimate.up)
    note = f"体幹から推定（傾き {estimate.lean_deg:.1f}°、{estimate.samples} フレーム、mode={estimate.mode}）"
    return GravityChoice(np.asarray(estimate.vector, dtype=np.float64), _opposite(up_label), up_label, "trunk",
                         f"{detail}。{note}" if detail else note)


def choose_gravity(
    trunk_ups,
    board_up,
    *,
    magnitude: float,
    mode: str = "axis",
    candidates: Sequence[str] | None = None,
    preferred_gravity: str = "Y-",
    ambiguity: float = 0.0,
) -> GravityChoice:
    """重力を決める。``trunk_ups`` は先頭の窓の体幹の上向き (N, 3)、``board_up`` は ``read_board_up`` の戻り値。

    盤があれば最寄りの軸（``candidates`` の中、上位 2 つの差が ``ambiguity`` 未満なら ``preferred_gravity``
    の逆）に吸着させる。符号は体幹で決め、盤と体幹がほぼ直交なら体幹に戻す。盤が無ければ
    ``estimate_gravity(trunk_ups, magnitude, mode)`` と同じ結果。体幹も取れなければ ``config.g``。
    """
    estimate = None
    trunk_error = "体幹の向きが無い"
    if trunk_ups is not None:
        try:
            estimate = estimate_gravity(trunk_ups, magnitude, mode)
        except ValueError as error:
            trunk_error = str(error)

    board = _unit(board_up) if board_up is not None else None
    if board is None:
        if estimate is None:
            return _default(magnitude, f"盤の向きが無く、体幹からも重力を決められない（{trunk_error}）")
        return _from_trunk(estimate)

    notes = []
    if estimate is not None:
        trunk = estimate.trunk_up
        cos = float(np.dot(board, trunk))
        if abs(cos) < MIN_BOARD_TRUNK_COS:
            return _from_trunk(estimate, f"盤と体幹がほぼ直交（cos={cos:.2f}）なので盤を使わない")
        if cos < 0.0:
            board = -board
            notes.append("盤の短辺の向きを体幹に合わせて反転")
    else:
        notes.append(f"体幹が取れず（{trunk_error}）、盤の符号をそのまま使う")

    axes = list(candidates) if candidates is not None else list(AXIS_LABELS)
    up_label, up_unit, axis_cos = nearest_axis(board, candidates=axes, preferred=_opposite(preferred_gravity),
                                               ambiguity=ambiguity)
    if estimate is not None and float(np.dot(up_unit, estimate.trunk_up)) <= 0.0:
        return _from_trunk(estimate, f"盤の吸着先 {up_label} が体幹と逆向き・直交なので盤を使わない")
    tilt = float(np.degrees(np.arccos(np.clip(axis_cos, -1.0, 1.0))))
    notes.append(f"盤の短辺を {up_label} に吸着（傾き {tilt:.1f}°）")
    vector = -up_unit * float(magnitude) + 0.0
    return GravityChoice(vector, _opposite(up_label), up_label, "checkerboard", "。".join(notes))
