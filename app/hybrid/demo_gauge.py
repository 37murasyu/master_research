"""混成のデモのゲージ（トルクを使わず、3D の肩の上昇と肘角の変化で針を動かす）。

USB の単眼デモ（``master_research_code.py`` の ``DEMO_MONO_GAUGE_ON``、460〜504 行・3484〜3539 行）は既定では
動いていなかった（肘角は 2D の古い並び順の添字、肩は Tasks 経路に world landmarks が無く常に None）。
混成では三角測量した 3D で同じ段階の規則を作り直す:

- 肩の上昇 = 肩の中点の「重力の上向き」への射影の、基準（EMA ``baseline_ema``）からの差 [m]
- 肘角の変化 = 3D の肘角（肩−肘と手首−肘のなす角）の、基準（EMA）からの差の絶対値 [°]（左右別）
- 段階の目標比: 肩 ≥ ``shoulder_full_m`` かつ肘 ≥ ``elbow_full_deg`` → ``ratio_full``、
  肩 ≥ ``shoulder_partial_m`` かつ肘 ≥ ``elbow_partial_deg`` → ``ratio_partial``、それ以外 0
- 1 フレームごとに +``up_step`` / −``down_step`` で目標へ近づける。同じ側の elbow と wrist に同じ比
- 比を J に直すとき、``ratio_full`` を帯の中央（(lo+hi)/2）に当てる（帯が無ければ ``unbanded_full_j``）

基準は USB と同じく毎フレーム EMA で追うので、持ち上げたまま止まると段階はやがて下がる。
関節の並びは ``config.pose_keypoints`` の昇順（``config.slot_of`` で引く）。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from config import slot_of

__all__ = ["DemoConfig", "DemoGauge"]

SIDES = ("L", "R")


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        return default
    return value if math.isfinite(value) else default


@dataclass(frozen=True)
class DemoConfig:
    """デモの段階の規則（既定と環境変数の名前は USB の DEMO_* と同じ）。"""

    shoulder_full_m: float = 0.10  # DEMO_SHOULDER_RISE_FULL_M
    elbow_full_deg: float = 45.0  # DEMO_ELBOW_DELTA_FULL_DEG
    shoulder_partial_m: float = 0.02  # DEMO_SHOULDER_RISE_PARTIAL_M
    elbow_partial_deg: float = 8.0  # DEMO_ELBOW_DELTA_PARTIAL_DEG
    ratio_full: float = 0.80  # DEMO_RATIO_FULL
    ratio_partial: float = 0.30  # DEMO_RATIO_PARTIAL
    up_step: float = 0.025  # DEMO_RATIO_UP_STEP
    down_step: float = 0.035  # DEMO_RATIO_DOWN_STEP
    baseline_ema: float = 0.01  # DEMO_BASELINE_EMA
    unbanded_full_j: float = 50.0  # 帯が無い部位で ratio_full に当てる J

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DemoConfig":
        env = os.environ if env is None else env
        d = cls()
        return cls(
            shoulder_full_m=_env_float(env, "DEMO_SHOULDER_RISE_FULL_M", d.shoulder_full_m),
            elbow_full_deg=_env_float(env, "DEMO_ELBOW_DELTA_FULL_DEG", d.elbow_full_deg),
            shoulder_partial_m=_env_float(env, "DEMO_SHOULDER_RISE_PARTIAL_M", d.shoulder_partial_m),
            elbow_partial_deg=_env_float(env, "DEMO_ELBOW_DELTA_PARTIAL_DEG", d.elbow_partial_deg),
            ratio_full=_env_float(env, "DEMO_RATIO_FULL", d.ratio_full),
            ratio_partial=_env_float(env, "DEMO_RATIO_PARTIAL", d.ratio_partial),
            up_step=_env_float(env, "DEMO_RATIO_UP_STEP", d.up_step),
            down_step=_env_float(env, "DEMO_RATIO_DOWN_STEP", d.down_step),
            baseline_ema=_env_float(env, "DEMO_BASELINE_EMA", d.baseline_ema),
        )


def _unit(vector) -> np.ndarray | None:
    if vector is None:
        return None
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        return None
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else None


def _elbow_angle_deg(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float | None:
    a = shoulder - elbow
    b = wrist - elbow
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _band_of(entry) -> tuple[float, float] | None:
    """``PartBand``（``band`` 属性）か (lo, hi) のどちらでも受ける。"""
    band = getattr(entry, "band", entry)
    if band is None:
        return None
    try:
        lo, hi = float(band[0]), float(band[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) else None


class DemoGauge:
    """``update`` を 1 フレームごとに呼ぶと、4 部位のデモの値 [J] を返す。"""

    def __init__(self, config: DemoConfig | None = None):
        self.config = config or DemoConfig()
        self._slots = {
            name: slot_of(name)
            for name in ("L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_WRIST", "R_WRIST")
        }
        self._shoulder_base: float | None = None
        self._elbow_base: dict[str, float | None] = {side: None for side in SIDES}
        self._ratio: dict[str, float] = {side: 0.0 for side in SIDES}

    @property
    def ratios(self) -> dict[str, float]:
        return dict(self._ratio)

    def _ema(self, base: float | None, now: float) -> float:
        if base is None:
            return now
        a = self.config.baseline_ema
        return float((1.0 - a) * base + a * now)

    def _shoulder_rise(self, points: np.ndarray, up: np.ndarray | None) -> float:
        if up is None:
            return 0.0
        mid = 0.5 * (points[self._slots["L_SHOULDER"]] + points[self._slots["R_SHOULDER"]])
        if not np.all(np.isfinite(mid)):
            return 0.0
        height = float(np.dot(mid, up))
        self._shoulder_base = self._ema(self._shoulder_base, height)
        return height - self._shoulder_base

    def _elbow_delta(self, points: np.ndarray, side: str) -> float:
        angle = _elbow_angle_deg(points[self._slots[f"{side}_SHOULDER"]], points[self._slots[f"{side}_ELBOW"]],
                                 points[self._slots[f"{side}_WRIST"]])
        if angle is None:
            return 0.0
        self._elbow_base[side] = self._ema(self._elbow_base[side], angle)
        return abs(angle - self._elbow_base[side])

    def _target(self, rise: float, delta: float) -> float:
        c = self.config
        if rise >= c.shoulder_full_m and delta >= c.elbow_full_deg:
            return c.ratio_full
        if rise >= c.shoulder_partial_m and delta >= c.elbow_partial_deg:
            return c.ratio_partial
        return 0.0

    def _to_joules(self, ratio: float, band) -> float:
        c = self.config
        span = _band_of(band)
        full = 0.5 * (span[0] + span[1]) if span is not None else c.unbanded_full_j
        return ratio / c.ratio_full * full if c.ratio_full > 0 else 0.0

    def update(self, points_3d, up_unit, bands: Mapping | None = None) -> dict[str, float]:
        """``points_3d`` は (16, 3)（``pose_keypoints`` 昇順）、``up_unit`` は重力の上向き、``bands`` は部位ごとの帯。"""
        c = self.config
        points = np.asarray(points_3d, dtype=np.float64)
        rise = self._shoulder_rise(points, _unit(up_unit))
        bands = bands or {}
        values: dict[str, float] = {}
        for side in SIDES:
            target = self._target(rise, self._elbow_delta(points, side))
            ratio = self._ratio[side]
            if target > ratio:
                ratio = min(target, ratio + c.up_step)
            else:
                ratio = max(target, ratio - c.down_step)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            self._ratio[side] = ratio
            for joint in ("elbow", "wrist"):
                part = f"{joint}_{side}"
                values[part] = self._to_joules(ratio, bands.get(part))
        return {part: values[part] for part in ("elbow_L", "elbow_R", "wrist_L", "wrist_R")}
