"""混成のゲージの閾値（論文 4.5.2 節）。

FB 尺度は 1 サイクルの正の仕事 W_pos = Σmax(τω, 0)Δt を理論 1RM 仕事で割ったスコア S。閾値は負荷率
c = 0.70・0.85 の理論仕事 W_c = theoretical_1rm_work(joint, 体重, 前腕長, c·1RM) で、スコアでは
S ≈ 0.72・0.86 に当たる（表 2）。ゲージはこの 2 点で状態を切り替える:

    v < W_0.70          不足（under）
    W_0.70 ≤ v < W_0.85 目標帯（target）
    v ≥ W_0.85          過負荷（over。表 2 は 0.86 ≦ S が最大筋力）

USB 経路のゲージは旧来の式（r_x 直書き、体重が 2 系統）のままで §2-6 の修正に追随していないので使わない。
1RM の列はオフラインのスコア（``compute_cycle_energy_elbow_wrist.ONE_RM_COLUMNS``）と同じ対応にする。

``compute_cycle_energy_elbow_wrist`` の import は約 0.6 秒かかる。子プロセスの起動時にメインスレッドで
読まれるので、先頭で import する（計測の途中で初めて読んで止まらないように）。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from app.gauge.protocol import PART_NAMES
from compute_cycle_energy_elbow_wrist import ONE_RM_COLUMNS, theoretical_1rm_work

# 部位の並びはゲージの行（protocol）と同じものを使う。``app/runners/hybrid_measure.py``
# などが ``thresholds.PARTS`` として import しているので名前は残す。
PARTS = PART_NAMES
LOAD_LO = 0.70
LOAD_HI = 0.85
# 前腕長（肘→手首）の人体の範囲 [m]。外なら 3D が壊れているとみなし帯を出さない
FOREARM_RANGE_M = (0.15, 0.40)


def subject_index(raw: str | None) -> int | None:
    """GUI の SUBJECT_ID（"00"・"0"・" 03 " など）を表の整数の番号に直す。数字でなければ None。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or not (text.isascii() and text.isdigit()):
        return None
    return int(text)


def _parse_kg(raw: str | None) -> float | None:
    """表の 1 セルを kg に。"none"・空欄・NaN・0 以下は None（帯を作れない）。"""
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() == "none":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _column(part: str) -> str:
    joint, side = part.split("_")
    return ONE_RM_COLUMNS[joint].format(side=side)


def load_one_rm(path: str | Path, subject: int) -> dict[str, float | None]:
    """1RM の表（``m_max_all_merged.csv``）から被験者の 4 部位の 1RM [kg] を引く。

    肘は ``elbow_{side}_outer``（伸展の力、§6-4）、手首は ``wrist_{side}``。番号が表に無ければ全部 None。
    表のファイルが無いときは ``FileNotFoundError`` をそのまま上げる（理由は呼び出し側がログする）。
    """
    result: dict[str, float | None] = {part: None for part in PARTS}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if subject_index(row.get("subject_id")) != subject:
                continue
            for part in PARTS:
                result[part] = _parse_kg(row.get(_column(part)))
            break
    return result


@dataclass(frozen=True)
class PartBand:
    """1 部位の帯。``band`` が None なら ``reason`` に理由（帯を出さない）。"""

    w1rm: float | None
    band: tuple[float, float] | None
    one_rm_kg: float | None
    forearm_m: float | None
    reason: str = ""


def _finite(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def part_bands(
    body_mass_kg: float,
    forearm_m: Mapping[str, float | None],
    one_rm: Mapping[str, float | None],
) -> dict[str, PartBand]:
    """4 部位の帯（W_0.70, W_0.85）と W_1RM。前腕長は左右ごとの実測（``{"L": m, "R": m}``）。

    前腕長が ``FOREARM_RANGE_M`` の外・1RM が無い・体重が正でないときは ``band=None`` と理由を返す。
    """
    mass = _finite(body_mass_kg)
    bands: dict[str, PartBand] = {}
    for part in PARTS:
        joint, side = part.split("_")
        length = _finite(forearm_m.get(side))
        m_db = _finite(one_rm.get(part))
        reason = ""
        if mass is None or mass <= 0.0:
            reason = f"体重 {body_mass_kg!r} kg が正でない"
        elif length is None or not (FOREARM_RANGE_M[0] <= length <= FOREARM_RANGE_M[1]):
            shown = "なし" if length is None else f"{length:.3f} m"
            reason = f"前腕長 {shown} が {FOREARM_RANGE_M[0]:.2f}〜{FOREARM_RANGE_M[1]:.2f} m の外"
        elif m_db is None or m_db <= 0.0:
            reason = f"1RM が無い（{_column(part)}）"
        if reason:
            bands[part] = PartBand(None, None, m_db, length, reason)
            continue
        lo = theoretical_1rm_work(joint, mass, length, LOAD_LO * m_db)
        hi = theoretical_1rm_work(joint, mass, length, LOAD_HI * m_db)
        w1rm = theoretical_1rm_work(joint, mass, length, m_db)
        bands[part] = PartBand(float(w1rm), (float(lo), float(hi)), m_db, length)
    return bands


def classify(value: float | None, band: tuple[float, float] | None) -> str | None:
    """値の状態。lo ちょうどは目標帯、hi ちょうどは過負荷。帯か値が無ければ None。"""
    if band is None:
        return None
    v = _finite(value)
    if v is None:
        return None
    lo, hi = band
    if v < lo:
        return "under"
    if v < hi:
        return "target"
    return "over"
