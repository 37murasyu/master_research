"""ゲージに出す仕事量 [J]。どの部位も「今のサイクルの正の仕事」で、サイクルを検出するたびに 0 に戻る。

- 肘: 肘角 θ と局所トルク τ_y から、正の τ·dθ だけを毎フレーム積む（``ElbowGaugeEnergy``）
- 手首: 仕事率の正の部分の時間積分 Σmax(P, 0)·dt（スコアの W_pos と同じ定義）
- 肩・体幹: 仕事率の時間積分 ΣP·dt

かつて肘だけ起動からの累積で、一度もリセットされなかった（KNOWN_ISSUES §6-2）。
手首・肩の履歴は ``master_research_code.py`` がサイクル確定時に空にしている。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

ELBOW_KEYS = ("elbow_R", "elbow_L")
WRIST_KEYS = ("wrist_R", "wrist_L")


class ElbowGaugeEnergy:
    """肘の正の仕事 Σmax(τ·dθ, 0) [J]。"""

    def __init__(self, keys: Iterable[str] = ELBOW_KEYS):
        self._last_theta: dict[str, float | None] = {key: None for key in keys}
        self._energy: dict[str, float] = {key: 0.0 for key in self._last_theta}

    def add(self, key: str, theta: float, tau: float) -> None:
        """今のフレームの肘角 [rad] とトルク [N·m] を足す。前後どちらかの角度が欠けたステップは数えない。"""
        previous = self._last_theta[key]
        if previous is not None and np.isfinite(previous) and np.isfinite(theta):
            work = tau * (theta - previous)
            if work > 0:
                self._energy[key] += work
        self._last_theta[key] = theta

    def reset_cycle(self) -> None:
        """仕事を 0 に戻す。角度は残す（消すと、次のフレームの増分を 1 つ落とす）。"""
        for key in self._energy:
            self._energy[key] = 0.0

    def value(self, key: str) -> float:
        return self._energy[key]


def gauge_values(
    keys: Iterable[str],
    elbow: ElbowGaugeEnergy,
    wrist_components: Mapping[str, Sequence[float]],
    power_history: Mapping[str, Sequence[float]],
    dt: float,
) -> dict[str, float]:
    """各部位のゲージの値 [J]。``wrist_components`` は max(P, 0) の履歴、``power_history`` は P の履歴。"""
    values = {}
    for key in keys:
        if key in ELBOW_KEYS:
            values[key] = float(elbow.value(key))
        elif key in WRIST_KEYS:
            values[key] = float(sum(wrist_components.get(key, []))) * dt
        else:
            values[key] = float(sum(power_history.get(key, []))) * dt
    return values
