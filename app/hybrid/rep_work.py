"""回ごとの仕事を 1 フレームずつ積む（Σ P_i·dt_i）。混成の計測（``app.runners.network_measure``）が使う。

かつて混成の計測は、サイクル確定のときに「サイクル全体の仕事率の和」に**確定したフレームの dt だけ**を
掛けていた。同期バッファ（``app.net.sync_buffer``）は 100 ms を超える穴で組を作らないので、組が抜けると
次の組の dt が 2〜数倍になり、その dt で全フレームを積んで仕事が数倍に化けた。

- 仕事はフレームごとの dt で積む。dt が ``MAX_STEP_S`` を超えるフレーム（長い抜けの直後）は積まない。
  その区間で腕がどう動いたかは分からないので、またいで積まない
- 部位ごとに W+ = Σmax(P,0)·dt（論文 4.5.2 節の FB 尺度の分子、ゲージの値）、W− = Σmin(P,0)·dt（負の値）、
  W± = W+ + W− を持つ
- 肘の濾波 E±（``energy_pipeline.compute_cycle_energy_filtered``）の材料として、肘角 θ と τ_y の列も持つ。
  長く回が閉じなくてもメモリを食わないよう、列は ``max_series`` フレームで打ち切る（古い方から捨てる）
- 力学の関所が閉じている間の直近 ``lookahead`` フレームは輪に置き（``hold``）、関所が開いたら流し込む
  （``release``）。押し上げの立ち上がりは関所が開く判定より数フレーム早い
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from app.hybrid.rep_detector import RepConfig
from app.net.sync_buffer import DEFAULT_GRID

__all__ = ["MAX_STEP_S", "LOOKAHEAD_FRAMES", "MAX_SERIES", "WorkSample", "PartWork", "RepAccumulator"]

# これより長い dt のフレームは積まない [s]。既定の格子の、同期バッファが補間で埋める穴の上限（100 ms）。
# 計測（NetworkMeasurement）は同期バッファと同じ GridSpec の max_gap_s を渡す
MAX_STEP_S = DEFAULT_GRID.max_gap_s
# 関所が開く前の輪の長さ（フレーム）。回の区切りの先読みの幅（RepConfig.lookback_frames）が正本
LOOKAHEAD_FRAMES = RepConfig().lookback_frames
# θ・τ_y の列の上限（30 Hz で 60 秒）
MAX_SERIES = 30 * 60


@dataclass(frozen=True)
class WorkSample:
    """1 フレームぶんの材料。``powers`` は部位 → 仕事率 [W]、``theta``・``tau_y`` は肘の部位 → 値。"""

    dt: float
    powers: Mapping[str, float]
    theta: Mapping[str, float] = field(default_factory=dict)
    tau_y: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PartWork:
    """1 部位の仕事 [J]。``neg`` は負の値（Σmin(P,0)·dt）。"""

    pos: float = 0.0
    neg: float = 0.0

    @property
    def net(self) -> float:
        return self.pos + self.neg


class RepAccumulator:
    """今の回の仕事を積む。受信スレッドだけから呼ぶ（ロックは持たない）。"""

    def __init__(
        self,
        parts: Iterable[str],
        *,
        max_step_s: float = MAX_STEP_S,
        lookahead: int = LOOKAHEAD_FRAMES,
        max_series: int = MAX_SERIES,
    ):
        self.parts = tuple(parts)
        self.max_step_s = float(max_step_s)
        self.max_series = int(max_series)
        self._held: deque[WorkSample] = deque(maxlen=max(0, int(lookahead)))
        self._pos = dict.fromkeys(self.parts, 0.0)
        self._neg = dict.fromkeys(self.parts, 0.0)
        self._theta: dict[str, deque[float]] = {}
        self._tau: dict[str, deque[float]] = {}
        # 今の回で積んだフレーム数と、dt が長すぎて積まなかったフレーム数
        self.frames = 0
        self.skipped = 0

    def counts(self, sample: WorkSample) -> bool:
        """このフレームを積むか（dt が正で ``max_step_s`` 以下）。"""
        return math.isfinite(sample.dt) and 0.0 < sample.dt <= self.max_step_s

    def add(self, sample: WorkSample) -> bool:
        """1 フレームを積む。積んだら True（dt が長すぎれば積まずに False）。"""
        if not self.counts(sample):
            self.skipped += 1
            return False
        for part in self.parts:
            power = sample.powers.get(part)
            if power is None or not math.isfinite(power):
                continue
            if power > 0.0:
                self._pos[part] += power * sample.dt
            else:
                self._neg[part] += power * sample.dt
        for part, value in sample.theta.items():
            tau = sample.tau_y.get(part, float("nan"))
            self._theta.setdefault(part, deque(maxlen=self.max_series)).append(float(value))
            self._tau.setdefault(part, deque(maxlen=self.max_series)).append(float(tau))
        self.frames += 1
        return True

    def hold(self, sample: WorkSample) -> None:
        """関所が閉じている間のフレームを輪に置く（積まない）。"""
        if self._held.maxlen:
            self._held.append(sample)

    def release(self) -> list[WorkSample]:
        """輪のフレームを積み、実際に積んだものを返す。"""
        held = list(self._held)
        self._held.clear()
        return [sample for sample in held if self.add(sample)]

    def drop_held(self) -> None:
        self._held.clear()

    def work(self) -> dict[str, PartWork]:
        return {part: PartWork(self._pos[part], self._neg[part]) for part in self.parts}

    def series(self, part: str) -> tuple[np.ndarray, np.ndarray]:
        """肘角 θ [rad] と τ_y [N·m] の列（積んだフレームだけ）。"""
        return (np.asarray(self._theta.get(part, ()), dtype=float),
                np.asarray(self._tau.get(part, ()), dtype=float))

    def reset(self) -> dict[str, PartWork]:
        """今の回の仕事を返して 0 から積み直す（輪も捨てる）。"""
        done = self.work()
        self._pos = dict.fromkeys(self.parts, 0.0)
        self._neg = dict.fromkeys(self.parts, 0.0)
        self._theta.clear()
        self._tau.clear()
        self._held.clear()
        self.frames = 0
        self.skipped = 0
        return done
