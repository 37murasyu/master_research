"""混成の計測（Mac＋Pixel）の EKF。USB 経路（``master_research_code.py``）と同じ ``LandmarkEKF`` を 30 Hz の格子で回す。

- 雑音は ``HYBRID_EKF_PROFILE`` の較正プロファイル。**空なら同梱の既定値**（``app.tuning.ekf_profile`` の builtin）。
  USB 経路は空なら環境変数のスカラー（``EKF_Q_ACC``・``EKF_R``）を使うが、GUI は USB 向けの 1e-3 を必ず子へ渡し、
  それを混成に使うと合成の押し上げで肘の W_pos が +52% になる（同梱値なら +10%）。混成は ``EKF_Q_ACC``・``EKF_R``・
  ``EKF_PROFILE`` を読まない
- ``EKF_ENABLE``・``EKF_GATE_STD``・``EKF_ROBUST_GATE``・``EKF_MAX_GAP_S``・``EKF_BPF_*`` は USB と共用
- 同期バッファ（``app.net.sync_buffer``）は 100 ms を超える穴で組を作らないので、次の組の時刻は格子（1/30 s）の
  n 倍跳ぶ。抜けた格子の数だけ NaN の観測と dt=1/30 で予測してから観測で更新する（``GridEkf.step``）。
  ``REBUILD_GAP_S`` を超える抜けは外挿が当てにならないので作り直す
- **発散の見張り**: 同梱の既定値（q=0.122、r=2.59e-5）で追える帯域は約 0.65 Hz で、頑健な門は予測から外れるほど
  更新を弱める（外れ幅 y に対して修正量が P·c²/y）。1 Hz・振幅 10 cm（最大 0.63 m/s）の動きで追従を失い、
  1 次元の試算で 0.8 m、振幅 20 cm で 6 m ずれたまま戻らなかった。押し上げ（約 0.5 Hz）は追えるが、手の置き直しの
  ような速い動きで数秒壊れる。そこで点ごとに、観測から ``DIVERGE_M`` 以上のずれが ``DIVERGE_FRAMES`` フレーム
  続いたら、その点を観測で初期化し直す。1〜2 フレームの飛び（2026-09-23 の実機の右肘の数 m）は今までどおり抑える
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from app.tuning.ekf_profile import AXES, RuntimeNoise, resolve_profile, runtime_noise
from extended_kalman_filter import EKFConfig, LandmarkEKF, SeriesNoise

__all__ = ["DT", "GRID_NS", "REBUILD_GAP_S", "DIVERGE_M", "DIVERGE_FRAMES", "EkfSettings", "GridEkf", "hybrid_noise"]

# 同期バッファの格子（30 Hz）
DT = 1.0 / 30.0
GRID_NS = 33_333_333
# これより長い抜けは EKF を作り直す [s]
REBUILD_GAP_S = 0.5
# 発散の見張り: 観測からのずれ [m] と、それが続くフレーム数
DIVERGE_M = 0.15
DIVERGE_FRAMES = 3

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    """``config.env_flag`` と同じ読み方（大文字小文字と前後の空白は問わない）。"""
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return True if value in _TRUE else False if value in _FALSE else default


def _number(env: Mapping[str, str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class EkfSettings:
    """混成の EKF の設定。既定値は USB 経路（``master_research_code.py`` の EKF_*）と同じ。"""

    enabled: bool = True
    # 較正プロファイル（ファイルかフォルダ）。None なら同梱の既定値
    profile: str | None = None
    gate_std: float = 3.0
    robust_gate: bool = True
    # 系列ごとの欠測の上限 [s]（LandmarkEKF の max_gap_s）。0 は無制限
    max_gap_s: float = 0.0
    bpf_low: float = 0.0
    bpf_high: float = 0.0
    bpf_order: int = 2

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EkfSettings":
        env = os.environ if env is None else env
        return cls(
            enabled=_flag(env, "EKF_ENABLE", True),
            profile=(env.get("HYBRID_EKF_PROFILE") or "").strip() or None,
            gate_std=_number(env, "EKF_GATE_STD", 3.0),
            robust_gate=_flag(env, "EKF_ROBUST_GATE", True),
            max_gap_s=_number(env, "EKF_MAX_GAP_S", 0.0),
            bpf_low=_number(env, "EKF_BPF_LOW", 0.0),
            bpf_high=_number(env, "EKF_BPF_HIGH", 0.0),
            bpf_order=int(_number(env, "EKF_BPF_ORDER", 2)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def hybrid_noise(settings: EkfSettings, landmark_ids: Sequence[int], *, dt: float = DT,
                 bpf_enabled: bool = False) -> RuntimeNoise:
    """混成の EKF の雑音と出どころ。プロファイルが無ければ同梱の既定値（``resolve_profile(None, dt=)`` 相当）。

    同梱の既定値には推定した門の広さが無いので、門だけ ``EKF_GATE_STD`` を使う。プロファイルを指定しても
    dt が合わない・BPF が有効で同梱の既定値に落ちたときも同じ（プロファイルが無いときと門を揃える）。
    """
    if settings.profile:
        noise = runtime_noise(settings.profile, dt=dt, bpf_enabled=bpf_enabled,
                              landmark_ids=landmark_ids, scalar=EKFConfig(gate_std=settings.gate_std))
        if noise.origin == "profile":
            return noise
        return replace(noise, cfg=_with_gate(noise.cfg, settings.gate_std))
    ids = tuple(sorted(int(lid) for lid in landmark_ids))
    resolution = resolve_profile(None, dt=dt, bpf_enabled=bpf_enabled)
    return RuntimeNoise(cfg=_with_gate(resolution.series_noise(ids), settings.gate_std), origin="builtin",
                        resolution=resolution, sources={"builtin": len(ids) * len(AXES)}, landmark_ids=ids)


def _with_gate(noise: SeriesNoise, gate_std: float) -> SeriesNoise:
    """系列ごとの雑音の門だけを ``gate_std`` にそろえる。"""
    return SeriesNoise(q_acc=noise.q_acc, r=noise.r, gate_std=np.full_like(noise.gate_std, gate_std))


class GridEkf:
    """30 Hz の格子の上で ``LandmarkEKF`` を回す。受信スレッドだけから呼ぶ。"""

    def __init__(self, settings: EkfSettings, landmark_ids: Sequence[int], *, dt: float = DT,
                 rebuild_gap_s: float = REBUILD_GAP_S):
        self.settings = settings
        self.landmark_ids = tuple(sorted(int(lid) for lid in landmark_ids))
        self.dt = float(dt)
        self.rebuild_gap_s = float(rebuild_gap_s)
        # 作り直した回数（0.5 s を超える抜け）と、NaN で予測した格子の数
        self.rebuilds = 0
        self.predicted = 0
        self.failures = 0
        # 発散の見張りで初期化し直した点の延べ数と、点ごとのずれが続いたフレーム数
        self.resets = 0
        self._drift = np.zeros(len(self.landmark_ids), dtype=int)
        self.scale_ratio: float | None = None
        self._ekf = self._new_filter(EKFConfig(gate_std=settings.gate_std))
        self.noise = hybrid_noise(settings, self.landmark_ids, dt=self.dt,
                                  bpf_enabled=self._ekf.bandpass_enabled)
        self._cfg = self.noise.cfg
        self._ekf.set_noise(self._cfg)

    def _new_filter(self, cfg) -> LandmarkEKF:
        s = self.settings
        return LandmarkEKF(
            len(self.landmark_ids), fs=1.0 / self.dt, cfg=cfg,
            bpf_low=s.bpf_low, bpf_high=s.bpf_high, bpf_order=s.bpf_order, vectorized=True,
            robust_gate=s.robust_gate, max_gap_s=s.max_gap_s)

    def set_scale(self, ratio: float) -> None:
        """体格の比 L_run / L_cal で較正値を掛け直す（プロファイルに基準長があるときだけ意味がある）。"""
        self.scale_ratio = float(ratio)
        self._cfg = self.noise.scaled(ratio)
        self._ekf.set_noise(self._cfg)

    def rebuild(self) -> None:
        self._ekf = self._new_filter(self._cfg)
        self._drift[:] = 0
        self.rebuilds += 1

    def _reinit(self, points: np.ndarray, raw: np.ndarray) -> None:
        """``points`` の点の状態を観測 ``raw`` で初期化し直す（位置 = 観測、速度・加速度 0、P = I）。

        ``LandmarkEKF`` のベクトル化の状態（``_X``・``_P``・``_init``・``_gap``）を直接書く。初めて観測したときの
        初期化（``_step_vectorized`` の 1)）と同じ値にする。
        """
        ekf = self._ekf
        rows = np.repeat(points, 3)
        z = raw.reshape(-1)
        ekf._X[rows] = 0.0
        ekf._X[rows, 0] = z[rows]
        ekf._P[rows] = np.eye(3)
        ekf._init[rows] = np.isfinite(z[rows])
        ekf._gap[rows] = 0.0

    def step(self, raw: np.ndarray, missing: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``missing`` 個の格子が抜けた後の観測 ``raw``（点 × 3）で更新し、(位置, 速度) を返す。"""
        raw = np.asarray(raw, dtype=float)
        try:
            if missing > 0 and (missing + 1) * self.dt > self.rebuild_gap_s:
                self.rebuild()
            elif missing > 0:
                blank = np.full(raw.shape, np.nan)
                for _ in range(int(missing)):
                    self._ekf.step(blank, self.dt)
                self.predicted += int(missing)
            pos, vel, _ = self._ekf.step(raw, self.dt)
        except (FloatingPointError, np.linalg.LinAlgError):
            # 数値の破綻だけを拾い、観測をそのまま使う（USB 経路と同じ。形の不整合などは握りつぶさない）
            self.failures += 1
            return raw.copy(), np.full(raw.shape, np.nan)
        if self._ekf.bandpass_enabled:
            # 前処理の BPF（EKF_BPF_*）が効くと、EKF が追うのは帯域を通した観測で、位置の直流分が抜ける。
            # 生の観測との差はいつも大きいので、見張りに掛けると全点を 3 フレームごとに初期化し直してしまう
            return pos, vel
        with np.errstate(invalid="ignore"):
            far = np.linalg.norm(pos - raw, axis=1) > DIVERGE_M   # 観測が NaN の点は数えない
        self._drift = np.where(far, self._drift + 1, 0)
        lost = self._drift >= DIVERGE_FRAMES
        if lost.any():
            self._reinit(lost, raw)
            pos[lost] = raw[lost]
            vel[lost] = 0.0
            self._drift[lost] = 0
            self.resets += int(lost.sum())
        return pos, vel

    def provenance(self) -> dict[str, Any]:
        """meta.json とサイドカーに残す出どころ。"""
        noise = self.noise.provenance()
        return {
            "enabled": True,
            "origin": noise["origin"],
            "path": noise["path"],
            "reason": noise["reason"],
            "scale_ratio": self.scale_ratio,
            "settings": self.settings.as_dict(),
            "rebuilds": self.rebuilds,
            "predicted_steps": self.predicted,
            "divergence_resets": self.resets,
        }
