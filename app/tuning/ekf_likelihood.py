"""EKF の対数尤度（予測誤差分解）。

推定中はゲートを完全に無効にした素の KF で、1 系列ぶんの対数尤度
``ℓ = −½ Σ [log(2π S_k) + y_k² / S_k]`` を計算する。欠測は predict だけ行い、和から落とす。

- F・Q は実行時の EKF と同じ ``extended_kalman_filter.constant_acceleration_model`` を使う
- 初期化も実行時の EKF に揃える（最初の観測で ``x = [z, 0, 0]``、``P = I``）。
  ``P = I`` は m 系ではほぼ無情報なので、直後の ``BURN_IN`` 回の更新は事前分布に
  引きずられる。そのイノベーションは和に入れない

尤度は推定の最適化で数百回呼ばれ、1 回に数千ステップ回す。3×3 の numpy 演算では
1 系列の推定に分単位かかるので、対称な共分散を 6 個の float で持つ素の Python で書く。
観測は位置だけ（H = [1, 0, 0]）なので、更新は P の第 0 列だけで閉じる。

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の「実装 3」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from extended_kalman_filter import constant_acceleration_model

# 位置・速度・加速度の 3 つが観測で決まるまでの更新回数
BURN_IN = 3


@dataclass(frozen=True, eq=False)
class InnovationResult:
    """``normalized`` は入力と同じ長さ。尤度に入った時刻だけ ``y/√S``、ほかは NaN。"""

    loglik: float
    n_eff: int
    normalized: np.ndarray


def innovation_loglik(z: np.ndarray, dt: float, q_acc: float, r: float, burn_in: int = BURN_IN) -> InnovationResult:
    series = np.asarray(z, dtype=float)
    F, q_unit = constant_acceleration_model(dt)
    if F[1, 0] or F[2, 0] or F[2, 1] or not (F[0, 0] == F[1, 1] == F[2, 2] == 1.0):
        raise ValueError("状態遷移が上三角・対角 1 の形でない。下の展開式は使えない")
    a, b, c = float(F[0, 1]), float(F[0, 2]), float(F[1, 2])
    q00, q01, q02 = (q_acc * float(v) for v in q_unit[0])
    q11, q12, q22 = q_acc * float(q_unit[1, 1]), q_acc * float(q_unit[1, 2]), q_acc * float(q_unit[2, 2])
    log_2pi = math.log(2.0 * math.pi)

    normalized = np.full(series.shape, np.nan)
    initialized = False
    x0 = x1 = x2 = 0.0
    p00 = p01 = p02 = p11 = p12 = p22 = 0.0
    total = 0.0
    n_eff = 0
    updates = 0

    for k, zk in enumerate(series.tolist()):
        finite = math.isfinite(zk)
        if not initialized:
            if finite:
                x0, x1, x2 = zk, 0.0, 0.0
                p00, p01, p02, p11, p12, p22 = 1.0, 0.0, 0.0, 1.0, 0.0, 1.0
                initialized = True
            continue

        # predict: x = F x、P = F P Fᵀ + q_acc·Q（F は上三角・対角 1）
        x0, x1 = x0 + a * x1 + b * x2, x1 + c * x2
        m00, m01, m02 = p00 + a * p01 + b * p02, p01 + a * p11 + b * p12, p02 + a * p12 + b * p22
        m11, m12 = p11 + c * p12, p12 + c * p22
        p00 = m00 + a * m01 + b * m02 + q00
        p01 = m01 + c * m02 + q01
        p02 = m02 + q02
        p11 = m11 + c * m12 + q11
        p12 = m12 + q12
        p22 = p22 + q22
        if not finite:
            continue

        # update（H = [1, 0, 0]）。最適ゲインでは Joseph 形式と P − K S Kᵀ は一致する
        s = p00 + r
        y = zk - x0
        k0, k1, k2 = p00 / s, p01 / s, p02 / s
        x0, x1, x2 = x0 + k0 * y, x1 + k1 * y, x2 + k2 * y
        p11 -= k1 * p01
        p12 -= k1 * p02
        p22 -= k2 * p02
        p01 -= k0 * p01
        p02 -= k0 * p02
        p00 -= k0 * p00

        updates += 1
        if updates <= burn_in:
            continue
        total += -0.5 * (log_2pi + math.log(s) + y * y / s)
        normalized[k] = y / math.sqrt(s)
        n_eff += 1

    return InnovationResult(loglik=total, n_eff=n_eff, normalized=normalized)


def innovation_autocorrelation(normalized: np.ndarray, max_lag: int = 5) -> np.ndarray:
    """ラグ 1〜``max_lag`` の自己相関。欠測（NaN）を挟むペアは使わない。"""
    e = np.asarray(normalized, dtype=float)
    finite = e[np.isfinite(e)]
    rho = np.full(max_lag, np.nan)
    if finite.size < 2:
        return rho
    mean = float(finite.mean())
    var = float(finite.var())
    if var <= 0.0:
        return rho
    for lag in range(1, max_lag + 1):
        head, tail = e[:-lag], e[lag:]
        ok = np.isfinite(head) & np.isfinite(tail)
        if ok.sum() > 1:
            rho[lag - 1] = float(np.mean((head[ok] - mean) * (tail[ok] - mean)) / var)
    return rho
