"""EKF の ``(q_acc, r)`` を系列ごとに最尤推定する。

``(log10 q_acc, log10 r)`` 上で Nelder-Mead を回し、``ekf_likelihood.innovation_loglik`` を最大化する。
探索範囲は初期値の前後 ``SPAN_DECADES`` 桁で、最適解がその端に張り付いたら ``at_bound`` で知らせる
（張り付いた系列を採用するかどうかは呼び出し側、S7 のフォールバックが決める）。

初期値は観測の差分の統計から取る。白色ジャーク（``q_acc``）で駆動される位置を、白色の
観測誤差（分散 ``r``）つきで観測したとき、フレーム間隔 ``m`` の 3 階差分
``∇³_m z_k = z_k − 3z_{k−m} + 3z_{k−2m} − z_{k−3m}`` は

- 分散 ``Var = 20·r + q_acc·(m·dt)⁵·11/20``（係数 1,3,3,1 の二乗和が 20。過程側は幅 m·dt の箱 3 つの畳み込み核）
- ラグ 3m の自己共分散 ``= −r``（過程側の核は幅 3m·dt なので、ずらすと重ならない）

を満たす。そこで ``r₀ = −γ₃``（m = 1）とする。``q_acc`` は m = 1 だと観測誤差に埋もれる
（間引きなしでは過程の寄与が約 20 万分の 1）ので、過程の寄与が観測誤差を十分上回るまで
m を倍々に広げてから見積もる。

どちらも行が dt ごとに並ぶ前提なので、生 CSV は frame 番号の抜け（処理が間に合わず取りこぼした行）を NaN の行に
戻してから使う（``capture_on_grid``）。詰めたままだと抜けの前後が 1 dt に縮み、q_acc が狂う（正弦の動きで 1 割の
取りこぼしなら約 2 倍）。

CLI（S6 の実測で使う）::

    python -m app.tuning.ekf_estimate output_data/kpts3d_raw_0913_120000.csv

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の「実装 3」。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from app.tuning.ekf_likelihood import innovation_autocorrelation, innovation_loglik
from app.tuning.raw_capture import RawCapture, read_raw_capture

# 探索範囲は初期値の前後この桁数。狭いと「端に張り付き」が誤発火する（設計メモ）
SPAN_DECADES = 3.0
# 端からこの距離（桁）以内で止まったら張り付きとみなす。範囲が狭いときは範囲の 1/10
BOUND_TOL_DECADES = 0.05
# Var(∇³_m x) = q_acc·(m·dt)⁵·UNIT_JERK_VAR
UNIT_JERK_VAR = 11.0 / 20.0
# q_acc の初期値を取る間隔 m は、過程の寄与が観測誤差のこの倍数を超える最小のもの
PROCESS_DOMINANCE = 100.0
# 初期値を取るのに要る 3 階差分の最少個数
MIN_DIFFS = 30
# frame 番号の抜けに挟む NaN の行の上限（30 Hz で 10 秒。これより長い抜けは状態の不確かさが事実上作り直しと同じ）
MAX_FILL_ROWS = 300


@dataclass(frozen=True)
class SeriesFit:
    q_acc: float
    r: float
    loglik: float
    n_eff: int
    at_bound: bool
    rho: tuple[float, ...]


def _third_difference(z: np.ndarray, stride: int) -> np.ndarray:
    return z[3 * stride:] - 3.0 * z[2 * stride:-stride] + 3.0 * z[stride:-2 * stride] - z[:-3 * stride]


def initial_guess(z: np.ndarray, dt: float) -> tuple[float, float]:
    """``(q_acc₀, r₀)``。探索の出発点と範囲の中心に使う。"""
    series = np.asarray(z, dtype=float)
    d3 = _third_difference(series, 1)
    finite = d3[np.isfinite(d3)]
    if finite.size < MIN_DIFFS:
        raise ValueError(f"連続した有限値が足りず初期値を取れない（3 階差分 {finite.size} 個、{MIN_DIFFS} 個以上が要る）")

    centered = d3 - finite.mean()
    lagged = centered[:-3] * centered[3:]
    gamma3 = float(np.mean(lagged[np.isfinite(lagged)]))
    # 過程の寄与や外れ値で符号が崩れたら、分散から粗く取る（どちらにせよ探索範囲は広く取る）
    r0 = -gamma3 if gamma3 < 0 else float(finite.var()) / 20.0

    q0 = None
    stride = 1
    while 3 * stride < series.size:
        dm = _third_difference(series, stride)
        dm = dm[np.isfinite(dm)]
        if dm.size < MIN_DIFFS:
            break
        process = float(dm.var()) - 20.0 * r0
        q0 = max(process, 0.0) / (UNIT_JERK_VAR * (stride * dt) ** 5)
        if process > PROCESS_DOMINANCE * 20.0 * r0:
            break
        stride *= 2
    if not q0:
        raise ValueError("どの間隔でも過程の寄与が観測誤差を上回らず、q_acc の初期値を取れない")
    return q0, r0


def fit_series(
    z: np.ndarray,
    dt: float,
    *,
    span_decades: float = SPAN_DECADES,
    start: tuple[float, float] | None = None,
) -> SeriesFit:
    """1 系列の最尤推定。``start`` を省くと ``initial_guess`` から始める。"""
    series = np.asarray(z, dtype=float)
    q0, r0 = start if start is not None else initial_guess(series, dt)
    center = np.log10([q0, r0])
    bounds = [(float(c - span_decades), float(c + span_decades)) for c in center]
    step = min(0.5, span_decades / 2.0)
    simplex = np.array([center, center + [step, 0.0], center + [0.0, step]])

    def negative_loglik(params: np.ndarray) -> float:
        return -innovation_loglik(series, dt, 10.0 ** params[0], 10.0 ** params[1]).loglik

    result = minimize(
        negative_loglik,
        center,
        method="Nelder-Mead",
        bounds=bounds,
        options={"initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-3, "maxiter": 400},
    )
    log_q, log_r = (float(v) for v in result.x)
    tol = min(BOUND_TOL_DECADES, span_decades / 10.0)
    at_bound = any(min(v - lo, hi - v) <= tol for v, (lo, hi) in zip((log_q, log_r), bounds))

    best = innovation_loglik(series, dt, 10.0 ** log_q, 10.0 ** log_r)
    rho = innovation_autocorrelation(best.normalized, max_lag=5)
    return SeriesFit(
        q_acc=10.0 ** log_q,
        r=10.0 ** log_r,
        loglik=best.loglik,
        n_eff=best.n_eff,
        at_bound=at_bound,
        rho=tuple(float(v) for v in rho),
    )


def frame_step(frame: np.ndarray) -> float:
    """frame 番号の差のうち最も多いもの（1 行ぶんの間引き幅。USB の 4 Hz 間引きなら 8、混成の格子なら 1）。"""
    diffs = np.diff(np.asarray(frame, dtype=float))
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if not positive.size:
        return 1.0
    values, counts = np.unique(positive, return_counts=True)
    return float(values[np.argmax(counts)])   # 同数なら小さい方


def on_frame_grid(frame: np.ndarray, values: np.ndarray) -> np.ndarray:
    """frame 番号の抜けを NaN の行で埋め、``values``（行が先頭の軸）を dt ごとの等間隔に並べ直す。

    推定（``innovation_loglik``・``initial_guess``）は行が dt ごとに並ぶ前提。処理が間に合わず取りこぼした行を
    詰めたまま推定すると、抜けの前後が 1 dt に縮んで q_acc が狂う。frame の差が間引き幅（``frame_step``）の k 倍なら
    k − 1 行の NaN を挟む。倍数でない差はいちばん近い倍数、0 以下の差は 1 行とみなす（どちらも普通は起こらない）。
    NaN を挟むのは ``MAX_FILL_ROWS`` 行まで（それより長い抜けは予測の分散が十分広がり、推定は変わらない）。
    """
    frame = np.asarray(frame, dtype=float)
    if frame.size < 2:
        return values
    advance = np.rint(np.diff(frame) / frame_step(frame))
    advance = np.clip(np.nan_to_num(advance, nan=1.0), 1, MAX_FILL_ROWS + 1).astype(int)
    if (advance == 1).all():
        return values
    rows = np.concatenate([[0], np.cumsum(advance)])
    grid = np.full((int(rows[-1]) + 1, *np.shape(values)[1:]), np.nan)
    grid[rows] = values
    return grid


def capture_on_grid(capture: RawCapture) -> np.ndarray:
    """生 CSV の点（行, 点, 3）を frame 番号の抜けを NaN の行で埋めて等間隔に並べ直したもの（推定・門・棄却率の入力）。"""
    return on_frame_grid(capture.frame, capture.points)


def fit_capture(capture: RawCapture) -> dict[tuple[int, str], SeriesFit | None]:
    """生 CSV 1 本の全系列を推定する。初期値すら取れない系列は None（S7 のフォールバック対象）。

    行は frame 番号の抜けを NaN で埋めてから使う（``capture_on_grid``）。
    """
    dt = float(capture.provenance["dt"])
    points = capture_on_grid(capture)
    fits: dict[tuple[int, str], SeriesFit | None] = {}
    for point, lid in enumerate(capture.landmark_ids):
        for axis_index, axis in enumerate(("x", "y", "z")):
            try:
                fits[(lid, axis)] = fit_series(points[:, point, axis_index], dt)
            except ValueError:
                fits[(lid, axis)] = None
    return fits


def format_report(fits: dict[tuple[int, str], SeriesFit | None], dt: float) -> str:
    """S6 の実測で読む表。dt は較正値の一部なので必ず出す（決定 6）。"""
    lines = [
        f"dt = {dt:.5f} s",
        f"{'ID':>3} {'軸':<2} {'q_acc':>10} {'r [m^2]':>10} {'σ [mm]':>7} {'n_eff':>6} {'端':<2} {'ρ1':>6} {'ρ2':>6}",
    ]
    for (lid, axis), fit in fits.items():
        if fit is None:
            lines.append(f"{lid:>3} {axis:<2} 推定不能（有限値が足りない）")
            continue
        lines.append(
            f"{lid:>3} {axis:<2} {fit.q_acc:10.3e} {fit.r:10.3e} {np.sqrt(fit.r) * 1e3:7.2f} "
            f"{fit.n_eff:6d} {'●' if fit.at_bound else '':<2} {fit.rho[0]:6.2f} {fit.rho[1]:6.2f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生 CSV から EKF の (q_acc, r) を系列ごとに推定して表にする")
    parser.add_argument("csv", help="計測時に書き出した kpts3d_raw_*.csv（同じ名前のサイドカー JSON が要る）")
    args = parser.parse_args(argv)

    capture = read_raw_capture(args.csv)
    print(format_report(fit_capture(capture), float(capture.provenance["dt"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())


