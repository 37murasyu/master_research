"""EKF の尤度（予測誤差分解）と、それを最大化する推定器を固定する。

**なぜこのテストがあるか。**

EKF の ``q_acc`` と ``r`` は全系列で単一の定数で、しかも桁から合っていなかった
（設計メモ 欠陥 1: 版 2 の推定で ``r`` は中央値の 39 倍、``q_acc`` は 1/122）。
系列の幅も ``r`` で 432 倍あるので、系列ごとに最尤推定で決める。

推定器は、実行時の EKF と同じ F・Q（``extended_kalman_filter.constant_acceleration_model``）で
尤度を計算する。別に書くと、推定した ``q_acc`` が実行時と違う意味になる。

合格条件は設計メモの検証 1: 合成データで既知の ``(q_acc, r)`` を **1 桁以内**で回収する。
dt は間引きなし（1/30 s）と 4Hz 間引き（8/30 s）の両方。較正値は dt ごとに別物として扱う
（決定 6）ので、どちらの dt でも推定が成り立たなければならない。

真値には版 2 で推定した中央値（``q_acc = 0.122``, ``r = 2.59e-5 m²`` = σ 5.1 mm）を使う。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from app.tuning.ekf_estimate import fit_capture, fit_series, initial_guess, main, on_frame_grid
from app.tuning.ekf_likelihood import BURN_IN, innovation_autocorrelation, innovation_loglik
from app.tuning.raw_capture import RawCapture, RawCaptureWriter, read_raw_capture
from extended_kalman_filter import EKFConfig, ExtendedKalman1D, constant_acceleration_model

Q_TRUE = 0.122
R_TRUE = 2.59e-5
# 系列の長さは、4Hz 間引きでも有効サンプルが 300（フォールバック閾値）を十分超えるようにする
SERIES = {"間引きなし": (1 / 30, 3000), "4Hz間引き": (8 / 30, 800)}


def _simulate(q_acc: float, r: float, dt: float, n: int, seed: int = 0) -> np.ndarray:
    """白色ジャークで駆動される等加速度運動を、白色の観測誤差つきで観測した 1 系列。"""
    rng = np.random.default_rng(seed)
    F, q_unit = constant_acceleration_model(dt)
    chol = np.linalg.cholesky(q_acc * q_unit)
    state = np.zeros(3)
    z = np.empty(n)
    for k in range(n):
        state = F @ state + chol @ rng.standard_normal(3)
        z[k] = state[0] + rng.normal(0.0, np.sqrt(r))
    return z


class TestLikelihood:
    """ゲート無しの素の KF による予測誤差分解の対数尤度。"""

    def test_true_parameters_score_higher_than_wrong_ones(self):
        dt, n = SERIES["間引きなし"]
        z = _simulate(Q_TRUE, R_TRUE, dt, n)
        truth = innovation_loglik(z, dt, Q_TRUE, R_TRUE).loglik
        assert truth > innovation_loglik(z, dt, Q_TRUE * 100, R_TRUE).loglik, "q_acc を 100 倍にしても尤度が下がらない"
        assert truth > innovation_loglik(z, dt, Q_TRUE, R_TRUE * 100).loglik, "r を 100 倍にしても尤度が下がらない"

    def test_missing_samples_only_predict_and_drop_out_of_the_sum(self):
        dt, n = SERIES["間引きなし"]
        z = _simulate(Q_TRUE, R_TRUE, dt, n)
        full = innovation_loglik(z, dt, Q_TRUE, R_TRUE)
        holed = z.copy()
        holed[1000:1045] = np.nan  # 版 2 で観測された 45 フレームの全欠測

        result = innovation_loglik(holed, dt, Q_TRUE, R_TRUE)
        assert np.isfinite(result.loglik), "欠測があると尤度が有限でなくなる"
        assert result.n_eff == full.n_eff - 45, "欠測の分だけ尤度に入る観測が減っていない"
        assert np.isnan(result.normalized[1000:1045]).all(), "欠測の時刻に正規化イノベーションが入っている"

    def test_innovations_are_white_at_the_true_parameters(self):
        # モデルが正しく指定されていれば、真値での正規化イノベーションは白色になる（検証 2 の基準）
        dt, n = SERIES["間引きなし"]
        z = _simulate(Q_TRUE, R_TRUE, dt, n)
        rho = innovation_autocorrelation(innovation_loglik(z, dt, Q_TRUE, R_TRUE).normalized, max_lag=5)
        assert rho.shape == (5,)
        assert np.abs(rho).max() < 0.1, f"真値でもイノベーションが白色でない: {np.round(rho, 3)}"


class TestEstimator:
    """(log10 q_acc, log10 r) 上の Nelder-Mead で最尤推定する。"""

    @pytest.mark.parametrize("dt, n", SERIES.values(), ids=SERIES.keys())
    def test_initial_guess_brackets_the_truth(self, dt, n):
        # 初期値そのものは粗くてよいが、探索範囲（初期値の前後）に真値が入っていなければならない
        z = _simulate(Q_TRUE, R_TRUE, dt, n)
        q0, r0 = initial_guess(z, dt)
        assert abs(np.log10(r0 / R_TRUE)) <= 1.0, f"r の初期値 {r0:.2e} が真値 {R_TRUE:.2e} から 1 桁以上ずれている"
        assert abs(np.log10(q0 / Q_TRUE)) <= 2.0, f"q_acc の初期値 {q0:.2e} が真値 {Q_TRUE:.2e} から 2 桁以上ずれている"

    @pytest.mark.parametrize("dt, n", SERIES.values(), ids=SERIES.keys())
    def test_recovers_known_parameters_within_one_decade(self, dt, n):
        z = _simulate(Q_TRUE, R_TRUE, dt, n)
        fit = fit_series(z, dt)

        assert abs(np.log10(fit.q_acc / Q_TRUE)) <= 1.0, f"q_acc = {fit.q_acc:.3e}（真値 {Q_TRUE:.3e}）"
        assert abs(np.log10(fit.r / R_TRUE)) <= 1.0, f"r = {fit.r:.3e}（真値 {R_TRUE:.3e}）"
        assert not fit.at_bound, "真値が探索範囲の内側にあるのに、端に張り付いたと判定された"
        assert len(fit.rho) == 5, "ラグ 1〜5 の自己相関が出力に含まれていない"

    def test_flags_a_fit_that_sticks_to_the_search_boundary(self):
        # 探索範囲を初期値の ±0.01 桁に狭めれば、最適解は必ず端に当たる
        dt, n = SERIES["間引きなし"]
        z = _simulate(Q_TRUE, R_TRUE, dt, n)
        fit = fit_series(z, dt, span_decades=0.01, start=(Q_TRUE * 1e3, R_TRUE))
        assert fit.at_bound, "範囲の端で止まった推定を、張り付きとして報告していない"


class TestDroppedFrames:
    """処理が間に合わず取りこぼした行は、生 CSV の frame 番号の差に残る（USB は間引き幅の倍数で跳ぶ）。推定は行が
    dt ごとに並ぶ前提なので、行を詰めたまま推定すると抜けの前後が 1 dt に縮み、q_acc が狂う（正弦の動きで 10% の
    取りこぼしなら約 2 倍、ランダムウォークでは桁で外れる）。frame の差から抜けを NaN の行に戻してから推定する。"""

    def test_gaps_in_the_frame_numbers_become_missing_rows(self):
        grid = on_frame_grid(np.array([0, 8, 16, 40, 48]), np.arange(5.0))
        np.testing.assert_array_equal(grid, [0.0, 1.0, 2.0, np.nan, np.nan, 3.0, 4.0])

    def test_steady_frames_are_left_as_they_are(self):
        values = np.arange(12.0).reshape(4, 3)
        np.testing.assert_array_equal(on_frame_grid(np.array([3, 4, 5, 6]), values), values)

    def test_a_capture_with_dropped_frames_gives_the_same_estimate(self):
        dt, n = SERIES["間引きなし"]
        z = _simulate(Q_TRUE, R_TRUE, dt, n)
        keep = np.flatnonzero(np.random.default_rng(1).random(n) > 0.10)   # 1 割を取りこぼした
        points = np.full((keep.size, 1, 3), np.nan)
        points[:, 0, 0] = z[keep]
        capture = RawCapture(landmark_ids=(11,), frame=keep, t=keep * dt, points=points,
                             provenance={"dt": dt, "stage": "pre_ekf", "landmark_ids": [11]})
        fit = fit_capture(capture)[(11, "x")]
        full = fit_series(z, dt)
        assert fit.q_acc == pytest.approx(full.q_acc, rel=0.2) and fit.r == pytest.approx(full.r, rel=0.2)
        packed = fit_series(z[keep], dt)
        assert abs(np.log10(packed.q_acc / full.q_acc)) > 0.3, "行を詰めても狂わないなら、この試験は何も確かめていない"


class TestSameFilterAsRuntime:
    """推定した q_acc が実行時にも同じ意味を持つには、状態推定そのものが一致していなければならない。"""

    def test_innovations_match_the_runtime_filter_step_by_step(self):
        dt, n = SERIES["4Hz間引き"]
        z = _simulate(Q_TRUE, R_TRUE, dt, n)[:200]
        z[50:60] = np.nan

        runtime = ExtendedKalman1D(EKFConfig(q_acc=Q_TRUE, r=R_TRUE, gate_std=0.0))
        expected = np.full(z.shape, np.nan)
        updates = 0
        for k, zk in enumerate(z):
            if runtime.initialized and np.isfinite(zk):
                probe = copy.deepcopy(runtime)
                probe._predict(dt)  # 更新直前の予測を覗く
                updates += 1
                if updates > BURN_IN:
                    expected[k] = (zk - probe.x[0]) / np.sqrt(probe.P[0, 0] + R_TRUE)
            runtime.step(float(zk) if np.isfinite(zk) else None, dt)

        got = innovation_loglik(z, dt, Q_TRUE, R_TRUE).normalized
        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected), err_msg="尤度に入る時刻が実行時と食い違う")
        np.testing.assert_allclose(got, expected, rtol=1e-9, err_msg="イノベーションが実行時の EKF と食い違う")


class TestCaptureReport:
    """生 CSV 1 本から全系列を推定し、表にする（S6 の実測で使う CLI）。"""

    @staticmethod
    def _capture_csv(tmp_path):
        dt, n = SERIES["4Hz間引き"]
        detected = np.stack([_simulate(Q_TRUE, R_TRUE, dt, n, seed=axis) for axis in range(3)], axis=1)
        never_detected = np.full((n, 3), np.nan)
        points = np.stack([detected, never_detected], axis=1)  # (フレーム数, 点数, 3)

        csv_path = tmp_path / "kpts3d_raw_0101_000001.csv"
        writer = RawCaptureWriter(csv_path, [11, 12], provenance={"dt": dt})
        for k in range(n):
            writer.append(8 * k, k * dt, points[k])
        writer.close()
        return csv_path

    def test_fits_every_series_and_reports_points_never_detected(self, tmp_path):
        fits = fit_capture(read_raw_capture(self._capture_csv(tmp_path)))

        assert set(fits) == {(lid, axis) for lid in (11, 12) for axis in "xyz"}, "系列の抜けか余りがある"
        assert all(fits[(11, axis)] is not None for axis in "xyz"), "推定できるはずの系列が推定されていない"
        assert all(fits[(12, axis)] is None for axis in "xyz"), "一度も検出されない点を推定したことにしている"

    def test_cli_prints_one_row_per_series(self, tmp_path, capsys):
        assert main([str(self._capture_csv(tmp_path))]) == 0
        out = capsys.readouterr().out

        assert "dt = 0.26667" in out, "表に dt が出ていない（dt ごとに別の較正値になるため必須）"
        for lid in (11, 12):
            for axis in "xyz":
                assert f" {lid} {axis} " in out, f"ID {lid} の {axis} 軸の行が無い"
        assert "推定不能" in out, "推定できなかった系列が表で分からない"
