"""``LandmarkEKF`` の逐次版とベクトル化版が同じ結果を返すことを固定する。

**なぜこのテストがあるか。**

``LandmarkEKF`` は、同じ EKF を 2 通りに実装している
（``vectorized=True`` の配列演算版と、``False`` の ``ExtendedKalman1D`` を並べた逐次版）。
一致を確かめるテストが無いまま併存しており（``KNOWN_ISSUES.md`` §4-3）、
NumPy 2 では逐次版が動いてすらいなかった（S2 で修正）。

S4 でクラスを ``master_research_code.py`` から ``extended_kalman_filter.py`` へ移し、
S8 で系列別の ``(q_acc, r)`` に配列化する。どちらも 2 実装をまとめて書き換えるので、
先に一致を固定しておく（S3）。S2 の修正をメモリ上で当てた確認では、外れ値 2%・欠測 5%・
60 フレームの穴・dt 2 通り・ゲートの有無のすべてで、最大相対差 1.4e-14、NaN の位置も一致した。
ベクトル化版に変異を入れると検出できることも確かめてある
（共分散更新の K r K^T 項を落とすと相対差 5.8e+02、未初期化の点を NaN にしないと NaN の位置が食い違う）。

S3 の時点では ``master_research_code.py`` を import できない（カメラを開く）ので ast で抜き出していた。
移設後は import する。

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の S3・S4。
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from extended_kalman_filter import EKFConfig, LandmarkEKF, SeriesNoise

REPO_ROOT = Path(__file__).resolve().parent.parent
N_POINTS = 16  # config.pose_keypoints の点数

# (q_acc, r, gate_std)。現行既定、版 2 で推定した中央値、そのゲート無し
CONFIGS = {
    "現行既定": (1e-3, 1e-3, 3.0),
    "推定中央値": (0.122, 2.59e-5, 3.0),
    "ゲート無し": (0.122, 2.59e-5, 0.0),
}
DTS = {"間引きなし": 1 / 30, "4Hz間引き": 8 / 30}


def _measurements(n_frames: int = 600, seed: int = 0) -> np.ndarray:
    """ゲート・欠測・初期化の遅れの分岐をすべて通る合成データ（m 系）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / 30
    truth = np.sin(2 * np.pi * 0.5 * t)[:, None, None] * 0.1 + rng.normal(0, 0.05, (1, N_POINTS, 3))
    meas = truth + rng.normal(0, 0.005, truth.shape)
    meas[rng.random(meas.shape) < 0.02] += 0.2  # 外れ値。ゲートに掛かる
    meas[rng.random((n_frames, N_POINTS)) < 0.05] = np.nan  # 点ごとの欠測
    meas[100:160, 3] = np.nan  # 長い穴。predict だけで進む
    meas[:20, 5] = np.nan  # 最初の観測が遅れ、しばらく未初期化のまま
    return meas


def _run(vectorized: bool, cfg: EKFConfig, dt: float, meas: np.ndarray):
    ekf = LandmarkEKF(N_POINTS, fs=30.0, cfg=cfg, vectorized=vectorized)
    outputs = [ekf.step(frame, dt) for frame in meas]
    return [np.stack(series) for series in zip(*outputs)]  # pos, vel, acc: (フレーム数, 点数, 3)


class TestSequentialAndVectorizedAgree:
    """同じ入力に対して、2 実装の位置・速度・加速度と NaN の位置が一致すること。"""

    @pytest.mark.parametrize("dt", DTS.values(), ids=DTS.keys())
    @pytest.mark.parametrize("params", CONFIGS.values(), ids=CONFIGS.keys())
    def test_outputs_match(self, params, dt):
        q_acc, r, gate_std = params
        cfg = EKFConfig(q_acc=q_acc, r=r, gate_std=gate_std)
        meas = _measurements()

        vectorized = _run(True, cfg, dt, meas)
        sequential = _run(False, cfg, dt, meas)

        for name, v, s in zip(("位置", "速度", "加速度"), vectorized, sequential):
            np.testing.assert_array_equal(np.isnan(v), np.isnan(s), err_msg=f"{name}の NaN の位置が食い違う")
            finite = np.isfinite(s)
            rel = np.abs(v[finite] - s[finite]) / (1.0 + np.abs(s[finite]))
            assert rel.max() <= 1e-12, f"{name}が 2 実装で食い違う（最大相対差 {rel.max():.2e}）"

    def test_the_data_exercises_the_gate(self):
        # ゲートの分岐を通っていなければ、上の一致はゲート抜きの一致でしかない
        meas = _measurements()
        with_gate = _run(True, EKFConfig(q_acc=0.122, r=2.59e-5, gate_std=3.0), 1 / 30, meas)[0]
        without_gate = _run(True, EKFConfig(q_acc=0.122, r=2.59e-5, gate_std=0.0), 1 / 30, meas)[0]
        assert not np.allclose(with_gate, without_gate, equal_nan=True), "合成データの外れ値がゲートに掛かっていない"


class TestPerSeriesParameters:
    """系列（点 × 軸）ごとに ``(q_acc, r, gate_std)`` を持てること（S8）。

    版 2 の推定では系列間の幅が ``r`` で 432 倍、``q_acc`` で 205 倍あった。単一の定数では
    原理的に合わせられないので、48 系列それぞれの値を持つ。
    """

    def test_uniform_arrays_behave_exactly_like_the_scalar_config(self):
        cfg = EKFConfig(q_acc=0.122, r=2.59e-5, gate_std=3.0)
        meas = _measurements()

        scalar = _run(True, cfg, 1 / 30, meas)
        arrays = _run(True, SeriesNoise.uniform(N_POINTS * 3, cfg), 1 / 30, meas)

        for name, a, b in zip(("位置", "速度", "加速度"), arrays, scalar):
            np.testing.assert_array_equal(a, b, err_msg=f"全系列同値の配列とスカラーで{name}が違う")

    def test_per_series_parameters_match_the_sequential_path(self):
        rng = np.random.default_rng(1)
        n = N_POINTS * 3
        noise = SeriesNoise(
            q_acc=10 ** rng.uniform(-2.0, 0.5, n),  # 版 2 の推定幅に合わせる
            r=10 ** rng.uniform(-5.5, -3.5, n),
            gate_std=rng.choice([0.0, 2.0, 3.0, 4.0], n),
        )
        meas = _measurements()

        vectorized = _run(True, noise, 1 / 30, meas)
        sequential = _run(False, noise, 1 / 30, meas)

        for name, v, s in zip(("位置", "速度", "加速度"), vectorized, sequential):
            np.testing.assert_array_equal(np.isnan(v), np.isnan(s), err_msg=f"{name}の NaN の位置が食い違う")
            finite = np.isfinite(s)
            rel = np.abs(v[finite] - s[finite]) / (1.0 + np.abs(s[finite]))
            assert rel.max() <= 1e-12, f"系列別の値で{name}が 2 実装で食い違う（最大相対差 {rel.max():.2e}）"

    def test_series_order_is_point_times_three_plus_axis(self):
        # 並びを取り違えても、2 実装が同じ取り違え方をすれば一致テストでは気づけない
        n = N_POINTS * 3
        noise = SeriesNoise(
            q_acc=np.full(n, 0.122),
            r=np.full(n, 2.59e-5),
            gate_std=np.zeros(n),
        )
        point, axis = 5, 2
        noise.r[point * 3 + axis] = 1e6  # この系列だけ観測をほとんど信用しない

        meas = _measurements()
        pos = _run(True, noise, 1 / 30, meas)[0]
        residual = np.nanmean(np.abs(pos - meas), axis=0)  # (点, 軸)

        # 観測を信用しない系列だけが観測から離れる。ほかの系列は観測ノイズ（σ 5 mm）と
        # 外れ値ぶんしか離れないので、実測では対象 7.95e-2 に対し中央値 1.01e-2 になる
        assert residual[point, axis] == np.nanmax(residual), "観測を信用しない系列が、いちばん観測から離れていない"
        assert residual[point, axis] > 5 * np.nanmedian(residual), "系列の並びが 点×3 + 軸 になっていない"

    def test_a_measurement_model_is_rejected(self):
        # LandmarkEKF は位置を直接観測する。以前はベクトル化版が h_fn を黙って無視していた
        # （KNOWN_ISSUES §4-3）。設定できないものは無視もできないよう、渡されたら止める
        with pytest.raises(ValueError, match="h_fn"):
            LandmarkEKF(
                N_POINTS,
                fs=30.0,
                cfg=EKFConfig(h_fn=lambda x: float(x[0]), h_jac_fn=lambda x: np.array([1.0, 0.0, 0.0])),
            )

    def test_a_wrong_length_is_rejected(self):
        short = SeriesNoise(q_acc=np.ones(5), r=np.ones(5), gate_std=np.ones(5))
        with pytest.raises(ValueError):
            LandmarkEKF(N_POINTS, fs=30.0, cfg=short)


class TestSingleDefinition:
    """移設はコピーではなく移動であること。2 つの定義が並ぶと、片方だけ直す事故が起きる。"""

    def test_measurement_script_imports_instead_of_defining(self):
        tree = ast.parse((REPO_ROOT / "master_research_code.py").read_text(encoding="utf-8"))
        defined = [n.lineno for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LandmarkEKF"]
        imported = any(
            isinstance(n, ast.ImportFrom)
            and n.module == "extended_kalman_filter"
            and any(alias.name == "LandmarkEKF" for alias in n.names)
            for n in tree.body
        )
        assert not defined, f"master_research_code.py:{defined} に LandmarkEKF の定義が残っている"
        assert imported, "master_research_code.py が extended_kalman_filter から LandmarkEKF を import していない"


def _one_series(values, vectorized: bool = True, dt: float = 1 / 30, **kwargs) -> np.ndarray:
    """1 点の x だけに意味のある観測を流し、x の推定位置の列を返す（y・z は 0 の定数）。"""
    cfg = EKFConfig(q_acc=kwargs.pop("q_acc", 1e-3), r=kwargs.pop("r", 1e-4), gate_std=kwargs.pop("gate_std", 3.0))
    ekf = LandmarkEKF(1, fs=1 / dt, cfg=cfg, vectorized=vectorized, **kwargs)
    out = []
    for v in values:
        frame = np.array([[v, 0.0, 0.0]], dtype=float)
        out.append(ekf.step(frame, dt)[0][0, 0])
    return np.array(out)


class TestRobustUpdate:
    """S10 門の外の観測を捨てず、分散を膨らませて取り込む（``robust_gate``）。

    門の外を捨てる方式は、ずれが大きくなるほど捨て続けて戻れない（欠陥 3。版 2 の実測で棄却率 99.87%）。
    門の外では S' = y²/c² として更新する。補正量は P00·c²/y で |y| に反比例して縮むので、
    外れ値 1 発には小さく反応し、段差が続けば少しずつ追いつく。
    """

    # 版 2 で推定した中央値。r は観測雑音 5 mm に相当する
    CALIBRATED = {"q_acc": 0.122, "r": 2.59e-5}

    @staticmethod
    def _sine(n: int = 900, seed: int = 0):
        """振幅 0.1 m・0.5 Hz の往復に 5 mm の観測雑音。プッシュアップの肩の上下に近い。"""
        rng = np.random.default_rng(seed)
        truth = 0.1 * np.sin(2 * np.pi * 0.5 * np.arange(n) / 30)
        return truth, truth + rng.normal(0, 0.005, n)

    def test_rejecting_gate_loses_track_of_a_smooth_motion(self):
        # 比較の前提: 捨てる方式は、ずれるほど捨て続けて戻れない（死のスパイラル）
        truth, meas = self._sine()
        est = _one_series(meas, **self.CALIBRATED)
        rms = np.sqrt(np.mean((est[30:] - truth[30:]) ** 2))
        assert rms > 0.1, f"前提が崩れている: 捨てる方式でも追従した（誤差 RMS {rms * 1000:.1f} mm）"

    def test_robust_gate_tracks_a_smooth_motion(self):
        truth, meas = self._sine()
        est = _one_series(meas, robust_gate=True, **self.CALIBRATED)
        rms = np.sqrt(np.mean((est[30:] - truth[30:]) ** 2))
        assert rms < 0.01, f"往復に追従できていない（誤差 RMS {rms * 1000:.1f} mm）"

    def test_robust_gate_settles_after_a_level_shift(self):
        """0.2 m の段差。捨てる方式は 2.4 m まで行き過ぎて 83 フレームかかる（実測）。"""
        est = _one_series(np.r_[np.zeros(60), np.full(300, 0.2)], robust_gate=True, **self.CALIBRATED)[60:]
        settled = next(i for i in range(len(est)) if np.all(np.abs(est[i:] - 0.2) < 0.01))
        assert settled <= 60, f"段差の後 ±1 cm に収まるまで {settled} フレームかかった"
        assert est.max() < 0.4, f"段差 0.2 m に対して {est.max():.2f} m まで行き過ぎた"

    def test_a_single_outlier_barely_moves_the_estimate(self):
        values = np.zeros(120)
        values[60] = 0.5
        est = _one_series(values, q_acc=1e-3, r=1e-4, robust_gate=True)
        assert abs(est[60]) < 0.05, f"外れ値 1 発で {est[60]:.3f} m 引きずられた"
        assert abs(est[-1]) < 1e-3, "外れ値の後に元へ戻らない"

    @pytest.mark.parametrize("dt", DTS.values(), ids=DTS.keys())
    def test_sequential_and_vectorized_agree(self, dt):
        cfg = EKFConfig(q_acc=1e-3, r=1e-3, gate_std=3.0)
        meas = _measurements()
        runs = []
        for vectorized in (True, False):
            ekf = LandmarkEKF(N_POINTS, fs=30.0, cfg=cfg, vectorized=vectorized, robust_gate=True)
            runs.append(np.stack([ekf.step(frame, dt)[0] for frame in meas]))
        v, s = runs
        np.testing.assert_array_equal(np.isnan(v), np.isnan(s))
        finite = np.isfinite(s)
        rel = np.abs(v[finite] - s[finite]) / (1.0 + np.abs(s[finite]))
        assert rel.max() <= 1e-12, f"ロバスト更新で 2 実装が食い違う（最大相対差 {rel.max():.2e}）"

    def test_the_default_keeps_the_rejecting_gate(self):
        """``run_ekf`` などの既存の契約は変えない。ロバスト更新は指定したときだけ。"""
        assert EKFConfig().robust_gate is False


class TestMaxGap:
    """S10 欠測が上限（秒）を超えたら外挿をやめて NaN を返し、次の観測でやり直す（欠陥 4）。"""

    @staticmethod
    def _with_gap(n_missing: int) -> np.ndarray:
        return np.r_[np.full(30, 0.1), np.full(n_missing, np.nan), np.full(10, 0.3)]

    @pytest.mark.parametrize("vectorized", [True, False], ids=["vectorized", "sequential"])
    def test_a_long_gap_returns_nan_then_restarts(self, vectorized):
        est = _one_series(self._with_gap(30), vectorized, max_gap_s=0.5)   # 30 フレーム = 1.0 s
        # 0.5 s ちょうど（15 フレーム目）は浮動小数の足し上げで揺れるので、判定はその前後で見る
        assert np.all(np.isfinite(est[30:44])), "上限（0.5 s = 15 フレーム）までは外挿を返す"
        assert np.all(np.isnan(est[46:60])), "上限を超えても外挿を返し続けている"
        assert est[60] == pytest.approx(0.3), "欠測の後の最初の観測で初期化し直していない"

    @pytest.mark.parametrize("vectorized", [True, False], ids=["vectorized", "sequential"])
    def test_a_short_gap_keeps_predicting(self, vectorized):
        est = _one_series(self._with_gap(10), vectorized, max_gap_s=0.5)
        assert np.all(np.isfinite(est))

    def test_the_limit_is_in_seconds_not_frames(self):
        # dt = 8/30 s なら 0.5 s は 2 フレーム弱。3 フレームの穴で NaN になる
        est = _one_series(self._with_gap(3), dt=8 / 30, max_gap_s=0.5)
        assert np.isnan(est[32])

    def test_zero_means_no_limit(self):
        est = _one_series(self._with_gap(60))
        assert np.all(np.isfinite(est))


class TestSetNoise:
    """S9 実行時に基準長の比が分かった時点で、雑音パラメータを差し替えられる。"""

    @pytest.mark.parametrize("vectorized", [True, False], ids=["vectorized", "sequential"])
    def test_matches_a_filter_built_with_the_new_values(self, vectorized):
        meas = _measurements(200)
        new = SeriesNoise.uniform(N_POINTS * 3, EKFConfig(q_acc=0.05, r=4e-5, gate_std=2.5))
        swapped = LandmarkEKF(N_POINTS, fs=30.0, cfg=EKFConfig(), vectorized=vectorized)
        fresh = LandmarkEKF(N_POINTS, fs=30.0, cfg=new, vectorized=vectorized)
        swapped.set_noise(new)
        a = np.stack([swapped.step(f, 1 / 30)[0] for f in meas])
        b = np.stack([fresh.step(f, 1 / 30)[0] for f in meas])
        np.testing.assert_array_equal(a, b)
