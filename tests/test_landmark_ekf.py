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

from extended_kalman_filter import EKFConfig, LandmarkEKF

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
