"""``extended_kalman_filter.py`` の 1 次元 EKF が NumPy 2 で動き、観測モデルを黙って無視しないことを固定する。

**なぜこのテストがあるか。**

``ExtendedKalman1D.step`` は ``S = float(H @ self.P @ H.T + self.cfg.r)`` で、(1,1) 配列を
``float()`` に渡していた。NumPy 2（2.5.3 で確認）はこれを ``TypeError`` にするので、
初期化の次の 2 ステップ目で必ず落ちる。影響は次の 3 つに及んでいた。

- ``EKF_VECTORIZED=0`` の逐次パス。``master_research_code.py`` の広い except が握りつぶし、
  EKF を通っていない生の値が黙って下流に流れる
- ``run_ekf``
- ``tmp_filter_pose_torque.py``

もう 1 つ、``_measure`` は ``h_fn`` と ``h_jac_fn`` の片方だけが与えられると、黙って恒等観測に
落ちていた（``KNOWN_ISSUES.md`` §4-3）。設定したつもりの観測モデルが無視されるので、例外にする。

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の「実装 2」（S2）。
"""

from __future__ import annotations

import numpy as np
import pytest

from extended_kalman_filter import EKFConfig, ExtendedKalman1D, run_ekf

DT = 1 / 30


class TestNumpy2:
    """(1,1) 配列を float() に渡さないこと。"""

    def test_step_runs_past_initialisation(self):
        f = ExtendedKalman1D(EKFConfig(q_acc=0.1, r=1e-4, gate_std=0.0))
        f.step(0.0, DT)  # 初期化だけ。ここは以前から通っていた
        pos, vel, acc = f.step(0.01, DT)  # 以前はここで TypeError

        assert np.isfinite([pos, vel, acc]).all(), "更新後の状態が有限でない"
        assert 0.0 < pos < 0.01, "観測 0.01 に向かって位置が更新されていない"

    def test_run_ekf_filters_a_whole_series(self):
        t = np.arange(90) * DT
        data = np.sin(2 * np.pi * 0.5 * t)[:, None] * 0.1
        pos, _vel, _acc = run_ekf(data, t, EKFConfig(q_acc=0.1, r=1e-4, gate_std=3.0))

        assert pos.shape == data.shape
        assert np.isfinite(pos).all(), "run_ekf が系列の途中で有限でない値を返した"
        assert np.median(np.abs(pos - data)) < 0.01, "ノイズの無い正弦波に追従していない"


class TestMeasurementModel:
    """観測モデルは両方そろって初めて使う。片方だけなら黙って恒等観測にせず知らせる。"""

    def test_h_fn_without_jacobian_is_rejected(self):
        with pytest.raises(ValueError, match="h_jac_fn"):
            ExtendedKalman1D(EKFConfig(h_fn=lambda x: float(x[0])))

    def test_jacobian_without_h_fn_is_rejected(self):
        with pytest.raises(ValueError, match="h_fn"):
            ExtendedKalman1D(EKFConfig(h_jac_fn=lambda x: np.array([1.0, 0.0, 0.0])))

    def test_custom_measurement_model_is_used_when_both_are_given(self):
        # h(x) = 2·位置。観測 2.0 を与え続ければ、位置は恒等観測の 2.0 ではなく 1.0 に寄る
        cfg = EKFConfig(
            q_acc=0.1,
            r=1e-6,
            gate_std=0.0,
            h_fn=lambda x: 2.0 * x[0],
            h_jac_fn=lambda x: np.array([2.0, 0.0, 0.0]),
        )
        f = ExtendedKalman1D(cfg)
        for _ in range(60):
            pos, _vel, _acc = f.step(2.0, DT)

        assert pos == pytest.approx(1.0, abs=0.05), "h_fn / h_jac_fn が使われていない"
