"""局所座標系を作れないときの扱いと、フォールバック軸の向きを固定する。

**なぜこのテストがあるか。**

``utils.compute_local_torque`` は、リンクが非有限・長さ 0 のとき全体座標の値をそのまま返す
（早期リターン）。100 N·m 級の全体座標の値が ``*_local_*`` 列に黙って紛れ、スコアに入り得た
（KNOWN_ISSUES §5-4、H-C）。2026-09-23 に「値は今のまま返し、警告を出す」と決めた。

もう 1 つは親リンクが無いときのフォールバック軸。基準軸を全体座標の z から順に試すので、
z が上のリアルタイム経路では局所 y が「リンクに直交する鉛直成分」になるが、y が鉛直で
z が奥行きのカメラ座標（オフラインの入力 CSV）では奥行き方向になっていた（§1-5）。
重力から決めた上向きを先頭の基準軸として渡せるようにした。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from utils import LocalFrameFallbackWarning, compute_joint_power, compute_local_torque

TORQUE = np.array([3.0, -4.0, 12.0])


class TestFallbackWarning:
    """局所座標系を作れないときは、値を変えずに警告する。"""

    @pytest.mark.parametrize("link", [
        np.array([np.nan, 0.0, 1.0]),
        np.array([0.0, np.inf, 1.0]),
        np.zeros(3),
    ], ids=["nan", "inf", "zero-length"])
    def test_returns_the_global_value_with_a_warning(self, link):
        with pytest.warns(LocalFrameFallbackWarning):
            result = compute_local_torque(TORQUE, link)
        np.testing.assert_array_equal(result, TORQUE)

    def test_a_regular_link_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", LocalFrameFallbackWarning)
            compute_local_torque(TORQUE, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
            compute_local_torque(TORQUE, np.array([0.3, 0.1, 0.2]))

    def test_joint_power_passes_the_warning_through(self):
        """仕事率も同じ関数で軸を作るので、軸を作れなければ同じく警告が出る。"""
        with pytest.warns(LocalFrameFallbackWarning):
            compute_joint_power(TORQUE, np.ones(3), None, np.zeros(3))


class TestFallbackAxisFollowsUp:
    """親リンクが無いとき、局所 y は「上向きのうちリンクに直交する成分」になる。"""

    def test_camera_coordinates_use_the_given_up_direction(self):
        # カメラ座標: y が鉛直下向きなので上は (0, -1, 0)。リンクは水平（x 方向）。
        # 局所 y = 上向き = (0, -1, 0) なので τ_y = τ·(0, -1, 0) = 4。
        local = compute_local_torque(TORQUE, np.array([1.0, 0.0, 0.0]), up_axis=np.array([0.0, -1.0, 0.0]))
        assert local[1] == pytest.approx(4.0, abs=1e-12)

    def test_the_up_direction_is_normalised(self):
        local = compute_local_torque(TORQUE, np.array([1.0, 0.0, 0.0]), up_axis=np.array([0.0, -9.81, 0.0]))
        assert local[1] == pytest.approx(4.0, abs=1e-12)

    def test_without_up_the_global_z_axis_is_used_as_before(self):
        # 既定（z が上の座標系を想定）: 局所 y = (0, 0, 1) なので τ_y = 12。
        local = compute_local_torque(TORQUE, np.array([1.0, 0.0, 0.0]))
        assert local[1] == pytest.approx(12.0, abs=1e-12)

    def test_a_link_along_up_falls_back_to_another_axis(self):
        """リンクが上向きとほぼ平行なら外積が潰れるので、次の基準軸に移る（値は有限）。"""
        local = compute_local_torque(TORQUE, np.array([0.0, -1.0, 0.01]), up_axis=np.array([0.0, -1.0, 0.0]))
        assert np.all(np.isfinite(local))
        assert np.linalg.norm(local) == pytest.approx(np.linalg.norm(TORQUE))

    def test_joint_power_uses_the_same_up_direction(self):
        # τ_y = 4、ω_y = (0, -2, 0)·(0, -1, 0) = 2 → P = 8
        power = compute_joint_power(
            TORQUE, np.array([0.0, -2.0, 5.0]), None, np.array([1.0, 0.0, 0.0]),
            up_axis=np.array([0.0, -1.0, 0.0]))
        assert power == pytest.approx(8.0, abs=1e-12)
