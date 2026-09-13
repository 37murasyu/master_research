"""関節の仕事率を、トルクと同じ局所軸で求めることを固定する。

**なぜこのテストがあるか。**

スコア経路（``compute_cycle_energy_elbow_wrist.py``）は、局所トルク ``*_local_y`` に
軸の作り方が別の角速度を掛けていた。

- 肘: τ_y（親リンクとの外積で作る軸）× 「+Y まわりの肘角」の微分
- 手首: τ_y（全体座標の基準軸から作る軸）× 「水平面からの前腕の傾き」の微分

左右は鏡像なので、正しく計算すれば仕事率は左右で一致する。ところが上の組み合わせでは、
鏡映したときに τ と ω の片方だけが符号を変える。被験者 3 の骨格を左右反転して確かめると、
反転した左の W_pos（301.94 J）が元の右の W_neg（−301.94 J）と一致した。
**左右で逆の相（押し出しと戻し）を積算していた。**

直し方は、角速度を「リンク側と親の相対角速度」にして、トルクと同じ軸に射影すること。
トルクも角速度も擬ベクトルなので、同じ軸（これも擬ベクトル）に射影すれば鏡映で揃って
符号を変え、積は変わらない。
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import compute_joint_power

MIRROR = np.array([-1.0, 1.0, 1.0])


class TestProjectionOnTheTorqueAxis:
    """τ と相対角速度を、compute_local_torque と同じ y 軸に射影して掛ける。"""

    def test_uses_the_relative_angular_velocity_about_the_local_y_axis(self):
        # link = z、parent = x のとき y = normalize(x × z) = (0, -1, 0)。
        #   τ_y = (5, 2, 0)·y = -2
        #   ω_rel = (0, 3, 7) - (0, 1, 0) = (0, 2, 7)、ω_rel·y = -2
        # よって P = (-2)(-2) = 4。x・z 成分（5 や 7）は効かない。
        power = compute_joint_power(
            torque_global=np.array([5.0, 2.0, 0.0]),
            omega_link=np.array([0.0, 3.0, 7.0]),
            omega_parent=np.array([0.0, 1.0, 0.0]),
            link_vec=np.array([0.0, 0.0, 1.0]),
            parent_vec=np.array([1.0, 0.0, 0.0]),
        )
        assert power == pytest.approx(4.0, abs=1e-12), (
            f"仕事率が τ_y × (ω_link − ω_parent)·y = 4 にならない（{power}）。"
            " 絶対角速度を使っているか、トルクと別の軸に射影していないか確認すること"
        )

    def test_rigid_co_rotation_does_no_work_at_the_joint(self):
        # 関節角が変わらず両リンクが一緒に回るだけなら、関節は仕事をしない。
        omega = np.array([0.4, -1.3, 2.2])
        power = compute_joint_power(
            torque_global=np.array([5.0, 2.0, -1.0]),
            omega_link=omega,
            omega_parent=omega.copy(),
            link_vec=np.array([0.2, -0.1, 0.9]),
            parent_vec=np.array([1.0, 0.3, 0.0]),
        )
        assert power == pytest.approx(0.0, abs=1e-12), (
            f"関節角が一定なのに仕事率が {power} になった。親の角速度を引いていない"
        )

    def test_a_fixed_parent_contributes_no_angular_velocity(self):
        # 親リンクが無いとき、y 軸は全体座標の基準軸から作る。link = z なら
        # z 軸は基準に使えず x 軸を使う: x = normalize(x̂ × ẑ) = (0, -1, 0)、y = ẑ × x = (1, 0, 0)。
        #   τ_y = (3, 1, 0)·y = 3、ω_y = (2, 5, 0)·y = 2 → P = 6
        power = compute_joint_power(
            torque_global=np.array([3.0, 1.0, 0.0]),
            omega_link=np.array([2.0, 5.0, 0.0]),
            omega_parent=None,
            link_vec=np.array([0.0, 0.0, 1.0]),
            parent_vec=None,
        )
        assert power == pytest.approx(6.0, abs=1e-12), (
            f"親が固定のとき、リンク側の角速度をそのまま y 軸に射影した 6 にならない（{power}）"
        )


class TestMirrorSymmetry:
    """左右を鏡映した関節でも、仕事率は同じ。"""

    @pytest.mark.parametrize("with_parent", [True, False], ids=["親リンクあり", "親リンクなし"])
    def test_power_is_unchanged_by_mirroring(self, with_parent):
        link = np.array([0.1, -0.2, 0.9])
        parent = np.array([0.8, 0.3, -0.1]) if with_parent else None
        torque = np.array([1.5, -2.0, 0.7])
        omega_link = np.array([0.3, 1.1, -0.4])
        omega_parent = np.array([-0.2, 0.5, 0.6]) if with_parent else None

        original = compute_joint_power(torque, omega_link, omega_parent, link, parent)
        # 位置の差（リンク）は M v、トルクと角速度（擬ベクトル）は -M v で移る。
        mirrored = compute_joint_power(
            -MIRROR * torque,
            -MIRROR * omega_link,
            None if omega_parent is None else -MIRROR * omega_parent,
            MIRROR * link,
            None if parent is None else MIRROR * parent,
        )
        assert abs(original) > 1e-6, "比較の前提として、仕事率が 0 でない配置を選ぶこと"
        assert mirrored == pytest.approx(original, rel=1e-12), (
            f"鏡映で仕事率が {original} → {mirrored} に変わった。"
            " トルクと角速度を別の軸に射影すると、左右で逆の相を積算する"
        )
