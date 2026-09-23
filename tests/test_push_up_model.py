"""座位プッシュアップの力学モデル（``push_up_model``）を固定する。

**なぜこのテストがあるか。**

同じモデルを 3 経路（オフライン・USB・スマホ）で別々に書いていたため、経路ごとに
重力の向き・関節の基準点・局所軸・体幹荷重の扱いが食い違っていた（KNOWN_ISSUES
§1-5、§5-1、§5-7、§5-8）。モデルを 1 か所にまとめたので、その約束をここで固定する。

- 重力は試技前（初期フレーム）の体幹の向きから決める（§1-5）
- 手首トルクは右手首まわり、肘は肘まわり（§5-7）。手を固定端に、前腕 → 上腕の鎖に
  体幹＋頭の荷重を肩で載せる（§2-1、§5-8）
- 手首の局所軸は手のひら（手の点）と前腕から作り、手の点が無ければ肘と同じ屈曲軸（§5-1）
- 慣性テンソルはリンクの向きに合わせて全体座標へ回す（§2-3）

期待値は、鎖の実装を通さずに「重さ × てこの腕」で手計算した数値で与える。
"""

from __future__ import annotations

import numpy as np
import pytest

from push_up_model import (
    SegmentState,
    estimate_gravity,
    hand_mass,
    hand_point,
    inertia_about_link,
    joint_axes,
    push_up_torques,
    torso_load_mass,
)
from utils import compute_local_torque

G = 9.81
MIRROR = np.array([-1.0, 1.0, 1.0])


class TestGravityFromTrunk:
    """重力は初期フレームの体幹（腰中点 → 肩中点）の向きの逆。"""

    def test_camera_coordinates_snap_to_plus_y(self):
        # 入力 CSV の実測（被験者 3 など）: 上向き ≈ [0.1, -0.98, -0.15]、y が鉛直下向き
        ups = np.tile([0.10, -0.98, -0.15], (30, 1))
        est = estimate_gravity(ups, magnitude=G)
        np.testing.assert_allclose(est.vector, [0.0, G, 0.0])

    def test_z_up_coordinates_snap_to_minus_z(self):
        # 5_1stereo_pose_scaled.csv だけ z が上: 上向き ≈ [0.11, -0.31, 0.95]
        ups = np.tile([0.11, -0.31, 0.95], (30, 1))
        est = estimate_gravity(ups, magnitude=G)
        np.testing.assert_allclose(est.vector, [0.0, 0.0, -G])

    def test_trunk_mode_uses_the_trunk_direction_itself(self):
        ups = np.tile([0.0, -0.6, -0.8], (5, 1))
        est = estimate_gravity(ups, magnitude=G, mode="trunk")
        np.testing.assert_allclose(est.vector, [0.0, 0.6 * G, 0.8 * G])

    def test_reports_how_far_the_trunk_leans_from_the_chosen_axis(self):
        ups = np.tile([0.0, -np.cos(np.radians(10)), np.sin(np.radians(10))], (5, 1))
        est = estimate_gravity(ups, magnitude=G)
        assert est.lean_deg == pytest.approx(10.0)

    def test_median_ignores_missing_and_outlying_frames(self):
        ups = np.tile([0.0, -1.0, 0.0], (9, 1))
        ups[2] = np.nan
        ups[5] = [0.0, 0.0, 5.0]   # 1 フレームだけ外れても多数決で決まる
        est = estimate_gravity(ups, magnitude=G)
        np.testing.assert_allclose(est.vector, [0.0, G, 0.0])

    def test_no_usable_frame_is_an_error(self):
        with pytest.raises(ValueError):
            estimate_gravity(np.full((4, 3), np.nan), magnitude=G)


class TestMasses:
    def test_torso_and_head_are_shared_by_the_two_arms(self):
        # Winter: 体幹 0.497 + 頭頸 0.081 = 0.578、60 kg で 34.68 kg、片腕 17.34 kg
        assert torso_load_mass(60.0) == pytest.approx(17.34)

    def test_torso_mass_override(self):
        assert torso_load_mass(60.0, share=0.5, torso_mass=30.0) == pytest.approx(15.0)

    def test_hand_mass(self):
        assert hand_mass(60.0) == pytest.approx(0.36)


class TestHandPoint:
    def test_midpoint_of_index_and_pinky(self):
        np.testing.assert_allclose(hand_point(np.array([0.0, 0.0, 0.0]), np.array([0.02, 0.08, 0.0])),
                                   [0.01, 0.04, 0.0])

    def test_missing_finger_gives_nan(self):
        assert not np.all(np.isfinite(hand_point(np.array([np.nan, 0, 0]), np.zeros(3))))


# 静止した右腕（x-z 平面、z が上）。手首が原点で前腕は鉛直、肩は少し前（+x）。
WRIST = np.array([0.0, 0.0, 0.0])
ELBOW = np.array([0.0, 0.0, 0.25])
SHOULDER = np.array([0.10, 0.0, 0.55])
OTHER_SHOULDER = np.array([0.10, 0.36, 0.55])
HAND = np.array([0.08, 0.0, 0.0])   # 手は前へ（手首は背屈 90°）
GRAVITY = np.array([0.0, 0.0, -G])

M_FOREARM = 60.0 * 0.016
M_UPPER_ARM = 60.0 * 0.0227


def _still(mass, com, link, inertia=None):
    zero = np.zeros(3)
    return SegmentState(
        inertia=np.diag([1e-3, 1e-3, 5e-4]) if inertia is None else inertia,
        mass=mass, omega=zero, domega=zero, com_acc=zero, com=com, link=link)


def _still_arm_torques(load=17.34, hand=0.36):
    forearm = _still(M_FOREARM, ELBOW + 0.430 * (WRIST - ELBOW), WRIST - ELBOW)
    upper_arm = _still(M_UPPER_ARM, SHOULDER + 0.436 * (ELBOW - SHOULDER), ELBOW - SHOULDER)
    return push_up_torques(forearm, upper_arm, WRIST, ELBOW, SHOULDER, GRAVITY, load, hand)


class TestStaticTorques:
    """静止姿勢では、各関節のトルク = 先の部位と荷重の重さ × 水平のてこの腕。

    重さ W（上向きに支える力 (0, 0, W)）が水平距離 d_x の位置にあると、y 成分は −d_x·W。
    """

    def test_wrist_torque_is_about_the_wrist(self):
        # 上腕の重心 x = 0.10 − 0.436·0.10 = 0.0564。前腕の重心は手首の真上（x = 0）
        # 体幹荷重は肩（x = 0.10）
        expected_y = -(0.0564 * M_UPPER_ARM * G + 0.10 * 17.34 * G)
        tau = _still_arm_torques()["wrist"]
        np.testing.assert_allclose(tau, [0.0, expected_y, 0.0], atol=1e-9)

    def test_elbow_torque_is_about_the_elbow(self):
        # 肘も x = 0 なので、てこの腕は手首と同じ。前腕は肘より手前（手首側）なので入らない
        expected_y = -(0.0564 * M_UPPER_ARM * G + 0.10 * 17.34 * G)
        tau = _still_arm_torques()["elbow"]
        np.testing.assert_allclose(tau, [0.0, expected_y, 0.0], atol=1e-9)

    def test_forearm_weight_acts_at_the_wrist_but_not_the_elbow(self):
        """前腕を傾けると、手首トルクにだけ前腕の重さが加わる。"""
        elbow = np.array([0.10, 0.0, 0.25])
        forearm = _still(M_FOREARM, elbow + 0.430 * (WRIST - elbow), WRIST - elbow)
        upper_arm = _still(M_UPPER_ARM, SHOULDER + 0.436 * (elbow - SHOULDER), elbow - SHOULDER)
        tau = push_up_torques(forearm, upper_arm, WRIST, elbow, SHOULDER, GRAVITY, 0.0, 0.0)
        forearm_com_x = 0.10 * 0.57
        np.testing.assert_allclose(tau["wrist"][1], -(forearm_com_x * M_FOREARM + 0.10 * M_UPPER_ARM) * G,
                                   atol=1e-9)
        np.testing.assert_allclose(tau["elbow"][1], 0.0, atol=1e-9)

    def test_shoulder_carries_the_hanging_arm_and_the_hand(self):
        # 自由振りの鎖（腕を肩から吊る）。手は手首の質点（分母と同じ近似、§2-2）。
        # 肩から見た水平距離: 上腕の重心 −0.0436、前腕の重心 −0.10、手（手首）−0.10
        expected_y = (0.0436 * M_UPPER_ARM + 0.10 * M_FOREARM + 0.10 * 0.36) * G
        tau = _still_arm_torques()["shoulder"]
        np.testing.assert_allclose(tau, [0.0, expected_y, 0.0], atol=1e-9)

    def test_the_torso_load_does_not_reach_the_shoulder(self):
        """体幹荷重は肩に載る外力なので、肩まわりのモーメントを作らない。"""
        with_load = _still_arm_torques(load=17.34)["shoulder"]
        without = _still_arm_torques(load=0.0)["shoulder"]
        np.testing.assert_allclose(with_load, without, atol=1e-12)


class TestJointAxes:
    """関節ごとの局所軸（z = リンク、y = 親 × z）。"""

    def _local_y(self, axes, joint):
        link, parent = axes[joint]
        return compute_local_torque(np.array([0.0, 1.0, 0.0]), link, parent)[1]

    def test_wrist_axis_is_perpendicular_to_the_forearm_and_the_hand(self):
        axes = joint_axes(SHOULDER, ELBOW, WRIST, hand=HAND, other_shoulder=OTHER_SHOULDER)
        link, parent = axes["wrist"]
        y = np.cross(parent, link)
        assert abs(np.dot(y, ELBOW - WRIST)) < 1e-12
        assert abs(np.dot(y, HAND - WRIST)) < 1e-12
        assert np.linalg.norm(y) > 0

    def test_without_the_hand_the_wrist_uses_the_elbow_flexion_axis(self):
        """手の点が無いと手のひらの向きは分からない。手を固定端に前腕が回る面は腕の面なので、
        肘と同じ屈曲軸（同じ向き）を使う。"""
        axes = joint_axes(SHOULDER, ELBOW, WRIST, hand=None)
        assert self._local_y(axes, "wrist") == pytest.approx(self._local_y(axes, "elbow"))
        assert abs(self._local_y(axes, "wrist")) == pytest.approx(1.0)

    def test_a_missing_hand_on_some_frames_falls_back_frame_by_frame(self):
        hands = np.stack([HAND, np.full(3, np.nan)])
        axes = joint_axes(np.stack([SHOULDER] * 2), np.stack([ELBOW] * 2), np.stack([WRIST] * 2), hand=hands)
        _, parent = axes["wrist"]
        np.testing.assert_allclose(parent[0], WRIST - HAND)
        np.testing.assert_allclose(parent[1], ELBOW - SHOULDER)

    @pytest.mark.parametrize("hand", [HAND, np.array([-0.08, 0.0, 0.0])], ids=["前へ", "後ろへ"])
    def test_the_palm_axis_points_like_the_elbow_axis(self, hand):
        """手のひらの軸と肘の屈曲軸を同じ向きに揃える。

        手のひらの軸は「手 × 前腕」なので、手が腕の面の前にあるか後ろにあるかで向きが反転する。
        手の点の有無や手首の曲がり具合でフレームごとに軸が切り替わると、τ_y に段差が入り、
        ノイズの分解（§1-6）では段差が「ノイズ」に化ける。仕事率は τ と ω を同じ軸に射影するので変わらない。
        """
        axes = joint_axes(SHOULDER, ELBOW, WRIST, hand=hand)
        assert self._local_y(axes, "wrist") == pytest.approx(self._local_y(axes, "elbow"))

    def test_elbow_axis_is_the_arm_plane_normal(self):
        axes = joint_axes(SHOULDER, ELBOW, WRIST)
        assert abs(self._local_y(axes, "elbow")) == pytest.approx(1.0)

    def test_shoulder_has_no_parent_without_the_other_shoulder(self):
        axes = joint_axes(SHOULDER, ELBOW, WRIST)
        assert axes["shoulder"][1] is None

    def test_local_y_keeps_its_sign_under_mirroring(self):
        """左右の鏡映で τ_y は変わらない（トルクも y 軸も擬ベクトル）。"""
        tau = np.array([0.3, -1.2, 0.5])
        axes = joint_axes(SHOULDER, ELBOW, WRIST, hand=HAND, other_shoulder=OTHER_SHOULDER)
        mirrored = joint_axes(SHOULDER * MIRROR, ELBOW * MIRROR, WRIST * MIRROR,
                              hand=HAND * MIRROR, other_shoulder=OTHER_SHOULDER * MIRROR)
        # 擬ベクトルの鏡映は −(M τ)
        tau_mirrored = -(tau * MIRROR)
        for joint in ("wrist", "elbow", "shoulder"):
            original = compute_local_torque(tau, *axes[joint])[1]
            reflected = compute_local_torque(tau_mirrored, *mirrored[joint])[1]
            assert reflected == pytest.approx(original), joint


class TestInertiaAboutLink:
    """部位固定系（z がリンク軸）の対角テンソルを、リンクの向きに合わせて全体座標へ回す。"""

    INERTIA = np.diag([0.0013, 0.0011, 0.0006])

    def test_link_along_z_keeps_the_long_axis_on_z(self):
        np.testing.assert_allclose(np.diag(inertia_about_link(self.INERTIA, np.array([0.0, 0.0, 0.3]))),
                                   [0.0012, 0.0012, 0.0006])

    def test_link_along_x_moves_the_long_axis_to_x(self):
        np.testing.assert_allclose(np.diag(inertia_about_link(self.INERTIA, np.array([-0.3, 0.0, 0.0]))),
                                   [0.0006, 0.0012, 0.0012])

    def test_spin_about_the_link_uses_the_long_axis_moment(self):
        link = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        rotated = inertia_about_link(self.INERTIA, link)
        np.testing.assert_allclose(rotated @ link, 0.0006 * link)

    def test_a_degenerate_link_leaves_the_tensor_unrotated(self):
        np.testing.assert_allclose(inertia_about_link(self.INERTIA, np.zeros(3)), self.INERTIA)


class TestNearestAxis:
    """上向きのベクトルに最も近い座標軸を、**符号つき**で選ぶ（USB 経路の重力の自動検出）。

    かつて ``master_research_code._pick_axis_from_vector`` は |cos| で比べていたので、同じ軸の + と − が
    必ず同点になり、同点のときの「優先ラベル」で常に 'Y+' が上になっていた。実行時の座標（−X, −Z, −Y）では
    y は奥行きなので、検出が終わると重力が水平に切り替わっていた。
    """

    @pytest.mark.parametrize("vector, label", [
        ([0.0, 0.1, 1.0], "Z+"),
        ([0.0, 0.1, -1.0], "Z-"),
        ([0.0, -1.0, 0.1], "Y-"),
        ([1.0, 0.0, 0.2], "X+"),
    ])
    def test_the_sign_is_kept(self, vector, label):
        from push_up_model import nearest_axis

        assert nearest_axis(np.array(vector))[0] == label

    def test_candidates_restrict_the_axes(self):
        from push_up_model import nearest_axis

        # x が最大でも、候補が Y と Z だけなら Z+
        assert nearest_axis(np.array([1.0, 0.0, 0.5]), candidates=("Y+", "Y-", "Z+", "Z-"))[0] == "Z+"

    def test_a_near_tie_prefers_the_given_label_only_among_the_tied(self):
        from push_up_model import nearest_axis

        v = np.array([0.0, 0.70, 0.72])
        assert nearest_axis(v, preferred="Y+", ambiguity=0.08)[0] == "Y+"
        # 優先ラベルが同点の組に入っていなければ、最も近い軸のまま
        assert nearest_axis(v, preferred="X+", ambiguity=0.08)[0] == "Z+"

    def test_the_result_matches_estimate_gravity(self):
        from push_up_model import nearest_axis

        for v in ([0.10, -0.98, -0.15], [0.11, -0.31, 0.95]):
            label, unit, _ = nearest_axis(np.array(v))
            np.testing.assert_allclose(estimate_gravity(np.tile(v, (3, 1)), magnitude=G).up, unit)


class TestPositiveWorkWhenLifting:
    """荷重を持ち上げる（肘を伸ばす）と肘は正の仕事をし、P = τ_y·dθ/dt になる（USB 経路の τ·dθ と同じ）。

    θ は上腕（肩→肘）と前腕（肘→手首）のなす角。伸ばすと θ が減る。
    """

    def test_extending_under_load_is_positive_work(self):
        from push_up_model import joint_axes, push_up_joint_powers

        theta = np.radians(60.0)
        wrist = np.zeros(3)
        elbow = np.array([0.0, 0.0, 0.25])
        # 上腕の向き u（肘→肩）は、前腕の延長（上向き）から θ だけ後ろ（−x）へ倒した方向
        u = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        shoulder = elbow + 0.30 * u
        # 肘を伸ばす: θ が減る。前腕は固定なので上腕が +y まわりに回る（−x → +z へ起き上がる）
        # u(θ) の θ 微分 = (−cos θ, 0, −sin θ)、dθ/dt = −1 rad/s なら du/dt = (cos θ, 0, sin θ)
        # ω_上腕 = u × du/dt = (0, sin²θ + cos²θ, 0) → 上腕は +y まわりに 1 rad/s
        omega_upper = np.array([0.0, 1.0, 0.0])
        upper = SegmentState(np.diag([1e-3, 1e-3, 5e-4]), M_UPPER_ARM, omega_upper, np.zeros(3), np.zeros(3),
                             elbow + 0.564 * (shoulder - elbow), shoulder - elbow)
        forearm = _still(M_FOREARM, elbow + 0.430 * (wrist - elbow), wrist - elbow)
        tau = push_up_torques(forearm, upper, wrist, elbow, shoulder, GRAVITY, 17.34, 0.36)
        axes = joint_axes(shoulder, elbow, wrist)
        power = push_up_joint_powers(tau, axes, forearm, upper, None)

        assert power["elbow"] > 0, f"荷重を持ち上げているのに肘の仕事率が {power['elbow']:.2f} W"
        # USB 経路は τ_y·dθ で肘のエネルギーを積む（dθ/dt = −1）
        tau_y = compute_local_torque(tau["elbow"], *axes["elbow"])[1]
        assert power["elbow"] == pytest.approx(tau_y * -1.0)
        # 大きさ: 荷重と上腕の重さ × 肘からの水平距離 × 角速度
        lever = 0.30 * np.sin(theta)
        assert power["elbow"] == pytest.approx((17.34 * lever + M_UPPER_ARM * 0.564 * lever) * G, rel=1e-3)

    def test_flexing_under_load_is_negative_work(self):
        from push_up_model import joint_axes, push_up_joint_powers

        theta = np.radians(60.0)
        wrist, elbow = np.zeros(3), np.array([0.0, 0.0, 0.25])
        shoulder = elbow + 0.30 * np.array([-np.sin(theta), 0.0, np.cos(theta)])
        upper = SegmentState(np.diag([1e-3, 1e-3, 5e-4]), M_UPPER_ARM, np.array([0.0, -1.0, 0.0]), np.zeros(3),
                             np.zeros(3), elbow + 0.564 * (shoulder - elbow), shoulder - elbow)
        forearm = _still(M_FOREARM, elbow + 0.430 * (wrist - elbow), wrist - elbow)
        tau = push_up_torques(forearm, upper, wrist, elbow, shoulder, GRAVITY, 17.34, 0.36)
        power = push_up_joint_powers(tau, joint_axes(shoulder, elbow, wrist), forearm, upper, None)
        assert power["elbow"] < 0
