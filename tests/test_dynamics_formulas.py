"""逆動力学の式が解析解と一致することを固定する。

**なぜこのテストがあるか。**

2026-09-08 の再検算（``力学計算_再検算_2026-09-08.md``）で、逆動力学の定式そのもの
── トルク連鎖式と ``F = m(a − g)`` ── は解析解と比 1.000000 で一致することを確認した。
誤っていたのは「式に何を入れているか」の側だけである。

この区別が重要で、続く一連の修正（§1-1〜§1-3、R-1〜R-7）は入力側だけを直す。
**定式そのものは変わってはいけない。** ここが壊れたら修正が行き過ぎている。

検証系は「始点を原点に固定し、終点が z 軸まわりに角速度 W で等速回転するリンク」。
この系は全量に閉じた解析解を持つ::

    r(t) = L(cos Wt, sin Wt, 0)     omega = (0, 0, W)     omega_dot = 0
    r''  = -W^2 r （大きさ W^2 L）  重心（中点）の加速度 = r'' / 2

手で追える再現は ``python tools/verify_dynamics_recheck.py``。
"""

from __future__ import annotations

import numpy as np
import pytest

from body_part_storage_module import BodyPartDataStorage
from config import g as GRAVITY
from utils_dynamic import calculate_individual_torques, calculate_M_and_F

DT = 1.0 / 30.0
LINK_LENGTH = 0.25
OMEGA_TRUE = 2.0
BODY_MASS = 60.0
FOREARM_MASS = BODY_MASS * 0.016


def rotating_link(n: int = 200, dt: float = DT, w: float = OMEGA_TRUE,
                  length: float = LINK_LENGTH) -> np.ndarray:
    """始点固定・終点が z 軸まわりに等速回転するリンクの点列 (n, 2, 3)。単位は m。"""
    frames = []
    for k in range(n):
        theta = w * k * dt
        frames.append(np.vstack([
            np.zeros(3),
            np.array([length * np.cos(theta), length * np.sin(theta), 0.0]),
        ]))
    return np.stack(frames)


class TestTorqueChain:
    """``tau_j = sum M + sum (r_g - p_j) x F - tau_E - (r_x - p_j) x f_E``。"""

    def _static_horizontal_link(self):
        """静止して水平に伸びたリンク。関節が原点、重心が (L/2, 0, 0)。"""
        storage = BodyPartDataStorage()
        storage.add_data(
            "seg",
            np.array([LINK_LENGTH, 0.0, 0.0]),   # relative_position_vector
            np.zeros(3),                          # velocity_vector
            np.zeros(3),                          # omega
            np.array([LINK_LENGTH / 2, 0.0, 0.0]),  # centroid
            np.zeros(3),                          # p1（関節位置）
            np.zeros(3),                          # dot_omega
            np.zeros(3),                          # dot_dot_pg
        )
        return storage

    def test_matches_the_closed_form_for_a_static_link(self):
        """静止した水平リンクの関節トルクは m*g*L/2 になる。"""
        storage = self._static_horizontal_link()
        force = FOREARM_MASS * (np.zeros(3) - GRAVITY)  # calculate_M_and_F と同じ F = m(a - g)
        torques = calculate_individual_torques(
            [np.zeros(3)], [force], [np.array([LINK_LENGTH / 2, 0.0, 0.0])],
            np.zeros(3), np.zeros(3), np.zeros(3), ["seg"], storage,
        )
        got = float(np.linalg.norm(torques[0][0]))
        want = FOREARM_MASS * abs(GRAVITY[2]) * LINK_LENGTH / 2
        assert got == pytest.approx(want, rel=1e-9), (
            f"静止水平リンクのトルクが解析解 m*g*L/2 = {want:.6f} N*m から外れた（実測 {got:.6f}）。"
            " 連鎖式そのものを変えてしまっていないか確認すること"
        )

    def test_torque_acts_about_the_axis_perpendicular_to_gravity_and_link(self):
        """トルクの向きは リンク方向 x 重力 の軸。x 方向のリンクと −z の重力なら −y。"""
        storage = self._static_horizontal_link()
        force = FOREARM_MASS * (np.zeros(3) - GRAVITY)
        torques = calculate_individual_torques(
            [np.zeros(3)], [force], [np.array([LINK_LENGTH / 2, 0.0, 0.0])],
            np.zeros(3), np.zeros(3), np.zeros(3), ["seg"], storage,
        )
        tau = torques[0][0]
        assert tau[0] == pytest.approx(0.0, abs=1e-12), f"x 成分が立った: {tau}"
        assert tau[2] == pytest.approx(0.0, abs=1e-12), f"z 成分が立った: {tau}"
        assert tau[1] < 0, f"y 成分の符号が反転している: {tau}"

    def test_external_force_at_the_joint_produces_no_moment(self):
        """外力の作用点が関節と一致するなら、その外力はモーメントを作らない。"""
        storage = self._static_horizontal_link()
        force = FOREARM_MASS * (np.zeros(3) - GRAVITY)
        common = ([np.zeros(3)], [force], [np.array([LINK_LENGTH / 2, 0.0, 0.0])])
        without = calculate_individual_torques(
            *common, np.zeros(3), np.zeros(3), np.zeros(3), ["seg"], storage)
        with_fe = calculate_individual_torques(
            *common, np.zeros(3), np.array([0.0, 0.0, 100.0]), np.zeros(3), ["seg"], storage)
        assert np.allclose(without[0][0], with_fe[0][0]), (
            "関節位置に作用する外力がトルクを変えた。r_x - p_j = 0 なら外積は 0 のはず"
        )


class TestNewtonEuler:
    """``M = I*omega_dot + omega x (I*omega)`` と ``F = m(a - g)``。"""

    def _storage_with(self, omega, dot_omega, dot_dot_pg):
        storage = BodyPartDataStorage()
        storage.add_data("seg", np.array([LINK_LENGTH, 0.0, 0.0]), np.zeros(3),
                         omega, np.array([LINK_LENGTH / 2, 0.0, 0.0]), np.zeros(3),
                         dot_omega, dot_dot_pg)
        return storage

    def test_force_is_mass_times_acceleration_minus_gravity(self):
        """静止した部位に働く力は重量そのもの。"""
        storage = self._storage_with(np.zeros(3), np.zeros(3), np.zeros(3))
        _, force, _ = calculate_M_and_F(
            np.zeros((3, 3)), FOREARM_MASS, storage.get_data("seg"), GRAVITY)
        assert np.allclose(force, FOREARM_MASS * (-GRAVITY)), (
            f"F = m(a - g) から外れた: {force}（期待 {FOREARM_MASS * (-GRAVITY)}）"
        )
        assert float(np.linalg.norm(force)) == pytest.approx(
            FOREARM_MASS * abs(GRAVITY[2]), rel=1e-9)

    def test_no_moment_for_constant_rotation_about_a_principal_axis(self):
        """対角慣性テンソルの主軸まわりの等速回転ではモーメントが厳密に 0 になる。"""
        inertia = np.diag([0.0046, 0.0045, 0.0007])
        storage = self._storage_with(np.array([0.0, 0.0, OMEGA_TRUE]), np.zeros(3), np.zeros(3))
        moment, _, _ = calculate_M_and_F(
            inertia, FOREARM_MASS, storage.get_data("seg"), GRAVITY)
        assert np.allclose(moment, np.zeros(3), atol=1e-12), (
            f"主軸まわりの等速回転でモーメントが立った: {moment}"
        )

    def test_moment_follows_the_inertia_times_angular_acceleration(self):
        """角加速度があるときのモーメントは I*omega_dot。"""
        inertia = np.diag([0.0046, 0.0045, 0.0007])
        alpha = 3.0
        storage = self._storage_with(np.zeros(3), np.array([0.0, 0.0, alpha]), np.zeros(3))
        moment, _, _ = calculate_M_and_F(
            inertia, FOREARM_MASS, storage.get_data("seg"), GRAVITY)
        assert moment[2] == pytest.approx(inertia[2, 2] * alpha, rel=1e-9), (
            f"M = I*omega_dot から外れた: {moment}"
        )


class TestOfflineSegmentKinematics:
    """``compute_torque_from_pose.compute_segment_kinematics``。

    論文のスコアはこの経路で出る。標準形 ``(r x r')/|r|^2`` を使っており、
    再検算で真値と一致することを確認済み。ここは修正対象ではないので変わってはいけない。
    """

    def _run(self):
        from compute_torque_from_pose import SegmentSpec, compute_segment_kinematics

        spec = SegmentSpec(
            name="test", proximal_joint=0, distal_joint=1,
            inertia_row=4, mass_fraction=0.016, com_fraction=0.5,
        )
        return compute_segment_kinematics(rotating_link(), [spec], DT)

    def test_angular_velocity_matches_the_true_value(self):
        omegas, _, _, _, _, _ = self._run()
        got = float(np.median(np.linalg.norm(omegas[10:-10, 0], axis=1)))
        assert got == pytest.approx(OMEGA_TRUE, rel=0.01), (
            f"オフライン経路の角速度が真値 {OMEGA_TRUE} から外れた（実測 {got:.4f}）"
        )

    def test_angular_acceleration_is_zero_for_constant_rotation(self):
        _, domegas, _, _, _, _ = self._run()
        got = float(np.median(np.linalg.norm(domegas[10:-10, 0], axis=1)))
        assert got == pytest.approx(0.0, abs=1e-6), (
            f"等速回転なのに角加速度が立った（実測 {got:.6f}）"
        )

    def test_com_acceleration_matches_the_true_value(self):
        _, _, _, com_acc, _, _ = self._run()
        got = float(np.median(np.linalg.norm(com_acc[10:-10, 0], axis=1)))
        want = OMEGA_TRUE ** 2 * LINK_LENGTH / 2   # com_fraction=0.5 なので中点
        assert got == pytest.approx(want, rel=0.01), (
            f"オフライン経路の重心加速度が真値 {want:.5f} から外れた（実測 {got:.5f}）"
        )
