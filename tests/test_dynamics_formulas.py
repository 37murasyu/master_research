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


class TestLinkVectorCalculator:
    """リアルタイム経路 ``LinkVectorCalculator`` の運動学。

    2026-09-08 の R-2・R-3 で修正した箇所を固定する。

    - R-2 角速度は標準形 ``(r × ṙ)/|r|²``。かつて外積の第 1 引数に前フレームの
      速度を入れており、``dt·|ω⊥|²·ω⊥`` という次元 1/s² の別物を返していた。
      信号を ``ω²·dt`` 倍に潰しつつ、位置ノイズを ``(σ/dt)²/|r|²`` で増幅していた。
    - R-3 ``acceleration`` は重心の 2 階微分。かつてリンクベクトル ``r`` の
      2 階微分を返しており、始点固定なら 2 倍、始点が動けば別のベクトルだった。
    """

    def _run(self, frames, dt=DT):
        from link_vector_calculator_module import LinkVectorCalculator

        calc = LinkVectorCalculator(0, 1)
        out = {"omega": [], "acc": [], "omega_vec": []}
        for i in range(len(frames)):
            result = calc.calculate_link_vectors(list(frames[: i + 1]), 1, i, dt)
            if result[0] is None:
                continue
            _, _, omega, _, _, acc, _ = result
            if omega is not None and np.all(np.isfinite(omega)):
                out["omega"].append(float(np.linalg.norm(omega)))
                out["omega_vec"].append(np.asarray(omega, dtype=float))
            if acc is not None and np.all(np.isfinite(acc)):
                out["acc"].append(float(np.linalg.norm(acc)))
        return out

    def test_angular_velocity_matches_the_true_value(self):
        """等速回転で角速度の真値がそのまま返る。"""
        got = float(np.median(self._run(rotating_link())["omega"][10:]))
        assert got == pytest.approx(OMEGA_TRUE, rel=0.02), (
            f"角速度が真値 {OMEGA_TRUE} rad/s から外れた（実測 {got:.4f}）"
        )

    def test_angular_velocity_is_not_the_cubed_form(self):
        """旧式 ``ω³·dt`` に戻っていないことを明示的に否定する。"""
        got = float(np.median(self._run(rotating_link())["omega"][10:]))
        assert got != pytest.approx(OMEGA_TRUE ** 3 * DT, rel=0.1), (
            f"角速度が ω³·dt ({OMEGA_TRUE ** 3 * DT:.4f}) になっている。"
            " cross(v_prev, v) を使う旧式が復活していないか確認すること"
        )

    def test_angular_velocity_points_along_the_rotation_axis(self):
        """回転軸が z でなくても、向きと大きさが一致する。

        リンクを回転軸に垂直に置く。``ω = (r × ṙ)/|r|²`` が厳密に ω を返すのは
        この配置のとき。リンクに軸方向の成分があると
        ``r × ṙ = ω|r⊥|² − r⊥(r∥·ω)`` の第 2 項が残り、向きがずれる
        ── これは 2 点から軸まわりの回転を復元できないという原理的な限界
        （KNOWN_ISSUES §2-4）であって実装の誤りではない。
        """
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)

        def rodrigues(u, angle):
            k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k

        r0 = np.array([LINK_LENGTH, 0.0, 0.0])
        r0 = r0 - np.dot(r0, axis) * axis          # 軸に垂直な成分だけ残す
        r0 = r0 / np.linalg.norm(r0) * LINK_LENGTH
        frames = np.stack([
            np.vstack([np.zeros(3), rodrigues(axis, OMEGA_TRUE * k * DT) @ r0])
            for k in range(120)
        ])
        result = self._run(frames)
        vecs = np.array(result["omega_vec"][10:])
        unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        assert float(np.median(unit @ axis)) == pytest.approx(1.0, abs=1e-3), (
            "角速度の向きが回転軸と一致していない"
        )
        assert float(np.median(result["omega"][10:])) == pytest.approx(OMEGA_TRUE, rel=0.02), (
            "z 軸以外の回転で角速度の大きさがずれた"
        )

    def test_pure_translation_gives_zero_angular_velocity(self):
        """向きが変わらない平行移動では角速度が 0 になる。"""
        frames = np.stack([
            np.vstack([np.array([0.0, 0.0, 0.1 * np.sin(3 * k * DT)]),
                       np.array([LINK_LENGTH, 0.0, 0.1 * np.sin(3 * k * DT)])])
            for k in range(120)
        ])
        got = float(np.max(self._run(frames)["omega"][10:]))
        assert got == pytest.approx(0.0, abs=1e-9), (
            f"平行移動なのに角速度が立った（実測 {got:.3e}）"
        )

    def test_acceleration_is_that_of_the_centre_of_mass(self):
        """返る加速度は重心のもので、リンクベクトルの 2 階微分ではない。"""
        got = float(np.median(self._run(rotating_link())["acc"][10:]))
        com_true = OMEGA_TRUE ** 2 * LINK_LENGTH / 2
        assert got == pytest.approx(com_true, rel=0.02), (
            f"重心加速度が真値 {com_true:.5f} m/s^2 から外れた（実測 {got:.5f}）"
        )
        assert got != pytest.approx(OMEGA_TRUE ** 2 * LINK_LENGTH, rel=0.1), (
            "リンクベクトルの 2 階微分（重心加速度の 2 倍）が返っている"
        )

    def test_acceleration_is_correct_when_the_proximal_end_moves(self):
        """始点が動く場合でも重心加速度になる（r̈ とは別物になるケース）。"""
        amp, w_shoulder = 0.05, 3.0
        frames = []
        for k in range(200):
            t = k * DT
            p0 = np.array([0.0, 0.0, amp * np.sin(w_shoulder * t)])
            p1 = p0 + np.array([LINK_LENGTH * np.cos(OMEGA_TRUE * t),
                                LINK_LENGTH * np.sin(OMEGA_TRUE * t), 0.0])
            frames.append(np.vstack([p0, p1]))
        frames = np.stack(frames)
        # 真値は中点の 2 階中心差分
        mid = (frames[:, 0] + frames[:, 1]) / 2
        truth = np.zeros_like(mid)
        truth[1:-1] = (mid[2:] - 2 * mid[1:-1] + mid[:-2]) / DT ** 2
        want = float(np.median(np.linalg.norm(truth[12:-12], axis=1)))
        got = float(np.median(self._run(frames)["acc"][12:-12]))
        assert got == pytest.approx(want, rel=0.05), (
            f"始点が動く系で重心加速度がずれた（実測 {got:.5f} / 真値 {want:.5f}）"
        )

    def test_noise_is_not_amplified(self):
        """位置ノイズを乗せても角速度が真値付近に留まる。

        旧式では 2 mm のノイズで真値と同じ大きさの偽信号が立っていた。
        """
        rng = np.random.default_rng(0)
        sigma = 0.002
        frames = []
        for k in range(200):
            theta = OMEGA_TRUE * k * DT
            frames.append(np.vstack([
                rng.normal(0, sigma, 3),
                np.array([LINK_LENGTH * np.cos(theta), LINK_LENGTH * np.sin(theta), 0.0])
                + rng.normal(0, sigma, 3),
            ]))
        got = float(np.median(self._run(np.stack(frames))["omega"][10:]))
        assert got == pytest.approx(OMEGA_TRUE, rel=0.15), (
            f"2 mm のノイズで角速度が {got:.4f} rad/s になった（真値 {OMEGA_TRUE}）。"
            " ノイズの外積を拾う旧式に戻っていないか確認すること"
        )


class TestCentreOfMassFraction:
    """R-4 重心比。中点固定ではなく体節ごとの文献値を使う。"""

    def _centroid(self, com_fraction: float) -> np.ndarray:
        from link_vector_calculator_module import LinkVectorCalculator

        # start=遠位(原点)、end=近位(x=1)。part_calculations と同じ向き。
        frames = [np.vstack([np.zeros(3), np.array([1.0, 0.0, 0.0])]) for _ in range(3)]
        calc = LinkVectorCalculator(0, 1, com_fraction)
        return calc.calculate_link_vectors(frames, 1, 1, DT)[3]

    def test_half_reproduces_the_midpoint(self):
        """0.5 なら従来どおり両端の中点になる（既存挙動との差分を切り分けるため）。"""
        assert self._centroid(0.5)[0] == pytest.approx(0.5), (
            "com_fraction=0.5 が中点にならない"
        )

    def test_fraction_is_measured_from_the_proximal_end(self):
        """重心比は近位端（end 側）から測る。"""
        frac = 0.436
        # 近位端は x=1 なので、そこから遠位（x=0）へ frac だけ寄る
        assert self._centroid(frac)[0] == pytest.approx(1.0 - frac), (
            f"重心が近位端から {frac} の位置に来ていない"
        )

    def test_gravity_moment_arm_shrinks_versus_the_midpoint(self):
        """文献値を使うと重力モーメント腕が中点より短くなる。

        近位端（関節）から重心までの距離がモーメント腕。中点なら 0.5、
        前腕の文献値なら 0.430 なので、中点は 16.3% 過大だった。
        """
        from config import COM_FRACTIONS

        # 近位端は x=1。そこから重心までの距離を測る。
        arm_mid = 1.0 - self._centroid(0.5)[0]
        arm_true = 1.0 - self._centroid(COM_FRACTIONS["forearm"])[0]
        assert arm_true < arm_mid, "文献値のモーメント腕が中点より長い"
        assert arm_mid / arm_true == pytest.approx(0.5 / COM_FRACTIONS["forearm"], rel=1e-9), (
            f"過大率が想定と違う（実測 {100 * (arm_mid / arm_true - 1):+.1f}%）"
        )


class TestInertiaLengthFromMedian:
    """R-6 慣性テンソルのリンク長は複数フレームの中央値で決める。"""

    def test_phone_path_uses_the_median_not_a_single_frame(self):
        """1 フレームだけ外れ値を混ぜても、確定するリンク長がほぼ動かない。"""
        from app.runners.network_measure import MeasurementConfig, NetworkMeasurement

        from config import pose_keypoints, slot_of

        config = MeasurementConfig(body_mass_kg=60.0)
        projection = np.hstack([np.eye(3), np.zeros((3, 1))])
        measurement = NetworkMeasurement(
            projection, projection, list(pose_keypoints), config)

        n = config.inertia_ready_frames
        # 関節はランドマーク名で置く。pose_keypoints の構成が変わっても追随する。
        clean = np.zeros((len(pose_keypoints), 3), dtype=float)
        for name, position in (
            ("L_SHOULDER", [-0.15, 0.0, 1.40]), ("R_SHOULDER", [0.15, 0.0, 1.40]),
            ("L_ELBOW", [-0.18, 0.0, 1.16]), ("L_WRIST", [-0.20, 0.0, 0.95]),
            ("L_HIP", [-0.10, 0.0, 1.00]), ("R_HIP", [0.10, 0.0, 1.00]),
            ("R_KNEE", [0.10, 0.0, 0.60]), ("R_ANKLE", [0.10, 0.0, 0.20]),
        ):
            clean[slot_of(name)] = position

        samples = np.stack([clean.copy() for _ in range(n)])
        measurement._build_inertia(samples)
        baseline = np.diag(measurement._inertia["forearm"]).copy()

        # 1 フレームだけ肘を大きく飛ばす（三角測量の外れ値を模す）
        spoiled = samples.copy()
        spoiled[n // 2, slot_of("L_ELBOW")] = [-1.50, 0.0, 1.16]
        measurement._inertia = {}
        measurement._build_inertia(spoiled)
        with_outlier = np.diag(measurement._inertia["forearm"])

        assert np.allclose(baseline, with_outlier, rtol=1e-9), (
            f"外れ値 1 フレームで慣性テンソルが動いた: {baseline} → {with_outlier}。"
            " 中央値ではなく平均や瞬時値を使っていないか確認すること"
        )

    def test_a_single_frame_would_have_been_wrong(self):
        """対照: その外れ値フレームだけで決めると値が大きく変わる。"""
        from utils_dynamic import calculate_inertia_tensor

        good = np.diag(calculate_inertia_tensor(4, 60.0, 0.21))
        bad = np.diag(calculate_inertia_tensor(4, 60.0, 1.32))
        assert not np.allclose(good, bad, rtol=0.1), (
            "外れ値のリンク長でも慣性テンソルが変わらない。テストの前提が崩れている"
        )


class TestTheoreticalWorkCoefficient:
    """R-5 理論仕事量の係数。2 系統で 9.3% 食い違っていたのを一本化した。"""

    def test_integral_matches_the_declared_angle_range(self):
        """係数は宣言した角度範囲の cos の積分そのもの。"""
        from config import WORK_ANGLE_RANGE_DEG, WORK_INTEGRAL_K

        low, high = np.radians(WORK_ANGLE_RANGE_DEG)
        assert WORK_INTEGRAL_K == pytest.approx(np.sin(high) - np.sin(low), rel=1e-12), (
            "係数が角度範囲から導かれていない"
        )

    def test_the_chosen_range_gives_the_verified_value(self):
        """-90°〜45° を採ったので √2/2 + 1 になる。"""
        from config import WORK_INTEGRAL_K

        assert WORK_INTEGRAL_K == pytest.approx(np.sqrt(2) / 2 + 1, rel=1e-12), (
            "係数が √2/2 + 1 から外れた。角度範囲を変えたなら"
            " 力学計算_検証結果.md の C 節と突き合わせること"
        )
        assert WORK_INTEGRAL_K != pytest.approx(np.sqrt(3) / 2 + 1, rel=1e-3), (
            "係数が √3/2 + 1 に戻っている（旧ゲージ閾値側の値）"
        )

    def test_coefficient_is_close_to_the_legacy_hardcoded_value(self):
        """従来の直書き 16.73 とほぼ一致する（g の桁を揃えた分だけ差が出る）。"""
        from config import THEORETICAL_WORK_COEFF

        assert THEORETICAL_WORK_COEFF == pytest.approx(16.73, rel=0.005), (
            f"係数が 16.73 から 0.5% 以上ずれた（実測 {THEORETICAL_WORK_COEFF:.4f}）"
        )

    def test_offline_and_realtime_agree(self):
        """ゲージ閾値側（CONST_K）と仕事量側（16.73 相当）が同じ積分係数を指す。"""
        import offline_wrist_energy
        from config import G_SCALAR, THEORETICAL_WORK_COEFF, WORK_INTEGRAL_K

        assert offline_wrist_energy.CONST_K == pytest.approx(WORK_INTEGRAL_K), (
            "offline_wrist_energy の CONST_K が config と食い違っている"
        )
        assert offline_wrist_energy.G == pytest.approx(G_SCALAR), (
            "重力加速度が config と食い違っている"
        )
        assert THEORETICAL_WORK_COEFF == pytest.approx(WORK_INTEGRAL_K * G_SCALAR), (
            "仕事量の係数が 積分係数 × g になっていない"
        )

    def test_effective_mass_coefficients_are_shared(self):
        """等価質量係数も 1 箇所で持つ。"""
        import offline_wrist_energy
        from config import EFFECTIVE_MASS_BY_JOINT

        assert offline_wrist_energy._DEF_COEFFS["wrist_R"] == pytest.approx(
            EFFECTIVE_MASS_BY_JOINT["wrist"]), "手首の等価質量係数が食い違っている"
        assert offline_wrist_energy._DEF_COEFFS["elbow_R"] == pytest.approx(
            EFFECTIVE_MASS_BY_JOINT["elbow"]), "肘の等価質量係数が食い違っている"


class TestInertiaRegressionGuard:
    """R-7 回帰式が適用範囲外で負を返したときのガード。"""

    def test_valid_range_is_untouched(self):
        """範囲内では警告も変更も起きない。"""
        import warnings

        from utils_dynamic import calculate_inertia_tensor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diag = np.diag(calculate_inertia_tensor(4, 60.0, 0.205))   # 前腕
        assert not caught, f"正常な入力で警告が出た: {[str(c.message) for c in caught]}"
        assert np.all(diag > 0), f"正常な入力で負の対角が出た: {diag}"

    def test_out_of_range_falls_back_to_a_uniform_rod(self):
        """範囲外では警告して一様棒近似に落ちる。"""
        import warnings

        from utils_dynamic import calculate_inertia_tensor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diag = np.diag(calculate_inertia_tensor(7, 60.0, 0.232))   # 下腿、範囲外
        assert caught, "範囲外なのに警告が出なかった"
        assert np.all(diag > 0), f"フォールバック後も負が残っている: {diag}"
        want = 60.0 * 0.0465 * 0.232 ** 2 / 12.0
        assert diag[0] == pytest.approx(want, rel=1e-9), (
            f"一様棒 m*L^2/12 = {want:.6f} になっていない（実測 {diag[0]:.6f}）"
        )

    def test_never_returns_a_negative_diagonal(self):
        """どの部位・どの長さでも負を返さない。"""
        import warnings

        from utils_dynamic import calculate_inertia_tensor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for row in range(9):
                for length in (0.05, 0.15, 0.25, 0.40, 0.60):
                    diag = np.diag(calculate_inertia_tensor(row, 60.0, length))
                    assert np.all(diag >= 0), (
                        f"行 {row}, L={length} で負の対角: {diag}"
                    )
