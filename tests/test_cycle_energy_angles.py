"""スコア経路の関節の仕事率が、正しい角速度・正しい軸・正しい符号で出ることを固定する。

**なぜこのテストがあるか。**

``compute_cycle_energy_elbow_wrist.py`` のサイクル仕事量は論文のスコアそのもの。
これまでに次の誤りがあった。

- §1-1 角速度に fps を掛けていた（ω が一律 30 倍。実測で平均 47.5 rad/s）
- §1-2 ``arctan2`` の角度を unwrap せずに微分していた（折り返しごとに 2π/dt のスパイク）
- 2026-09-13 に判明: 局所トルク ``*_local_y`` に、軸の作り方が別の角速度
  （肘は「+Y まわりの肘角」、手首は「水平面からの前腕の傾き」の微分）を掛けていた。
  左右を鏡映すると片方だけ符号が反転し、左右で逆の相を積算していた
  （被験者 3 を左右反転すると、左の W_pos 301.94 J が右の W_neg と一致した）

いまは、トルクを出した鎖（``--wrist-base``: 手を固定端として前腕 → 上腕）と同じ部位の
相対角速度を、トルクと同じ局所軸に射影して掛ける（``utils.compute_joint_power``）。
角度を経由しないので、fps の掛け戻しも折り返しも構造的に起きない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compute_cycle_energy_elbow_wrist import LEFT, RIGHT, _joint_powers

DT = 1.0 / 30.0
UPPER_ARM = 0.30
FOREARM = 0.25
BODY_MASS = 60.0
GRAVITY = np.array([0.0, 0.0, -9.81])
MIRROR = np.array([-1.0, 1.0, 1.0])


def _pose_df(side: dict, wrist: np.ndarray, elbow: np.ndarray, shoulder: np.ndarray) -> pd.DataFrame:
    """姿勢 CSV と同じ列（joint_{id}_{axis}）の DataFrame を作る。"""
    columns = {}
    for name, series in (("wrist", wrist), ("elbow", elbow), ("shoulder", shoulder)):
        for axis, label in enumerate("xyz"):
            columns[f"joint_{side[name]}_{label}"] = series[:, axis]
    return pd.DataFrame(columns)


def _torque_df(side_name: str, elbow: np.ndarray, wrist: np.ndarray) -> pd.DataFrame:
    """トルク CSV と同じ列（{part}_{side}_{axis}、全体座標）の DataFrame を作る。"""
    columns = {}
    for part, series in (("elbow", elbow), ("wrist", wrist)):
        for axis, label in enumerate("xyz"):
            columns[f"{part}_{side_name}_{label}"] = series[:, axis]
    return pd.DataFrame(columns)


def _upper_arm_rotating_about_y(n: int, omega: float, theta0: float = 0.3):
    """手首と肘を固定し（前腕は鉛直）、上腕だけが y 軸まわりに角速度 omega で回る。

    肩 = 肘 + L (sin θ, 0, cos θ)、θ = θ0 + ω t なので、上腕の r × ṙ / |r|² = (0, ω, 0)。
    前腕は動かないので、肘の相対角速度も (0, ω, 0)。
    """
    theta = theta0 + omega * DT * np.arange(n)
    wrist = np.zeros((n, 3))
    elbow = np.tile([0.0, 0.0, FOREARM], (n, 1))
    shoulder = elbow + UPPER_ARM * np.stack([np.sin(theta), np.zeros(n), np.cos(theta)], axis=1)
    return wrist, elbow, shoulder


def _elbow_power_under_constant_torque(n: int, omega: float, dt: float = DT) -> np.ndarray:
    wrist, elbow, shoulder = _upper_arm_rotating_about_y(n, omega)
    torque = np.tile([0.0, 2.0, 0.0], (n, 1))
    elbow_power, _ = _joint_powers(
        _pose_df(RIGHT, wrist, elbow, shoulder), _torque_df("R", torque, np.zeros((n, 3))), "R", dt)
    return elbow_power


def _lifting_motion(n: int = 121):
    """前腕を鉛直に固定し、上腕を 30° → 90° へ静止から静止まで持ち上げる（x-z 平面内）。"""
    progress = (1.0 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2.0
    phi = np.pi / 6 + (np.pi / 3) * progress
    wrist = np.zeros((n, 3))
    elbow = np.tile([0.0, 0.0, FOREARM], (n, 1))
    shoulder = elbow + UPPER_ARM * np.stack([np.cos(phi), np.zeros(n), np.sin(phi)], axis=1)
    return wrist, elbow, shoulder


def _reaching_motion(n: int = 90):
    """手首を固定し、前腕と上腕の両方が面に乗らずに動く。"""
    t = DT * np.arange(n)
    a = 0.3 + 0.25 * np.sin(2 * np.pi * 0.8 * t)
    b = 0.2 * np.sin(2 * np.pi * 0.5 * t + 0.4)
    phi = 0.9 + 0.3 * np.sin(2 * np.pi * 0.8 * t + 1.0)
    wrist = np.zeros((n, 3))
    elbow = FOREARM * np.stack([np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)], axis=1)
    direction = np.stack([np.cos(phi), np.full(n, 0.3), np.sin(phi)], axis=1) / np.sqrt(1.09)
    return wrist, elbow, elbow + UPPER_ARM * direction


def _wrist_base_powers(wrist, elbow, shoulder, side_name: str, load_kg: float):
    """--wrist-base と同じ鎖でトルクを出し、そのトルク CSV 相当から仕事率を求める。"""
    from compute_torque_from_pose import (
        WRIST_BASE_SEGMENTS_LEFT,
        WRIST_BASE_SEGMENTS_RIGHT,
        compute_side_torques,
    )

    side = RIGHT if side_name == "R" else LEFT
    segments = WRIST_BASE_SEGMENTS_RIGHT if side_name == "R" else WRIST_BASE_SEGMENTS_LEFT
    n = len(wrist)
    pose = np.zeros((n, 17, 3))
    pose[:, side["wrist"]], pose[:, side["elbow"]], pose[:, side["shoulder"]] = wrist, elbow, shoulder
    tau, _ = compute_side_torques(
        pose, segments, BODY_MASS, DT, GRAVITY,
        external_force=np.tile(load_kg * GRAVITY, (n, 1)), external_point=shoulder)
    return _joint_powers(
        _pose_df(side, wrist, elbow, shoulder), _torque_df(side_name, tau[:, 1], tau[:, 0]), side_name, DT)


class TestAngularVelocityScale:
    """§1-1 仕事率は rad/s の角速度で決まり、fps を掛けない。"""

    def test_power_is_torque_times_angular_velocity(self):
        # τ = (0, 2, 0) N·m、ω_rel = (0, 1.5, 0) rad/s、局所 y 軸 = ±ŷ → P = 3.0 W
        # （中心差分は正弦波の振幅を sin(ωdt)/(ωdt) = 0.9996 倍にする）
        power = _elbow_power_under_constant_torque(60, omega=1.5)
        got = float(np.median(power[2:-2]))
        assert got == pytest.approx(3.0, rel=0.01), (
            f"仕事率が τω = 3.0 W から外れた（{got:.4f}）。"
            " 30 倍（90 W）なら角速度に fps を掛け戻している（§1-1 の再発）"
        )

    def test_the_sampling_interval_is_honoured(self):
        fast = float(np.median(_elbow_power_under_constant_torque(60, 1.5, DT)[2:-2]))
        slow = float(np.median(_elbow_power_under_constant_torque(60, 1.5, DT * 2)[2:-2]))
        assert fast / slow == pytest.approx(2.0, rel=0.01), "dt を 2 倍にしたのに仕事率が半分になっていない"


class TestNoWrappingSpikes:
    """§1-2 何回転しても角速度にスパイクが立たない。"""

    def test_power_stays_constant_over_several_turns(self):
        # ω = 6 rad/s で 90 フレーム（約 2.9 回転）。P = 2 × 6 × sin(0.2)/0.2 = 11.92 W
        power = _elbow_power_under_constant_torque(90, omega=6.0)[2:-2]
        np.testing.assert_allclose(power, 11.92, rtol=0.01, err_msg=(
            "回転を重ねると仕事率が一定でなくなる。角度の折り返しを微分していないか確認すること（§1-2）"))


class TestLiftingTheTrunk:
    """手を固定端として体幹を持ち上げる動きで、仕事の大きさと符号が物理と合う。"""

    def test_positive_elbow_work_equals_the_potential_energy_gained(self):
        # 持ち上げる位置エネルギー:
        #   荷重 20 kg × g × 肩の上昇 0.30 (sin 90° − sin 30°) = 20 × 9.81 × 0.15
        #   上腕 60 × 0.0227 kg × g × 重心の上昇 0.564 × 0.15（重心は肘から 0.564 L）
        #   合計 9.81 × (3.0 + 1.362 × 0.0846) = 30.56 J
        # 静止から静止までなので運動エネルギーの差は 0。
        elbow_power, _ = _wrist_base_powers(*_lifting_motion(), "R", load_kg=20.0)
        work_pos = float(np.sum(np.clip(elbow_power, 0, None)) * DT)
        work_neg = float(np.sum(np.clip(elbow_power, None, 0)) * DT)
        assert work_pos == pytest.approx(30.56, rel=0.02), (
            f"持ち上げたときの肘の正の仕事 {work_pos:.2f} J が位置エネルギーの増加 30.56 J と合わない。"
            " 負なら相対角速度の向き（リンク − 親）が逆"
        )
        assert abs(work_neg) < 0.3, f"単調に持ち上げているのに負の仕事 {work_neg:.2f} J が出た"

    def test_a_still_forearm_does_no_work_at_the_wrist(self):
        # 手は固定端、前腕は動かないので手首の相対角速度は 0
        _, wrist_power = _wrist_base_powers(*_lifting_motion(), "R", load_kg=20.0)
        assert float(np.sum(np.abs(wrist_power)) * DT) < 1e-9, "前腕が静止しているのに手首が仕事をした"


def _forearm_tilting_about_the_wrist(n: int = 20, omega: float = 0.3, theta0: float = 0.05):
    """手首を固定し、鉛直に近い前腕が x-z 面（腕の面）内で手首まわりに ω で傾く。肘角は一定。

    前腕 r = L (sin θ, 0, cos θ) なので r × ṙ / |r|² = (0, ω, 0)。
    """
    theta = theta0 + omega * DT * np.arange(n)
    wrist = np.zeros((n, 3))
    elbow = FOREARM * np.stack([np.sin(theta), np.zeros(n), np.cos(theta)], axis=1)
    shoulder = elbow + UPPER_ARM * np.stack([np.sin(theta + 0.8), np.zeros(n), np.cos(theta + 0.8)], axis=1)
    return wrist, elbow, shoulder


def _wrist_power_with_hand(hand_offset=None, omega: float = 0.3) -> np.ndarray:
    wrist, elbow, shoulder = _forearm_tilting_about_the_wrist(omega=omega)
    n = len(wrist)
    pose = _pose_df(RIGHT, wrist, elbow, shoulder)
    if hand_offset is not None:
        # 右小指 18・右人差指 20。中点が wrist + hand_offset になるよう少しずらして置く
        for jid, jitter in ((18, -0.01), (20, 0.01)):
            point = wrist + np.asarray(hand_offset) + np.array([jitter, 0.0, 0.0])
            for axis, label in enumerate("xyz"):
                pose[f"joint_{jid}_{label}"] = point[:, axis]
    torque = _torque_df("R", np.zeros((n, 3)), np.tile([0.0, 2.0, 0.0], (n, 1)))
    _, wrist_power = _joint_powers(pose, torque, "R", DT)
    return wrist_power


class TestWristAxis:
    """§5-1 手首の仕事率は手首の屈曲軸で取る。

    かつて手首の局所軸は、前腕リンクと全体座標の基準軸（z → x → y の順）から作っていた。
    前腕が鉛直に近いと局所 y が水平の面内方向を向き、腕の面内で前腕が倒れる動き
    （プッシュアップの手首の屈曲そのもの）の仕事が 0 になっていた。
    """

    def test_without_hand_points_the_arm_plane_normal_is_used(self):
        # τ = (0, 2, 0)、ω_前腕 = (0, 0.3, 0)、手は固定 → P = 0.6 W
        power = _wrist_power_with_hand(None)
        assert float(np.median(power[2:-2])) == pytest.approx(0.6, rel=0.01), (
            "手の点が無いとき、手首の軸が腕の面の法線（肘と同じ屈曲軸）になっていない")

    def test_hand_points_set_the_axis_from_the_palm(self):
        """手が腕の面から 45° 横へ出ていれば、手首の軸（前腕と手に直交）も 45° 回る。

        手 = (a, a, 0)、前腕 ≈ ẑ のとき y ∝ 手 × 前腕 ∝ (−a, a, 0)/√2。τ_y と ω_y がそれぞれ
        1/√2 倍になるので P ≈ 0.6 / 2 = 0.3 W（前腕の傾き分だけわずかに小さい）。
        腕の面の法線なら 0.6 W、全体座標の基準軸なら 0 W になる。
        """
        power = _wrist_power_with_hand([0.06, 0.06, 0.0])
        assert float(np.median(power[2:-2])) == pytest.approx(0.3, rel=0.03), (
            "手の点があるのに手のひら（前腕と手に直交する軸）から軸を作っていない")


class TestMirrorSymmetry:
    """左右を鏡映した動きなら、左の仕事率は右と同じ。"""

    def test_left_matches_right_for_mirrored_motion(self):
        wrist, elbow, shoulder = _reaching_motion()
        right = _wrist_base_powers(wrist, elbow, shoulder, "R", load_kg=17.0)
        left = _wrist_base_powers(wrist * MIRROR, elbow * MIRROR, shoulder * MIRROR, "L", load_kg=17.0)
        assert np.max(np.abs(right[1])) > 1e-3, "比較の前提として、手首が仕事をする動きにすること"
        for name, r, l in (("肘", right[0], left[0]), ("手首", right[1], left[1])):
            np.testing.assert_allclose(l, r, rtol=1e-9, atol=1e-9, err_msg=(
                f"{name}: 鏡映した動きで左の仕事率が右と一致しない。"
                " トルクと角速度を別の軸に射影すると、左右で逆の相を積算する"))


class TestInertiaTensorArgument:
    """§1-3 慣性回帰式には全身体重を渡す。"""

    def test_inertia_tensors_are_positive_definite(self):
        """部位質量ではなく体重を渡すので、対角成分が正になる。"""
        from compute_torque_from_pose import RIGHT_SEGMENTS, build_side_inverse_inputs

        n = 60
        pose = np.zeros((n, 17, 3), dtype=float)
        for k in range(n):
            pose[k, 12] = [0.0, 0.0, 0.0]        # 右肩
            pose[k, 14] = [UPPER_ARM, 0.0, 0.0]  # 右肘
            pose[k, 16] = [UPPER_ARM + FOREARM, 0.0, 0.0]  # 右手首
        inertia, _, _, _, _, _, _, _ = build_side_inverse_inputs(
            pose, RIGHT_SEGMENTS, body_mass=60.0, dt=DT, gravity=np.array([0.0, 0.0, -9.81]))
        for idx, seg in enumerate(RIGHT_SEGMENTS):
            diag = np.diag(inertia[idx])
            assert np.all(diag > 0), (
                f"{seg.name} の慣性テンソル対角に負がある: {diag}。"
                " 回帰式 I = a*w + b*l + c の w に部位質量を渡していないか確認すること"
            )
