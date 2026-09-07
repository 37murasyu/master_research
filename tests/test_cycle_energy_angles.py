"""サイクル仕事量の角度と角速度が正しく出ることを固定する。

**なぜこのテストがあるか。**

``力学計算_検証結果.md`` §A-1・§A-2（= ``KNOWN_ISSUES.md`` §1-1・§1-2）で確定した
2 つの誤りを、再発しないよう留める。この 2 つは論文のスコアを **約 350 倍** 押し上げていた。

- §1-1 ``_gradient(angle, dt)`` の戻り値は既に rad/s なのに、さらに fps を掛けていた。
  ω が一律 30 倍になり、実測で平均 47.5 rad/s（= 7.6 回転/秒）という肘では起こり得ない値が出ていた。
- §1-2 関節角が ``arctan2`` 由来で値域 ±π なのに unwrap していなかった。
  折り返し 1 回につき ω に ``2π/dt`` のスパイクが立ち、仕事量が ``max(τω, 0)`` と
  正側だけを拾う（整流する）ため、符号がランダムなスパイクでも必ず加算されていた。

検証系は「肩と肘を固定し、手首を y 軸まわりに一定角速度で回す」合成データ。
``_angle_about_y`` の定義から肘角度は ``-θ`` になるので、角速度の大きさが真値と一致するはず。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compute_cycle_energy_elbow_wrist import RIGHT, _compute_angles, _compute_omega, _gradient

DT = 1.0 / 30.0
UPPER_ARM = 0.30
FOREARM = 0.25


def _swinging_forearm(n: int, omega: float, dt: float = DT) -> pd.DataFrame:
    """肩と肘を固定し、手首が y 軸まわりに角速度 omega で回る点列。

    肘を原点、肩を +x 方向に置く。手首は xz 平面内を回るので
    ``_angle_about_y`` が拾う軸と一致する。
    """
    rows = []
    for k in range(n):
        theta = omega * k * dt
        elbow = np.zeros(3)
        shoulder = np.array([UPPER_ARM, 0.0, 0.0])
        wrist = np.array([FOREARM * np.cos(theta), 0.0, FOREARM * np.sin(theta)])
        row = {}
        for idx, point in ((RIGHT["shoulder"], shoulder),
                           (RIGHT["elbow"], elbow),
                           (RIGHT["wrist"], wrist)):
            row[f"joint_{idx}_x"], row[f"joint_{idx}_y"], row[f"joint_{idx}_z"] = point
        rows.append(row)
    return pd.DataFrame(rows)


class TestAngularVelocityScale:
    """§1-1 角速度に fps を掛けてはいけない。"""

    def test_computed_omega_is_in_radians_per_second(self):
        """既知の一定角速度を与えると、真値がそのまま出る。"""
        omega_true = 1.5
        pose = _swinging_forearm(90, omega_true)
        elbow_omega, _ = _compute_omega(pose, RIGHT, DT)
        got = float(np.median(np.abs(elbow_omega)[5:-5]))
        assert got == pytest.approx(omega_true, rel=0.02), (
            f"角速度が真値 {omega_true} rad/s から外れた（実測 {got:.4f}）。"
            " _gradient に fps を掛け戻していないか確認すること"
        )

    def test_angular_velocity_is_not_scaled_by_fps(self):
        """fps を掛けた値（30 倍）になっていないことを明示的に否定する。"""
        omega_true = 1.5
        pose = _swinging_forearm(90, omega_true)
        elbow_omega, _ = _compute_omega(pose, RIGHT, DT)
        got = float(np.median(np.abs(elbow_omega)[5:-5]))
        assert got != pytest.approx(omega_true * 30.0, rel=0.1), (
            f"角速度が真値の 30 倍（{omega_true * 30:.1f} rad/s）になっている。"
            " これは KNOWN_ISSUES §1-1 の再発"
        )

    def test_the_sampling_interval_is_honoured(self):
        """dt を変えれば角速度もその分だけ変わる（積分係数として効いている）。"""
        omega_true = 1.5
        pose = _swinging_forearm(90, omega_true)
        fast = float(np.median(np.abs(_compute_omega(pose, RIGHT, DT)[0])[5:-5]))
        slow = float(np.median(np.abs(_compute_omega(pose, RIGHT, DT * 2)[0])[5:-5]))
        assert fast / slow == pytest.approx(2.0, rel=0.01), (
            "dt を 2 倍にしたのに角速度が半分になっていない"
        )


class TestAngleUnwrapping:
    """§1-2 関節角は unwrap してから微分する。"""

    def test_angles_are_unwrapped(self):
        """±π をまたいでも角度が連続になる。"""
        # 3 回転させれば必ず折り返しが起きる
        pose = _swinging_forearm(200, omega=3.0 * 2 * np.pi / (200 * DT) * 3)
        elbow_angle, _ = _compute_angles(pose, RIGHT)
        span = float(np.max(elbow_angle) - np.min(elbow_angle))
        assert span > 2 * np.pi, (
            f"角度の振れ幅が {span:.2f} rad しかない。unwrap されていれば"
            " 複数回転で 2π を超えるはず（±π に折り返されている）"
        )

    def test_no_spike_from_wrapping(self):
        """折り返し由来の巨大な角速度スパイクが出ない。"""
        omega_true = 6.0   # 1 秒で約 1 回転。90 フレームで 3 回転する
        pose = _swinging_forearm(90, omega_true)
        omega = np.abs(_compute_omega(pose, RIGHT, DT)[0])[3:-3]
        spike_threshold = 2 * np.pi / DT * 0.5   # 折り返し 1 回分の半分
        assert float(np.max(omega)) < spike_threshold, (
            f"角速度の最大が {np.max(omega):.1f} rad/s に達している"
            f"（折り返しスパイクの目安 {spike_threshold:.1f}）。unwrap が効いていない"
        )
        assert float(np.median(omega)) == pytest.approx(omega_true, rel=0.05), (
            f"折り返しを含む系列で角速度の中央値がずれた（実測 {np.median(omega):.3f}）"
        )


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
