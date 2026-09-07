"""力学計算のサンプル間隔 ``dt`` が正しく決まることを固定する。

**なぜこのテストがあるか。**

``config.py`` には長らく ``dt = 0.3  # 0.1秒ごと`` と書かれていた。値もコメントも
``fps = 30`` も三者三様に食い違っており、初期コミットから一度も検証されていなかった。
この値が ``master_research_code.py`` の EKF・リンクベクトル計算・エネルギー積分の
すべてに配られていた。

微分と積分が必要とするのは、カメラのフレーム間隔ではなく
**連続して「処理される」フレームの実時間間隔**である。実行時は既定でフレームを
間引くため（``RT_POSE_FIXED_HZ_ON`` が既定 ON で 4Hz）、正しい値は設定に依存する。

| 設定 | 処理間隔 | 旧 ``0.3`` の誤差 |
|---|---|---|
| 素の既定（4Hz 間引き） | 0.267s | 12% 過大 |
| アプリ既定（間引きなし） | 0.0333s | **9 倍過大** |

``1/30`` への一律置換は素の設定では 8 倍小さすぎるので、
``config.resolve_dynamics_dt`` が間引き係数から決める。

影響の実測（右前腕、30Hz 収録データ、``(r × ṙ)/|r|²`` の**正しい**角速度で計算）:

- 角速度 |ω| 中央値 0.072 → 0.652 rad/s
- 慣性トルク 中央値 ≈0.000 → 0.014 N·m（重力トルク 0.803 N·m の 1.7%）

修正後の 1.7% は ``KNOWN_ISSUES.md`` §2-3 が独立に述べている「慣性項の寄与は
1.7〜4.6%」と一致する。**旧値は慣性項を実質ゼロに潰していた。**

ただしリポジトリの実装は角速度に別の式を使っており、dt の効き方が一段強い。
詳細は :class:`TestScalingLaws` の注記を参照。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import resolve_dynamics_dt


class TestResolveDynamicsDt:
    """間引き設定から処理フレーム間隔を決める。"""

    def test_fixed_hz_skipping_widens_the_interval(self):
        # RT_POSE_FIXED_HZ_ON=1 / 4Hz → 8 フレームに 1 回だけ処理する
        value, _ = resolve_dynamics_dt(30.0, fixed_hz_on=True, fixed_skip=7)
        assert value == pytest.approx(8 / 30.0), "4Hz 間引き時の間隔が 8 フレーム分になっていない"

    def test_no_skipping_gives_the_frame_interval(self):
        # アプリの CURATED はこのフラグを 0 に落とすので、こちらが本番の既定
        value, _ = resolve_dynamics_dt(30.0, fixed_hz_on=False, fixed_skip=7)
        assert value == pytest.approx(1 / 30.0), "間引き無効時に 1/fps になっていない"

    def test_skip_frames_is_honoured(self):
        value, _ = resolve_dynamics_dt(30.0, fixed_hz_on=False, fixed_skip=7, skip_mod=3)
        assert value == pytest.approx(3 / 30.0), "SKIP_FRAMES による間引きが反映されていない"

    def test_override_wins(self):
        # 旧挙動の再現手段。既発表 CSV を作り直して照合するために残してある
        value, source = resolve_dynamics_dt(30.0, fixed_hz_on=True, fixed_skip=7, override="0.3")
        assert value == pytest.approx(0.3), "DT_SEC による上書きが効いていない"
        assert "override" in source, "由来の説明に上書きであることが出ていない"

    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_non_positive_override_is_rejected(self, bad):
        # 0 や負の dt は 0 除算や時間反転を静かに引き起こすので、入口で弾く
        with pytest.raises(ValueError):
            resolve_dynamics_dt(30.0, fixed_hz_on=False, fixed_skip=0, override=bad)

    def test_unusable_fps_falls_back_to_config(self):
        # cap.get(CAP_PROP_FPS) は 0 や -1 を返すことがある（特にファイル入力）
        value, _ = resolve_dynamics_dt(0.0, fixed_hz_on=False, fixed_skip=0)
        assert value == pytest.approx(1 / 30.0), "fps が取れないときに config.fps へ落ちていない"

    def test_source_string_explains_where_the_value_came_from(self):
        # 起動時に [DT] 行として出る。実機で「なぜこの値か」を追えるようにするため
        _, source = resolve_dynamics_dt(30.0, fixed_hz_on=True, fixed_skip=7)
        assert "8frame" in source and "30" in source, f"由来が読み取れない: {source}"


def _rotating_link(n: int, dt: float, omega: float = 2.0, length: float = 0.25):
    """始点固定・終点が z 軸まわりに一定角速度で回るリンクの点列。単位は m。

    直線的に往復する運動では ``v_prev`` と ``v`` が平行になり、
    ``cross(v_prev, v)`` が恒等的にゼロになってしまう（実装が外積を使うため）。
    スケーリングを見るには本物の回転が要る。
    """
    pts = []
    for k in range(n):
        th = omega * k * dt
        pts.append(
            np.vstack([np.zeros(3), np.array([length * np.cos(th), length * np.sin(th), 0.0])])
        )
    return pts


class TestScalingLaws:
    """``dt`` を変えたときに各量がどう動くかを固定する。

    解析的には自明な関係だが、**どの量が dt に比例し、どの量が相殺するのか**が
    今回の修正の勘所なので、退行しないようテストで留める。

    .. warning::
       ``link_vector_calculator_module.py:100`` の角速度は
       ``cross(v_prev, v) / |r|²`` で計算されており、標準形の ``(r × ṙ) / |r|²``
       ではない。既知の回転（ω=2.0 rad/s, L=0.25m, dt=1/30）を与えると
       コードは **0.2664** を返す。これは ``ω³·dt = 8 × 0.0333 = 0.2667`` に一致する。
       つまりこの量は角速度ではなく ``ω³·dt`` で、次元が 1/s²、しかも ω の 3 乗。

       ``KNOWN_ISSUES.md`` §2-4 はこれを ``(r × ṙ)/|r|²`` だと想定して
       「式の展開自体は正しい（検算済み）」と書いており、**この食い違いは
       未カタログ**。物理式の修正は本作業のスコープ外なので、ここでは
       現状の挙動を固定するにとどめる。式を直したら下の 2 つが落ちるので、
       そのときにこの注記ごと更新すること。

       この式のため、``v ∝ 1/dt`` が 2 回入って **ω は 1/dt ではなく 1/dt²**
       でスケールする。dt 誤りの影響が正しい式より一段強く出る。
    """

    def _series(self, dt: float, n: int = 120):
        from link_vector_calculator_module import LinkVectorCalculator

        calc = LinkVectorCalculator(0, 1)
        pts = _rotating_link(n, 1 / 30.0)  # 運動そのものは常に 30Hz 刻み
        omegas, ang_accs = [], []
        for i in range(n):
            result = calc.calculate_link_vectors(pts[: i + 1], True, i, dt)
            if result[0] is None:
                continue
            omega, ang_acc = result[2], result[6]
            if omega is not None and np.all(np.isfinite(omega)):
                omegas.append(np.linalg.norm(np.asarray(omega, dtype=float)))
            if ang_acc is not None and np.all(np.isfinite(ang_acc)):
                ang_accs.append(np.linalg.norm(np.asarray(ang_acc, dtype=float)))
        return np.array(omegas), np.array(ang_accs)

    def test_angular_velocity_formula_is_the_cubic_one(self):
        """コードの「角速度」が ``ω³·dt`` であることを固定する。

        既知の一定回転を与えて確かめる。直したらここが落ちる。
        """
        from link_vector_calculator_module import LinkVectorCalculator

        w_true, length, dt, n = 2.0, 0.25, 1 / 30.0, 60
        pts = []
        for k in range(n):
            th = w_true * k * dt
            pts.append(
                np.vstack([np.zeros(3), np.array([length * np.cos(th), length * np.sin(th), 0.0])])
            )
        calc = LinkVectorCalculator(0, 1)
        got = []
        for i in range(n):
            result = calc.calculate_link_vectors(pts[: i + 1], True, i, dt)
            if result[0] is None or result[2] is None:
                continue
            if np.all(np.isfinite(result[2])):
                got.append(abs(result[2][2]))
        measured = float(np.median(got[5:]))
        assert measured == pytest.approx(w_true**3 * dt, rel=0.02), (
            f"角速度が ω³·dt ({w_true**3 * dt:.4f}) から外れた（実測 {measured:.4f}）。"
            f" 式を直したならこのテストと docstring を更新すること"
        )
        assert measured != pytest.approx(w_true, rel=0.1), (
            "角速度が真値と一致した。式が直ったなら docstring の注記ごと更新すること"
        )

    def test_angular_velocity_scales_with_the_square_of_dt(self):
        # cross(v_prev, v) に v = Δr/dt が 2 回入るので 1/dt²。
        # 正しい式 (r × ṙ)/|r|² なら 1/dt だった。
        w_fast, _ = self._series(1 / 30.0)
        w_slow, _ = self._series(0.3)
        assert len(w_fast) > 10 and len(w_slow) > 10, "角速度が十分に得られていない"
        ratio = float(np.median(w_slow[5:]) / np.median(w_fast[5:]))
        assert ratio == pytest.approx(1 / 81, rel=0.05), (
            f"dt を 9 倍にしたとき角速度が 1/81 になっていない（実測 {ratio:.5f} 倍）"
        )

    def test_angular_acceleration_adds_one_more_power_of_dt(self):
        # ω̇ = Δω/dt なので、ω の 1/dt² とあわせて 1/dt³。
        # 慣性トルク I·ω̇ もここに乗るため、dt 誤りは三乗で効く。
        _, a_fast = self._series(1 / 30.0)
        _, a_slow = self._series(0.3)
        assert len(a_fast) > 10 and len(a_slow) > 10, "角加速度が十分に得られていない"
        ratio = float(np.median(a_slow[5:]) / np.median(a_fast[5:]))
        assert ratio == pytest.approx(1 / 729, rel=0.05), (
            f"dt を 9 倍にしたとき角加速度が 1/729 になっていない（実測 {ratio:.6f} 倍）"
        )

    def test_impulse_scales_directly_with_dt(self):
        """力積は ``Σ τ·dt`` なので dt に比例し、**相殺しない**。

        エネルギー ``Σ τ·ω·dt`` は ``ω = Δθ/dt`` と相殺するのに対し、
        力積は dt 誤りがそのまま出力の誤りになる。ここが両者の決定的な違い。
        """
        from utils_dynamic import compute_impulse

        torque = pd.Series(np.sin(np.linspace(0, 8 * np.pi, 300)))
        pos_fast, neg_fast = compute_impulse(torque, 1 / 30.0)
        pos_slow, neg_slow = compute_impulse(torque, 0.3)
        assert pos_slow / pos_fast == pytest.approx(9.0, rel=1e-6), "正の力積が dt に比例していない"
        assert neg_slow / neg_fast == pytest.approx(9.0, rel=1e-6), "負の力積が dt に比例していない"
