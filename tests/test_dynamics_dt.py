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

角速度の式そのものも 2026-09-08 に修正済み（R-2）。以前は ``cross(v_prev, v)/|r|²``
という次元 1/s² の別物を返しており、dt の効き方が一段強かった。
現在は標準形 ``(r × ṙ)/|r|²`` なので、下のスケーリングも素直な冪になる。
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

    直線的に往復する運動では r と ṙ が平行になり ``cross(r, ṙ)`` が 0 になる。
    これは物理的に正しい（向きが変わらないなら角速度は 0）が、
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
    dt 修正の勘所なので、退行しないようテストで留める。

    角速度は標準形 ``ω = (r × ṙ)/|r|²``（R-2 で修正済み）。``ṙ = Δr/dt`` が
    1 回だけ入るので ``ω ∝ 1/dt``、``ω̇ = Δω/dt`` でもう 1 つ乗って ``ω̇ ∝ 1/dt²``。

    修正前は ``cross(v_prev, v)/|r|²`` という別式で、``v ∝ 1/dt`` が 2 回入り
    ``ω ∝ 1/dt²``・``ω̇ ∝ 1/dt³`` と一段強くスケールしていた。
    既知の回転（ω=2.0 rad/s, L=0.25 m, dt=1/30）に対して **0.2664**、すなわち
    ``ω³·dt`` を返しており、次元も 1/s² で角速度になっていなかった。
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

    def test_angular_velocity_matches_the_true_value(self):
        """既知の一定回転を与えると、角速度の真値がそのまま返る。"""
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
        assert measured == pytest.approx(w_true, rel=0.02), (
            f"角速度が真値 {w_true} rad/s から外れた（実測 {measured:.4f}）。"
            f" 外積の第 1 引数がリンクベクトルになっているか確認すること"
        )
        assert measured != pytest.approx(w_true**3 * dt, rel=0.1), (
            f"角速度が ω³·dt ({w_true**3 * dt:.4f}) に戻っている。"
            f" cross(v_prev, v) を使う旧式が復活していないか確認すること"
        )

    def test_angular_velocity_scales_inversely_with_dt(self):
        # (r × ṙ)/|r|² に ṙ = Δr/dt が 1 回だけ入るので 1/dt。
        # 旧式 cross(v_prev, v) では v が 2 回入り 1/dt² だった。
        w_fast, _ = self._series(1 / 30.0)
        w_slow, _ = self._series(0.3)
        assert len(w_fast) > 10 and len(w_slow) > 10, "角速度が十分に得られていない"
        ratio = float(np.median(w_slow[5:]) / np.median(w_fast[5:]))
        assert ratio == pytest.approx(1 / 9, rel=0.05), (
            f"dt を 9 倍にしたとき角速度が 1/9 になっていない（実測 {ratio:.5f} 倍）"
        )

    def test_angular_acceleration_scales_with_the_square_of_dt(self):
        # ω̇ = Δω/dt なので、ω の 1/dt とあわせて 1/dt²。
        # 慣性トルク I·ω̇ もここに乗るため、dt 誤りは二乗で効く。
        _, a_fast = self._series(1 / 30.0)
        _, a_slow = self._series(0.3)
        assert len(a_fast) > 10 and len(a_slow) > 10, "角加速度が十分に得られていない"
        ratio = float(np.median(a_slow[5:]) / np.median(a_fast[5:]))
        assert ratio == pytest.approx(1 / 81, rel=0.05), (
            f"dt を 9 倍にしたとき角加速度が 1/81 になっていない（実測 {ratio:.6f} 倍）"
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
