#!/usr/bin/env python3
"""``dt`` を変えたときに力学量がどれだけ変わるかを表にする。

なぜ必要か。``config.py`` には長らく ``dt = 0.3`` と書かれており、実行時の
EKF・リンクベクトル計算・エネルギー積分のすべてにこの値が配られていた。
実際の処理フレーム間隔は間引き設定に依存し、アプリの既定（間引き無効）では
1/30 = 0.0333 秒である。**旧値は 9 倍過大だった。**

修正すると出力の数値が変わる。どの量がどれだけ変わるのかを、収録済みの
実データで確かめられるようにするのがこのスクリプト。修論の既発表値を
どう訂正するかの判断材料になる。

物理計算は既存モジュールをそのまま使う（再実装しない）:
``link_vector_calculator_module.LinkVectorCalculator``、
``utils_dynamic.calculate_inertia_tensor``、``utils_dynamic.compute_impulse``。

使い方::

    python tools/compare_dt.py --input "output_data/kpts3d_XXXX.csv"
    python tools/compare_dt.py --input ... --old-dt 0.3 --new-dt 0.0333
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import resolve_dynamics_dt, w as BODY_MASS_DEFAULT  # noqa: E402
from link_vector_calculator_module import LinkVectorCalculator  # noqa: E402
from utils_dynamic import calculate_inertia_tensor, compute_impulse  # noqa: E402

# master_research_code.py の part_calculations と同じ組
PART_LINKS: dict[str, tuple[int, int]] = {
    "upper_arm_R": (3, 1),
    "forearm_R": (5, 3),
    "both_shoulder": (0, 1),
    "both_hip": (6, 7),
    "up_arm_l": (2, 0),
    "forearm_L": (4, 2),
    "upper_Leg_R": (7, 9),
    "upper_Leg_L": (6, 8),
}


def load_points(path: str) -> tuple[np.ndarray, str]:
    """CSV を (n, joints, 3) の **メートル**にして返す。

    収録 CSV は経路によって cm のものと m のものがある。人体のリンク長は
    0.15〜0.5 m に収まるので、代表リンクの長さで判定する。
    """
    frame = pd.read_csv(path)
    cols = [c for c in frame.columns if c.startswith("joint_")]
    if not cols:
        raise SystemExit(f"joint_* 列が見つかりません: {path}")
    points = frame[cols].to_numpy().reshape(len(frame), len(cols) // 3, 3)

    a, b = PART_LINKS["forearm_R"]
    if points.shape[1] <= max(a, b):
        raise SystemExit(f"関節数が足りません: {points.shape[1]}")
    span = float(np.nanmedian(np.linalg.norm(points[:, a] - points[:, b], axis=1)))
    if span > 1.0:  # 0.2 m のリンクが 20 と出ているなら cm
        return points * 0.01, f"cm と判定（代表リンク長 {span:.1f} → {span * 0.01:.3f} m）"
    return points, f"m と判定（代表リンク長 {span:.3f} m）"


def link_series(points: np.ndarray, start: int, end: int, dt: float) -> dict[str, np.ndarray]:
    """1 本のリンクについて |ω|、|ω̇|、|a| の系列を出す。"""
    calc = LinkVectorCalculator(start, end)
    out: dict[str, list[float]] = {"omega": [], "ang_acc": [], "acc": []}
    frames = [points[i] for i in range(len(points))]
    for i in range(len(frames)):
        result = calc.calculate_link_vectors(frames[: i + 1], True, i, dt)
        if result[0] is None:
            continue
        for key, value in (("omega", result[2]), ("acc", result[5]), ("ang_acc", result[6])):
            if value is not None:
                vec = np.asarray(value, dtype=float)
                if np.all(np.isfinite(vec)):
                    out[key].append(float(np.linalg.norm(vec)))
    return {k: np.array(v) for k, v in out.items()}


def inertial_torque(series: dict[str, np.ndarray], inertia: np.ndarray) -> float:
    """``I·ω̇ + ω×(I·ω)`` の大きさの中央値。utils_dynamic の式と同じ形。"""
    n = min(len(series["omega"]), len(series["ang_acc"]))
    if n == 0:
        return float("nan")
    # ω と ω̇ は生成タイミングが 1 フレームずれるので末尾を揃える
    omega = series["omega"][-n:]
    ang_acc = series["ang_acc"][-n:]
    scale = float(np.linalg.norm(np.diag(inertia)))
    # スカラー近似（各成分の大きさで代表させる）。比を見るのが目的なので十分。
    return float(np.nanmedian(scale * ang_acc + scale * omega**2))


def median(values: np.ndarray) -> float:
    return float(np.nanmedian(values)) if len(values) else float("nan")


def ratio(new: float, old: float) -> str:
    if not np.isfinite(new) or not np.isfinite(old) or old == 0:
        return "—"
    r = new / old
    return f"×{r:,.1f}" if r >= 1 else f"÷{1 / r:,.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="kpts3d_*.csv")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--old-dt", type=float, default=0.3, help="修正前の値")
    parser.add_argument("--new-dt", type=float, default=None, help="省略時は間引き無効で算出")
    parser.add_argument("--body-mass", type=float, default=float(BODY_MASS_DEFAULT))
    args = parser.parse_args(argv)

    new_dt = args.new_dt
    if new_dt is None:
        new_dt, _ = resolve_dynamics_dt(args.fps, fixed_hz_on=False, fixed_skip=0)

    points, unit_note = load_points(args.input)

    print(f"# dt 変更による力学量の差分\n")
    print(f"- 入力: `{args.input}`（{len(points)} フレーム、{unit_note}）")
    print(f"- 旧 dt = {args.old_dt} 秒 / 新 dt = {new_dt:.5f} 秒（{args.old_dt / new_dt:.1f} 倍の違い）")
    print(f"- 体重 = {args.body_mass} kg\n")

    a, b = PART_LINKS["forearm_R"]
    forearm_len = float(np.nanmedian(np.linalg.norm(points[:, a] - points[:, b], axis=1)))
    inertia = calculate_inertia_tensor(4, args.body_mass, forearm_len)

    print("## リンクごとの微分量（中央値）\n")
    print("| リンク | 量 | 旧 dt | 新 dt | 比 |")
    print("|---|---|---:|---:|---:|")
    for part, (start, end) in PART_LINKS.items():
        if max(start, end) >= points.shape[1]:
            continue
        old = link_series(points, start, end, args.old_dt)
        new = link_series(points, start, end, new_dt)
        for key, label in (("omega", "角速度"), ("ang_acc", "角加速度"), ("acc", "加速度")):
            o, n = median(old[key]), median(new[key])
            print(f"| {part} | {label} | {o:.5g} | {n:.5g} | {ratio(n, o)} |")

    print("\n## 前腕（右）の慣性トルクと力積\n")
    old = link_series(points, *PART_LINKS["forearm_R"], args.old_dt)
    new = link_series(points, *PART_LINKS["forearm_R"], new_dt)
    t_old = inertial_torque(old, inertia)
    t_new = inertial_torque(new, inertia)

    # 力積 Σ τ·dt の dt 依存は 2 通りに分けて見ないと誤解する。
    #   (1) トルク系列を固定して dt だけ変える  → dt に正比例（÷9）
    #   (2) トルクも新しい dt で計算し直す      → τ 側の ×729 が乗って ×81
    # 実際の修正で起きるのは (2)。(1) は「積分係数としての dt」の効きだけを見る対照。
    fixed_series = pd.Series(inertia[0, 0] * new["ang_acc"])
    imp_coeff_old = compute_impulse(fixed_series, args.old_dt)[0]
    imp_coeff_new = compute_impulse(fixed_series, new_dt)[0]
    imp_full_old = compute_impulse(pd.Series(inertia[0, 0] * old["ang_acc"]), args.old_dt)[0]
    imp_full_new = compute_impulse(fixed_series, new_dt)[0]

    print("| 量 | 旧 dt | 新 dt | 比 |")
    print("|---|---:|---:|---:|")
    print(f"| 慣性トルク中央値 [N·m] | {t_old:.5g} | {t_new:.5g} | {ratio(t_new, t_old)} |")
    print(
        f"| 力積: 積分係数の dt だけ [N·m·s] | {imp_coeff_old:.5g} | {imp_coeff_new:.5g} "
        f"| {ratio(imp_coeff_new, imp_coeff_old)} |"
    )
    print(
        f"| 力積: トルクも再計算 [N·m·s] | {imp_full_old:.5g} | {imp_full_new:.5g} "
        f"| {ratio(imp_full_new, imp_full_old)} |"
    )
    print(f"\n- `forearm_R` として使ったリンクの長さ（中央値）= {forearm_len:.3f} m")
    if not (0.15 <= forearm_len <= 0.35):
        print(
            f"  > **注意**: 前腕としては不自然な長さ。`PART_LINKS` の索引は"
            f" リアルタイム経路の関節順を前提にしているので、この CSV の列順とは"
            f" 一致していない可能性がある。**比（右端の列）は単位にも索引にも依らない**ので"
            f" そのまま読めるが、絶対値は参考値として扱うこと。"
        )
    print(f"- 慣性テンソル対角 = {np.diag(inertia)}")
    print(
        "\n> 角速度が `1/dt²`、角加速度が `1/dt³` でスケールするのは、"
        "`link_vector_calculator_module.py:100` が角速度を `cross(v_prev, v)/|r|²` で"
        "計算しているため。標準形 `(r × ṙ)/|r|²` なら順に `1/dt`、`1/dt²` になる。"
        "詳細は `tests/test_dynamics_dt.py` の `TestScalingLaws` を参照。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
