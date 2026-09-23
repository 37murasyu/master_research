#!/usr/bin/env python3
"""力学の式を解析解と突き合わせて検算する。

``力学計算_検証結果.md``（2026-09-02）はオフライン経路を中心に検証したもので、
リアルタイム経路（``master_research_code.py`` → ``LinkVectorCalculator``）は範囲外だった。
2026-09-08 に両経路を同じ解析解へ通し、10 件の誤りを確定して修正した
（``力学計算_再検算_2026-09-08.md``）。

**このスクリプトは修正が効いていることを確認し、退行したら気づけるようにするためのもの。**
各節に「修正前はこうだった」を併記してあるので、値が旧側に戻っていないかを目で追える。
自動判定が要る場合は ``tests/test_dynamics_formulas.py`` と
``tests/test_keypoint_order.py`` が同じことを assert している。

検証系は「始点を原点に固定し、終点が z 軸まわりに角速度 W で等速回転するリンク」。
この系は全量に閉じた解析解を持つ::

    r(t) = L(cos Wt, sin Wt, 0)     omega = (0,0,W)     omega_dot = 0
    r''  = -W^2 r  （大きさ W^2 L）  重心（中点）の加速度 = r''/2

使い方::

    python tools/verify_dynamics_recheck.py
    python tools/verify_dynamics_recheck.py --pose output_data/stereo_2_....npy
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body_part_storage_module import BodyPartDataStorage  # noqa: E402
from config import (  # noqa: E402
    COM_FRACTIONS,
    G_SCALAR,
    THEORETICAL_WORK_COEFF,
    WORK_ANGLE_RANGE_DEG,
    WORK_INTEGRAL_K,
)
from config import g as G_VEC  # noqa: E402
from config import part_calculations, pose_keypoints  # noqa: E402
from link_vector_calculator_module import LinkVectorCalculator  # noqa: E402
from utils_dynamic import calculate_individual_torques, calculate_inertia_tensor  # noqa: E402

DT, N, L, W = 1.0 / 30.0, 300, 0.25, 2.0
BODY_MASS = 60.0

MP_NAME = {16: "右手首", 14: "右肘", 12: "右肩", 11: "左肩", 13: "左肘", 15: "左手首",
           24: "右腰", 23: "左腰", 25: "左膝", 26: "右膝", 27: "左足首", 28: "右足首",
           # 手の点（8e20586 で pose_keypoints に追加）。無いと関節索引の節が KeyError で止まる
           17: "左小指", 18: "右小指", 19: "左人差指", 20: "右人差指"}


def head(title: str) -> None:
    print("\n" + "=" * 76 + f"\n{title}\n" + "=" * 76)


def verdict(ok: bool, detail: str = "") -> str:
    return ("  → OK  " if ok else "  → 退行 ") + detail


def rotating_link(n=N, dt=DT, w=W, length=L, sigma=0.0, rng=None):
    """等速回転するリンクの点列。sigma を与えると位置に正規ノイズを乗せる。"""
    rng = rng or np.random.default_rng(0)
    out = []
    for k in range(n):
        th = w * k * dt
        p0 = rng.normal(0, sigma, 3) if sigma else np.zeros(3)
        p1 = np.array([length * np.cos(th), length * np.sin(th), 0.0])
        out.append(np.vstack([p0, p1 + (rng.normal(0, sigma, 3) if sigma else 0.0)]))
    return out


def realtime_series(points, dt=DT, com_fraction=0.5):
    """LinkVectorCalculator を通して |omega| |omega_dot| |acc| を集める。"""
    calc = LinkVectorCalculator(0, 1, com_fraction)
    out = {"omega": [], "ang_acc": [], "acc": []}
    for i in range(len(points)):
        res = calc.calculate_link_vectors(points[: i + 1], 1, i, dt)
        if res[0] is None:
            continue
        for key, val in (("omega", res[2]), ("acc", res[5]), ("ang_acc", res[6])):
            if val is not None and np.all(np.isfinite(val)):
                out[key].append(float(np.linalg.norm(val)))
    return {k: np.array(v) for k, v in out.items()}


def reference_omega(points, dt=DT):
    """標準形 (r x r') / |r|^2。compute_segment_kinematics と同じ式の参照実装。"""
    P = np.stack(points)
    link = P[:, 1] - P[:, 0]
    vel = np.gradient(link, dt, axis=0)
    return np.linalg.norm(
        np.cross(link, vel) / np.sum(link * link, axis=1, keepdims=True), axis=1
    )


def check_torque_chain() -> None:
    head("1. 逆動力学の連鎖式  tau_j = sum M + sum (r_g - p_j) x F - tau_E - (r_x - p_j) x f_E")
    print("  この式は再検算で正しいと確認済み。修正対象ではないので変わってはいけない。\n")
    mass = BODY_MASS * 0.016
    storage = BodyPartDataStorage()
    storage.add_data("seg", np.array([L, 0, 0]), np.zeros(3), np.zeros(3),
                     np.array([L / 2, 0, 0]), np.zeros(3), np.zeros(3), np.zeros(3))
    force = mass * (np.zeros(3) - G_VEC)          # calculate_M_and_F と同じ F = m(a - g)
    torque = calculate_individual_torques(
        [np.zeros(3)], [force], [np.array([L / 2, 0, 0])],
        np.zeros(3), np.zeros(3), np.zeros(3), ["seg"], storage,
    )
    got = float(np.linalg.norm(torque[0][0]))
    want = mass * abs(G_VEC[2]) * L / 2
    print(f"  静止水平リンク m={mass:.3f} kg, L={L} m")
    print(f"    実装 |tau| = {got:.6f} N*m   解析解 m*g*L/2 = {want:.6f} N*m")
    print(f"    F = {force}（|F| = {np.linalg.norm(force):.4f} N = m*g）")
    print(verdict(abs(got / want - 1) < 1e-9, f"比 = {got / want:.6f}"))


def check_angular_velocity() -> None:
    head("2. 角速度（R-2）  真値 omega = 2.0 rad/s の等速回転を与える")
    pts = rotating_link()
    impl = float(np.median(realtime_series(pts)["omega"][10:]))
    ref = float(np.median(reference_omega(pts)[10:]))
    print(f"  実装 (r x r')/|r|^2       : {impl:.6f} rad/s")
    print(f"  参照実装（中心差分）       : {ref:.6f} rad/s")
    print(f"  真値                      : {W:.6f} rad/s")
    print(verdict(abs(impl / W - 1) < 0.02))
    print(f"\n  修正前は cross(v_prev, v)/|r|^2 で {W ** 3 * DT:.4f} = omega^3*dt を返していた。")
    print("  剛体回転 r' = omega x r を代入すると dt*|omega_perp|^2*omega_perp となり、")
    print("  次元は 1/s^2 で角速度ではない。信号は omega^2*dt 倍に潰れていた:\n")
    print(f"  {'真の omega [rad/s]':>20} | {'修正前の値':>12} | {'真値に対する比':>16}")
    for w in (0.5, 1.0, 1.6, 2.0, 3.0):
        print(f"  {w:>20.1f} | {w ** 3 * DT:>12.4f} | {w ** 2 * DT:>16.4f}")
    print(f"\n  実測の omega 平均 1.59 rad/s なら真値の {100 * 1.59 ** 2 * DT:.1f}% しか出ていなかった。")


def check_noise_response() -> None:
    head("3. ノイズ応答（R-2 の副次的な効果）")
    print(f"  真の |omega| = {W} rad/s、リンク長 {L} m、dt = 1/30 s")
    print("  修正前の式は信号を潰す一方でノイズを増幅していた。2 mm のノイズで")
    print("  真値と同じ大きさの偽信号が立ち、出力は角速度ではなくノイズだった。\n")
    print(f"  {'位置ノイズ':>10} | {'実装 中央':>10} | {'実装 最大':>10} |"
          f" {'参照 中央':>10} | {'参照 最大':>10}")
    print("  " + "-" * 60)
    worst = 0.0
    for sigma in (0.0, 0.0005, 0.001, 0.002, 0.005):
        rng = np.random.default_rng(0)
        pts = rotating_link(sigma=sigma, rng=rng)
        impl, ref = realtime_series(pts)["omega"][10:], reference_omega(pts)[10:]
        worst = max(worst, abs(float(np.median(impl)) / W - 1))
        print(f"  {sigma * 1000:>7.1f} mm | {np.median(impl):>10.4f} | {np.max(impl):>10.3f} |"
              f" {np.median(ref):>10.4f} | {np.max(ref):>10.3f}")
    print(verdict(worst < 0.3, f"最大の中央値ずれ {100 * worst:.1f}%（5 mm ノイズ時）"))


def check_com_acceleration() -> None:
    head("4. dot_dot_pg（R-3）  F = m(a - g) に入る加速度")
    impl = float(np.median(realtime_series(rotating_link())["acc"][10:]))
    com_true = W ** 2 * L / 2
    print(f"  実装が渡す値          : {impl:.5f} m/s^2")
    print(f"  重心（中点）の加速度   : {com_true:.5f} m/s^2  ← これが正しい")
    print(f"  r'' = W^2 L           : {W ** 2 * L:.5f} m/s^2  ← 修正前はこれを渡していた")
    print(verdict(abs(impl / com_true - 1) < 0.02))
    print("\n  リンクベクトル r の 2 階微分は r'' = p_end'' - p_start'' なので、")
    print("  始点が固定なら重心加速度のちょうど 2 倍、始点が動けば別のベクトルになる。")

    head("4b. 重心の位置（R-4）  中点固定ではなく体節ごとの文献値")
    print(f"  {'部位':<6} {'重心比':>8} {'中点との比':>12}")
    for label, key in (("上腕", "upper_arm"), ("前腕", "forearm"), ("大腿", "thigh")):
        frac = COM_FRACTIONS[key]
        print(f"  {label:<6} {frac:>8.3f} {0.5 / frac:>11.4f}x")
    used = {name: spec["com_fraction"] for name, spec in part_calculations.items()}
    limbs = [v for k, v in used.items() if k not in ("both_shoulder", "both_hip")]
    print(f"\n  part_calculations が使う値: {used}")
    print(verdict(all(abs(v - 0.5) > 1e-9 for v in limbs),
                  "四肢が中点 (0.5) から離れている"))


def check_joint_indices(pose_path: str | None) -> None:
    head("5. 関節索引（R-1）  抽出はランドマーク ID の昇順")
    ordered = sorted(pose_keypoints)
    idx_name = {i: MP_NAME[pid] for i, pid in enumerate(ordered)}
    print(f"  config.pose_keypoints = {list(pose_keypoints)}")
    print("  このリストの並びに意味は無い。元実装 TemugeB/bodypose3d は enumerate で")
    print("  ランドマーク ID の昇順に走査しており、フィルタとしてしか使わない。")
    print(f"  したがって 3D 点列の並びは sorted = {ordered}\n")
    print("   " + "  ".join(f"[{i}]={idx_name[i]}" for i in range(6)))
    print("   " + "  ".join(f"[{i}]={idx_name[i]}" for i in range(6, 12)))
    print("\n  part_calculations が結ぶ点:")
    for name, spec in part_calculations.items():
        s, e = spec["start"], spec["end"]
        print(f"    {name:<14} {idx_name[s]} → {idx_name[e]}")
    print("\n  修正前は抽出がリストの並び順で回っており、[0]=右手首 [1]=右肘 [2]=右肩 …")
    print("  という別の並びになっていた。8 リンク中 7 本が体を斜めに横切っていた。")

    if not pose_path:
        return
    arr = np.load(pose_path)
    if arr.ndim != 3 or arr.shape[1] < 33:
        print(f"\n  （{pose_path} は MediaPipe 33 点ではないので実測は省略）")
        return
    raw = arr.astype(np.float64) * 0.01
    sub = raw[:, ordered, :]
    span = lambda P, a, b: float(np.nanmedian(np.linalg.norm(P[:, b] - P[:, a], axis=1)))
    expect = {"upper_arm_R": (0.15, 0.40), "forearm_R": (0.13, 0.35),
              "up_arm_l": (0.15, 0.40), "forearm_L": (0.13, 0.35),
              "both_shoulder": (0.20, 0.50), "both_hip": (0.12, 0.40),
              "upper_Leg_R": (0.20, 0.55), "upper_Leg_L": (0.20, 0.55)}
    print(f"\n  実データ {os.path.basename(pose_path)} {arr.shape} で測ったリンク長 [m]:")
    all_ok = True
    for name, spec in part_calculations.items():
        s, e = spec["start"], spec["end"]
        length = span(sub, s, e)
        low, high = expect[name]
        ok = low <= length <= high
        all_ok &= ok
        mark = "OK" if ok else f"×（想定 {low}〜{high}）"
        print(f"    {name:<14} {idx_name[s] + '→' + idx_name[e]:<20} {length:.3f}  {mark}")
    print(verdict(all_ok, "全リンクが解剖学的に妥当な長さ"))


def check_inertia_regression() -> None:
    head("6. 慣性モーメント回帰式（§1-3・R-7）  I = a*w + b*l + c（w は全身体重）")
    print("  §1-3: 部位質量ではなく全身体重を渡す。部位質量だと対角成分が負になる。")
    print("  R-7:  体重を渡しても短いリンクでは負になるので、警告して一様棒に落とす。\n")
    for label, row, length, frac in (("上腕", 3, 0.237, 0.0227), ("前腕", 4, 0.205, 0.0160),
                                     ("下腿", 7, 0.232, 0.0465)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            i_body = np.diag(calculate_inertia_tensor(row, BODY_MASS, length))
            fell_back = bool(caught)
        note = "  ← 適用範囲外。一様棒 m*L^2/12 にフォールバック" if fell_back else ""
        print(f"  {label} (L={length} m): {np.round(i_body, 6)}{note}")
    print("\n  体重 60 kg で対角が正になる最小リンク長:")
    print("    上腕 0.205 / 前腕 0.168 / 大腿 0.249 / 下腿 0.283 m")
    print("  実測の下腿 0.232 m はこの範囲に入らない。")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        negative = any(
            np.any(np.diag(calculate_inertia_tensor(r, BODY_MASS, ln)) < 0)
            for r in range(9) for ln in (0.05, 0.15, 0.25, 0.40, 0.60)
        )
    print(verdict(not negative, "どの部位・長さでも負を返さない"))


def check_work_coefficient() -> None:
    head("7. 理論仕事量の係数（R-5）  1 サイクルの角度範囲にわたる cos の積分")
    low, high = WORK_ANGLE_RANGE_DEG
    print(f"  角度範囲 = {low}°〜{high}°")
    print(f"  積分係数 K = sin({high}°) - sin({low}°) = {WORK_INTEGRAL_K:.6f}")
    print(f"  重力加速度 = {G_SCALAR}")
    print(f"  理論仕事量の係数 = K * g = {THEORETICAL_WORK_COEFF:.4f}")
    print("\n  修正前は 2 系統が 9.3% 食い違ったまま併存していた:")
    print(f"    master_research_code.py / offline_wrist_energy.py : √3/2+1 = {np.sqrt(3) / 2 + 1:.4f}")
    print(f"    compute_cycle_energy_elbow_wrist.py ほか 3 箇所   : 16.73 =(√2/2+1)x9.8")
    ok = abs(WORK_INTEGRAL_K - (np.sqrt(2) / 2 + 1)) < 1e-9
    print(verdict(ok, f"従来の直書き 16.73 との差 {100 * (THEORETICAL_WORK_COEFF / 16.73 - 1):+.2f}%"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", default="output_data/stereo_2_20250925_162436.npy",
                        help="MediaPipe 33 点の 3D pose (.npy)。実データ照合に使う")
    args = parser.parse_args(argv)
    pose = args.pose if args.pose and os.path.exists(args.pose) else None
    if args.pose and not pose:
        print(f"[注意] {args.pose} が見つからないので実データ照合を省略します")

    check_torque_chain()
    check_angular_velocity()
    check_noise_response()
    check_com_acceleration()
    check_joint_indices(pose)
    check_inertia_regression()
    check_work_coefficient()
    print("\n" + "=" * 76)
    print("自動判定は tests/test_dynamics_formulas.py と tests/test_keypoint_order.py にある。")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
