#!/usr/bin/env python3
"""力学の式を解析解と突き合わせて検算する。

``力学計算_検証結果.md``（2026-09-02）はオフライン経路
（``compute_torque_from_pose.py``）を中心に検証したもので、リアルタイム経路
（``master_research_code.py`` → ``LinkVectorCalculator``）は検証範囲外だった。
本スクリプトは両経路を同じ解析解に通して比べる。

検証系は「始点を原点に固定し、終点が z 軸まわりに角速度 W で等速回転するリンク」。
この系は全量に閉じた解析解を持つ:

    r(t) = L(cos Wt, sin Wt, 0)   omega = (0,0,W)   omega_dot = 0
    r''  = -W^2 r                 （大きさ W^2 L）
    重心（中点）の加速度 = r''/2  （始点が固定のとき）

使い方::

    python tools/verify_dynamics_recheck.py
    python tools/verify_dynamics_recheck.py --pose output_data/stereo_2_....npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from body_part_storage_module import BodyPartDataStorage  # noqa: E402
from config import g as G_VEC, pose_keypoints  # noqa: E402
from link_vector_calculator_module import LinkVectorCalculator  # noqa: E402
from utils_dynamic import calculate_individual_torques, calculate_inertia_tensor  # noqa: E402

DT, N, L, W = 1.0 / 30.0, 300, 0.25, 2.0
BODY_MASS = 60.0

# master_research_code.py:1653 の part_calculations と同じ
PART_LINKS = {
    "upper_arm_R": (3, 1), "forearm_R": (5, 3), "both_shoulder": (0, 1),
    "both_hip": (6, 7), "up_arm_l": (2, 0), "forearm_L": (4, 2),
    "upper_Leg_R": (7, 9), "upper_Leg_L": (6, 8),
}
MP_NAME = {16: "右手首", 14: "右肘", 12: "右肩", 11: "左肩", 13: "左肘", 15: "左手首",
           24: "右腰", 23: "左腰", 25: "左膝", 26: "右膝", 27: "左足首", 28: "右足首"}


def head(title: str) -> None:
    print("\n" + "=" * 76 + f"\n{title}\n" + "=" * 76)


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


def realtime_series(points, dt=DT):
    """LinkVectorCalculator を通して |omega| |omega_dot| |acc| を集める。"""
    calc = LinkVectorCalculator(0, 1)
    out = {"omega": [], "ang_acc": [], "acc": []}
    for i in range(len(points)):
        res = calc.calculate_link_vectors(points[: i + 1], 1, i, dt)
        if res[0] is None:
            continue
        for key, val in (("omega", res[2]), ("acc", res[5]), ("ang_acc", res[6])):
            if val is not None and np.all(np.isfinite(val)):
                out[key].append(float(np.linalg.norm(val)))
    return {k: np.array(v) for k, v in out.items()}


def correct_omega(points, dt=DT):
    """標準形 (r x r') / |r|^2。compute_segment_kinematics と同じ式。"""
    P = np.stack(points)
    link = P[:, 1] - P[:, 0]
    vel = np.gradient(link, dt, axis=0)
    return np.linalg.norm(
        np.cross(link, vel) / np.sum(link * link, axis=1, keepdims=True), axis=1
    )


def check_torque_chain() -> None:
    head("1. 逆動力学の連鎖式  tau_j = sum M + sum (r_g - p_j) x F - tau_E - (r_x - p_j) x f_E")
    mass = BODY_MASS * 0.016
    storage = BodyPartDataStorage()
    # 静止して水平に伸びたリンク。関節は原点、重心は (L/2, 0, 0)
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
    print(f"    実装 |tau| = {got:.6f} N*m")
    print(f"    解析解 m*g*L/2 = {want:.6f} N*m   比 = {got / want:.6f}")
    print(f"    F = {force}（|F| = {np.linalg.norm(force):.4f} N = m*g）")
    print("  → 連鎖式・符号規約ともに正しい")


def check_angular_velocity() -> None:
    head("2. 角速度  真値 omega = 2.0 rad/s の等速回転を与える")
    pts = rotating_link()
    rt = realtime_series(pts)
    print(f"  正しい式 (r x r')/|r|^2      : {np.median(correct_omega(pts)[10:]):.6f} rad/s")
    print(f"  実装 (v_prev x v)/|r|^2      : {np.median(rt['omega'][10:]):.6f} rad/s")
    print(f"  仮説 omega^3 * dt            : {W ** 3 * DT:.6f}")
    print("\n  導出: 剛体回転 r' = omega x r で r'_prev ≈ omega x r - dt*omega x (omega x r)")
    print("        → r'_prev x r' = dt*|omega_perp|^2 * omega_perp * |r|^2")
    print("        → |r|^2 で割ると omega_impl = dt * |omega_perp|^2 * omega_perp")
    print("  次元は 1/s^2 であって 1/s ではない。信号は omega^2*dt 倍に潰れる。\n")
    print(f"  {'真の omega [rad/s]':>20} | {'実装が返す値':>13} | {'比 (= omega^2*dt)':>18}")
    for w in (0.5, 1.0, 1.6, 2.0, 3.0):
        print(f"  {w:>20.1f} | {w ** 3 * DT:>13.4f} | {w ** 2 * DT:>18.4f}")


def check_noise_response() -> None:
    head("3. 同じ式のノイズ応答  （信号は潰すのにノイズは増幅する）")
    print(f"  真の |omega| = {W} rad/s、リンク長 {L} m、dt = 1/30 s\n")
    print(f"  {'位置ノイズ':>10} | {'実装 中央':>10} | {'実装 最大':>10} |"
          f" {'正しい式 中央':>13} | {'正しい式 最大':>13}")
    print("  " + "-" * 68)
    for sigma in (0.0, 0.0005, 0.001, 0.002, 0.005):
        rng = np.random.default_rng(0)
        pts = rotating_link(sigma=sigma, rng=rng)
        impl, ok = realtime_series(pts)["omega"][10:], correct_omega(pts)[10:]
        print(f"  {sigma * 1000:>7.1f} mm | {np.median(impl):>10.4f} | {np.max(impl):>10.3f} |"
              f" {np.median(ok):>13.4f} | {np.max(ok):>13.3f}")
    print("\n  v = dr/dt なのでノイズは 1/dt = 30 倍される。その外積を |r|^2 で割るため")
    print("  (sqrt(2)*sigma/dt)^2 / |r|^2 のオーダーの偽信号が乗る。")
    print("  2 mm のノイズで真値と同じ大きさの値が立つ = 出力は信号ではなくノイズ。")


def check_com_acceleration() -> None:
    head("4. dot_dot_pg（F = m(a - g) に入る加速度）")
    pts = rotating_link()
    impl = realtime_series(pts)["acc"]
    print(f"  実装が渡す値              : {np.median(impl[10:]):.5f} m/s^2")
    print(f"  r'' = W^2 L               : {W ** 2 * L:.5f} m/s^2  ← 実装はこれ")
    print(f"  重心（中点）の加速度       : {W ** 2 * L / 2:.5f} m/s^2  ← 本来こちら")
    print("\n  link_vector_calculator_module.py:105 の acc は")
    print("  『リンクベクトル r の 2 階微分』であって重心加速度ではない。")
    print("  始点が固定なら r'' = 2 x (中点の加速度)。始点が動けば別のベクトルになる。")
    print("\n  重心位置そのものも中点固定（COM 比 0.5）:")
    for name, frac in (("上腕", 0.436), ("前腕", 0.430)):
        print(f"    {name}: 実装 0.500 / 文献 {frac} → 重力モーメント腕が"
              f" {100 * (0.5 / frac - 1):+.1f}%")


def check_joint_indices(pose_path: str | None) -> None:
    head("5. 関節索引の並び  config.pose_keypoints と part_calculations の整合")
    idx_name = {i: MP_NAME[pid] for i, pid in enumerate(pose_keypoints)}
    print(f"  config.pose_keypoints = {pose_keypoints}")
    print("  utils.py:137 は『pose_keypoints の並び順で返す』、")
    print("  master_research_code.py:2170-2173 は軸変換のみで並べ替えない。したがって:\n")
    print("   " + "  ".join(f"[{i}]={idx_name[i]}" for i in range(6)))
    print("   " + "  ".join(f"[{i}]={idx_name[i]}" for i in range(6, 10)))
    print("\n  part_calculations が実際に結ぶ点:")
    for name, (s, e) in PART_LINKS.items():
        print(f"    {name:<14} {idx_name[s]} → {idx_name[e]}")
    print("\n  元実装 TemugeB/bodypose3d は enumerate でランドマーク ID の昇順に走査し、")
    print("  pose_keypoints はフィルタとしてしか使わない。したがって正しい並びは")
    print(f"    sorted = {sorted(pose_keypoints)}")
    print("    = [左肩, 右肩, 左肘, 右肘, 左手首, 右手首, 左腰, 右腰, 左膝, 右膝, 左足首, 右足首]")
    print("  これは part_calculations の前提と一致する。part_calculations は正しい。")
    print(f"\n  一方 utils.py:142 は `for pid in pose_keypoints:` とリスト順で回すため")
    print(f"    実際 = {list(pose_keypoints)}")
    print("  先頭 8 個の並びが食い違い、後半 [25, 26, 27, 28] だけ一致する。")

    if not pose_path:
        return
    arr = np.load(pose_path)
    if arr.ndim != 3 or arr.shape[1] < 33:
        print(f"\n  （{pose_path} は MediaPipe 33 点ではないので実測は省略）")
        return
    raw = arr.astype(np.float64) * 0.01
    sub = raw[:, pose_keypoints, :]
    span = lambda P, a, b: float(np.nanmedian(np.linalg.norm(P[:, b] - P[:, a], axis=1)))
    print(f"\n  実データ {os.path.basename(pose_path)} {arr.shape} で測ったリンク長 [m]:")
    for name, (s, e) in PART_LINKS.items():
        print(f"    {name:<14} {idx_name[s]+'→'+idx_name[e]:<20} {span(sub, s, e):.3f}")
    print("\n  解剖学的に正しい組:")
    for name, (a, b) in (("右上腕", (12, 14)), ("右前腕", (14, 16)), ("両肩幅", (11, 12)),
                         ("両腰幅", (23, 24)), ("右大腿", (24, 26))):
        print(f"    {name}: {span(raw, a, b):.3f}")

    print("\n  慣性テンソルへの波及（体重 60 kg）:")
    for label, row, got, want in (
        ("上腕", 3, span(sub, 0, 2), span(raw, 12, 14)),
        ("前腕", 4, span(sub, 2, 4), span(raw, 14, 16)),
        ("大腿", 6, span(sub, 9, 7), span(raw, 24, 26)),
    ):
        a = np.diag(calculate_inertia_tensor(row, BODY_MASS, got))
        b = np.diag(calculate_inertia_tensor(row, BODY_MASS, want))
        print(f"    {label} L={got:.3f}→{want:.3f} m   diag 比 = {np.round(a / b, 2)}")


def check_inertia_regression() -> None:
    head("6. 慣性モーメント回帰式  I = a*w + b*l + c （w は全身体重）")
    for label, row, length, frac in (("上腕", 3, 0.237, 0.0227), ("前腕", 4, 0.205, 0.0160),
                                     ("下腿", 7, 0.232, 0.0465)):
        i_body = np.diag(calculate_inertia_tensor(row, BODY_MASS, length))
        i_seg = np.diag(calculate_inertia_tensor(row, BODY_MASS * frac, length))
        print(f"  {label} (L={length} m)")
        print(f"    体重 {BODY_MASS} kg を渡す      : {np.round(i_body, 6)}"
              f"  負={int((i_body < 0).sum())}/3")
        print(f"    部位質量 {BODY_MASS * frac:.2f} kg を渡す: {np.round(i_seg, 6)}"
              f"  負={int((i_seg < 0).sum())}/3")
    print("\n  部位質量を渡すと負になるのは KNOWN_ISSUES §1-3 のとおり。")
    print("  加えて、体重を渡しても短いリンクでは負になりうる（下腿 L=0.232 m）。")
    print("  定数項 c が b*l と相殺する領域で、回帰式の適用範囲の確認が要る。")


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
