"""estimate_joint_torque_from_pose.py
上肢（肩・肘・手首）のトルクを、姿勢CSV (3D, MediaPipeのjoint_ID列) から簡易推定して出力します。

前提と近似:
- セグメントを剛体の棒とみなし、関節角の2階微分(角加速度)×慣性モーメント I_end=(1/3)mL^2 からトルク大きさを近似。
- 軸は2セグメントが張る平面の法線（cross(上腕, 前腕) など）方向。
- 重力項や床反力は未考慮。スケールが未確定な座標でも相対的パターン比較に有効。
- 手部セグメントがないため手首トルクは0ベクトルを出力（列は保持）。

MediaPipe ID:
  右: 肩=12, 肘=14, 手首=16
  左: 肩=11, 肘=13, 手首=15
  骨盤(左右): 24,23 （肩トルク軸の参照用に使用可だが本推定では未使用）

出力: aim_torque_vec_<ID>_est.csv（列は frame, wrist_R/elbow_R/shoulder_R, wrist_L/elbow_L/shoulder_L の各x,y,z）
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np


MP_IDS = {
    "R": {"shoulder": 12, "elbow": 14, "wrist": 16},
    "L": {"shoulder": 11, "elbow": 13, "wrist": 15},
}

# Legacy 0..5 indices used in older CSVs (matching other tools in this repo)
LEGACY_IDS = {
    "R": {"shoulder": 0, "elbow": 2, "wrist": 4},
    "L": {"shoulder": 1, "elbow": 3, "wrist": 5},
}


def safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(v)
    return v / n


def central_diff(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """1階・2階の中心差分（端点は片側差分）を返す。
    x: (T,) 時系列
    戻り値: (dx/dt, d2x/dt2)
    """
    T = len(x)
    dx = np.zeros(T)
    ddx = np.zeros(T)
    if T >= 3:
        dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        ddx[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / (dt * dt)
    if T >= 2:
        dx[0] = (x[1] - x[0]) / dt
        dx[-1] = (x[-1] - x[-2]) / dt
        ddx[0] = (x[2] - 2 * x[1] + x[0]) / (dt * dt) if T >= 3 else 0.0
        ddx[-1] = (x[-1] - 2 * x[-2] + x[-3]) / (dt * dt) if T >= 3 else 0.0
    return dx, ddx


def angle_series(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """単位ベクトル列 u, v (T,3) のなす角 [rad] を返す。"""
    dot = np.sum(u * v, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def estimate_side(
    p3d: np.ndarray,
    ids: Dict[str, int],
    fps: float,
    m_upper: float,
    m_fore: float,
    up_axis: np.ndarray | None = None,
    include_gravity: bool = False,
    g_mag: float = 9.81,
    com_frac_upper: float = 0.5,
    com_frac_fore: float = 0.5,
    hand_load_kg: float = 0.0,
) -> Dict[str, np.ndarray]:
    """片側の肩・肘・手首トルクベクトル系列を推定して返す。
    p3d: (T,33,3) または (T,J,3) で IDに対応するインデックスが含まれていること
    戻り: dict keys ['wrist','elbow','shoulder'] 各 (T,3)
    """
    T = p3d.shape[0]
    out = {
        "wrist": np.zeros((T, 3), dtype=float),
        "elbow": np.zeros((T, 3), dtype=float),
        "shoulder": np.zeros((T, 3), dtype=float),
    }

    sh, el, wr = ids["shoulder"], ids["elbow"], ids["wrist"]
    # IDが存在しない場合はゼロのまま返す
    if max(sh, el, wr) >= p3d.shape[1]:
        return out

    S = p3d[:, sh, :]
    E = p3d[:, el, :]
    W = p3d[:, wr, :]

    # 単位方向ベクトル
    u_upper = np.vstack([safe_unit(E[t] - S[t]) for t in range(T)])
    u_fore = np.vstack([safe_unit(W[t] - E[t]) for t in range(T)])

    # セグメント長（平均値を使う）
    L_upper = float(np.nanmedian(np.linalg.norm(E - S, axis=1)))
    L_fore = float(np.nanmedian(np.linalg.norm(W - E, axis=1)))

    dt = 1.0 / max(fps, 1e-6)
    # 肘角度（上腕と前腕のなす角）
    theta_e = angle_series(-u_upper, u_fore)  # 肘の屈曲角に近い
    _, ddtheta_e = central_diff(theta_e, dt)
    # 肘の軸（平面法線）
    axis_e = np.vstack([safe_unit(np.cross(u_upper[t], u_fore[t])) for t in range(T)])
    # トルク大きさ: I_end * 角加速度（端回りを仮定）
    I_fore = (1.0 / 3.0) * m_fore * (L_fore ** 2)
    tau_e_mag = I_fore * ddtheta_e
    out["elbow"] = axis_e * tau_e_mag[:, None]

    # 肩角度（上腕 vs 上向き軸）
    up = np.array([0.0, 0.0, 1.0], dtype=float) if up_axis is None else up_axis.astype(float)
    up = safe_unit(up)
    uprep = np.tile(up[None, :], (T, 1))
    theta_s = angle_series(u_upper, uprep)
    _, ddtheta_s = central_diff(theta_s, dt)
    # 上腕方向と上向き軸の平面法線
    axis_s = np.vstack([safe_unit(np.cross(u_upper[t], up)) for t in range(T)])
    I_upper = (1.0 / 3.0) * m_upper * (L_upper ** 2)
    tau_s_mag = I_upper * ddtheta_s
    out["shoulder"] = axis_s * tau_s_mag[:, None]

    # 手部はアームレストに固定: 前腕ベクトルの角加速度から手首トルクを推定
    # u_fore(t) の角速度 ω_w = u × du/dt, 角加速度 α_w = u × d^2u/dt^2 （近似）
    # 慣性は端回りの棒: I_end = (1/3) m L^2 を等方テンソルとみなし M = I_end * α_w
    # ベクトルの時間微分（各軸に中央差分を適用）
    dt = 1.0 / max(fps, 1e-6)
    ddU = np.zeros_like(u_fore)
    for k in range(3):
        _, dd = central_diff(u_fore[:, k], dt)
        # du/dt は不要（ω計算に厳密には必要だが、等方Iではω×(Iω)=0のため αのみ使用）
        ddU[:, k] = dd
    alpha_w = np.vstack([np.cross(u_fore[t], ddU[t]) for t in range(T)])
    I_fore_end = (1.0 / 3.0) * m_fore * (L_fore ** 2)
    out["wrist"] = alpha_w * I_fore_end

    # === 重力トルクの追加（オプション） ===
    if include_gravity and g_mag > 0.0:
        g_vec = -g_mag * up  # 上向きに対して下向きに重力
        # 肘: 前腕（＋手荷重）の重力トルク（肘まわり）
        r_e_com_fore = com_frac_fore * (W - E)  # E -> fore COM
        tau_g_elbow = np.cross(r_e_com_fore, m_fore * g_vec)
        # 手に外部荷重がある場合（手首位置に集中荷重と仮定）
        if hand_load_kg > 0.0:
            tau_g_elbow += np.cross(W - E, hand_load_kg * g_vec)
        out["elbow"] = out["elbow"] + tau_g_elbow

        # 肩: 上腕 + 前腕（＋手荷重）の重力トルク（肩まわり）
        r_s_com_upper = com_frac_upper * (E - S)  # S -> upper COM
        r_s_com_fore = (E - S) + com_frac_fore * (W - E)  # S -> fore COM
        tau_g_shoulder = np.cross(r_s_com_upper, m_upper * g_vec) + np.cross(r_s_com_fore, m_fore * g_vec)
        if hand_load_kg > 0.0:
            tau_g_shoulder += np.cross(W - S, hand_load_kg * g_vec)
        out["shoulder"] = out["shoulder"] + tau_g_shoulder

    return out


def infer_id_from_pose_path(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith("stereo_") and base.endswith("_pose.csv"):
        return base[len("stereo_"):-len("_pose.csv")]
    return os.path.splitext(base)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="姿勢CSVから上肢トルク(簡易)を推定してCSV出力")
    ap.add_argument("--pose-csv", required=True)
    ap.add_argument("--fps", type=float, default=30.0, help="撮影フレームレートの仮定 [Hz]")
    ap.add_argument("--body-mass", type=float, default=60.0, help="体重[kg]")
    ap.add_argument("--upperarm-frac", type=float, default=0.028, help="上腕質量比 (体重×係数)")
    ap.add_argument("--forearm-frac", type=float, default=0.016, help="前腕質量比 (体重×係数)")
    ap.add_argument("--up-axis", choices=["x", "y", "z"], default="y", help="ワールドの上向き軸（重力はその反対向きに作用）")
    ap.add_argument("--include-gravity", action="store_true", help="重力トルクを加算する")
    ap.add_argument("--g", type=float, default=9.81, help="重力加速度の大きさ [m/s^2]")
    ap.add_argument("--com-frac-upper", type=float, default=0.5, help="上腕の重心位置（近位からの比率）")
    ap.add_argument("--com-frac-fore", type=float, default=0.5, help="前腕の重心位置（近位からの比率）")
    ap.add_argument("--hand-load-kg", type=float, default=0.0, help="手にかかる外部荷重（質量換算, kg）")
    ap.add_argument("--prefer-raw", action="store_true", help="_f（フィルタ済み）より生列を優先して微分を強める")
    ap.add_argument("--target-upperarm-m", type=float, default=0.0, help="上腕の目標長[m]（>0で長さキャリブレーションを適用）")
    ap.add_argument("--out", default=None, help="出力CSV（未指定なら output_data/aim_torque_vec_<ID>_est.csv）")
    args = ap.parse_args(argv)

    import pandas as pd
    df = pd.read_csv(args.pose_csv)
    if "frame" not in df.columns:
        raise ValueError("pose CSVに'frame'列がありません")

    # 関節IDの列から 3D 配列を構築
    # joint_{id}_{axis} もしくは joint_{id}_{axis}_f（フィルタ済み）を全探索（_f を優先採用）
    joint_ids = sorted({
        int(c.split("_")[1])
        for c in df.columns
        if c.startswith("joint_")
           and c.split("_")[1].isdigit()
           and (
               c.endswith("_x") or c.endswith("_y") or c.endswith("_z") or
               c.endswith("_x_f") or c.endswith("_y_f") or c.endswith("_z_f")
           )
    })
    if not joint_ids:
        raise ValueError("joint_* 列が見つかりません")
    # ID->列名
    def coln(jid: int, ax: str) -> str:
        # 優先順: prefer-raw 指定時は生列, それ以外は _f を優先
        c = f"joint_{jid}_{ax}"
        c_f = f"joint_{jid}_{ax}_f"
        if args.prefer_raw:
            return c if c in df.columns else c_f
        else:
            return c_f if c_f in df.columns else c

    T = len(df)
    J = len(joint_ids)
    p3d = np.zeros((T, J, 3), dtype=float)
    for j_idx, jid in enumerate(joint_ids):
        cols = [coln(jid, ax) for ax in ("x", "y", "z")]
        # いずれかの列が存在しない場合は NaN を入れる
        if not all(c in df.columns for c in cols):
            p3d[:, j_idx, :] = np.nan
            continue
        p3d[:, j_idx, 0] = df[cols[0]].to_numpy(dtype=float)
        p3d[:, j_idx, 1] = df[cols[1]].to_numpy(dtype=float)
        p3d[:, j_idx, 2] = df[cols[2]].to_numpy(dtype=float)

    # ID -> 行インデックスの対応表
    id2row = {jid: i for i, jid in enumerate(joint_ids)}
    # p3d を 0..maxID に拡張して ID でアクセス可能に
    max_id = max(joint_ids)
    p3d_full = np.full((T, max_id + 1, 3), np.nan, dtype=float)
    for jid, idx in id2row.items():
        p3d_full[:, jid, :] = p3d[:, idx, :]

    m_upper = args.body_mass * args.upperarm_frac
    m_fore = args.body_mass * args.forearm_frac

    # Decide ID mapping: prefer MediaPipe if available, else fallback to legacy 0..5
    have_mp = all(j in id2row for j in (11, 12, 13, 14, 15, 16))
    ids_map = MP_IDS if have_mp else LEGACY_IDS

    # 上向き軸ベクトル
    up_axis_map = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
    up_vec = up_axis_map[args.up_axis]

    # 長さキャリブレーション（任意）: 上腕長の中央値を目標値に合わせるスケール s を求め、全座標を s 倍。
    if args.target_upperarm_m and args.target_upperarm_m > 0.0:
        # 可能な側の上腕長を収集
        lengths = []
        for side in ("R", "L"):
            ids = ids_map[side]
            sh, el = ids["shoulder"], ids["elbow"]
            if max(sh, el) < p3d_full.shape[1]:
                S = p3d_full[:, sh, :]
                E = p3d_full[:, el, :]
                L = np.linalg.norm(E - S, axis=1)
                if np.isfinite(L).any():
                    lengths.append(np.nanmedian(L))
        if lengths:
            L_med = float(np.nanmedian(np.array(lengths)))
            if L_med > 1e-6:
                s = args.target_upperarm_m / L_med
                p3d_full = p3d_full * s
                print(f"[estimate] length scale applied: s={s:.3f} (upperarm med {L_med:.3f} -> {args.target_upperarm_m:.3f} m)")

    side_out = {}
    for side in ("R", "L"):
        side_out[side] = estimate_side(
            p3d_full,
            ids_map[side],
            args.fps,
            m_upper,
            m_fore,
            up_axis=up_vec,
            include_gravity=args.include_gravity,
            g_mag=args.g,
            com_frac_upper=args.com_frac_upper,
            com_frac_fore=args.com_frac_fore,
            hand_load_kg=args.hand_load_kg,
        )

    # 出力DataFrame
    frames = df["frame"].to_numpy(dtype=np.int64)
    out = {
        "frame": frames,
        # 右
        "wrist_R_x": side_out["R"]["wrist"][:, 0],
        "wrist_R_y": side_out["R"]["wrist"][:, 1],
        "wrist_R_z": side_out["R"]["wrist"][:, 2],
        "elbow_R_x": side_out["R"]["elbow"][:, 0],
        "elbow_R_y": side_out["R"]["elbow"][:, 1],
        "elbow_R_z": side_out["R"]["elbow"][:, 2],
        "shoulder_R_x": side_out["R"]["shoulder"][:, 0],
        "shoulder_R_y": side_out["R"]["shoulder"][:, 1],
        "shoulder_R_z": side_out["R"]["shoulder"][:, 2],
        # 左
        "wrist_L_x": side_out["L"]["wrist"][:, 0],
        "wrist_L_y": side_out["L"]["wrist"][:, 1],
        "wrist_L_z": side_out["L"]["wrist"][:, 2],
        "elbow_L_x": side_out["L"]["elbow"][:, 0],
        "elbow_L_y": side_out["L"]["elbow"][:, 1],
        "elbow_L_z": side_out["L"]["elbow"][:, 2],
        "shoulder_L_x": side_out["L"]["shoulder"][:, 0],
        "shoulder_L_y": side_out["L"]["shoulder"][:, 1],
        "shoulder_L_z": side_out["L"]["shoulder"][:, 2],
    }
    out_df = pd.DataFrame(out)

    if args.out is None:
        id_str = infer_id_from_pose_path(args.pose_csv)
        out_path = os.path.join(os.path.dirname(args.pose_csv), os.pardir)
        out_path = os.path.normpath(out_path)
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"aim_torque_vec_{id_str}_est.csv")
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved estimated torque CSV: {out_path} (T={len(frames)})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
