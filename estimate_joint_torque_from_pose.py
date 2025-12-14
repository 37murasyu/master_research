"""estimate_joint_torque_from_pose.py

姿勢CSVから上肢トルクを推定するユーティリティ。従来の簡易モデルでは慣性項のみを考慮していましたが、
本版では `compute_torque_from_pose` の逆動力学パイプラインを呼び出し、慣性・遠心/コリオリ・重力・
セグメント間の力伝播を含む完全な剛体モデルでトルクを算出します。
出力CSVの列構成は従来と同じ (frame + 各部位の x/y/z トルク) です。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from compute_torque_from_pose import (
    LEFT_SEGMENTS,
    OUTPUT_PART_ORDER,
    RIGHT_SEGMENTS,
    build_output,
    compute_side_torques,
    interpolate_and_smooth,
    load_pose_csv,
)
from utils import compute_local_torque


LEGACY_TO_MP = {
    0: 12,  # shoulder_R
    2: 14,  # elbow_R
    4: 16,  # wrist_R
    1: 11,  # shoulder_L
    3: 13,  # elbow_L
    5: 15,  # wrist_L
}

FOREARM_MASS_FRAC = 0.0160  # 体重に対する前腕質量比


def infer_id_from_pose_path(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith("stereo_") and base.endswith("_pose.csv"):
        return base[len("stereo_"):-len("_pose.csv")]
    return os.path.splitext(base)[0]


def promote_legacy_indices(pose: np.ndarray) -> np.ndarray:
    required_max = 16
    if pose.shape[1] > required_max:
        return pose
    T = pose.shape[0]
    promoted = np.full((T, required_max + 1, 3), np.nan, dtype=pose.dtype)
    promoted[:, :pose.shape[1], :] = pose
    for legacy, mp in LEGACY_TO_MP.items():
        if legacy >= pose.shape[1]:
            continue
        has_modern = mp < pose.shape[1] and np.isfinite(pose[:, mp, :]).any()
        if has_modern:
            promoted[:, mp, :] = pose[:, mp, :]
            continue
        promoted[:, mp, :] = pose[:, legacy, :]
    return promoted


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="姿勢CSVから上肢トルク(逆動力学)を推定してCSV出力")
    ap.add_argument("--pose-csv", required=True, help="joint_* 列を含む姿勢CSV")
    ap.add_argument("--fps", type=float, default=30.0, help="サンプリング周波数[Hz]")
    ap.add_argument("--body-mass", type=float, default=60.0, help="体重[kg]")
    ap.add_argument("--upperarm-frac", type=float, default=0.028, help="互換確保用: 利用しません")
    ap.add_argument("--forearm-frac", type=float, default=0.016, help="互換確保用: 利用しません")
    ap.add_argument("--up-axis", choices=["x", "y", "z"], default="y", help="重力が作用する軸 (正向きが上)")
    ap.add_argument("--include-gravity", action="store_true", help="重力トルクを含める")
    ap.add_argument("--g", type=float, default=9.81, help="重力加速度 [m/s^2]")
    ap.add_argument("--com-frac-upper", type=float, default=0.5, help="互換確保用: 利用しません")
    ap.add_argument("--com-frac-fore", type=float, default=0.5, help="互換確保用: 利用しません")
    ap.add_argument("--hand-load-kg", type=float, default=0.0, help="互換確保用: 現行パイプラインでは未対応")
    ap.add_argument("--prefer-raw", action="store_true", help="joint_*_f より生データを優先して使用")
    ap.add_argument("--skip-smoothing", action="store_true", help="サビツキー・ゴレイ平滑を無効化")
    ap.add_argument("--savgol-window", type=int, default=7, help="Savitzky-Golay 窓長 (奇数)" )
    ap.add_argument("--savgol-poly", type=int, default=3, help="Savitzky-Golay 多項式次数")
    ap.add_argument("--target-upperarm-m", type=float, default=0.0, help="上腕中央値をこの長さにスケール")
    ap.add_argument("--keep-local", action="store_true", help="_local_* 列もCSVに残す")
    ap.add_argument(
        "--support-body-weight",
        action="store_true",
        help="両手首で体重を支持する前提で、手首/肘/肩トルクに追加荷重を反映",
    )
    ap.add_argument(
        "--support-share",
        type=float,
        default=0.5,
        help="胴体質量を片腕がどの割合で支えるか (0.0-1.0)",
    )
    ap.add_argument(
        "--support-moment-arm",
        type=float,
        default=0.05,
        help="手首支持荷重のモーメントアーム[m] (0 で追加モーメント無し)",
    )
    ap.add_argument("--out", default=None, help="出力CSVパス")
    return ap.parse_args(argv)


def gravity_vector(include_gravity: bool, g_mag: float, up_axis: str) -> np.ndarray:
    if not include_gravity or g_mag <= 0.0:
        return np.zeros(3, dtype=np.float64)
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    up_dir = axis_map[up_axis]
    return -abs(g_mag) * up_dir


def apply_upperarm_scale(pose: np.ndarray, target_m: float) -> float:
    if target_m <= 0.0:
        return 1.0
    lengths: List[float] = []
    for seg in list(RIGHT_SEGMENTS) + list(LEFT_SEGMENTS):
        if "upper_arm" not in seg.name:
            continue
        prox = pose[:, seg.proximal_joint, :]
        dist = pose[:, seg.distal_joint, :]
        seg_len = np.linalg.norm(dist - prox, axis=1)
        if np.isfinite(seg_len).any():
            lengths.append(float(np.nanmedian(seg_len)))
    if not lengths:
        return 1.0
    current = float(np.nanmedian(np.array(lengths)))
    if current <= 1e-6:
        return 1.0
    scale = target_m / current
    pose *= scale
    return scale


def global_columns() -> List[str]:
    cols: List[str] = []
    for part in OUTPUT_PART_ORDER:
        cols.extend([f"{part}_x", f"{part}_y", f"{part}_z"])
    return cols


def local_columns() -> List[str]:
    cols: List[str] = []
    for part in OUTPUT_PART_ORDER:
        cols.extend([f"{part}_local_x", f"{part}_local_y", f"{part}_local_z"])
    return cols


def write_meta(meta_path: str, frames: int, body_mass: float, gravity: np.ndarray, fps: float, dt: float, shapes: Dict[str, tuple]) -> None:
    meta = {
        "frames": frames,
        "body_mass": float(body_mass),
        "gravity": gravity.tolist(),
        "fps": float(fps),
        "dt": float(dt),
        "columns": {
            "global": global_columns(),
            "local": local_columns(),
        },
        "shapes": {k: list(v) for k, v in shapes.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as fw:
        json.dump(meta, fw, ensure_ascii=False, indent=2)


def safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(v)
    return v / n


def central_diff_scalar(series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(series, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    T = len(x)
    first = np.zeros(T, dtype=float)
    second = np.zeros(T, dtype=float)
    if T >= 3:
        first[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        second[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt * dt)
    if T >= 2:
        first[0] = (x[1] - x[0]) / dt
        first[-1] = (x[-1] - x[-2]) / dt
        if T >= 3:
            second[0] = (x[2] - 2.0 * x[1] + x[0]) / (dt * dt)
            second[-1] = (x[-1] - 2.0 * x[-2] + x[-3]) / (dt * dt)
    return first, second


def central_diff_vec(series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    vel = np.zeros_like(series, dtype=float)
    acc = np.zeros_like(series, dtype=float)
    for ax in range(series.shape[1]):
        vel[:, ax], acc[:, ax] = central_diff_scalar(series[:, ax], dt)
    return vel, acc


def compute_wrist_torque_simple(
    pose: np.ndarray,
    side: str,
    body_mass: float,
    dt: float,
    support_force_vec: Optional[np.ndarray] = None,
    support_moment_arm: float = 0.0,
) -> np.ndarray:
    mp_ids = {
        "R": {"elbow": 14, "wrist": 16},
        "L": {"elbow": 13, "wrist": 15},
    }
    ids = mp_ids[side]
    if ids["elbow"] >= pose.shape[1] or ids["wrist"] >= pose.shape[1]:
        return np.zeros((pose.shape[0], 3), dtype=float)
    elbow = pose[:, ids["elbow"], :]
    wrist = pose[:, ids["wrist"], :]
    link = wrist - elbow
    link_norm = np.linalg.norm(link, axis=1)
    valid = link_norm > 1e-6
    if not np.any(valid):
        return np.zeros((pose.shape[0], 3), dtype=float)
    unit = np.zeros_like(link)
    unit[valid] = (link[valid].T / link_norm[valid]).T
    _, dd_unit = central_diff_vec(unit, dt)
    alpha = np.cross(unit, dd_unit)
    forearm_mass = body_mass * FOREARM_MASS_FRAC
    length = float(np.nanmedian(link_norm[valid])) if np.any(valid) else 0.0
    if length <= 1e-6 or forearm_mass <= 0.0:
        return np.zeros((pose.shape[0], 3), dtype=float)
    inertia = (1.0 / 3.0) * forearm_mass * (length ** 2)
    tau = alpha * inertia
    tau = np.where(np.isfinite(tau), tau, 0.0)

    if support_force_vec is not None:
        force_vec = np.asarray(support_force_vec, dtype=float)
        if force_vec.shape == (3,):
            force_mag = float(np.linalg.norm(force_vec))
            if force_mag > 0.0:
                force_unit = force_vec / force_mag
                # 直交方向ベクトルとモーメントアーム（水平成分が極端に小さい場合は support_moment_arm を下限値として使用）
                axes = np.cross(unit, force_unit)
                axes_norm = np.linalg.norm(axes, axis=1)
                moment_arm = link_norm * axes_norm
                if support_moment_arm > 0.0:
                    moment_arm = np.maximum(moment_arm, support_moment_arm)
                else:
                    moment_arm = np.where(moment_arm > 0.0, moment_arm, 0.0)
                tau_support = np.zeros_like(tau)
                valid_axes = axes_norm > 1e-6
                if np.any(valid_axes):
                    axes_unit = np.zeros_like(axes)
                    axes_unit[valid_axes] = (axes[valid_axes].T / axes_norm[valid_axes]).T
                    tau_support[valid_axes] = axes_unit[valid_axes] * (force_mag * moment_arm[valid_axes])[:, None]
                tau += tau_support
    return tau


def compute_local_from_global(tau_global: np.ndarray, link: np.ndarray) -> np.ndarray:
    local = np.zeros_like(tau_global, dtype=float)
    for i in range(len(tau_global)):
        local[i] = compute_local_torque(tau_global[i], link[i])
    return local


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.hand_load_kg != 0.0:
        raise SystemExit("--hand-load-kg は現行の逆動力学実装では未対応です")
    if args.upperarm_frac != 0.028 or args.forearm_frac != 0.016:
        print("[WARN] upperarm/forearm frac は固定SegmentSpecを使用するため無視します")
    if args.com_frac_upper != 0.5 or args.com_frac_fore != 0.5:
        print("[WARN] com fraction 引数は無視されます (SegmentSpec定義を使用)")

    fps = args.fps if args.fps > 0 else 30.0
    dt = 1.0 / fps

    frames, pose_raw = load_pose_csv(args.pose_csv, prefer_filtered=not args.prefer_raw)
    pose_full = promote_legacy_indices(pose_raw)
    scale = apply_upperarm_scale(pose_full, args.target_upperarm_m)
    if scale != 1.0:
        print(f"[INFO] length scale applied: s={scale:.4f}")

    pose_interp = interpolate_and_smooth(
        pose_full,
        skip_smoothing=args.skip_smoothing,
        window=args.savgol_window,
        poly=args.savgol_poly,
    )

    support_force_vec = None
    support_joint_masses = None
    gravity = gravity_vector(args.include_gravity or args.support_body_weight, args.g, args.up_axis)
    body_mass = args.body_mass

    support_mass_each = 0.0
    upper_mass_each = body_mass * RIGHT_SEGMENTS[0].mass_fraction
    fore_mass_each = body_mass * RIGHT_SEGMENTS[1].mass_fraction
    if args.support_body_weight:
        arm_mass_frac = sum(seg.mass_fraction for seg in RIGHT_SEGMENTS)
        arm_mass = body_mass * arm_mass_frac
        torso_mass = max(body_mass - 2.0 * arm_mass, 0.0)
        share = float(np.clip(args.support_share, 0.0, 1.0))
        support_mass_each = torso_mass * share
        if np.linalg.norm(gravity) <= 0.0:
            gravity = gravity_vector(True, args.g, args.up_axis)
        if support_mass_each > 0.0 and np.linalg.norm(gravity) > 0.0:
            support_mass_elbow = support_mass_each + upper_mass_each
            support_joint_masses = np.array([support_mass_each, support_mass_elbow], dtype=float)
            total_wrist_mass = support_mass_elbow + fore_mass_each
            support_force_vec = -total_wrist_mass * gravity
            print(
                "[INFO] support load per wrist: torso={:.3f} kg, upper={:.3f} kg, forearm={:.3f} kg, |F|={:.2f} N".format(
                    support_mass_each,
                    upper_mass_each,
                    fore_mass_each,
                    np.linalg.norm(support_force_vec),
                )
            )

    tau_g_right, tau_l_right = compute_side_torques(
        pose_interp,
        RIGHT_SEGMENTS,
        body_mass,
        dt,
        gravity,
        support_joint_masses=support_joint_masses,
    )
    tau_g_left, tau_l_left = compute_side_torques(
        pose_interp,
        LEFT_SEGMENTS,
        body_mass,
        dt,
        gravity,
        support_joint_masses=support_joint_masses,
    )

    # Wrist torques are not part of the rigid-body chain; compute heuristic values to retain energy pipeline compatibility.
    wrist_R_global = compute_wrist_torque_simple(
        pose_interp,
        "R",
        body_mass,
        dt,
        support_force_vec=support_force_vec,
        support_moment_arm=args.support_moment_arm,
    )
    wrist_L_global = compute_wrist_torque_simple(
        pose_interp,
        "L",
        body_mass,
        dt,
        support_force_vec=support_force_vec,
        support_moment_arm=args.support_moment_arm,
    )
    wrist_R_link = pose_interp[:, 16, :] - pose_interp[:, 14, :] if pose_interp.shape[1] > 16 else np.zeros_like(wrist_R_global)
    wrist_L_link = pose_interp[:, 15, :] - pose_interp[:, 13, :] if pose_interp.shape[1] > 15 else np.zeros_like(wrist_L_global)
    wrist_R_local = compute_local_from_global(wrist_R_global, wrist_R_link)
    wrist_L_local = compute_local_from_global(wrist_L_global, wrist_L_link)

    df_all, meta_shapes = build_output(frames, tau_g_right, tau_l_right, tau_g_left, tau_l_left)

    for side, g_vals, l_vals in (("R", wrist_R_global, wrist_R_local), ("L", wrist_L_global, wrist_L_local)):
        df_all[f"wrist_{side}_x"] = g_vals[:, 0]
        df_all[f"wrist_{side}_y"] = g_vals[:, 1]
        df_all[f"wrist_{side}_z"] = g_vals[:, 2]
        df_all[f"wrist_{side}_local_x"] = l_vals[:, 0]
        df_all[f"wrist_{side}_local_y"] = l_vals[:, 1]
        df_all[f"wrist_{side}_local_z"] = l_vals[:, 2]

    if args.keep_local:
        df_out = df_all
    else:
        df_out = df_all[["frame", *global_columns()]]

    if args.out is None:
        id_str = infer_id_from_pose_path(args.pose_csv)
        base_dir = os.path.dirname(os.path.abspath(args.pose_csv))
        out_dir = os.path.normpath(os.path.join(base_dir, os.pardir, "torque"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"aim_torque_vec_{id_str}_est.csv")
    else:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    df_out.to_csv(out_path, index=False)

    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    write_meta(meta_path, len(frames), body_mass, gravity, fps, dt, meta_shapes)

    print(f"Saved estimated torque CSV: {out_path} (frames={len(frames)})")
    print(f"Saved metadata JSON: {meta_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
