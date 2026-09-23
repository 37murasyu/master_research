"""Offline inverse-dynamics for prerecorded 3D pose CSVs.

Reads a MediaPipe-style pose CSV, interpolates missing samples, derives upper
limb segment kinematics, and invokes `utils_dynamic` helpers to recover joint
torques in both global and local frames.

既定では座位プッシュアップのモデル（``push_up_model``、KNOWN_ISSUES §2-1）で解く:
手を固定端に前腕 → 上腕の鎖を解き、体幹＋頭の荷重を肩に載せる。重力は初期フレームの
体幹の向きから決める（§1-5）。入力の座標系（カメラ座標で y が下、など）には依らない。

Example:
    python compute_torque_from_pose.py \
        --pose-csv output_data/poses/kpts3d_subject5_20250925_133228_filtpos.csv \
        --out-dir output_data/torque --save-npy
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import INERTIA_LENGTH_FRAMES
from config import MP_LANDMARK
from config import OUTPUT_SCHEMA_VERSION
from config import SUPPORT_SHARE_DEFAULT
from config import g as CONFIG_GRAVITY
from config import w as CONFIG_BODY_MASS
from push_up_model import (
    GRAVITY_MODES,
    GravityEstimate,
    SegmentState,
    estimate_gravity,
    hand_mass,
    hand_point,
    inertia_about_link,
    joint_axes,
    push_up_torques,
    torso_load_mass,
    trunk_up_vectors,
    wrist_hand_mask,
)
from utils import LocalFrameFallbackWarning, compute_local_torque
from utils_dynamic import calculate_inertia_tensor, compute_MF_batch_native, compute_tau_chain_native

# ---------------------------------------------------------------------------
# Segment definitions and configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentSpec:
    """Rigid segment definition used by the inverse-dynamics chain."""

    name: str
    proximal_joint: int
    distal_joint: int
    inertia_row: int
    mass_fraction: float
    com_fraction: float


RIGHT_SEGMENTS: Tuple[SegmentSpec, ...] = (
    SegmentSpec("upper_arm_R", 12, 14, 3, 0.0227, 0.436),
    SegmentSpec("forearm_R", 14, 16, 4, 0.0160, 0.430),
)
LEFT_SEGMENTS: Tuple[SegmentSpec, ...] = (
    SegmentSpec("upper_arm_L", 11, 13, 3, 0.0227, 0.436),
    SegmentSpec("forearm_L", 13, 15, 4, 0.0160, 0.430),
)

# 手を固定端にした鎖（0: 前腕、関節 = 手首 / 1: 上腕、関節 = 肘）。push_up_model に渡す部位の並び。
# 重心比は近位端（肘・肩）から測った文献値を、手首・肘側から測り直したもの。
WRIST_BASE_SEGMENTS_RIGHT: Tuple[SegmentSpec, ...] = (
    SegmentSpec("forearm_R_wrist", 16, 14, 4, 0.0160, 1.0 - 0.430),
    SegmentSpec("upper_arm_R_wrist", 14, 12, 3, 0.0227, 1.0 - 0.436),
)
WRIST_BASE_SEGMENTS_LEFT: Tuple[SegmentSpec, ...] = (
    SegmentSpec("forearm_L_wrist", 15, 13, 4, 0.0160, 1.0 - 0.430),
    SegmentSpec("upper_arm_L_wrist", 13, 11, 3, 0.0227, 1.0 - 0.436),
)

# --no-wrist-base の自由振りの鎖（腕を肩から吊る）で、部位 → 出力する関節。
SEGMENT_TO_OUTPUT = {
    "upper_arm_R": "shoulder_R",
    "forearm_R": "elbow_R",
    "upper_arm_L": "shoulder_L",
    "forearm_L": "elbow_L",
}

JOINTS = ("wrist", "elbow", "shoulder")

# どの部位がどの部位に加えるトルクか（config.OUTPUT_SCHEMA_VERSION の v2 の説明と同じ）
TORQUE_CONVENTION = (
    "wrist: hand->forearm, elbow: forearm->upper_arm (hand is the fixed end), "
    "shoulder: trunk->hanging arm; local y = parent x link (push_up_model.joint_axes)"
)

OUTPUT_PART_ORDER = [
    "wrist_R",
    "elbow_R",
    "shoulder_R",
    "wrist_L",
    "elbow_L",
    "shoulder_L",
]
OUTPUT_GLOBAL_COLS = [f"{part}_{axis}" for part in OUTPUT_PART_ORDER for axis in ("x", "y", "z")]
OUTPUT_LOCAL_COLS = [f"{part}_local_{axis}" for part in OUTPUT_PART_ORDER for axis in ("x", "y", "z")]

# Defaults sourced from config
DEFAULT_FPS = 30.0
DEFAULT_BODY_MASS = float(CONFIG_BODY_MASS)
DEFAULT_GRAVITY = np.array(CONFIG_GRAVITY, dtype=np.float64)
COLUMN_TEMPLATES = ("joint_{jid}_{axis}_f", "joint_{jid}_{axis}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute joint torques from pose CSV via inverse dynamics")
    parser.add_argument("--pose-csv", required=True, help="Input pose CSV with joint_{id}_{axis} columns")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Sampling rate [Hz] (default: 30)")
    parser.add_argument("--body-mass", type=float, default=DEFAULT_BODY_MASS, help="Body mass [kg] (config default)")
    parser.add_argument("--gravity", type=float, default=None, help="Override gravity magnitude (positive scalar)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: pose_dir/../torque)")
    parser.add_argument("--prefix", default=None, help="Output prefix (default: pose filename stem)")
    parser.add_argument("--save-npy", action="store_true", help="Also save NumPy arrays for torques and frames")
    parser.add_argument("--skip-smoothing", action="store_true", help="Disable Savitzky-Golay smoothing")
    parser.add_argument("--window", type=int, default=7, help="Savitzky-Golay window length (odd >=5)")
    parser.add_argument("--poly", type=int, default=3, help="Savitzky-Golay polynomial order")
    parser.add_argument("--debug", action="store_true", help="Verbose diagnostics")
    parser.add_argument("--pos-scale", type=float, default=1.0, help="Scale factor applied to positions (e.g., 0.01 if CSV is cm)")
    parser.add_argument("--dumbbell-mass-right", type=float, default=0.0,
                        help="External load mass at right wrist [kg] (--no-wrist-base only)")
    parser.add_argument("--dumbbell-mass-left", type=float, default=0.0,
                        help="External load mass at left wrist [kg] (--no-wrist-base only)")
    parser.add_argument(
        "--wrist-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="座位プッシュアップのモデル（手を固定端に前腕→上腕、体幹＋頭の荷重を肩に載せる）で解く。"
             "既定オン。--no-wrist-base で腕を肩から吊る旧来の鎖だけを解く（手首は 0）",
    )
    parser.add_argument(
        "--gravity-mode",
        choices=GRAVITY_MODES,
        default="axis",
        help="重力の決め方。axis: 初期フレームの体幹の向きに最も近い座標軸（カメラが水平な前提、既定）。"
             "trunk: 体幹の向きそのもの",
    )
    parser.add_argument(
        "--gravity-frames",
        type=int,
        default=INERTIA_LENGTH_FRAMES,
        help="重力の推定に使う先頭のフレーム数（試技前の安静座位）",
    )
    parser.add_argument(
        "--support-share",
        type=float,
        default=SUPPORT_SHARE_DEFAULT,
        help="Fraction of torso mass assigned to a single arm when wrist-base mode is active (0-1)",
    )
    parser.add_argument(
        "--torso-mass",
        type=float,
        default=None,
        help="Override torso mass [kg] treated as external load in wrist-base mode",
    )
    # かつて既定 0.01（N·cm → N·m 用）で、m 単位の入力（Adjusted 3D Pose/*.csv）では
    # トルクが黙って 1/100 になっていた。
    parser.add_argument(
        "--torque-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to all output torques (default 1: positions in metres give N·m)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# CSV ingestion and smoothing
# ---------------------------------------------------------------------------


def load_pose_csv(path: str, prefer_filtered: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "frame" not in df.columns:
        raise ValueError("Pose CSV must contain a 'frame' column")
    frames = df["frame"].to_numpy(dtype=np.int64)

    joint_ids = sorted(
        {
            int(col.split("_")[1])
            for col in df.columns
            if col.startswith("joint_") and col.split("_")[1].isdigit()
        }
    )
    if not joint_ids:
        raise ValueError("No joint_* columns found in pose CSV")

    max_id = max(joint_ids)
    pose = np.full((len(frames), max_id + 1, 3), np.nan, dtype=np.float64)
    for jid in joint_ids:
        for axis_idx, axis in enumerate(("x", "y", "z")):
            col_name = None
            candidates: Iterable[str] = COLUMN_TEMPLATES if prefer_filtered else reversed(COLUMN_TEMPLATES)
            for tmpl in candidates:
                candidate = tmpl.format(jid=jid, axis=axis)
                if candidate in df.columns:
                    col_name = candidate
                    break
            if col_name is None:
                raise ValueError(f"Missing column for joint {jid} axis {axis}")
            # copy=True が要る。pandas の Copy-on-Write では to_numpy() が
            # 読み取り専用のビューを返すことがあり、次行の代入が
            # ValueError: assignment destination is read-only で落ちる。
            series = df[col_name].to_numpy(dtype=np.float64, copy=True)
            series[series == -1.0] = np.nan
            pose[:, jid, axis_idx] = series
    return frames, pose


def interpolate_and_smooth(
    pose: np.ndarray,
    skip_smoothing: bool,
    window: int,
    poly: int,
) -> np.ndarray:
    clean = pose.copy()
    T, J, _ = clean.shape
    for jid in range(J):
        for axis in range(3):
            series = clean[:, jid, axis]
            mask = ~np.isfinite(series)
            if mask.all():
                continue
            if mask.any():
                valid_idx = np.where(~mask)[0]
                fill_idx = np.where(mask)[0]
                clean[mask, jid, axis] = np.interp(fill_idx, valid_idx, series[valid_idx])
    if skip_smoothing or T < 5:
        return clean
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return clean
    window = max(5, window if window % 2 == 1 else window + 1)
    window = min(window, T if T % 2 == 1 else T - 1)
    if window < 5:
        return clean
    poly = max(2, min(poly, window - 1))

    # load_pose_csv は max(joint_id) + 1 の疎配列を作るので、使わない関節
    # （MediaPipe の 0〜10 や 17〜22）が全 NaN のまま残る。上のループはそれを
    # `if mask.all(): continue` で飛ばすだけなので NaN が残り、そのまま
    # savgol_filter に渡すと ValueError: array must not contain infs or NaNs で落ちる。
    # 有効な関節だけに適用する。NaN を 0 で埋めると偽の座標が下流に流れるので避ける。
    usable = np.isfinite(clean).all(axis=(0, 2))
    if not usable.any():
        return clean
    smoothed = clean.copy()
    smoothed[:, usable, :] = savgol_filter(
        clean[:, usable, :], window_length=window, polyorder=poly, axis=0, mode="interp"
    )
    return smoothed


# ---------------------------------------------------------------------------
# Kinematics utilities
# ---------------------------------------------------------------------------


def central_diff(series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    first = np.zeros_like(series)
    second = np.zeros_like(series)
    if len(series) >= 3:
        first[1:-1] = (series[2:] - series[:-2]) / (2 * dt)
        second[1:-1] = (series[2:] - 2 * series[1:-1] + series[:-2]) / (dt * dt)
    if len(series) >= 2:
        first[0] = (series[1] - series[0]) / dt
        first[-1] = (series[-1] - series[-2]) / dt
        if len(series) >= 3:
            second[0] = (series[2] - 2 * series[1] + series[0]) / (dt * dt)
            second[-1] = (series[-1] - 2 * series[-2] + series[-3]) / (dt * dt)
    return first, second


def central_diff_vec(series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    vel = np.zeros_like(series)
    acc = np.zeros_like(series)
    for axis in range(series.shape[1]):
        vel[:, axis], acc[:, axis] = central_diff(series[:, axis], dt)
    return vel, acc


def compute_segment_kinematics(
    pose: np.ndarray,
    segments: Sequence[SegmentSpec],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = pose.shape[0]
    N = len(segments)
    omegas = np.zeros((T, N, 3), dtype=np.float64)
    domegas = np.zeros((T, N, 3), dtype=np.float64)
    com_pos = np.zeros((T, N, 3), dtype=np.float64)
    com_acc = np.zeros((T, N, 3), dtype=np.float64)
    joint_pos = np.zeros((T, N, 3), dtype=np.float64)
    link_vec = np.zeros((T, N, 3), dtype=np.float64)

    for idx, seg in enumerate(segments):
        prox = pose[:, seg.proximal_joint, :]
        dist = pose[:, seg.distal_joint, :]
        link = dist - prox
        link_vec[:, idx, :] = link
        joint_pos[:, idx, :] = prox
        com_pos[:, idx, :] = prox + seg.com_fraction * link

        link_vel, link_acc = central_diff_vec(link, dt)
        cross_r_v = np.cross(link, link_vel)
        cross_r_a = np.cross(link, link_acc)
        dot_r_v = np.sum(link * link_vel, axis=1, keepdims=True)
        norm_sq = np.sum(link * link, axis=1, keepdims=True)
        safe_norm_sq = np.where(norm_sq > 1e-8, norm_sq, 1e-8)
        inv_norm_sq = 1.0 / safe_norm_sq

        omegas[:, idx, :] = cross_r_v * inv_norm_sq
        domegas[:, idx, :] = cross_r_a * inv_norm_sq - 2.0 * dot_r_v * cross_r_v * (inv_norm_sq ** 2)

        _, com_acc[:, idx, :] = central_diff_vec(com_pos[:, idx, :], dt)

    return omegas, domegas, com_pos, com_acc, joint_pos, link_vec


# ---------------------------------------------------------------------------
# Inverse dynamics helpers
# ---------------------------------------------------------------------------


def build_side_inverse_inputs(
    pose: np.ndarray,
    segments: Sequence[SegmentSpec],
    body_mass: float,
    dt: float,
    gravity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _ = gravity  # reserved for potential future use (keeps signature aligned with callers)
    omegas, domegas, com_pos, com_acc, joint_pos, link_vec = compute_segment_kinematics(pose, segments, dt)
    N = len(segments)
    inertia_tensors = np.zeros((N, 3, 3), dtype=np.float64)
    masses = np.zeros(N, dtype=np.float64)
    lengths = np.linalg.norm(link_vec, axis=2)
    for idx, seg in enumerate(segments):
        length = float(np.nanmedian(lengths[:, idx]))
        length = max(length, 1e-4)
        mass = body_mass * seg.mass_fraction
        masses[idx] = mass
        # 回帰式 I = a*w + b*l + c（utils_dynamic.py:50）の w は全身体重であって
        # 部位質量ではない。部位質量を渡すと a*w の項が小さすぎ、定数項 c が
        # 効いて対角成分が負になる（前腕は全軸で負になっていた）。
        inertia_tensors[idx] = calculate_inertia_tensor(seg.inertia_row, body_mass, length)
    ddpg = com_acc
    return inertia_tensors, masses, omegas, domegas, ddpg, com_pos, joint_pos, link_vec
def _broadcast_vector(value: Optional[np.ndarray], length: int, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"{name} must have length 3")
        return np.broadcast_to(arr, (length, 3)).astype(np.float64, copy=False)
    if arr.ndim == 2:
        if arr.shape != (length, 3):
            raise ValueError(f"{name} must have shape ({length}, 3)")
        return arr.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be shape (3,) or (T, 3)")


def run_side_inverse_dynamics(
    inertia_tensors: np.ndarray,
    masses: np.ndarray,
    omegas: np.ndarray,
    domegas: np.ndarray,
    ddpg: np.ndarray,
    com_pos: np.ndarray,
    joint_pos: np.ndarray,
    link_vec: np.ndarray,
    gravity_vec: np.ndarray,
    segments: Sequence[SegmentSpec],
    support_joint_masses: Optional[np.ndarray] = None,
    external_force: Optional[np.ndarray] = None,
    external_point: Optional[np.ndarray] = None,
    root_parent_vec: Optional[np.ndarray] = None,
    up_axis: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    T, N, _ = omegas.shape
    tau_global = np.zeros((T, N, 3), dtype=np.float64)
    tau_local = np.zeros((T, N, 3), dtype=np.float64)
    zero3 = np.zeros(3, dtype=np.float64)
    gravity = np.asarray(gravity_vec, dtype=np.float64)

    ext_force_arr = _broadcast_vector(external_force, T, "external_force")
    ext_point_arr = _broadcast_vector(external_point, T, "external_point")
    root_parent_arr = _broadcast_vector(root_parent_vec, T, "root_parent_vec")

    support_forces = None
    segment_names = [seg.name for seg in segments]
    if support_joint_masses is not None:
        joint_mass = np.asarray(support_joint_masses, dtype=np.float64)
        if joint_mass.ndim == 1 and joint_mass.shape[0] == N:
            support_forces = np.zeros((N, 3), dtype=np.float64)
            for idx in range(N):
                if idx == N - 1:
                    mass = joint_mass[idx]
                else:
                    mass = joint_mass[idx] - joint_mass[idx + 1]
                if mass == 0.0:
                    continue
                support_forces[idx] = -mass * gravity

    support_abs_accum = None
    base_abs_accum = None
    support_frame_count = 0
    if support_forces is not None:
        support_abs_accum = np.zeros((N, 3), dtype=np.float64)
        base_abs_accum = np.zeros((N, 3), dtype=np.float64)

    for t in range(T):
        f_ext = ext_force_arr[t] if ext_force_arr is not None else zero3
        r_x = ext_point_arr[t] if ext_point_arr is not None else zero3
        # 部位固定系の慣性テンソルを、このフレームのリンクの向きに合わせて回す（§2-3）
        inertia_now = np.stack([inertia_about_link(inertia_tensors[i], link_vec[t, i]) for i in range(N)])
        M, F_base = compute_MF_batch_native(
            inertia_now,
            masses,
            omegas[t],
            domegas[t],
            ddpg[t],
            gravity,
        )
        tau = compute_tau_chain_native(
            Ms=M,
            Fs=F_base,
            r_gs=com_pos[t],
            p1s=joint_pos[t],
            tau_E=zero3,
            f_E=f_ext,
            r_x=r_x,
        )
        if support_forces is not None:
            F_with = F_base + support_forces
            tau_with = compute_tau_chain_native(
                Ms=M,
                Fs=F_with,
                r_gs=com_pos[t],
                p1s=joint_pos[t],
                tau_E=zero3,
                f_E=f_ext,
                r_x=r_x,
            )
            support_delta = tau_with - tau
            support_abs_accum += np.abs(support_delta)
            base_abs_accum += np.abs(tau)
            support_frame_count += 1
            tau = tau_with
        tau_global[t] = tau
        for n in range(N):
            parent_vec = link_vec[t, n - 1] if n > 0 else None
            if parent_vec is None and root_parent_arr is not None:
                parent_vec = root_parent_arr[t]
            tau_local[t, n] = compute_local_torque(tau[n], link_vec[t, n], parent_vec, up_axis)

    if support_forces is not None and support_frame_count > 0:
        for idx in range(N):
            part_name = segment_names[idx] if idx < len(segment_names) else f"segment_{idx}"
            avg_support = support_abs_accum[idx] / support_frame_count
            avg_base = base_abs_accum[idx] / support_frame_count
            print(
                "[DEBUG] support_vs_base",
                part_name,
                {
                    "avg_support_tau": avg_support.tolist(),
                    "avg_base_tau": avg_base.tolist(),
                },
            )
    return tau_global, tau_local


def compute_side_torques(
    pose: np.ndarray,
    segments: Sequence[SegmentSpec],
    body_mass: float,
    dt: float,
    gravity: np.ndarray,
    support_joint_masses: Optional[np.ndarray] = None,
    external_force: Optional[np.ndarray] = None,
    external_point: Optional[np.ndarray] = None,
    root_parent_vec: Optional[np.ndarray] = None,
    up_axis: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    inputs = build_side_inverse_inputs(pose, segments, body_mass, dt, gravity)
    return run_side_inverse_dynamics(
        *inputs,
        gravity,
        segments,
        support_joint_masses=support_joint_masses,
        external_force=external_force,
        external_point=external_point,
        root_parent_vec=root_parent_vec,
        up_axis=up_axis,
    )


# ---------------------------------------------------------------------------
# 座位プッシュアップのモデル（push_up_model）
# ---------------------------------------------------------------------------


def _landmark(pose: np.ndarray, name: str) -> Optional[np.ndarray]:
    """ランドマーク名の点列 (T, 3)。CSV に無い・全フレーム欠測なら None。"""
    jid = MP_LANDMARK[name]
    if pose.shape[1] <= jid or not np.isfinite(pose[:, jid]).any():
        return None
    return pose[:, jid]


def estimate_pose_gravity(pose: np.ndarray, magnitude: float, frames: int, mode: str) -> GravityEstimate:
    """先頭 frames フレーム（試技前の安静座位）の体幹の向きから重力を決める（§1-5）。

    かつて重力を全体座標の −z に固定していた。入力の Adjusted 3D Pose/*.csv はカメラ座標で
    y が鉛直下向き・z が奥行きなので、重力が奥行き方向を向いていた。一方
    5_1stereo_pose_scaled.csv だけは z が上で、座標系が混在している。ファイルごとに
    体幹から決めれば、どちらでも鉛直下向きになる。
    """
    points = [_landmark(pose, n) for n in ("L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP")]
    if any(p is None for p in points):
        raise ValueError(
            "両肩（ID 11・12）と両腰（ID 23・24）が無いので、体幹の向きから重力を決められない")
    head = slice(0, max(1, int(frames)))
    return estimate_gravity(trunk_up_vectors(*(p[head] for p in points)), magnitude, mode)


def _hand(pose: np.ndarray, side: str) -> Optional[np.ndarray]:
    pinky = _landmark(pose, f"{side}_PINKY")
    index = _landmark(pose, f"{side}_INDEX")
    if pinky is None or index is None:
        return None
    return hand_point(pinky, index)


def compute_push_up_side(
    pose: np.ndarray,
    side: str,
    body_mass: float,
    dt: float,
    gravity: np.ndarray,
    load_mass: float,
    hand_mass_kg: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """片腕の手首・肘・肩のトルクを ``push_up_model`` で解く。

    Returns
    -------
    global_map, local_map : {"wrist" | "elbow" | "shoulder": (T, 3)}
    info : {"local_frame_fallbacks": 局所座標系を作れなかった件数,
            "wrist_hand": 手首の軸を手のひらから作ったフレーム数}
    """
    segments = WRIST_BASE_SEGMENTS_RIGHT if side == "R" else WRIST_BASE_SEGMENTS_LEFT
    inertia, masses, omegas, domegas, com_acc, com_pos, _, link_vec = build_side_inverse_inputs(
        pose, segments, body_mass, dt, gravity)
    wrist = pose[:, MP_LANDMARK[f"{side}_WRIST"]]
    elbow = pose[:, MP_LANDMARK[f"{side}_ELBOW"]]
    shoulder = pose[:, MP_LANDMARK[f"{side}_SHOULDER"]]
    other_shoulder = _landmark(pose, f"{'L' if side == 'R' else 'R'}_SHOULDER")
    hand = _hand(pose, side)
    axes = joint_axes(shoulder, elbow, wrist, hand=hand, other_shoulder=other_shoulder)
    up = -np.asarray(gravity, dtype=np.float64)

    T = pose.shape[0]
    global_map = {j: np.zeros((T, 3)) for j in JOINTS}
    local_map = {j: np.zeros((T, 3)) for j in JOINTS}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LocalFrameFallbackWarning)
        for t in range(T):
            forearm, upper_arm = (
                SegmentState(inertia[i], masses[i], omegas[t, i], domegas[t, i],
                             com_acc[t, i], com_pos[t, i], link_vec[t, i])
                for i in (0, 1))
            tau = push_up_torques(forearm, upper_arm, wrist[t], elbow[t], shoulder[t],
                                  gravity, load_mass, hand_mass_kg)
            for joint in JOINTS:
                link, parent = axes[joint]
                global_map[joint][t] = tau[joint]
                local_map[joint][t] = compute_local_torque(
                    tau[joint], link[t], None if parent is None else parent[t], up)
    fallbacks = 0
    for record in caught:
        if issubclass(record.category, LocalFrameFallbackWarning):
            fallbacks += 1
        else:
            warnings.warn_explicit(record.message, record.category, record.filename, record.lineno)
    hand_frames = 0 if hand is None else int(np.sum(wrist_hand_mask(elbow, wrist, hand)))
    return global_map, local_map, {"local_frame_fallbacks": fallbacks, "wrist_hand": hand_frames}


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def build_output(
    frames: np.ndarray,
    global_map: Dict[str, np.ndarray],
    local_map: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """関節名（wrist_R など）→ (T, 3) の辞書から出力の表を作る。無い関節は 0。"""
    T = len(frames)
    data = {"frame": frames.astype(np.int64)}
    for part in OUTPUT_PART_ORDER:
        g_vals = global_map.get(part, np.zeros((T, 3)))
        l_vals = local_map.get(part, np.zeros((T, 3)))
        for label, vals in (("", g_vals), ("local_", l_vals)):
            if vals.shape != (T, 3):
                raise ValueError(f"{part} must have shape ({T}, 3)")
            for axis_idx, axis in enumerate(("x", "y", "z")):
                data[f"{part}_{label}{axis}"] = vals[:, axis_idx]
    return pd.DataFrame(data)


def save_outputs(
    df: pd.DataFrame,
    out_dir: str,
    prefix: str,
    save_npy: bool,
    global_map: Dict[str, np.ndarray],
    local_map: Dict[str, np.ndarray],
    frames: np.ndarray,
    meta: Dict[str, object],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{prefix}_torque.csv"), index=False)

    meta = {
        **meta,
        "frames": int(len(frames)),
        "columns": {"global": OUTPUT_GLOBAL_COLS, "local": OUTPUT_LOCAL_COLS},
        "npy_joint_order": list(JOINTS),
    }
    with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if save_npy:
        T = len(frames)
        for side, name in (("R", "right"), ("L", "left")):
            for kind, source in (("global", global_map), ("local", local_map)):
                stacked = np.stack([source.get(f"{j}_{side}", np.zeros((T, 3))) for j in JOINTS], axis=1)
                np.save(os.path.join(out_dir, f"{prefix}_tau_{kind}_{name}.npy"), stacked)
        np.save(os.path.join(out_dir, f"{prefix}_frames.npy"), frames)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _free_swing_side(pose, side, args, body_mass, dt, gravity, up):
    """--no-wrist-base: 腕を肩から吊る鎖（上腕 → 前腕）。手は手首の質点（§2-2）。"""
    segments = RIGHT_SEGMENTS if side == "R" else LEFT_SEGMENTS
    dumbbell = args.dumbbell_mass_right if side == "R" else args.dumbbell_mass_left
    load = max(float(dumbbell), 0.0) + hand_mass(body_mass)
    T = pose.shape[0]
    tau_g, tau_l = compute_side_torques(
        pose, segments, body_mass, dt, gravity,
        external_force=np.tile(load * gravity, (T, 1)),
        external_point=pose[:, segments[-1].distal_joint],
        up_axis=up,
    )
    global_map, local_map = {}, {}
    for idx, seg in enumerate(segments):
        part = SEGMENT_TO_OUTPUT[seg.name]
        global_map[part] = tau_g[:, idx, :]
        local_map[part] = tau_l[:, idx, :]
    return global_map, local_map


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fps = args.fps if args.fps > 0 else DEFAULT_FPS
    dt = 1.0 / fps
    body_mass = args.body_mass
    gravity_mag = abs(args.gravity) if args.gravity is not None else float(np.linalg.norm(DEFAULT_GRAVITY))

    frames, pose_full = load_pose_csv(args.pose_csv)
    pos_scale = max(1e-6, float(args.pos_scale))
    if pos_scale != 1.0:
        pose_full = pose_full * pos_scale
        if args.debug:
            print(f"[DEBUG] position scaled by {pos_scale}")
    pose_interp = interpolate_and_smooth(
        pose_full,
        skip_smoothing=args.skip_smoothing,
        window=args.window,
        poly=args.poly,
    )

    gravity_est = estimate_pose_gravity(pose_interp, gravity_mag, args.gravity_frames, args.gravity_mode)
    gravity = gravity_est.vector
    up = gravity_est.up
    print(f"[GRAVITY] 先頭 {gravity_est.samples} フレームの体幹から推定: g={np.round(gravity, 4).tolist()} "
          f"(mode={gravity_est.mode}, 体幹の傾き {gravity_est.lean_deg:.1f}°)")
    if gravity_est.mode == "axis" and gravity_est.lean_deg > 30.0:
        warnings.warn(
            f"体幹が最寄りの座標軸から {gravity_est.lean_deg:.1f}° 傾いている。カメラが水平でない可能性がある"
            "（--gravity-mode trunk を検討）", RuntimeWarning, stacklevel=1)

    torque_scale = float(args.torque_scale)
    global_map: Dict[str, np.ndarray] = {}
    local_map: Dict[str, np.ndarray] = {}
    meta: Dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "body_mass": body_mass,
        "fps": fps,
        "dt": dt,
        "torque_scale": torque_scale,
        "gravity": {
            "vector": gravity.tolist(),
            "mode": gravity_est.mode,
            "trunk_up": gravity_est.trunk_up.tolist(),
            "lean_deg": gravity_est.lean_deg,
            "frames": gravity_est.samples,
        },
        "hand_mass": hand_mass(body_mass),
    }

    if args.wrist_base:
        if args.dumbbell_mass_right or args.dumbbell_mass_left:
            warnings.warn(
                "--dumbbell-mass-* は腕を肩から吊る鎖（--no-wrist-base）でだけ使う。座位プッシュアップの"
                "モデルでは無視する（dumbbell mass is ignored with --wrist-base）", UserWarning, stacklevel=1)
        load = torso_load_mass(body_mass, args.support_share, args.torso_mass)
        meta["model"] = "wrist_base"
        meta["torque_convention"] = TORQUE_CONVENTION
        meta["torso_load_mass_per_arm"] = load
        meta["local_frame_fallbacks"] = {}
        meta["wrist_axis"] = {}
        for side in ("R", "L"):
            g_side, l_side, info = compute_push_up_side(
                pose_interp, side, body_mass, dt, gravity, load, hand_mass(body_mass))
            for joint in JOINTS:
                global_map[f"{joint}_{side}"] = g_side[joint] * torque_scale
                local_map[f"{joint}_{side}"] = l_side[joint] * torque_scale
            meta["local_frame_fallbacks"][side] = info["local_frame_fallbacks"]
            meta["wrist_axis"][side] = {
                "hand": info["wrist_hand"],
                "elbow_plane": int(len(frames) - info["wrist_hand"]),
            }
            if info["local_frame_fallbacks"]:
                print(f"[WARN] {side}: 局所座標系を作れず全体座標の値をそのまま使った件数 "
                      f"{info['local_frame_fallbacks']}（KNOWN_ISSUES §5-4）")
        if args.debug:
            print("[DEBUG] wrist-base load", {"torso_load_mass_per_arm": load, "share": args.support_share})
    else:
        meta["model"] = "free_swing"
        for side in ("R", "L"):
            g_side, l_side = _free_swing_side(pose_interp, side, args, body_mass, dt, gravity, up)
            for part in g_side:
                global_map[part] = g_side[part] * torque_scale
                local_map[part] = l_side[part] * torque_scale

    df_out = build_output(frames, global_map, local_map)

    out_dir = args.out_dir
    if out_dir is None:
        base_dir = os.path.dirname(os.path.abspath(args.pose_csv))
        out_dir = os.path.normpath(os.path.join(base_dir, os.pardir, "torque"))
    prefix = args.prefix or os.path.splitext(os.path.basename(args.pose_csv))[0]

    save_outputs(df_out, out_dir, prefix, args.save_npy, global_map, local_map, frames, meta)

    print(f"Saved torque outputs to {out_dir} (prefix='{prefix}', frames={len(frames)})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
