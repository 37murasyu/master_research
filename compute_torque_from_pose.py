"""Offline inverse-dynamics for prerecorded 3D pose CSVs.

Reads a MediaPipe-style pose CSV, interpolates missing samples, derives upper
limb segment kinematics, and invokes `utils_dynamic` helpers to recover joint
torques in both global and local frames.

Example:
    python compute_torque_from_pose.py \
        --pose-csv output_data/poses/kpts3d_subject5_20250925_133228_filtpos.csv \
        --out-dir output_data/torque --save-npy
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import g as CONFIG_GRAVITY
from config import w as CONFIG_BODY_MASS
from utils import compute_local_torque
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

RIGHT_SHOULDER_IDX_LEGACY = 0
LEFT_SHOULDER_IDX_LEGACY = 1
RIGHT_ELBOW_IDX_LEGACY = 2
LEFT_ELBOW_IDX_LEGACY = 3
RIGHT_WRIST_IDX_LEGACY = 4
LEFT_WRIST_IDX_LEGACY = 5

LEGACY_RIGHT_SEGMENTS: Tuple[SegmentSpec, ...] = (
    SegmentSpec("upper_arm_R", RIGHT_SHOULDER_IDX_LEGACY, RIGHT_ELBOW_IDX_LEGACY, 3, 0.0227, 0.436),
    SegmentSpec("forearm_R", RIGHT_ELBOW_IDX_LEGACY, RIGHT_WRIST_IDX_LEGACY, 4, 0.0160, 0.430),
)
LEGACY_LEFT_SEGMENTS: Tuple[SegmentSpec, ...] = (
    SegmentSpec("upper_arm_L", LEFT_SHOULDER_IDX_LEGACY, LEFT_ELBOW_IDX_LEGACY, 3, 0.0227, 0.436),
    SegmentSpec("forearm_L", LEFT_ELBOW_IDX_LEGACY, LEFT_WRIST_IDX_LEGACY, 4, 0.0160, 0.430),
)

WRIST_BASE_SEGMENTS_RIGHT: Tuple[SegmentSpec, ...] = (
    SegmentSpec("forearm_R_wrist", 16, 14, 4, 0.0160, 1.0 - 0.430),
    SegmentSpec("upper_arm_R_wrist", 14, 12, 3, 0.0227, 1.0 - 0.436),
)
WRIST_BASE_SEGMENTS_LEFT: Tuple[SegmentSpec, ...] = (
    SegmentSpec("forearm_L_wrist", 15, 13, 4, 0.0160, 1.0 - 0.430),
    SegmentSpec("upper_arm_L_wrist", 13, 11, 3, 0.0227, 1.0 - 0.436),
)

LEGACY_WRIST_BASE_SEGMENTS_RIGHT: Tuple[SegmentSpec, ...] = (
    SegmentSpec("forearm_R_wrist", RIGHT_WRIST_IDX_LEGACY, RIGHT_ELBOW_IDX_LEGACY, 4, 0.0160, 1.0 - 0.430),
    SegmentSpec("upper_arm_R_wrist", RIGHT_ELBOW_IDX_LEGACY, RIGHT_SHOULDER_IDX_LEGACY, 3, 0.0227, 1.0 - 0.436),
)
LEGACY_WRIST_BASE_SEGMENTS_LEFT: Tuple[SegmentSpec, ...] = (
    SegmentSpec("forearm_L_wrist", LEFT_WRIST_IDX_LEGACY, LEFT_ELBOW_IDX_LEGACY, 4, 0.0160, 1.0 - 0.430),
    SegmentSpec("upper_arm_L_wrist", LEFT_ELBOW_IDX_LEGACY, LEFT_SHOULDER_IDX_LEGACY, 3, 0.0227, 1.0 - 0.436),
)

SEGMENT_TO_OUTPUT = {
    "upper_arm_R": "shoulder_R",
    "forearm_R": "elbow_R",
    "upper_arm_L": "shoulder_L",
    "forearm_L": "elbow_L",
}

SEGMENT_TO_OUTPUT_WRIST_BASE = {
    "forearm_R_wrist": "wrist_R",
    "upper_arm_R_wrist": "elbow_R",
    "forearm_L_wrist": "wrist_L",
    "upper_arm_L_wrist": "elbow_L",
}

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
    parser.add_argument("--dumbbell-mass-right", type=float, default=0.0, help="External load mass at right wrist [kg]")
    parser.add_argument("--dumbbell-mass-left", type=float, default=0.0, help="External load mass at left wrist [kg]")
    parser.add_argument(
        "--wrist-base",
        action="store_true",
        help="Also solve a wrist-anchored two-link chain (forearm+upper arm) supporting torso load",
    )
    parser.add_argument(
        "--support-share",
        type=float,
        default=0.5,
        help="Fraction of torso mass assigned to a single arm when wrist-base mode is active (0-1)",
    )
    parser.add_argument(
        "--torso-mass",
        type=float,
        default=None,
        help="Override torso mass [kg] treated as external load in wrist-base mode",
    )
    parser.add_argument(
        "--torque-scale",
        type=float,
        default=0.01,
        help="Scale factor applied to all output torques (use 0.01 if upstream produces N-cm)",
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
            series = df[col_name].to_numpy(dtype=np.float64)
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
    return savgol_filter(clean, window_length=window, polyorder=poly, axis=0, mode="interp")


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
        inertia_tensors[idx] = calculate_inertia_tensor(seg.inertia_row, mass, length)
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
        M, F_base = compute_MF_batch_native(
            inertia_tensors,
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
            tau_local[t, n] = compute_local_torque(tau[n], link_vec[t, n], parent_vec)

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
    )


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def build_output(
    frames: np.ndarray,
    tau_g_right: np.ndarray,
    tau_l_right: np.ndarray,
    tau_g_left: np.ndarray,
    tau_l_left: np.ndarray,
    right_segments: Sequence[SegmentSpec],
    left_segments: Sequence[SegmentSpec],
    override_global: Optional[Dict[str, np.ndarray]] = None,
    override_local: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, ...]]]:
    T = len(frames)
    global_map: Dict[str, np.ndarray] = {part: np.zeros((T, 3), dtype=np.float64) for part in OUTPUT_PART_ORDER}
    local_map: Dict[str, np.ndarray] = {part: np.zeros((T, 3), dtype=np.float64) for part in OUTPUT_PART_ORDER}
    for idx, seg in enumerate(right_segments):
        part = SEGMENT_TO_OUTPUT[seg.name]
        global_map[part] = tau_g_right[:, idx, :]
        local_map[part] = tau_l_right[:, idx, :]
    for idx, seg in enumerate(left_segments):
        part = SEGMENT_TO_OUTPUT[seg.name]
        global_map[part] = tau_g_left[:, idx, :]
        local_map[part] = tau_l_left[:, idx, :]

    if override_global:
        for part, arr in override_global.items():
            if arr.shape[0] != T or arr.shape[1] != 3:
                raise ValueError(f"override_global[{part!s}] must have shape ({T}, 3)")
            global_map[part] = arr
    if override_local:
        for part, arr in override_local.items():
            if arr.shape[0] != T or arr.shape[1] != 3:
                raise ValueError(f"override_local[{part!s}] must have shape ({T}, 3)")
            local_map[part] = arr

    data = {"frame": frames.astype(np.int64)}
    for part in OUTPUT_PART_ORDER:
        g_vals = global_map[part]
        l_vals = local_map[part]
        data[f"{part}_x"] = g_vals[:, 0]
        data[f"{part}_y"] = g_vals[:, 1]
        data[f"{part}_z"] = g_vals[:, 2]
        data[f"{part}_local_x"] = l_vals[:, 0]
        data[f"{part}_local_y"] = l_vals[:, 1]
        data[f"{part}_local_z"] = l_vals[:, 2]
    df = pd.DataFrame(data)
    meta = {part: global_map[part].shape for part in OUTPUT_PART_ORDER}
    return df, meta


def save_outputs(
    df: pd.DataFrame,
    meta_shapes: Dict[str, Tuple[int, ...]],
    out_dir: str,
    prefix: str,
    save_npy: bool,
    tau_g_right: np.ndarray,
    tau_l_right: np.ndarray,
    tau_g_left: np.ndarray,
    tau_l_left: np.ndarray,
    frames: np.ndarray,
    body_mass: float,
    gravity: np.ndarray,
    fps: float,
    dt: float,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{prefix}_torque.csv")
    df.to_csv(csv_path, index=False)

    meta = {
        "frames": int(len(frames)),
        "body_mass": body_mass,
        "gravity": gravity.tolist(),
        "fps": fps,
        "dt": dt,
        "columns": {
            "global": OUTPUT_GLOBAL_COLS,
            "local": OUTPUT_LOCAL_COLS,
        },
        "shapes": meta_shapes,
    }
    with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if save_npy:
        np.save(os.path.join(out_dir, f"{prefix}_tau_global_right.npy"), tau_g_right)
        np.save(os.path.join(out_dir, f"{prefix}_tau_global_left.npy"), tau_g_left)
        np.save(os.path.join(out_dir, f"{prefix}_tau_local_right.npy"), tau_l_right)
        np.save(os.path.join(out_dir, f"{prefix}_tau_local_left.npy"), tau_l_left)
        np.save(os.path.join(out_dir, f"{prefix}_frames.npy"), frames)
        if extra_arrays:
            for name, arr in extra_arrays.items():
                np.save(os.path.join(out_dir, f"{prefix}_{name}.npy"), arr)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fps = args.fps if args.fps > 0 else DEFAULT_FPS
    dt = 1.0 / fps
    body_mass = args.body_mass
    gravity_mag = abs(args.gravity) if args.gravity is not None else float(np.linalg.norm(DEFAULT_GRAVITY))
    gravity = np.array([0.0, 0.0, -gravity_mag], dtype=np.float64)

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

    max_required_modern = max(
        max(seg.proximal_joint, seg.distal_joint)
        for seg in (*RIGHT_SEGMENTS, *LEFT_SEGMENTS, *WRIST_BASE_SEGMENTS_RIGHT, *WRIST_BASE_SEGMENTS_LEFT)
    )
    use_legacy = pose_interp.shape[1] <= max_required_modern

    if use_legacy:
        right_segments = LEGACY_RIGHT_SEGMENTS
        left_segments = LEGACY_LEFT_SEGMENTS
        wrist_segments_right = LEGACY_WRIST_BASE_SEGMENTS_RIGHT
        wrist_segments_left = LEGACY_WRIST_BASE_SEGMENTS_LEFT
        if args.debug:
            print("[DEBUG] Using legacy joint indices (0-based 0..5 mapping) for torque computation")
    else:
        right_segments = RIGHT_SEGMENTS
        left_segments = LEFT_SEGMENTS
        wrist_segments_right = WRIST_BASE_SEGMENTS_RIGHT
        wrist_segments_left = WRIST_BASE_SEGMENTS_LEFT

    def _build_external(mass_kg: float, segs: Sequence[SegmentSpec]):
        if mass_kg <= 0:
            return None, None
        fvec = np.array([0.0, 0.0, -mass_kg * gravity_mag], dtype=np.float64)
        T = pose_interp.shape[0]
        farr = np.repeat(fvec[np.newaxis, :], T, axis=0)
        wrist_idx = segs[-1].distal_joint
        point = np.asarray(pose_interp[:, wrist_idx, :], dtype=np.float64)
        return farr, point

    ext_force_right, ext_point_right = _build_external(args.dumbbell_mass_right, right_segments)
    ext_force_left, ext_point_left = _build_external(args.dumbbell_mass_left, left_segments)

    tau_g_right, tau_l_right = compute_side_torques(
        pose_interp,
        right_segments,
        body_mass,
        dt,
        gravity,
        external_force=ext_force_right,
        external_point=ext_point_right,
    )
    tau_g_left, tau_l_left = compute_side_torques(
        pose_interp,
        left_segments,
        body_mass,
        dt,
        gravity,
        external_force=ext_force_left,
        external_point=ext_point_left,
    )

    torque_scale = float(args.torque_scale)
    if torque_scale != 1.0:
        tau_g_right *= torque_scale
        tau_l_right *= torque_scale
        tau_g_left *= torque_scale
        tau_l_left *= torque_scale

    override_global: Dict[str, np.ndarray] = {}
    override_local: Dict[str, np.ndarray] = {}
    extra_arrays: Dict[str, np.ndarray] = {}

    if args.wrist_base:
        share = float(np.clip(args.support_share, 0.0, 1.0))
        arm_mass_frac = sum(seg.mass_fraction for seg in RIGHT_SEGMENTS)
        torso_mass_default = max(body_mass - 2.0 * arm_mass_frac * body_mass, 0.0)
        torso_mass_total = args.torso_mass if args.torso_mass is not None else torso_mass_default
        torso_mass_total = max(torso_mass_total, 0.0)
        torso_mass_each = torso_mass_total * share
        external_force_vec = torso_mass_each * gravity
        T = pose_interp.shape[0]
        external_force_arr = np.repeat(external_force_vec[np.newaxis, :], T, axis=0)
        shoulder_idx_right = wrist_segments_right[-1].distal_joint
        shoulder_idx_left = wrist_segments_left[-1].distal_joint
        external_point_right = np.asarray(pose_interp[:, shoulder_idx_right, :], dtype=np.float64)
        external_point_left = np.asarray(pose_interp[:, shoulder_idx_left, :], dtype=np.float64)

        tau_g_right_wrist, tau_l_right_wrist = compute_side_torques(
            pose_interp,
            wrist_segments_right,
            body_mass,
            dt,
            gravity,
            external_force=external_force_arr,
            external_point=external_point_right,
        )
        tau_g_left_wrist, tau_l_left_wrist = compute_side_torques(
            pose_interp,
            wrist_segments_left,
            body_mass,
            dt,
            gravity,
            external_force=external_force_arr,
            external_point=external_point_left,
        )

        if torque_scale != 1.0:
            tau_g_right_wrist *= torque_scale
            tau_l_right_wrist *= torque_scale
            tau_g_left_wrist *= torque_scale
            tau_l_left_wrist *= torque_scale

        for idx, seg in enumerate(wrist_segments_right):
            part = SEGMENT_TO_OUTPUT_WRIST_BASE.get(seg.name)
            if part:
                override_global[part] = tau_g_right_wrist[:, idx, :]
                override_local[part] = tau_l_right_wrist[:, idx, :]
        for idx, seg in enumerate(wrist_segments_left):
            part = SEGMENT_TO_OUTPUT_WRIST_BASE.get(seg.name)
            if part:
                override_global[part] = tau_g_left_wrist[:, idx, :]
                override_local[part] = tau_l_left_wrist[:, idx, :]

        extra_arrays.update({
            "tau_global_right_wristbase": tau_g_right_wrist,
            "tau_local_right_wristbase": tau_l_right_wrist,
            "tau_global_left_wristbase": tau_g_left_wrist,
            "tau_local_left_wristbase": tau_l_left_wrist,
        })

        if args.debug:
            print(
                "[DEBUG] wrist-base load",
                {
                    "torso_mass_total": torso_mass_total,
                    "share": share,
                    "torso_mass_each": torso_mass_each,
                    "force_norm": float(np.linalg.norm(external_force_vec)),
                },
            )

    df_out, meta_shapes = build_output(
        frames,
        tau_g_right,
        tau_l_right,
        tau_g_left,
        tau_l_left,
        right_segments,
        left_segments,
        override_global=override_global if override_global else None,
        override_local=override_local if override_local else None,
    )

    out_dir = args.out_dir
    if out_dir is None:
        base_dir = os.path.dirname(os.path.abspath(args.pose_csv))
        out_dir = os.path.normpath(os.path.join(base_dir, os.pardir, "torque"))
    prefix = args.prefix or os.path.splitext(os.path.basename(args.pose_csv))[0]

    save_outputs(
        df_out,
        meta_shapes,
        out_dir,
        prefix,
        args.save_npy,
        tau_g_right,
        tau_l_right,
        tau_g_left,
        tau_l_left,
        frames,
        body_mass,
        gravity,
        fps,
        dt,
        extra_arrays=extra_arrays if extra_arrays else None,
    )

    print(f"Saved torque outputs to {out_dir} (prefix='{prefix}', frames={len(frames)})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
