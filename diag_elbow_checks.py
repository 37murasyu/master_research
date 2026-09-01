import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

G = 9.81
FOREARM_MASS_FRAC = 0.0160
FOREARM_COM_FRAC = 0.430


def load_pose(pose_csv: str) -> Tuple[np.ndarray, np.ndarray, float]:
    df = pd.read_csv(pose_csv)
    for col in (
        "frame",
        "joint_12_x",
        "joint_12_y",
        "joint_12_z",
        "joint_14_x",
        "joint_14_y",
        "joint_14_z",
        "joint_16_x",
        "joint_16_y",
        "joint_16_z",
    ):
        if col not in df.columns:
            raise SystemExit(f"missing column {col} in pose csv {pose_csv}")
    frames = df["frame"].to_numpy(int)
    # assume constant dt from frames
    frame_step = np.median(np.diff(frames)) if len(frames) > 1 else 1.0
    fps = 30.0 if frame_step == 0 else 30.0 / frame_step
    p12 = df[["joint_12_x", "joint_12_y", "joint_12_z"]].to_numpy(float)
    p14 = df[["joint_14_x", "joint_14_y", "joint_14_z"]].to_numpy(float)
    p16 = df[["joint_16_x", "joint_16_y", "joint_16_z"]].to_numpy(float)
    return frames, np.stack([p12, p14, p16], axis=1), fps


def elbow_angle(p_sh: np.ndarray, p_el: np.ndarray, p_wr: np.ndarray) -> np.ndarray:
    v1 = p_sh - p_el
    v2 = p_wr - p_el
    num = np.einsum("ij,ij->i", v1, v2)
    den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    den = np.where(den < 1e-9, np.nan, den)
    cos_th = np.clip(num / den, -1.0, 1.0)
    return np.arccos(cos_th)


def gradient(series: np.ndarray, dt: float) -> np.ndarray:
    if len(series) < 2:
        return np.zeros_like(series)
    return np.gradient(series, dt)


def align_with_offset(a: np.ndarray, b: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
    # returns (a_aligned, b_aligned) with offset samples discarded
    if offset > 0:
        n = min(len(a), len(b) - offset)
        return a[:n], b[offset:offset + n]
    elif offset < 0:
        k = -offset
        n = min(len(a) - k, len(b))
        return a[k:k + n], b[:n]
    else:
        n = min(len(a), len(b))
        return a[:n], b[:n]


def per_cycle(work: np.ndarray, cycle_idx: np.ndarray, dt: float, skip: set[int]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    cycles = np.unique(cycle_idx)
    for c in cycles:
        if c < 1 or c in skip:
            continue
        mask = cycle_idx == c
        w_signed = float(np.nansum(work[mask] * dt))
        w_pos = float(np.nansum(np.clip(work[mask], 0, None) * dt))
        w_neg = float(np.nansum(np.clip(work[mask], None, 0) * dt))
        out[int(c)] = {"work_signed": w_signed, "work_pos": w_pos, "work_neg": w_neg}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Elbow diagnostics: axis/sign/offset/external check")
    ap.add_argument("--pose-csv", required=True)
    ap.add_argument("--torque-csv", required=True)
    ap.add_argument("--cycles-csv", required=True)
    ap.add_argument("--body-mass", type=float, required=True)
    ap.add_argument("--dumbbell-mass", type=float, required=True)
    ap.add_argument("--forearm-len", type=float, default=None, help="Fixed forearm length [m]. Default: median")
    ap.add_argument("--mass-frac", type=float, default=FOREARM_MASS_FRAC)
    ap.add_argument("--com-frac", type=float, default=FOREARM_COM_FRAC)
    ap.add_argument("--offsets", nargs="*", type=int, default=[0, 2, 4, 6, 8])
    ap.add_argument("--skip-cycles", nargs="*", type=int, default=[])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    frames, pose_arr, fps = load_pose(args.pose_csv)
    dt = 1.0 / fps
    p12, p14, p16 = pose_arr[:, 0], pose_arr[:, 1], pose_arr[:, 2]
    ang = elbow_angle(p12, p14, p16)
    ang_vel = gradient(ang, dt)

    forearm_len = np.linalg.norm(p16 - p14, axis=1)
    forearm_len_med = float(np.nanmedian(forearm_len))
    forearm_len_use = float(args.forearm_len) if args.forearm_len is not None else forearm_len_med
    r_x = forearm_len_use
    r_g = forearm_len_use * float(args.com_frac)

    df_tau = pd.read_csv(args.torque_csv)
    if "elbow_R_local_y" not in df_tau.columns:
        raise SystemExit("torque csv lacks elbow_R_local_y")
    tau_y = df_tau["elbow_R_local_y"].to_numpy(float)

    df_cyc = pd.read_csv(args.cycles_csv)
    if "cycle_index" not in df_cyc.columns:
        raise SystemExit("cycles csv lacks cycle_index")
    cyc = df_cyc["cycle_index"].to_numpy(int)

    # truncate to common length
    n = min(len(ang_vel), len(tau_y), len(cyc))
    ang_vel = ang_vel[:n]
    tau_y = tau_y[:n]
    cyc = cyc[:n]

    m_x = args.body_mass * float(args.mass_frac)
    theor = (m_x * r_g + args.dumbbell_mass * r_x) * 16.73

    skip = set(int(x) for x in args.skip_cycles)

    offset_results: List[Dict[str, float]] = []
    for off in args.offsets:
        tau_a, w_a = align_with_offset(tau_y, ang_vel, off)
        cyc_a = cyc[: len(tau_a)]
        power = tau_a * w_a
        cycles = per_cycle(power, cyc_a, dt, skip)
        if cycles:
            ratios = [c["work_pos"] / theor for c in cycles.values() if theor > 0]
            signed = [c["work_signed"] / theor for c in cycles.values() if theor > 0]
            offset_results.append(
                {
                    "offset": off,
                    "mean_ratio_pos": float(np.nanmean(ratios)) if len(ratios) else np.nan,
                    "min_ratio_pos": float(np.nanmin(ratios)) if len(ratios) else np.nan,
                    "max_ratio_pos": float(np.nanmax(ratios)) if len(ratios) else np.nan,
                    "mean_ratio_signed": float(np.nanmean(signed)) if len(signed) else np.nan,
                    "n_cycles": len(ratios),
                }
            )
        else:
            offset_results.append({"offset": off, "n_cycles": 0})

    # sign alignment and rms
    tau_rms = float(np.sqrt(np.nanmean(tau_y ** 2)))
    vel_rms = float(np.sqrt(np.nanmean(ang_vel ** 2)))
    same_sign = float(np.nanmean(np.sign(tau_y) == np.sign(ang_vel)))
    pos_power_frac = float(np.nanmean(np.clip(tau_y * ang_vel, 0, None) > 0))

    # baseline: low angular velocity frames
    low_mask = np.abs(ang_vel) < 0.05
    tau_low_mean = float(np.nanmean(tau_y[low_mask])) if low_mask.any() else np.nan
    tau_low_std = float(np.nanstd(tau_y[low_mask])) if low_mask.any() else np.nan

    # external moment vs tau
    ext_moment = args.dumbbell_mass * G * r_x
    tau_med = float(np.nanmedian(np.abs(tau_y)))

    out = {
        "pose_csv": os.path.abspath(args.pose_csv),
        "torque_csv": os.path.abspath(args.torque_csv),
        "cycles_csv": os.path.abspath(args.cycles_csv),
        "body_mass": args.body_mass,
        "dumbbell_mass": args.dumbbell_mass,
        "mass_frac": float(args.mass_frac),
        "com_frac": float(args.com_frac),
        "forearm_len_median": forearm_len_med,
        "forearm_len_used": forearm_len_use,
        "r_x": r_x,
        "r_g": r_g,
        "theoretical_work_J": theor,
        "tau_rms": tau_rms,
        "ang_vel_rms": vel_rms,
        "sign_alignment_frac": same_sign,
        "positive_power_fraction": pos_power_frac,
        "baseline_tau_mean_if_low_angvel": tau_low_mean,
        "baseline_tau_std_if_low_angvel": tau_low_std,
        "ext_moment_Nm": ext_moment,
        "tau_abs_median_Nm": tau_med,
        "offset_sweep": offset_results,
    }

    print(json.dumps(out, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
