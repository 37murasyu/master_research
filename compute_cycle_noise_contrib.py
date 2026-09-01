from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_FPS = 30.0

RIGHT = {
    "shoulder": 12,
    "elbow": 14,
    "wrist": 16,
}
LEFT = {
    "shoulder": 11,
    "elbow": 13,
    "wrist": 15,
}


def _col_triplet(idx: int) -> List[str]:
    return [f"joint_{idx}_x", f"joint_{idx}_y", f"joint_{idx}_z"]


def _angle_about_y(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    cross = np.cross(v1, v2)
    cross_y = cross[:, 1]
    dot = np.einsum("ij,ij->i", v1, v2)
    return np.arctan2(cross_y, dot)


def _angle_from_xz_plane(v: np.ndarray) -> np.ndarray:
    vy = v[:, 1]
    vxz = np.linalg.norm(v[:, [0, 2]], axis=1)
    return np.arctan2(vy, vxz)


def _gradient(series: np.ndarray, dt: float) -> np.ndarray:
    if len(series) < 2:
        return np.zeros_like(series)
    return np.gradient(series, dt)


def _unit_scale(unit: str) -> float:
    if unit == "m":
        return 1.0
    if unit == "cm":
        return 0.01
    if unit == "mm":
        return 0.001
    return 1.0


def _auto_pose_unit(pos: np.ndarray) -> str:
    med = float(np.nanmedian(np.abs(pos)))
    if med > 5:
        unit = "cm"
        if med > 50:
            unit = "mm"
    else:
        unit = "m"
    return unit


def _parse_subject_id(stem: str) -> int | None:
    if stem.startswith("kpts3d_"):
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    head = stem.split("_")[0]
    return int(head) if head.isdigit() else None


def _map_torque_csv(torque_dir: Path, pose_with_cycles: Path) -> Path:
    stem = pose_with_cycles.stem.replace("_with_cycles", "")
    if stem.endswith("_lpf"):
        torque_stem = stem.replace("_lpf", "_torque_lpf")
    else:
        torque_stem = stem + "_torque_lpf"
    cand = torque_dir / f"{torque_stem}.csv"
    if cand.exists():
        return cand
    return torque_dir / f"{stem}_torque.csv"


def _prepare_cycle_map(cycles_df: pd.DataFrame) -> Dict[int, int]:
    if "frame" not in cycles_df.columns or "cycle_index" not in cycles_df.columns:
        raise ValueError("cycles csv must have frame and cycle_index")
    return {int(f): int(c) for f, c in zip(cycles_df["frame"], cycles_df["cycle_index"])}


def _merge_by_frame(df: pd.DataFrame, frame_map: Dict[int, int]) -> np.ndarray:
    frames = df["frame"].to_numpy(int) if "frame" in df.columns else np.arange(len(df))
    return np.array([frame_map.get(int(f), -1) for f in frames], dtype=int)


def _compute_angles(pose_df: pd.DataFrame, side: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    p_sh = pose_df[_col_triplet(side["shoulder"])].to_numpy(float)
    p_el = pose_df[_col_triplet(side["elbow"])].to_numpy(float)
    p_wr = pose_df[_col_triplet(side["wrist"])].to_numpy(float)
    v1 = p_sh - p_el
    v2 = p_wr - p_el
    elbow_angle = _angle_about_y(v1, v2)
    forearm = v2
    wrist_angle = _angle_from_xz_plane(forearm)
    return elbow_angle, wrist_angle


def _split_low_high(x: np.ndarray, fs: float, fc: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return x.copy(), x.copy()
    n = len(x)
    x_f = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    low_mask = freqs <= fc
    x_low = np.fft.irfft(x_f * low_mask, n=n)
    x_high = x - x_low
    return x_low, x_high


def _noise_power_ratio(x: np.ndarray, x_high: np.ndarray) -> float:
    var_total = float(np.nanvar(x))
    if var_total <= 0:
        return np.nan
    return float(np.nanvar(x_high) / var_total)


def _aggregate_cycles(power: np.ndarray, cycle_index: np.ndarray, dt: float) -> pd.DataFrame:
    cycles = np.unique(cycle_index)
    cycles = cycles[cycles >= 1]
    rows = []
    for c in cycles:
        mask = cycle_index == c
        w = float(np.nansum(power[mask] * dt))
        w_pos = float(np.nansum(np.clip(power[mask], 0, None) * dt))
        w_neg = float(np.nansum(np.clip(power[mask], None, 0) * dt))
        rows.append({
            "cycle_index": int(c),
            "work_J_signed": w,
            "work_J_pos": w_pos,
            "work_J_neg": w_neg,
        })
    if not rows:
        return pd.DataFrame(columns=["cycle_index", "work_J_signed", "work_J_pos", "work_J_neg"])
    return pd.DataFrame(rows)


def _noise_contrib_per_cycle(
    tau_sig: np.ndarray,
    tau_noi: np.ndarray,
    omg_sig: np.ndarray,
    omg_noi: np.ndarray,
    cycle_idx: np.ndarray,
    dt: float,
) -> pd.DataFrame:
    p_sig = tau_sig * omg_sig
    p_cross = tau_sig * omg_noi + tau_noi * omg_sig
    p_noi = tau_noi * omg_noi
    p_total = p_sig + p_cross + p_noi

    rows = []
    cycles = np.unique(cycle_idx)
    cycles = cycles[cycles >= 1]
    for c in cycles:
        mask = cycle_idx == c
        w_total = float(np.nansum(p_total[mask] * dt))
        w_sig = float(np.nansum(p_sig[mask] * dt))
        w_cross = float(np.nansum(p_cross[mask] * dt))
        w_noi = float(np.nansum(p_noi[mask] * dt))
        w_pos = float(np.nansum(np.clip(p_total[mask], 0, None) * dt))
        w_pos_noise = float(np.nansum(np.clip((p_cross + p_noi)[mask], 0, None) * dt))
        denom = abs(w_total) if abs(w_total) > 0 else np.nan
        denom_pos = w_pos if w_pos > 0 else np.nan
        rows.append({
            "cycle_index": int(c),
            "work_J_signed": w_total,
            "work_J_pos": w_pos,
            "work_sig": w_sig,
            "work_cross": w_cross,
            "work_noi": w_noi,
            "noise_ratio_abs": (abs(w_cross) + abs(w_noi)) / denom if np.isfinite(denom) else np.nan,
            "noise_ratio_pos": w_pos_noise / denom_pos if np.isfinite(denom_pos) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate per-cycle noise contribution in work (tau*omega)")
    ap.add_argument("--pose-dir", default="output_data/filtered_pose_lpf", help="pose dir with *_with_cycles.csv")
    ap.add_argument("--torque-dir", default="output_data/filtered_torque_lpf_recalc", help="torque dir with *_torque_lpf.csv")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="fps")
    ap.add_argument("--pose-unit", default="auto", choices=["auto", "m", "cm", "mm"], help="pose length unit")
    ap.add_argument("--torque-scale", type=float, default=1.0, help="scale torque (e.g., 0.01 if N*cm -> N*m)")
    ap.add_argument("--fc", type=float, default=3.0, help="cutoff frequency [Hz] for noise split")
    ap.add_argument("--out-dir", default="output_data/cycle_energy_noise", help="output directory")
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir)
    torque_dir = Path(args.torque_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(pose_dir.glob("*_with_cycles.csv"))
    for pose_path in pose_files:
        stem = pose_path.stem.replace("_with_cycles", "")
        subject_id = _parse_subject_id(stem)
        if subject_id is None or subject_id == 4:
            continue

        torque_path = _map_torque_csv(torque_dir, pose_path)
        if not torque_path.exists():
            print(f"[SKIP] torque not found: {torque_path}")
            continue

        pose_df = pd.read_csv(pose_path)
        torque_df = pd.read_csv(torque_path)
        if "cycle_index" not in pose_df.columns:
            print(f"[SKIP] cycle_index missing: {pose_path}")
            continue

        cycle_map = _prepare_cycle_map(pose_df[["frame", "cycle_index"]])
        torque_cycle = _merge_by_frame(torque_df, cycle_map)

        dt = 1.0 / (args.fps if args.fps > 0 else DEFAULT_FPS)

        pose_cols = [c for c in pose_df.columns if c.startswith("joint_") and c.endswith(("_x", "_y", "_z"))]
        pose_vals = pose_df[pose_cols].to_numpy(float) if pose_cols else np.array([])
        unit = args.pose_unit
        if unit == "auto":
            unit = _auto_pose_unit(pose_vals) if pose_vals.size else "m"
        pos_scale = _unit_scale(unit)

        for side_name, side in ("R", RIGHT), ("L", LEFT):
            pose_scaled = pose_df.copy()
            if pos_scale != 1.0:
                for idx in (side["shoulder"], side["elbow"], side["wrist"]):
                    for ax in ("x", "y", "z"):
                        col = f"joint_{idx}_{ax}"
                        if col in pose_scaled.columns:
                            pose_scaled[col] = pose_scaled[col].to_numpy(float) * pos_scale

            elbow_angle, wrist_angle = _compute_angles(pose_scaled, side)
            elbow_omega = _gradient(elbow_angle, dt) * (args.fps if args.fps > 0 else DEFAULT_FPS)
            wrist_omega = _gradient(wrist_angle, dt) * (args.fps if args.fps > 0 else DEFAULT_FPS)

            n = min(len(torque_df), len(pose_df))
            elbow_omega = elbow_omega[:n]
            wrist_omega = wrist_omega[:n]
            cycle_idx = torque_cycle[:n]

            elbow_tau_col = f"elbow_{side_name}_local_y"
            wrist_tau_col = f"wrist_{side_name}_local_y"
            if elbow_tau_col not in torque_df.columns or wrist_tau_col not in torque_df.columns:
                print(f"[SKIP] missing torque columns for {stem} {side_name}")
                continue

            elbow_tau = torque_df[elbow_tau_col].to_numpy(float)[:n] * args.torque_scale
            wrist_tau = torque_df[wrist_tau_col].to_numpy(float)[:n] * args.torque_scale

            for part_name, tau, omg in (
                (f"elbow_{side_name}", elbow_tau, elbow_omega),
                (f"wrist_{side_name}", wrist_tau, wrist_omega),
            ):
                tau_sig, tau_noi = _split_low_high(tau, args.fps, args.fc)
                omg_sig, omg_noi = _split_low_high(omg, args.fps, args.fc)

                noise_ratio_tau = _noise_power_ratio(tau, tau_noi)
                noise_ratio_omg = _noise_power_ratio(omg, omg_noi)

                df_cycles = _noise_contrib_per_cycle(tau_sig, tau_noi, omg_sig, omg_noi, cycle_idx, dt)
                df_cycles["part"] = part_name
                df_cycles["subject_id"] = subject_id
                df_cycles["fc_hz"] = args.fc
                df_cycles["rho_noise_tau"] = noise_ratio_tau
                df_cycles["rho_noise_omega"] = noise_ratio_omg

                out_path = out_dir / f"cycle_noise_{stem}_s{subject_id}_{part_name}.csv"
                df_cycles.to_csv(out_path, index=False)
                print(f"[OUT] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
