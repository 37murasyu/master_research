"""サイクル仕事を「信号 × 信号」「交差項」「ノイズ × ノイズ」に分解する。

分解する仕事はスコア（``compute_cycle_energy_elbow_wrist.py``）と同じもの: トルクと関節の相対角速度を
同じ局所 y 軸に射影した積（``_joint_projections``）。かつてこちらだけ、角度（``arctan2``）を
経由して微分した角速度に fps を掛けており（30 倍、KNOWN_ISSUES §1-1 と同じ誤り）、
トルクとは別の軸の角速度を掛けていた（§5-2）。スコア側を直したときに取り残されていた（§1-6）。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from compute_cycle_energy_elbow_wrist import HAND, _joint_projections

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


def side_noise_contrib(
    pose_df: pd.DataFrame,
    torque_df: pd.DataFrame,
    side_name: str,
    dt: float,
    fps: float,
    fc: float,
    cycle_idx: np.ndarray,
    torque_scale: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """片側の肘・手首について、サイクルごとの仕事の分解を返す（キーは elbow_R など）。"""
    proj = _joint_projections(pose_df, torque_df, side_name, dt)
    out: Dict[str, pd.DataFrame] = {}
    for joint in ("elbow", "wrist"):
        tau, omg = proj[joint]
        tau = tau * torque_scale
        n = len(tau)
        idx = np.asarray(cycle_idx)[:n]
        tau_sig, tau_noi = _split_low_high(tau, fps, fc)
        omg_sig, omg_noi = _split_low_high(omg, fps, fc)
        df_cycles = _noise_contrib_per_cycle(tau_sig, tau_noi, omg_sig, omg_noi, idx, dt)
        df_cycles["part"] = f"{joint}_{side_name}"
        df_cycles["fc_hz"] = fc
        df_cycles["rho_noise_tau"] = _noise_power_ratio(tau, tau_noi)
        df_cycles["rho_noise_omega"] = _noise_power_ratio(omg, omg_noi)
        out[f"{joint}_{side_name}"] = df_cycles
    return out


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
                for idx in (side["shoulder"], side["elbow"], side["wrist"], *HAND[side_name]):
                    for ax in ("x", "y", "z"):
                        col = f"joint_{idx}_{ax}"
                        if col in pose_scaled.columns:
                            pose_scaled[col] = pose_scaled[col].to_numpy(float) * pos_scale

            needed = [f"{part}_{side_name}_{ax}" for part in ("elbow", "wrist") for ax in "xyz"]
            if any(col not in torque_df.columns for col in needed):
                print(f"[SKIP] missing torque columns for {stem} {side_name}")
                continue

            results = side_noise_contrib(
                pose_scaled, torque_df, side_name, dt, args.fps, args.fc, torque_cycle, args.torque_scale)
            for part_name, df_cycles in results.items():
                df_cycles["subject_id"] = subject_id
                out_path = out_dir / f"cycle_noise_{stem}_s{subject_id}_{part_name}.csv"
                df_cycles.to_csv(out_path, index=False)
                print(f"[OUT] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
