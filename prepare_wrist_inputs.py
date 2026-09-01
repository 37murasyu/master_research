"""Prepare forearm & wrist torque npy files from existing CSV exports.

Input expectations:
    - kpts3d_<timestamp>.csv : columns frame, joint_0_x, joint_0_y, joint_0_z, ... joint_11_z (12 joints)
        Index mapping expected to match main pipeline: (0: shoulder_base_R, 1: shoulder_base_L, 2: elbow_R, 3: elbow_L, 4: wrist_R, 5: wrist_L)
        We only need joints 2 and 4 for right forearm, 3 and 5 for left forearm, and (0,2,4)/(1,3) for upper-arm scaling when height is provided.
  - aim_torque_vec_<timestamp>.csv : columns frame, wrist_R_x,y,z, elbow_R_x,y,z, shoulder_R_x,y,z, wrist_L_x,y,z, ...
    We take wrist_R_y (local torque y) and wrist_L_y.

Output:
  forearm_R_<timestamp>.npy (N,3)
  tau_wrist_R_<timestamp>.npy (N,)
  (optionally left side if --left)

Usage:
  python prepare_wrist_inputs.py --kpts kpts3d_20250101T120000.csv \
      --torque aim_torque_vec_20250101T120000.csv --out-dir . --left
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict

import numpy as np
import pandas as pd
from utils import compute_local_torque

# Legacy indices (older CSVs)
RIGHT_ELBOW_IDX_LEG = 2
RIGHT_WRIST_IDX_LEG = 4
LEFT_ELBOW_IDX_LEG = 3
LEFT_WRIST_IDX_LEG = 5
RIGHT_SHOULDER_IDX_LEG = 0
LEFT_SHOULDER_IDX_LEG = 1

# Anthropometric length ratios (relative to stature; Dempster-like averages)
UPPER_ARM_RATIO = 0.186  # shoulder->elbow
FOREARM_RATIO = 0.146    # elbow->wrist

# MediaPipe indices (preferred when available)
RIGHT_ELBOW_ID_MP = 14
RIGHT_WRIST_ID_MP = 16
LEFT_ELBOW_ID_MP = 13
LEFT_WRIST_ID_MP = 15
RIGHT_SHOULDER_ID_MP = 12
LEFT_SHOULDER_ID_MP = 11


def load_kpts3d_df(path: str) -> pd.DataFrame:
    """Load kpts3d CSV as DataFrame. Supports both legacy (0..11) and MediaPipe (11,12,13,14,15,16,...) joint IDs.

    We will extract only the joints actually present.
    """
    return pd.read_csv(path)


def load_torque(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _resolve_joint_cols(df: pd.DataFrame, jid: int):
    """Return (cx, cy, cz) column names for a joint id.

    Supports either plain columns: joint_{id}_x/y/z or filtered: joint_{id}_x_f/y_f/z_f.
    Returns None if not found.
    """
    cand_sets = [
        (f"joint_{jid}_x", f"joint_{jid}_y", f"joint_{jid}_z"),
        (f"joint_{jid}_x_f", f"joint_{jid}_y_f", f"joint_{jid}_z_f"),
    ]
    for cx, cy, cz in cand_sets:
        if cx in df.columns and cy in df.columns and cz in df.columns:
            return cx, cy, cz
    return None


def has_joint(df: pd.DataFrame, jid: int) -> bool:
    return _resolve_joint_cols(df, jid) is not None


def target_segment_lengths(height_cm: float) -> Dict[str, float]:
    """Return target segment lengths (meters) given stature in cm."""
    h_m = float(height_cm) / 100.0
    return {
        "upper_arm": UPPER_ARM_RATIO * h_m,
        "forearm": FOREARM_RATIO * h_m,
    }


def scale_vectors_to_length(vecs: np.ndarray, target_len_m: float, label: str):
    """Scale vectors so median length matches target_len_m. Returns (scaled, factor)."""
    lens = np.linalg.norm(vecs, axis=1)
    med = float(np.nanmedian(lens)) if lens.size else float('nan')
    if not np.isfinite(med) or med < 1e-9:
        print(f"[WARN] {label}: median length invalid (med={med}) -> skip scaling")
        return vecs, None
    factor = target_len_m / med
    return vecs * factor, factor


def extract_vector_series_df(df: pd.DataFrame, start_jid: int, end_jid: int) -> np.ndarray:
    """Return vectors from start joint to end joint (end - start)."""
    start_cols = _resolve_joint_cols(df, start_jid)
    end_cols = _resolve_joint_cols(df, end_jid)
    if start_cols is None or end_cols is None:
        raise ValueError(f"Required joints not found in kpts3d CSV: start {start_jid}, end {end_jid}")
    start_pos = df[list(start_cols)].to_numpy(dtype=float)
    end_pos = df[list(end_cols)].to_numpy(dtype=float)
    return end_pos - start_pos


def extract_forearm_series_df(df: pd.DataFrame, elbow_jid: int, wrist_jid: int) -> np.ndarray:
    """Extract forearm vectors (N,3) using joint IDs present in DataFrame."""
    return extract_vector_series_df(df, elbow_jid, wrist_jid)


def extract_upper_arm_series_df(df: pd.DataFrame, elbow_jid: int, shoulder_jid: int) -> np.ndarray:
    """Extract upper-arm vectors pointing from elbow to shoulder (N,3)."""
    return extract_vector_series_df(df, elbow_jid, shoulder_jid)


def extract_joint_tau_series(df_torque: pd.DataFrame, joint: str, side: str) -> np.ndarray:
    """Return joint torque columns if available.

    Prefer (x,y,z) components; if only *_y exists, return (N,1).
    """
    cols_xyz = [f"{joint}_{side}_x", f"{joint}_{side}_y", f"{joint}_{side}_z"]
    if all(c in df_torque.columns for c in cols_xyz):
        return df_torque[cols_xyz].to_numpy(dtype=float)
    col_y = f"{joint}_{side}_y"
    if col_y in df_torque.columns:
        return df_torque[[col_y]].to_numpy(dtype=float)
    raise ValueError(f"{joint}_{side} torque columns not found in torque CSV")


def extract_wrist_tau_series(df_torque: pd.DataFrame, side: str) -> np.ndarray:
    return extract_joint_tau_series(df_torque, "wrist", side)


def extract_elbow_tau_series(df_torque: pd.DataFrame, side: str) -> np.ndarray:
    return extract_joint_tau_series(df_torque, "elbow", side)


def infer_timestamp_from_name(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r'(\d{8}T\d{6})', base)
    if m:
        return m.group(1)
    # fallback remove extension
    return os.path.splitext(base)[0].replace('kpts3d_','').replace('aim_torque_vec_','')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kpts', required=True, help='kpts3d_<timestamp>.csv path')
    ap.add_argument('--torque', required=True, help='aim_torque_vec_<timestamp>.csv path')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--left', action='store_true', help='Also output left wrist series')
    ap.add_argument('--tau-mode', choices=['local_y','global_y'], default='local_y', help='Export local_y (preferred) or fallback global_y')
    ap.add_argument('--prefix', default='', help='Optional prefix for output filenames')
    ap.add_argument('--csv', default=None, help='Optional CSV filename (or path) to export per-frame torque values')
    ap.add_argument('--shoulder-z', action='store_true', help='Also output shoulder Z time series (R/L if --left).')
    ap.add_argument('--pose-unit', choices=['auto','m','cm','mm'], default='auto',
                    help='Unit of pose values in the CSV. auto: infer from magnitude (>|10| -> mm, >|1| -> cm). Used only for shoulder-Z export; values are saved in meters.')
    ap.add_argument('--height-cm', type=float, default=None,
                    help='If set (e.g., 170), scale arm segments to anthropometric lengths (upper-arm=0.186*H, forearm=0.146*H).')
    args = ap.parse_args()

    kpts_df = load_kpts3d_df(args.kpts)
    torque_df = load_torque(args.torque)


    if len(kpts_df) != len(torque_df):
        print(f"[WARN] length mismatch kpts={len(kpts_df)} torque={len(torque_df)} -> trunc to min")
    N = min(len(kpts_df), len(torque_df))
    kpts_df = kpts_df.iloc[:N]
    torque_df = torque_df.iloc[:N]
    export_series: Dict[str, np.ndarray] = {}

    # Prefer MediaPipe joints; fallback to legacy indices
    if has_joint(kpts_df, RIGHT_ELBOW_ID_MP) and has_joint(kpts_df, RIGHT_WRIST_ID_MP):
        forearm_R = extract_forearm_series_df(kpts_df, RIGHT_ELBOW_ID_MP, RIGHT_WRIST_ID_MP)
        upperarm_R = extract_upper_arm_series_df(kpts_df, RIGHT_ELBOW_ID_MP, RIGHT_SHOULDER_ID_MP) if has_joint(kpts_df, RIGHT_SHOULDER_ID_MP) else None
    else:
        if not (has_joint(kpts_df, RIGHT_ELBOW_IDX_LEG) and has_joint(kpts_df, RIGHT_WRIST_IDX_LEG)):
            raise ValueError("Right elbow/wrist joints not found in kpts3d CSV (neither MediaPipe 14/16 nor legacy 2/4).")
        forearm_R = extract_forearm_series_df(kpts_df, RIGHT_ELBOW_IDX_LEG, RIGHT_WRIST_IDX_LEG)
        upperarm_R = extract_upper_arm_series_df(kpts_df, RIGHT_ELBOW_IDX_LEG, RIGHT_SHOULDER_IDX_LEG) if has_joint(kpts_df, RIGHT_SHOULDER_IDX_LEG) else None
    if upperarm_R is None:
        print("[WARN] Right shoulder joint not found; falling back to forearm vector for elbow torque localization")
        upperarm_R = forearm_R.copy()
    tau_R_series = extract_wrist_tau_series(torque_df, 'R')
    tau_elbow_R_series = extract_elbow_tau_series(torque_df, 'R')

    # Optional anthropometric scaling (e.g., height 170 cm male)
    if args.height_cm:
        tgt = target_segment_lengths(args.height_cm)
        forearm_R, sf_fr = scale_vectors_to_length(forearm_R, tgt["forearm"], "R forearm")
        upperarm_R, sf_ur = scale_vectors_to_length(upperarm_R, tgt["upper_arm"], "R upper-arm")
        if sf_fr:
            print(f"[SCALE] R forearm median-> {tgt['forearm']:.4f} m (factor={sf_fr:.4f})")
        if sf_ur:
            print(f"[SCALE] R upper-arm median-> {tgt['upper_arm']:.4f} m (factor={sf_ur:.4f})")

    ts = infer_timestamp_from_name(args.kpts)
    prefix = (args.prefix + '_') if args.prefix else ''

    out_forearm_R = os.path.join(args.out_dir, f"{prefix}forearm_R_{ts}.npy")
    out_tau_R = os.path.join(args.out_dir, f"{prefix}tau_wrist_R_{ts}.npy")
    np.save(out_forearm_R, forearm_R)
    # tau export (prefer local y if xyz available)
    if tau_R_series.shape[1] == 3 and args.tau_mode == 'local_y':
        M = min(len(forearm_R), len(tau_R_series))
        tau_local = np.zeros((M, 3), dtype=float)
        for i in range(M):
            tau_local[i] = compute_local_torque(tau_R_series[i], forearm_R[i])
        tau_local_y = tau_local[:, 1]
        np.save(out_tau_R, tau_local_y)
        print(f"[OUT] R wrist tau_y(local) -> {out_tau_R}  shape={tau_local_y.shape}")
        export_series['tau_wrist_R_local_x'] = tau_local[:, 0]
        export_series['tau_wrist_R_local_y'] = tau_local_y
        export_series['tau_wrist_R_local_z'] = tau_local[:, 2]
    else:
        # global fallback: if only y provided, it's at [:,0]; else use [:,1]
        if tau_R_series.shape[1] == 1:
            tau_y = tau_R_series[:, 0]
            np.save(out_tau_R, tau_y)
            print(f"[OUT] R wrist tau_y(global) -> {out_tau_R}  shape={tau_y.shape}")
            export_series['tau_wrist_R_global_y'] = tau_y
        else:
            tau_global = tau_R_series[:, :3]
            np.save(out_tau_R, tau_global[:, 1])
            print(f"[OUT] R wrist tau_y(global) -> {out_tau_R}  shape={tau_global[:,1].shape}")
            export_series['tau_wrist_R_global_x'] = tau_global[:, 0]
            export_series['tau_wrist_R_global_y'] = tau_global[:, 1]
            export_series['tau_wrist_R_global_z'] = tau_global[:, 2]
    print(f"[OUT] R forearm vecs -> {out_forearm_R}  shape={forearm_R.shape}")

    out_tau_elbow_R = os.path.join(args.out_dir, f"{prefix}tau_elbow_R_{ts}.npy")
    if tau_elbow_R_series.shape[1] == 3 and args.tau_mode == 'local_y':
        M_elbow = min(len(upperarm_R), len(tau_elbow_R_series))
        tau_elbow_local = np.zeros((M_elbow, 3), dtype=float)
        for i in range(M_elbow):
            tau_elbow_local[i] = compute_local_torque(tau_elbow_R_series[i], upperarm_R[i])
        tau_elbow_local_y = tau_elbow_local[:, 1]
        np.save(out_tau_elbow_R, tau_elbow_local_y)
        print(f"[OUT] R elbow tau_y(local) -> {out_tau_elbow_R}  shape={tau_elbow_local_y.shape}")
        export_series['tau_elbow_R_local_x'] = tau_elbow_local[:, 0]
        export_series['tau_elbow_R_local_y'] = tau_elbow_local_y
        export_series['tau_elbow_R_local_z'] = tau_elbow_local[:, 2]
    else:
        if tau_elbow_R_series.shape[1] == 1:
            tau_elbow_y = tau_elbow_R_series[:, 0]
            np.save(out_tau_elbow_R, tau_elbow_y)
            print(f"[OUT] R elbow tau_y(global) -> {out_tau_elbow_R}  shape={tau_elbow_y.shape}")
            export_series['tau_elbow_R_global_y'] = tau_elbow_y
        else:
            tau_elbow_global = tau_elbow_R_series[:, :3]
            np.save(out_tau_elbow_R, tau_elbow_global[:, 1])
            print(f"[OUT] R elbow tau_y(global) -> {out_tau_elbow_R}  shape={tau_elbow_global[:,1].shape}")
            export_series['tau_elbow_R_global_x'] = tau_elbow_global[:, 0]
            export_series['tau_elbow_R_global_y'] = tau_elbow_global[:, 1]
            export_series['tau_elbow_R_global_z'] = tau_elbow_global[:, 2]

    # Optional shoulder Z export
    if args.shoulder_z:
        def _unit_scale_to_m(unit: str, series: np.ndarray) -> float:
            if unit == 'm':
                return 1.0
            if unit == 'cm':
                return 0.01
            if unit == 'mm':
                return 0.001
            # auto detect: use median absolute value heuristic
            med = float(np.nanmedian(np.abs(series))) if series.size else 0.0
            # If values are around hundreds, assume mm; if a few units, assume cm
            if med > 10.0:
                return 0.001  # mm -> m
            if med > 1.0:
                return 0.01   # cm -> m
            return 1.0        # already meters
        def _extract_joint_z(df, jid):
            cols = _resolve_joint_cols(df, jid)
            if cols is None:
                raise ValueError(f"Shoulder joint {jid} not found in kpts3d CSV")
            _cx, _cy, cz = cols
            return df[cz].to_numpy(dtype=float)
        if has_joint(kpts_df, RIGHT_SHOULDER_ID_MP):
            shR_z = _extract_joint_z(kpts_df, RIGHT_SHOULDER_ID_MP)
            scale_R = _unit_scale_to_m(args.pose_unit, shR_z)
            if scale_R != 1.0:
                shR_z = shR_z * scale_R
            out_shR = os.path.join(args.out_dir, f"{prefix}shoulderZ_R_{ts}.npy")
            np.save(out_shR, shR_z)
            unit_msg = f" (scaled to meters, factor={scale_R})" if args.pose_unit != 'm' or scale_R != 1.0 else ""
            print(f"[OUT] R shoulder Z -> {out_shR}  shape={shR_z.shape}{unit_msg}")
        else:
            print(f"[WARN] failed to export R shoulder Z: joint {RIGHT_SHOULDER_ID_MP} unavailable")

    if args.left:
        if has_joint(kpts_df, LEFT_ELBOW_ID_MP) and has_joint(kpts_df, LEFT_WRIST_ID_MP):
            forearm_L = extract_forearm_series_df(kpts_df, LEFT_ELBOW_ID_MP, LEFT_WRIST_ID_MP)
            upperarm_L = extract_upper_arm_series_df(kpts_df, LEFT_ELBOW_ID_MP, LEFT_SHOULDER_ID_MP) if has_joint(kpts_df, LEFT_SHOULDER_ID_MP) else None
        else:
            if not (has_joint(kpts_df, LEFT_ELBOW_IDX_LEG) and has_joint(kpts_df, LEFT_WRIST_IDX_LEG)):
                raise ValueError("Left elbow/wrist joints not found in kpts3d CSV (neither MediaPipe 13/15 nor legacy 3/5).")
            forearm_L = extract_forearm_series_df(kpts_df, LEFT_ELBOW_IDX_LEG, LEFT_WRIST_IDX_LEG)
            upperarm_L = extract_upper_arm_series_df(kpts_df, LEFT_ELBOW_IDX_LEG, LEFT_SHOULDER_IDX_LEG) if has_joint(kpts_df, LEFT_SHOULDER_IDX_LEG) else None
        if upperarm_L is None:
            print("[WARN] Left shoulder joint not found; falling back to forearm vector for elbow torque localization")
            upperarm_L = forearm_L.copy()
        if args.height_cm:
            tgt = target_segment_lengths(args.height_cm)
            forearm_L, sf_fl = scale_vectors_to_length(forearm_L, tgt["forearm"], "L forearm")
            upperarm_L, sf_ul = scale_vectors_to_length(upperarm_L, tgt["upper_arm"], "L upper-arm")
            if sf_fl:
                print(f"[SCALE] L forearm median-> {tgt['forearm']:.4f} m (factor={sf_fl:.4f})")
            if sf_ul:
                print(f"[SCALE] L upper-arm median-> {tgt['upper_arm']:.4f} m (factor={sf_ul:.4f})")
        tau_L_series = extract_wrist_tau_series(torque_df, 'L')
        tau_elbow_L_series = extract_elbow_tau_series(torque_df, 'L')
        out_forearm_L = os.path.join(args.out_dir, f"{prefix}forearm_L_{ts}.npy")
        out_tau_L = os.path.join(args.out_dir, f"{prefix}tau_wrist_L_{ts}.npy")
        np.save(out_forearm_L, forearm_L)
        if tau_L_series.shape[1] == 3 and args.tau_mode == 'local_y':
            M = min(len(forearm_L), len(tau_L_series))
            tau_local_L = np.zeros((M, 3), dtype=float)
            for i in range(M):
                tau_local_L[i] = compute_local_torque(tau_L_series[i], forearm_L[i])
            tau_local_y_L = tau_local_L[:, 1]
            np.save(out_tau_L, tau_local_y_L)
            print(f"[OUT] L wrist tau_y(local) -> {out_tau_L}  shape={tau_local_y_L.shape}")
            export_series['tau_wrist_L_local_x'] = tau_local_L[:, 0]
            export_series['tau_wrist_L_local_y'] = tau_local_y_L
            export_series['tau_wrist_L_local_z'] = tau_local_L[:, 2]
        else:
            if tau_L_series.shape[1] == 1:
                tau_y_L = tau_L_series[:, 0]
                np.save(out_tau_L, tau_y_L)
                print(f"[OUT] L wrist tau_y(global) -> {out_tau_L}  shape={tau_y_L.shape}")
                export_series['tau_wrist_L_global_y'] = tau_y_L
            else:
                tau_global_L = tau_L_series[:, :3]
                np.save(out_tau_L, tau_global_L[:, 1])
                print(f"[OUT] L wrist tau_y(global) -> {out_tau_L}  shape={tau_global_L[:,1].shape}")
                export_series['tau_wrist_L_global_x'] = tau_global_L[:, 0]
                export_series['tau_wrist_L_global_y'] = tau_global_L[:, 1]
                export_series['tau_wrist_L_global_z'] = tau_global_L[:, 2]
        print(f"[OUT] L forearm vecs -> {out_forearm_L}  shape={forearm_L.shape}")

        out_tau_elbow_L = os.path.join(args.out_dir, f"{prefix}tau_elbow_L_{ts}.npy")
        if tau_elbow_L_series.shape[1] == 3 and args.tau_mode == 'local_y':
            M_elbow_L = min(len(upperarm_L), len(tau_elbow_L_series))
            tau_elbow_local_L = np.zeros((M_elbow_L, 3), dtype=float)
            for i in range(M_elbow_L):
                tau_elbow_local_L[i] = compute_local_torque(tau_elbow_L_series[i], upperarm_L[i])
            tau_elbow_local_y_L = tau_elbow_local_L[:, 1]
            np.save(out_tau_elbow_L, tau_elbow_local_y_L)
            print(f"[OUT] L elbow tau_y(local) -> {out_tau_elbow_L}  shape={tau_elbow_local_y_L.shape}")
            export_series['tau_elbow_L_local_x'] = tau_elbow_local_L[:, 0]
            export_series['tau_elbow_L_local_y'] = tau_elbow_local_y_L
            export_series['tau_elbow_L_local_z'] = tau_elbow_local_L[:, 2]
        else:
            if tau_elbow_L_series.shape[1] == 1:
                tau_elbow_y_L = tau_elbow_L_series[:, 0]
                np.save(out_tau_elbow_L, tau_elbow_y_L)
                print(f"[OUT] L elbow tau_y(global) -> {out_tau_elbow_L}  shape={tau_elbow_y_L.shape}")
                export_series['tau_elbow_L_global_y'] = tau_elbow_y_L
            else:
                tau_elbow_global_L = tau_elbow_L_series[:, :3]
                np.save(out_tau_elbow_L, tau_elbow_global_L[:, 1])
                print(f"[OUT] L elbow tau_y(global) -> {out_tau_elbow_L}  shape={tau_elbow_global_L[:,1].shape}")
                export_series['tau_elbow_L_global_x'] = tau_elbow_global_L[:, 0]
                export_series['tau_elbow_L_global_y'] = tau_elbow_global_L[:, 1]
                export_series['tau_elbow_L_global_z'] = tau_elbow_global_L[:, 2]

        if args.shoulder_z:
            if has_joint(kpts_df, LEFT_SHOULDER_ID_MP):
                shL_z = _extract_joint_z(kpts_df, LEFT_SHOULDER_ID_MP)
                scale_L = _unit_scale_to_m(args.pose_unit, shL_z)
                if scale_L != 1.0:
                    shL_z = shL_z * scale_L
                out_shL = os.path.join(args.out_dir, f"{prefix}shoulderZ_L_{ts}.npy")
                np.save(out_shL, shL_z)
                unit_msg = f" (scaled to meters, factor={scale_L})" if args.pose_unit != 'm' or scale_L != 1.0 else ""
                print(f"[OUT] L shoulder Z -> {out_shL}  shape={shL_z.shape}{unit_msg}")
            else:
                print(f"[WARN] failed to export L shoulder Z: joint {LEFT_SHOULDER_ID_MP} unavailable")

    if args.csv and export_series:
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(args.out_dir, csv_path)
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        max_len = max(len(arr) for arr in export_series.values()) if export_series else 0
        data = {'frame': np.arange(max_len, dtype=int)} if max_len else {'frame': np.array([], dtype=int)}
        for col_name, series in export_series.items():
            if len(series) == max_len:
                data[col_name] = series
            else:
                padded = np.full(max_len, np.nan, dtype=float)
                padded[:len(series)] = series
                data[col_name] = padded
        df_csv = pd.DataFrame(data)
        df_csv.to_csv(csv_path, index=False)
        print(f"[OUT] local torque CSV -> {csv_path}  cols={list(export_series.keys())}")

    print('[DONE] prepare_wrist_inputs')

if __name__ == '__main__':
    main()
