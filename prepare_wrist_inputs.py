"""Prepare forearm & wrist torque npy files from existing CSV exports.

Input expectations:
  - kpts3d_<timestamp>.csv : columns frame, joint_0_x, joint_0_y, joint_0_z, ... joint_11_z (12 joints)
    Index mapping expected to match main pipeline: (0: shoulder_base_R?, 2: elbow_R, 4: wrist_R, 3: elbow_L, 5: wrist_L ... )
    We only need joints 2 and 4 for right forearm, 3 and 5 for left forearm.
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
import numpy as np
import pandas as pd
from utils import compute_local_torque

# Legacy indices (older CSVs)
RIGHT_ELBOW_IDX_LEG = 2
RIGHT_WRIST_IDX_LEG = 4
LEFT_ELBOW_IDX_LEG = 3
LEFT_WRIST_IDX_LEG = 5

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


def extract_forearm_series_df(df: pd.DataFrame, elbow_jid: int, wrist_jid: int) -> np.ndarray:
    """Extract forearm vectors (N,3) using joint IDs present in DataFrame."""
    ec = _resolve_joint_cols(df, elbow_jid)
    wc = _resolve_joint_cols(df, wrist_jid)
    if ec is None or wc is None:
        raise ValueError(f"Required joints not found in kpts3d CSV: elbow {elbow_jid}, wrist {wrist_jid}")
    e = df[list(ec)].to_numpy(dtype=float)
    w = df[list(wc)].to_numpy(dtype=float)
    return w - e


def extract_wrist_tau_series(df_torque: pd.DataFrame, side: str) -> np.ndarray:
    """Return wrist torque columns if available.
    Prefer (x,y,z); if only y exists, return (N,1).
    """
    cols_xyz = [f"wrist_{side}_x", f"wrist_{side}_y", f"wrist_{side}_z"]
    if all(c in df_torque.columns for c in cols_xyz):
        return df_torque[cols_xyz].to_numpy(dtype=float)
    col_y = f"wrist_{side}_y"
    if col_y in df_torque.columns:
        return df_torque[[col_y]].to_numpy(dtype=float)
    raise ValueError(f"wrist_{side} torque columns not found in torque CSV")


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
    ap.add_argument('--shoulder-z', action='store_true', help='Also output shoulder Z time series (R/L if --left).')
    ap.add_argument('--pose-unit', choices=['auto','m','cm','mm'], default='auto',
                    help='Unit of pose values in the CSV. auto: infer from magnitude (>|10| -> mm, >|1| -> cm). Used only for shoulder-Z export; values are saved in meters.')
    args = ap.parse_args()

    kpts_df = load_kpts3d_df(args.kpts)
    torque_df = load_torque(args.torque)

    if len(kpts_df) != len(torque_df):
        print(f"[WARN] length mismatch kpts={len(kpts_df)} torque={len(torque_df)} -> trunc to min")
    N = min(len(kpts_df), len(torque_df))
    kpts_df = kpts_df.iloc[:N]
    torque_df = torque_df.iloc[:N]

    # Prefer MediaPipe joints; fallback to legacy indices
    if has_joint(kpts_df, RIGHT_ELBOW_ID_MP) and has_joint(kpts_df, RIGHT_WRIST_ID_MP):
        forearm_R = extract_forearm_series_df(kpts_df, RIGHT_ELBOW_ID_MP, RIGHT_WRIST_ID_MP)
    else:
        # legacy fallback
        # Build minimal (N,12,3) with potential NaNs to keep backward compat not desired; instead, raise clear error
        if not (has_joint(kpts_df, RIGHT_ELBOW_IDX_LEG) and has_joint(kpts_df, RIGHT_WRIST_IDX_LEG)):
            raise ValueError("Right elbow/wrist joints not found in kpts3d CSV (neither MediaPipe 14/16 nor legacy 2/4).")
        forearm_R = extract_forearm_series_df(kpts_df, RIGHT_ELBOW_IDX_LEG, RIGHT_WRIST_IDX_LEG)
    tau_R_series = extract_wrist_tau_series(torque_df, 'R')

    ts = infer_timestamp_from_name(args.kpts)
    prefix = (args.prefix + '_') if args.prefix else ''

    out_forearm_R = os.path.join(args.out_dir, f"{prefix}forearm_R_{ts}.npy")
    out_tau_R = os.path.join(args.out_dir, f"{prefix}tau_wrist_R_{ts}.npy")
    np.save(out_forearm_R, forearm_R)
    # tau export (prefer local y if xyz available)
    if tau_R_series.shape[1] == 3 and args.tau_mode == 'local_y':
        M = min(len(forearm_R), len(tau_R_series))
        tau_local_y = np.zeros((M,), dtype=float)
        for i in range(M):
            tau_local = compute_local_torque(tau_R_series[i], forearm_R[i])
            tau_local_y[i] = float(tau_local[1])
        np.save(out_tau_R, tau_local_y)
        print(f"[OUT] R wrist tau_y(local) -> {out_tau_R}  shape={tau_local_y.shape}")
    else:
        # global_y fallback: if only y provided, it's at [:,0]; else use [:,1]
        tau_y = tau_R_series[:, 0] if tau_R_series.shape[1] == 1 else tau_R_series[:, 1]
        np.save(out_tau_R, tau_y)
        print(f"[OUT] R wrist tau_y(global) -> {out_tau_R}  shape={tau_y.shape}")
    print(f"[OUT] R forearm vecs -> {out_forearm_R}  shape={forearm_R.shape}")

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
        try:
            shR_z = _extract_joint_z(kpts_df, RIGHT_SHOULDER_ID_MP)
            scale_R = _unit_scale_to_m(args.pose_unit, shR_z)
            if scale_R != 1.0:
                shR_z = shR_z * scale_R
            out_shR = os.path.join(args.out_dir, f"{prefix}shoulderZ_R_{ts}.npy")
            np.save(out_shR, shR_z)
            unit_msg = f" (scaled to meters, factor={scale_R})" if args.pose_unit != 'm' or scale_R != 1.0 else ""
            print(f"[OUT] R shoulder Z -> {out_shR}  shape={shR_z.shape}{unit_msg}")
        except Exception as e:
            print(f"[WARN] failed to export R shoulder Z: {e}")

    if args.left:
        if has_joint(kpts_df, LEFT_ELBOW_ID_MP) and has_joint(kpts_df, LEFT_WRIST_ID_MP):
            forearm_L = extract_forearm_series_df(kpts_df, LEFT_ELBOW_ID_MP, LEFT_WRIST_ID_MP)
        else:
            if not (has_joint(kpts_df, LEFT_ELBOW_IDX_LEG) and has_joint(kpts_df, LEFT_WRIST_IDX_LEG)):
                raise ValueError("Left elbow/wrist joints not found in kpts3d CSV (neither MediaPipe 13/15 nor legacy 3/5).")
            forearm_L = extract_forearm_series_df(kpts_df, LEFT_ELBOW_IDX_LEG, LEFT_WRIST_IDX_LEG)
        tau_L_series = extract_wrist_tau_series(torque_df, 'L')
        out_forearm_L = os.path.join(args.out_dir, f"{prefix}forearm_L_{ts}.npy")
        out_tau_L = os.path.join(args.out_dir, f"{prefix}tau_wrist_L_{ts}.npy")
        np.save(out_forearm_L, forearm_L)
        if tau_L_series.shape[1] == 3 and args.tau_mode == 'local_y':
            M = min(len(forearm_L), len(tau_L_series))
            tau_local_y_L = np.zeros((M,), dtype=float)
            for i in range(M):
                tau_local = compute_local_torque(tau_L_series[i], forearm_L[i])
                tau_local_y_L[i] = float(tau_local[1])
            np.save(out_tau_L, tau_local_y_L)
            print(f"[OUT] L wrist tau_y(local) -> {out_tau_L}  shape={tau_local_y_L.shape}")
        else:
            tau_y_L = tau_L_series[:, 0] if tau_L_series.shape[1] == 1 else tau_L_series[:, 1]
            np.save(out_tau_L, tau_y_L)
            print(f"[OUT] L wrist tau_y(global) -> {out_tau_L}  shape={tau_y_L.shape}")
        print(f"[OUT] L forearm vecs -> {out_forearm_L}  shape={forearm_L.shape}")

        if args.shoulder_z:
            try:
                shL_z = _extract_joint_z(kpts_df, LEFT_SHOULDER_ID_MP)
                scale_L = _unit_scale_to_m(args.pose_unit, shL_z)
                if scale_L != 1.0:
                    shL_z = shL_z * scale_L
                out_shL = os.path.join(args.out_dir, f"{prefix}shoulderZ_L_{ts}.npy")
                np.save(out_shL, shL_z)
                unit_msg = f" (scaled to meters, factor={scale_L})" if args.pose_unit != 'm' or scale_L != 1.0 else ""
                print(f"[OUT] L shoulder Z -> {out_shL}  shape={shL_z.shape}{unit_msg}")
            except Exception as e:
                print(f"[WARN] failed to export L shoulder Z: {e}")

    print('[DONE] prepare_wrist_inputs')

if __name__ == '__main__':
    main()
