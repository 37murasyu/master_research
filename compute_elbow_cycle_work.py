import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

FOREARM_MASS_FRAC = 0.0160  # body mass fraction for forearm
FOREARM_COM_FRAC = 0.430    # COM distance fraction from elbow toward wrist
DEFAULT_FPS = 30.0


def read_mmax(csv_path: str, subject_id: int) -> float:
    df = pd.read_csv(csv_path)
    row = df.loc[df['subject_id'] == subject_id]
    if row.empty:
        raise ValueError(f"subject_id {subject_id} not found in {csv_path}")
    val = row['elbow_R_outer'].iloc[0]
    if pd.isna(val):
        raise ValueError(f"elbow_R_outer is NaN for subject_id {subject_id} in {csv_path}")
    return float(val)


def compute_elbow_flexion_angle(p_sh: np.ndarray, p_el: np.ndarray, p_wr: np.ndarray) -> np.ndarray:
    v1 = p_sh - p_el
    v2 = p_wr - p_el
    num = np.einsum('ij,ij->i', v1, v2)
    den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    den = np.where(den < 1e-9, np.nan, den)
    cos_th = np.clip(num / den, -1.0, 1.0)
    return np.arccos(cos_th)  # radians


def gradient(series: np.ndarray, dt: float) -> np.ndarray:
    if len(series) < 2:
        return np.zeros_like(series)
    return np.gradient(series, dt)


def aggregate_cycles(frames: np.ndarray, power: np.ndarray, cycle_index: np.ndarray, dt: float) -> pd.DataFrame:
    cycles = np.unique(cycle_index)
    cycles = cycles[cycles >= 1]
    rows = []
    for c in cycles:
        mask = cycle_index == c
        w = float(np.nansum(power[mask] * dt))
        w_pos = float(np.nansum(np.clip(power[mask], 0, None) * dt))
        w_neg = float(np.nansum(np.clip(power[mask], None, 0) * dt))
        rows.append({
            'cycle_index': int(c),
            'work_J_signed': w,
            'work_J_pos': w_pos,
            'work_J_neg': w_neg,
        })
    return pd.DataFrame(rows)


def compute_theoretical_work(m_body: float, m_max: float, r_g: float, r_x: float, mass_frac: float = FOREARM_MASS_FRAC) -> float:
    return (m_body * mass_frac * r_g + m_max * r_x) * 16.73


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Elbow cycle work from local tau_y and elbow flexion angular velocity')
    ap.add_argument('--pose-csv', required=True, help='Pose CSV (joint_* columns, GC spline済み推奨)')
    ap.add_argument('--torque-csv', required=True, help='Torque CSV (needs elbow_R_local_y)')
    ap.add_argument('--cycles-csv', required=True, help='cycle_index付きCSV')
    ap.add_argument('--mmax-csv', required=True, help='m_max_all_merged.csv 等')
    ap.add_argument('--subject-id', type=int, default=0, help='m_max lookup subject id (default: 0)')
    ap.add_argument('--body-mass', type=float, required=True, help='被験者体重 [kg]')
    ap.add_argument('--shoulder-idx', type=int, default=12, help='肩関節のjointインデックス (デフォルト: 12; 3-5-7系なら7)')
    ap.add_argument('--elbow-idx', type=int, default=14, help='肘関節のjointインデックス (デフォルト: 14; 3-5-7系なら5)')
    ap.add_argument('--wrist-idx', type=int, default=16, help='手首関節のjointインデックス (デフォルト: 16; 3-5-7系なら3)')
    ap.add_argument('--fps', type=float, default=DEFAULT_FPS, help='fps (default 30)')
    ap.add_argument('--out-csv', required=True, help='出力CSVパス')
    ap.add_argument('--forearm-len', type=float, default=None, help='前腕長[m] を固定指定（未指定時は入力CSVの中央値）')
    ap.add_argument('--com-frac', type=float, default=FOREARM_COM_FRAC, help='COM距離係数 r_g = len * com_frac (default 0.430)')
    ap.add_argument('--mass-frac', type=float, default=FOREARM_MASS_FRAC, help='等価質量係数 m_x = body_mass * mass_frac (default 0.0160)')
    args = ap.parse_args(argv)

    # load pose
    df_pose = pd.read_csv(args.pose_csv)
    # allow custom joint index set (e.g., 7/5/3 instead of 12/14/16)
    def col_triplet(idx: int) -> list[str]:
        return [f'joint_{idx}_x', f'joint_{idx}_y', f'joint_{idx}_z']

    req_cols = col_triplet(args.shoulder_idx) + col_triplet(args.elbow_idx) + col_triplet(args.wrist_idx)
    missing = [c for c in req_cols if c not in df_pose.columns]
    if missing:
        raise SystemExit(f'missing column(s) {missing} in pose csv')

    p12 = df_pose[col_triplet(args.shoulder_idx)].to_numpy(float)
    p14 = df_pose[col_triplet(args.elbow_idx)].to_numpy(float)
    p16 = df_pose[col_triplet(args.wrist_idx)].to_numpy(float)
    frames_pose = df_pose['frame'].to_numpy(int) if 'frame' in df_pose.columns else np.arange(len(df_pose))

    # elbow flexion angle + angular velocity
    ang = compute_elbow_flexion_angle(p12, p14, p16)
    dt = 1.0 / (args.fps if args.fps > 0 else DEFAULT_FPS)
    ang_vel = gradient(ang, dt)

    # forearm length stats for r_g, r_x
    forearm_len = np.linalg.norm(p16 - p14, axis=1)
    forearm_len_med = float(np.nanmedian(forearm_len))
    forearm_len_use = float(args.forearm_len) if args.forearm_len is not None else forearm_len_med
    r_x = forearm_len_use
    r_g = forearm_len_use * float(args.com_frac)

    # torque
    df_tau = pd.read_csv(args.torque_csv)
    if 'elbow_R_local_y' not in df_tau.columns:
        raise SystemExit('torque csv lacks elbow_R_local_y')
    tau_y = df_tau['elbow_R_local_y'].to_numpy(float)
    frames_tau = df_tau['frame'].to_numpy(int) if 'frame' in df_tau.columns else np.arange(len(df_tau))

    # align length
    n = min(len(ang_vel), len(tau_y))
    ang_vel = ang_vel[:n]
    tau_y = tau_y[:n]
    frames = np.arange(n)
    # cycle index
    df_cyc = pd.read_csv(args.cycles_csv)
    if 'cycle_index' not in df_cyc.columns:
        raise SystemExit('cycles csv lacks cycle_index')
    cyc = df_cyc['cycle_index'].to_numpy(int)[:n]

    power = tau_y * ang_vel

    df_out = aggregate_cycles(frames, power, cyc, dt)

    m_max = read_mmax(args.mmax_csv, args.subject_id)
    theor = compute_theoretical_work(args.body_mass, m_max, r_g, r_x, mass_frac=float(args.mass_frac))
    df_out['theoretical_1rm_J'] = theor
    df_out['ratio_meas_pos_vs_theor'] = df_out['work_J_pos'] / theor if theor > 0 else np.nan

    meta = {
        'pose_csv': os.path.abspath(args.pose_csv),
        'torque_csv': os.path.abspath(args.torque_csv),
        'cycles_csv': os.path.abspath(args.cycles_csv),
        'mmax_csv': os.path.abspath(args.mmax_csv),
        'subject_id': args.subject_id,
        'body_mass': args.body_mass,
        'fps': args.fps,
        'dt': dt,
        'forearm_len_median': forearm_len_med,
        'forearm_len_used': forearm_len_use,
        'r_x': r_x,
        'r_g': r_g,
        'm_max_weight': m_max,
        'mass_frac': float(args.mass_frac),
        'com_frac': float(args.com_frac),
        'formula': '(m_x*r_g + m_maxweight*r_x)*16.73 with m_x = body_mass*mass_frac',
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    meta_path = os.path.splitext(args.out_csv)[0] + '_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'[OUT] cycles={len(df_out)} theor_J={theor:.3f} -> {args.out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
