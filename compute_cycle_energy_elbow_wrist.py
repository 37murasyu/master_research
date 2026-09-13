from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from compute_torque_from_pose import (
    WRIST_BASE_SEGMENTS_LEFT,
    WRIST_BASE_SEGMENTS_RIGHT,
    compute_segment_kinematics,
)
from config import THEORETICAL_WORK_COEFF
from utils import compute_joint_power

FOREARM_MASS_FRAC = 0.0160
HAND_MASS_FRAC = 0.0060
FOREARM_COM_FRAC = 0.430
HAND_COM_FRAC = 0.506
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


def _parse_subject_id(stem: str) -> int | None:
    # examples: 2_stereo_pose_lpf, 3_0stereo_pose..., kpts3d_9_20250925_...
    if stem.startswith("kpts3d_"):
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
    head = stem.split("_")[0]
    return int(head) if head.isdigit() else None


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


def _map_torque_csv(torque_dir: Path, pose_with_cycles: Path) -> Path:
    stem = pose_with_cycles.stem.replace("_with_cycles", "")
    if stem.endswith("_lpf"):
        torque_stem = stem.replace("_lpf", "_torque_lpf")
    else:
        torque_stem = stem + "_torque_lpf"
    cand = torque_dir / f"{torque_stem}.csv"
    if cand.exists():
        return cand
    # fallback: wrist-base outputs prefix_torque.csv
    return torque_dir / f"{stem}_torque.csv"


def _compute_lengths(pose_df: pd.DataFrame, side: Dict[str, int]) -> Tuple[np.ndarray, float, float]:
    p_el = pose_df[_col_triplet(side["elbow"])].to_numpy(float)
    p_wr = pose_df[_col_triplet(side["wrist"])].to_numpy(float)
    forearm = p_wr - p_el
    forearm_len = np.linalg.norm(forearm, axis=1)
    forearm_len_med = float(np.nanmedian(forearm_len))
    r_x = forearm_len_med
    return forearm, forearm_len_med, r_x


def _pose_array(pose_df: pd.DataFrame, joint_ids) -> np.ndarray:
    """姿勢 CSV の列を (フレーム, 最大 ID + 1, 3) の配列にする。使わない関節は NaN。"""
    pose = np.full((len(pose_df), max(joint_ids) + 1, 3), np.nan)
    for jid in joint_ids:
        pose[:, jid, :] = pose_df[_col_triplet(jid)].to_numpy(float)
    return pose


def _joint_powers(
    pose_df: pd.DataFrame, torque_df: pd.DataFrame, side_name: str, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """肘と手首の仕事率 [W] を (肘, 手首) で返す。

    トルク CSV は ``compute_torque_from_pose.py --wrist-base`` の出力（全体座標の列）を前提にする。
    手を固定端として前腕 → 上腕の順に解いた鎖なので、関節の相対角速度も同じ鎖・同じリンクから取る:

    - 手首: 前腕の角速度（親の手は固定）。局所軸は前腕リンクから作る
    - 肘: 上腕の角速度 − 前腕の角速度。局所軸は上腕リンクと親の前腕リンクから作る

    どちらもトルクと同じ局所軸に射影して掛ける（``utils.compute_joint_power``）。
    角度を経由しないので、fps の掛け戻し（§1-1）も ±π の折り返し（§1-2）も起きない。

    かつて ``*_local_y`` に「+Y まわりの肘角」「水平面からの前腕の傾き」の微分を掛けていた。
    軸の作り方が τ と別なので、左右を鏡映すると片方だけ符号が反転し、左右で逆の相を積算していた。
    """
    segments = WRIST_BASE_SEGMENTS_RIGHT if side_name == "R" else WRIST_BASE_SEGMENTS_LEFT
    joint_ids = sorted({seg.proximal_joint for seg in segments} | {seg.distal_joint for seg in segments})
    # 角速度とリンクは、トルクを出した compute_torque_from_pose と同じ関数で求める（0: 前腕、1: 上腕）
    omegas, _, _, _, _, links = compute_segment_kinematics(_pose_array(pose_df, joint_ids), segments, dt)
    wrist_tau = torque_df[[f"wrist_{side_name}_{ax}" for ax in "xyz"]].to_numpy(float)
    elbow_tau = torque_df[[f"elbow_{side_name}_{ax}" for ax in "xyz"]].to_numpy(float)
    n = min(len(pose_df), len(torque_df))
    wrist = np.array([
        compute_joint_power(wrist_tau[t], omegas[t, 0], None, links[t, 0]) for t in range(n)])
    elbow = np.array([
        compute_joint_power(elbow_tau[t], omegas[t, 1], omegas[t, 0], links[t, 1], links[t, 0])
        for t in range(n)])
    return elbow, wrist


def _aggregate_cycles(frame_idx: np.ndarray, power: np.ndarray, cycle_index: np.ndarray, dt: float) -> pd.DataFrame:
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


def _theoretical_work(m_x: float, m_db: float, r_g: float, r_x: float) -> float:
    # 係数は config に集約（1 サイクルの角度範囲にわたる cos の積分 × g）。
    # かつて 16.73 と直書きされており、ゲージ閾値側の √3/2+1 と 9.3% 食い違っていた。
    return (m_x * r_g + m_db * r_x) * THEORETICAL_WORK_COEFF


def _load_mmax(mmax_df: pd.DataFrame, subject_id: int, col: str) -> float:
    row = mmax_df.loc[mmax_df["subject_id"] == subject_id]
    if row.empty:
        raise ValueError(f"subject_id {subject_id} not found")
    val = row[col].iloc[0]
    if isinstance(val, str) and val.strip().lower() == "none":
        raise ValueError(f"{col} is none for subject_id {subject_id}")
    if pd.isna(val):
        raise ValueError(f"{col} is NaN for subject_id {subject_id}")
    return float(val)


def _prepare_cycle_map(cycles_df: pd.DataFrame) -> Dict[int, int]:
    if "frame" not in cycles_df.columns or "cycle_index" not in cycles_df.columns:
        raise ValueError("cycles csv must have frame and cycle_index")
    return {int(f): int(c) for f, c in zip(cycles_df["frame"], cycles_df["cycle_index"])}


def _merge_by_frame(df: pd.DataFrame, frame_map: Dict[int, int]) -> np.ndarray:
    frames = df["frame"].to_numpy(int) if "frame" in df.columns else np.arange(len(df))
    return np.array([frame_map.get(int(f), -1) for f in frames], dtype=int)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute per-cycle elbow/wrist energy from tau_x and omega_x")
    ap.add_argument("--pose-dir", default="output_data/filtered_pose_lpf", help="pose dir with *_with_cycles.csv")
    ap.add_argument("--torque-dir", default="output_data/filtered_torque_lpf_recalc", help="torque dir with *_torque_lpf.csv")
    ap.add_argument("--mmax-csv", default="m_max_all_merged.csv", help="1RM source csv")
    ap.add_argument("--body-mass", type=float, default=60.0, help="body mass [kg]")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="fps")
    ap.add_argument("--pose-unit", default="auto", choices=["auto", "m", "cm", "mm"], help="pose length unit")
    ap.add_argument("--torque-scale", type=float, default=1.0, help="scale torque (e.g., 0.01 if N*cm -> N*m)")
    ap.add_argument("--out-dir", default="output_data/cycle_energy", help="output directory")
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir)
    torque_dir = Path(args.torque_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mmax_df = pd.read_csv(args.mmax_csv)

    pose_files = sorted(pose_dir.glob("*_with_cycles.csv"))
    for pose_path in pose_files:
        stem = pose_path.stem.replace("_with_cycles", "")
        subject_id = _parse_subject_id(stem)
        if subject_id is None:
            continue
        if subject_id == 4:
            continue

        torque_path = _map_torque_csv(torque_dir, pose_path)
        if not torque_path.exists():
            print(f"[SKIP] torque not found: {torque_path}")
            continue

        pose_df = pd.read_csv(pose_path)
        torque_df = pd.read_csv(torque_path)
        cycles_df = pose_df[["frame", "cycle_index"]] if "cycle_index" in pose_df.columns else None
        if cycles_df is None:
            print(f"[SKIP] cycle_index missing: {pose_path}")
            continue

        cycle_map = _prepare_cycle_map(cycles_df)
        torque_cycle = _merge_by_frame(torque_df, cycle_map)
        pose_cycle = _merge_by_frame(pose_df, cycle_map)

        dt = 1.0 / (args.fps if args.fps > 0 else DEFAULT_FPS)

        # pose unit scaling
        pose_cols = [c for c in pose_df.columns if c.startswith("joint_") and c.endswith(('_x','_y','_z'))]
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
            # 仕事率はトルク（全体座標の列）と姿勢から、関節の相対角速度で求める
            needed = [f"{part}_{side_name}_{ax}" for part in ("elbow", "wrist") for ax in "xyz"]
            if any(col not in torque_df.columns for col in needed):
                print(f"[SKIP] missing torque columns for {stem} {side_name}")
                continue
            elbow_power, wrist_power = _joint_powers(pose_scaled, torque_df, side_name, dt)
            n = len(elbow_power)
            elbow_power = elbow_power * args.torque_scale
            wrist_power = wrist_power * args.torque_scale
            cycle_idx = torque_cycle[:n]

            elbow_cycles = _aggregate_cycles(np.arange(n), elbow_power, cycle_idx, dt)
            wrist_cycles = _aggregate_cycles(np.arange(n), wrist_power, cycle_idx, dt)

            # lever arms (forearm length as elbow->wrist distance)
            forearm_vec, forearm_len_med, r_x_elbow = _compute_lengths(pose_scaled, side)
            # equivalent masses
            m_forearm = args.body_mass * FOREARM_MASS_FRAC
            m_hand = args.body_mass * HAND_MASS_FRAC
            m_x_elbow = m_forearm + m_hand
            m_x_wrist = m_hand
            # COM distances from joint
            r_forearm = forearm_len_med * FOREARM_COM_FRAC
            # hand COM distance from elbow: assume COM at wrist (hand length unknown)
            r_hand_from_elbow = forearm_len_med
            r_g_elbow = (m_forearm * r_forearm + m_hand * r_hand_from_elbow) / max(m_x_elbow, 1e-9)
            # wrist COM distance (hand COM from wrist). Use forearm length as proxy.
            r_g_wrist = forearm_len_med * HAND_COM_FRAC
            # external force point at wrist
            r_x_wrist = 0.0

            # 1RM from m_max_all_merged
            try:
                m_db_elbow = _load_mmax(mmax_df, subject_id, f"elbow_{side_name}_outer")
            except Exception as e:
                print(f"[WARN] {stem} {side_name} elbow mmax missing: {e}")
                m_db_elbow = np.nan
            try:
                m_db_wrist = _load_mmax(mmax_df, subject_id, f"wrist_{side_name}")
            except Exception as e:
                print(f"[WARN] {stem} {side_name} wrist mmax missing: {e}")
                m_db_wrist = np.nan

            theor_elbow = _theoretical_work(m_x_elbow, m_db_elbow, r_g_elbow, r_x_elbow) if np.isfinite(m_db_elbow) else np.nan
            theor_wrist = _theoretical_work(m_x_wrist, m_db_wrist, r_g_wrist, r_x_wrist) if np.isfinite(m_db_wrist) else np.nan

            elbow_cycles["part"] = f"elbow_{side_name}"
            elbow_cycles["subject_id"] = subject_id
            elbow_cycles["theoretical_1rm_J"] = theor_elbow
            elbow_cycles["ratio_pos_vs_1rm"] = elbow_cycles["work_J_pos"] / theor_elbow if np.isfinite(theor_elbow) and theor_elbow > 0 else np.nan

            wrist_cycles["part"] = f"wrist_{side_name}"
            wrist_cycles["subject_id"] = subject_id
            wrist_cycles["theoretical_1rm_J"] = theor_wrist
            wrist_cycles["ratio_pos_vs_1rm"] = wrist_cycles["work_J_pos"] / theor_wrist if np.isfinite(theor_wrist) and theor_wrist > 0 else np.nan

            out_df = pd.concat([elbow_cycles, wrist_cycles], ignore_index=True)
            out_path = out_dir / f"cycle_energy_{stem}_s{subject_id}_{side_name}.csv"
            out_df.to_csv(out_path, index=False)
            print(f"[OUT] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
