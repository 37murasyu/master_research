"""Recalculate elbow local torque using forearm/upper-arm plane normal (y-axis).

Updates elbow_R_local_* and elbow_L_local_* in torque CSVs based on pose CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import compute_local_torque

MP_JOINTS = {
    "R_SHOULDER": 12,
    "R_ELBOW": 14,
    "R_WRIST": 16,
    "L_SHOULDER": 11,
    "L_ELBOW": 13,
    "L_WRIST": 15,
}
LEGACY_JOINTS = {
    "R_SHOULDER": 0,
    "R_ELBOW": 2,
    "R_WRIST": 4,
    "L_SHOULDER": 1,
    "L_ELBOW": 3,
    "L_WRIST": 5,
}


def _joint_cols(jid: int) -> Tuple[str, str, str]:
    return (f"joint_{jid}_x", f"joint_{jid}_y", f"joint_{jid}_z")


def _detect_joint_map(pose_df: pd.DataFrame) -> dict:
    if all(c in pose_df.columns for c in _joint_cols(MP_JOINTS["R_WRIST"])):
        return MP_JOINTS
    if all(c in pose_df.columns for c in _joint_cols(LEGACY_JOINTS["R_WRIST"])):
        return LEGACY_JOINTS
    raise ValueError("Pose CSV does not contain expected joint columns (MediaPipe or legacy)")


def _extract_vecs(pose_df: pd.DataFrame, joint_map: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def cols(name: str) -> List[str]:
        return list(_joint_cols(joint_map[name]))

    r_elbow = pose_df[cols("R_ELBOW")].to_numpy(float)
    r_wrist = pose_df[cols("R_WRIST")].to_numpy(float)
    r_shoulder = pose_df[cols("R_SHOULDER")].to_numpy(float)

    l_elbow = pose_df[cols("L_ELBOW")].to_numpy(float)
    l_wrist = pose_df[cols("L_WRIST")].to_numpy(float)
    l_shoulder = pose_df[cols("L_SHOULDER")].to_numpy(float)

    forearm_r = r_wrist - r_elbow
    upperarm_r = r_shoulder - r_elbow
    forearm_l = l_wrist - l_elbow
    upperarm_l = l_shoulder - l_elbow

    return forearm_r, upperarm_r, forearm_l, upperarm_l


def _ensure_local_cols(df: pd.DataFrame, prefix: str) -> None:
    for ax in ("x", "y", "z"):
        col = f"{prefix}_local_{ax}"
        if col not in df.columns:
            df[col] = np.nan


def _recalc_side(
    torque_df: pd.DataFrame,
    forearm: np.ndarray,
    upperarm: np.ndarray,
    side: str,
) -> int:
    torque_cols = [f"elbow_{side}_x", f"elbow_{side}_y", f"elbow_{side}_z"]
    for col in torque_cols:
        if col not in torque_df.columns:
            raise ValueError(f"Missing torque column: {col}")

    _ensure_local_cols(torque_df, f"elbow_{side}")
    local_cols = [f"elbow_{side}_local_x", f"elbow_{side}_local_y", f"elbow_{side}_local_z"]

    tau_global = torque_df[torque_cols].to_numpy(float)
    local_existing = torque_df[local_cols].to_numpy(float)
    local_new = local_existing.copy()

    valid = (
        np.isfinite(forearm).all(axis=1)
        & np.isfinite(upperarm).all(axis=1)
        & np.isfinite(tau_global).all(axis=1)
    )

    for i in np.where(valid)[0]:
        local_new[i] = compute_local_torque(tau_global[i], forearm[i], upperarm[i])

    torque_df[local_cols] = local_new
    return int(valid.sum())


def recalc_pair(pose_csv: Path, torque_csv: Path, out_dir: Path | None) -> None:
    pose_df = pd.read_csv(pose_csv)
    torque_df = pd.read_csv(torque_csv)
    if "frame" not in pose_df.columns or "frame" not in torque_df.columns:
        raise ValueError("Both pose and torque CSVs must contain 'frame' column")

    joint_map = _detect_joint_map(pose_df)
    pose_aligned = torque_df[["frame"]].merge(pose_df, on="frame", how="left")

    forearm_r, upperarm_r, forearm_l, upperarm_l = _extract_vecs(pose_aligned, joint_map)

    cnt_r = _recalc_side(torque_df, forearm_r, upperarm_r, "R")
    cnt_l = _recalc_side(torque_df, forearm_l, upperarm_l, "L")

    out_path = torque_csv
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / torque_csv.name

    torque_df.to_csv(out_path, index=False)
    print(f"[OK] {torque_csv.name} -> {out_path} (R:{cnt_r} rows, L:{cnt_l} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalculate elbow local torque using forearm/upper-arm plane normal.")
    parser.add_argument("--pose-csv", action="append", required=True, help="Pose CSV path (repeatable)")
    parser.add_argument("--torque-csv", action="append", required=True, help="Torque CSV path (repeatable)")
    parser.add_argument("--out-dir", default=None, help="Optional output directory (overwrite if omitted)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_paths = [Path(p) for p in args.pose_csv]
    torque_paths = [Path(p) for p in args.torque_csv]
    if len(pose_paths) != len(torque_paths):
        raise ValueError("Number of --pose-csv and --torque-csv entries must match")

    out_dir = Path(args.out_dir) if args.out_dir else None
    for pose_csv, torque_csv in zip(pose_paths, torque_paths):
        recalc_pair(pose_csv, torque_csv, out_dir)


if __name__ == "__main__":
    main()
