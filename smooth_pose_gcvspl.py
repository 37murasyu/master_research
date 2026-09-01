"""Smooth 3D pose CSVs with GCV spline and output pos/vel/acc (dt=1/30 by default).

If the optional ``gcvspline`` package is available, it is used first. Otherwise the
code falls back to SciPy's ``UnivariateSpline`` for a light smoothing + derivatives.

Outputs (all CSV):
- *_gcvspl.csv          : smoothed positions, same columns as input (joint_{id}_{axis})
- *_gcvspl_vel.csv      : first derivatives, columns joint_{id}_{axis}_vel
- *_gcvspl_acc.csv      : second derivatives, columns joint_{id}_{axis}_acc
- *_gcvspl_angles.csv   : elbow angles (raw/smoothed) in degrees when joints exist
- *_gcvspl_summary.json : paths + dt/fps
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _load_pose_csv(path: str) -> Tuple[np.ndarray, List[int], List[str]]:
    df = pd.read_csv(path)
    joint_ids = sorted(
        {
            int(col.split("_")[1])
            for col in df.columns
            if col.startswith("joint_")
            and col.split("_")[1].isdigit()
            and col.split("_")[-1] in ("x", "y", "z")
        }
    )
    if not joint_ids:
        raise ValueError("No joint_*_x/y/z columns found")

    axes = ["x", "y", "z"]
    T = len(df)
    J = max(joint_ids) + 1
    pos = np.full((T, J, 3), np.nan, dtype=float)
    for jid in joint_ids:
        for a_idx, ax in enumerate(axes):
            col = f"joint_{jid}_{ax}"
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")
            s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            s[s == -1.0] = np.nan
            pos[:, jid, a_idx] = s
    return pos, joint_ids, axes


def _interpolate_nan(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    mask = ~np.isfinite(y)
    if not mask.any():
        return y, mask
    idx = np.arange(len(y))
    good = ~mask
    if good.any():
        y_filled = np.interp(idx, idx[good], y[good])
    else:
        y_filled = np.zeros_like(y)
    return y_filled, mask


def _smooth_with_gcvspline(t: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        import gcvspline  # type: ignore
    except Exception:
        return None

    try:
        # charlesll/gcvspline exposes gcvspline.gcv_spline
        out = gcvspline.gcv_spline(t, y, m=2)  # returns coeffs, yhat, info
        yhat = out[1]
        # Derivatives via gcvspline.spline_derivatives if available
        if hasattr(gcvspline, "spline_derivatives"):
            dy1, dy2 = gcvspline.spline_derivatives(t, out[0], der=2)
        else:
            dy1 = np.gradient(yhat, t)
            dy2 = np.gradient(dy1, t)
        return yhat, dy1, dy2
    except Exception:
        return None


def _smooth_with_scipy(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.interpolate import UnivariateSpline  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy missing
        raise RuntimeError("SciPy not available and gcvspline missing") from exc

    # light smoothing: s scaled to length
    s_val = max(len(y) * 1e-3, 1e-6)
    spl = UnivariateSpline(t, y, s=s_val, k=3)
    yhat = spl(t)
    dy1 = spl.derivative(1)(t)
    dy2 = spl.derivative(2)(t)
    return yhat, dy1, dy2


def _smooth_axis(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_filled, nanmask = _interpolate_nan(y)
    res = _smooth_with_gcvspline(t, y_filled)
    if res is None:
        res = _smooth_with_scipy(t, y_filled)
    yhat, dy1, dy2 = res
    if nanmask.any():
        yhat[nanmask] = np.nan
        dy1[nanmask] = np.nan
        dy2[nanmask] = np.nan
    return yhat, dy1, dy2


def _compute_elbow_angles(pos_raw: np.ndarray, pos_smooth: np.ndarray) -> Dict[str, np.ndarray]:
    def angle_deg(s: np.ndarray, e: np.ndarray, w: np.ndarray) -> np.ndarray:
        v1 = s - e
        v2 = w - e
        num = np.einsum("ij,ij->i", v1, v2)
        den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        den = np.where(den < 1e-9, np.nan, den)
        cos_th = np.clip(num / den, -1.0, 1.0)
        return np.degrees(np.arccos(cos_th))

    out: Dict[str, np.ndarray] = {}
    T = pos_raw.shape[0]
    def has_joints(ids: Sequence[int]) -> bool:
        return all(j < pos_raw.shape[1] and np.isfinite(pos_raw[:, j, 0]).any() for j in ids)

    if has_joints((12, 14, 16)):
        out["elbow_R_angle_raw_deg"] = angle_deg(pos_raw[:, 12], pos_raw[:, 14], pos_raw[:, 16])
        out["elbow_R_angle_smooth_deg"] = angle_deg(pos_smooth[:, 12], pos_smooth[:, 14], pos_smooth[:, 16])
    if has_joints((11, 13, 15)):
        out["elbow_L_angle_raw_deg"] = angle_deg(pos_raw[:, 11], pos_raw[:, 13], pos_raw[:, 15])
        out["elbow_L_angle_smooth_deg"] = angle_deg(pos_smooth[:, 11], pos_smooth[:, 13], pos_smooth[:, 15])
    return out


def process_file(path: str, fps: float, out_dir: Optional[str], suffix: str) -> Dict[str, str]:
    pos_raw, joint_ids, axes = _load_pose_csv(path)
    dt = 1.0 / fps
    t = np.arange(pos_raw.shape[0], dtype=float) * dt

    pos_f = np.full_like(pos_raw, np.nan)
    vel = np.full_like(pos_raw, np.nan)
    acc = np.full_like(pos_raw, np.nan)

    for j in joint_ids:
        for a_idx, ax in enumerate(axes):
            y = pos_raw[:, j, a_idx]
            yhat, dy1, dy2 = _smooth_axis(t, y)
            pos_f[:, j, a_idx] = yhat
            vel[:, j, a_idx] = dy1
            acc[:, j, a_idx] = dy2

    base = os.path.splitext(os.path.basename(path))[0]
    out_root = out_dir or os.path.dirname(path) or "."
    os.makedirs(out_root, exist_ok=True)

    pos_path = os.path.join(out_root, f"{base}{suffix}.csv")
    vel_path = os.path.join(out_root, f"{base}{suffix}_vel.csv")
    acc_path = os.path.join(out_root, f"{base}{suffix}_acc.csv")
    ang_path = os.path.join(out_root, f"{base}{suffix}_angles.csv")
    summary_path = os.path.join(out_root, f"{base}{suffix}_summary.json")

    def dump_positions(arr: np.ndarray, path_out: str) -> None:
        T = arr.shape[0]
        df_out = pd.DataFrame()
        df_out["frame"] = np.arange(T, dtype=int)
        for j in joint_ids:
            for a_idx, ax in enumerate(axes):
                df_out[f"joint_{j}_{ax}"] = arr[:, j, a_idx]
        df_out.to_csv(path_out, index=False)

    dump_positions(pos_f, pos_path)
    # velocities
    df_vel = pd.DataFrame()
    df_vel["frame"] = np.arange(pos_raw.shape[0], dtype=int)
    for j in joint_ids:
        for a_idx, ax in enumerate(axes):
            df_vel[f"joint_{j}_{ax}_vel"] = vel[:, j, a_idx]
    df_vel.to_csv(vel_path, index=False)

    df_acc = pd.DataFrame()
    df_acc["frame"] = np.arange(pos_raw.shape[0], dtype=int)
    for j in joint_ids:
        for a_idx, ax in enumerate(axes):
            df_acc[f"joint_{j}_{ax}_acc"] = acc[:, j, a_idx]
    df_acc.to_csv(acc_path, index=False)

    angles = _compute_elbow_angles(pos_raw, pos_f)
    if angles:
        df_ang = pd.DataFrame({"frame": np.arange(pos_raw.shape[0], dtype=int), **angles})
        df_ang.to_csv(ang_path, index=False)
    else:
        ang_path = ""

    summary = {
        "pose_csv": os.path.relpath(pos_path, os.getcwd()),
        "velocity_csv": os.path.relpath(vel_path, os.getcwd()),
        "acceleration_csv": os.path.relpath(acc_path, os.getcwd()),
        "angles_csv": os.path.relpath(ang_path, os.getcwd()) if ang_path else None,
        "dt": dt,
        "fps": fps,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "pose_csv": pos_path,
        "velocity_csv": vel_path,
        "acceleration_csv": acc_path,
        "angles_csv": ang_path,
        "summary": summary_path,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Smooth pose CSV with GCV spline and output pos/vel/acc")
    ap.add_argument("--csv", required=True, nargs="+", help="Input pose CSV paths (joint_* columns)")
    ap.add_argument("--fps", type=float, default=30.0, help="Sampling rate [Hz] (default 30)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: same as input)")
    ap.add_argument("--suffix", default="_gcvspl", help="Suffix for outputs before extension")
    args = ap.parse_args(argv)

    for p in args.csv:
        outs = process_file(p, args.fps, args.out_dir, args.suffix)
        print(f"[OUT] {os.path.basename(p)} -> {os.path.basename(outs['pose_csv'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())