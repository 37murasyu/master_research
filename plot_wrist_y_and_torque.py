import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mad_clip(series: np.ndarray, threshold: float = 3.5, force_nan_first: bool = False, drop_first_n: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return cleaned series and boolean mask of outliers using MAD z-score."""
    med = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - med))
    if mad < 1e-12:
        mask = np.zeros_like(series, dtype=bool)
        series = series.copy()
        n = min(drop_first_n, len(series))
        if n > 0:
            mask[:n] = True
            series[:n] = np.nan
        if force_nan_first and len(series) > 0:
            mask[0] = True
            series[0] = np.nan
        return series, mask

    z = np.abs(series - med) / (1.4826 * mad)
    outliers = z > threshold
    outliers = outliers.copy()
    n = min(drop_first_n, len(series))
    if n > 0:
        outliers[:n] = True
    if force_nan_first and len(series) > 0:
        outliers[0] = True
    cleaned = series.copy()
    cleaned[outliers] = np.nan

    if outliers.any():
        idx = np.arange(len(cleaned))
        ok = np.isfinite(cleaned)
        if ok.any():
            cleaned = np.interp(idx, idx[ok], cleaned[ok])
        else:
            cleaned = series
    return cleaned, outliers


def percentile_ylim(data: np.ndarray, low: float = 1.0, high: float = 99.0, pad: float = 0.05) -> Optional[Tuple[float, float]]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    p_low, p_high = np.percentile(finite, [low, high])
    if p_high - p_low < 1e-9:
        span = max(1e-3, abs(p_high) * 0.1)
        return p_low - span, p_high + span
    extra = (p_high - p_low) * pad
    return p_low - extra, p_high + extra


def plot_wrist_y(pose_csv: Path, out_dir: Path, mass: Optional[float], drop_first_n: int) -> Path:
    df = pd.read_csv(pose_csv)
    frames = df["frame"].to_numpy()
    y = df["joint_16_y"].to_numpy(float)

    y_clean, outliers = mad_clip(y, drop_first_n=drop_first_n)

    mask = np.arange(len(frames)) >= drop_first_n
    frames_p = frames[mask]
    y_p = y[mask]
    y_clean_p = y_clean[mask]
    outliers_p = outliers[mask]

    plt.figure(figsize=(12, 4))
    plt.plot(frames_p, y_p, label="raw joint_16_y", alpha=0.6)
    plt.plot(frames_p, y_clean_p, label="outlier-clipped", linewidth=2)
    if outliers_p.any():
        plt.scatter(frames_p[outliers_p], y_p[outliers_p], color="red", s=12, label="outliers")
    title_mass = f" (mass={mass} kg)" if mass is not None else ""
    plt.title(f"Right wrist y{title_mass}")
    plt.xlabel("frame")
    plt.ylabel("joint_16_y (m)")
    ylim = percentile_ylim(y_clean_p)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"{pose_csv.stem}_wrist_y_outlier_clipped.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_torque_local_y(torque_csv: Path, out_dir: Path, mass: Optional[float], drop_first_n: int) -> Path:
    df = pd.read_csv(torque_csv)
    frames = df["frame"].to_numpy()

    sh = df["shoulder_R_local_y"].to_numpy(float)
    el = df["elbow_R_local_y"].to_numpy(float)

    sh_clean, sh_out = mad_clip(sh, force_nan_first=True, drop_first_n=drop_first_n)
    el_clean, el_out = mad_clip(el, force_nan_first=True, drop_first_n=drop_first_n)

    mask = np.arange(len(frames)) >= drop_first_n
    frames_p = frames[mask]
    sh_p, el_p = sh[mask], el[mask]
    sh_clean_p, el_clean_p = sh_clean[mask], el_clean[mask]
    sh_out_p, el_out_p = sh_out[mask], el_out[mask]

    plt.figure(figsize=(12, 4))
    plt.plot(frames_p, sh_p, label="shoulder_R_local_y (raw)", alpha=0.45)
    plt.plot(frames_p, sh_clean_p, label="shoulder_R_local_y (clipped)", linewidth=2)
    if sh_out_p.any():
        plt.scatter(frames_p[sh_out_p], sh_p[sh_out_p], color="red", s=10)

    plt.plot(frames_p, el_p, label="elbow_R_local_y (raw)", alpha=0.45)
    plt.plot(frames_p, el_clean_p, label="elbow_R_local_y (clipped)", linewidth=2)
    if el_out_p.any():
        plt.scatter(frames_p[el_out_p], el_p[el_out_p], color="orange", s=10)

    title_mass = f" (mass={mass} kg)" if mass is not None else ""
    plt.title(f"Right shoulder/elbow local y torque{title_mass}")
    plt.xlabel("frame")
    plt.ylabel("torque (Nm)")
    ylim = percentile_ylim(np.concatenate([sh_clean_p, el_clean_p]))
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"{torque_csv.stem}_torque_local_y.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_torque_local_xyz(torque_csv: Path, out_dir: Path, mass: Optional[float], drop_first_n: int) -> Path:
    df = pd.read_csv(torque_csv)
    frames = df["frame"].to_numpy()

    comps = {
        "shoulder_R": df[["shoulder_R_local_x", "shoulder_R_local_y", "shoulder_R_local_z"]].to_numpy(float),
        "elbow_R": df[["elbow_R_local_x", "elbow_R_local_y", "elbow_R_local_z"]].to_numpy(float),
    }

    plt.figure(figsize=(12, 6))
    for idx, (name, arr) in enumerate(comps.items(), start=1):
        clean = np.empty_like(arr)
        masks = []
        for c in range(3):
            clean[:, c], mask_c = mad_clip(arr[:, c], force_nan_first=True, drop_first_n=drop_first_n)
            masks.append(mask_c)

        mask = np.arange(len(frames)) >= drop_first_n
        frames_p = frames[mask]
        clean_p = clean[mask]
        raw_p = arr[mask]
        mask_p = [m[mask] for m in masks]

        ax = plt.subplot(2, 1, idx)
        ax.plot(frames_p, raw_p[:, 0], label=f"{name}_x raw", alpha=0.4, color="tab:blue")
        ax.plot(frames_p, clean_p[:, 0], label=f"{name}_x clipped", linewidth=2, color="tab:blue")
        if mask_p[0].any():
            ax.scatter(frames_p[mask_p[0]], raw_p[mask_p[0], 0], color="tab:blue", s=10)

        ax.plot(frames_p, raw_p[:, 1], label=f"{name}_y raw", alpha=0.4, color="tab:orange")
        ax.plot(frames_p, clean_p[:, 1], label=f"{name}_y clipped", linewidth=2, color="tab:orange")
        if mask_p[1].any():
            ax.scatter(frames_p[mask_p[1]], raw_p[mask_p[1], 1], color="tab:orange", s=10)

        ax.plot(frames_p, raw_p[:, 2], label=f"{name}_z raw", alpha=0.4, color="tab:green")
        ax.plot(frames_p, clean_p[:, 2], label=f"{name}_z clipped", linewidth=2, color="tab:green")
        if mask_p[2].any():
            ax.scatter(frames_p[mask_p[2]], raw_p[mask_p[2], 2], color="tab:green", s=10)

        ylim = percentile_ylim(clean_p.flatten())
        if ylim:
            ax.set_ylim(*ylim)
        title_mass = f" (mass={mass} kg)" if mass is not None else ""
        ax.set_title(f"{name} local torque xyz{title_mass}")
        ax.set_xlabel("frame")
        ax.set_ylabel("torque (Nm)")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(False)

    plt.tight_layout()
    out_path = out_dir / f"{torque_csv.stem}_torque_local_xyz.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def central_diff(series: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    vel = np.zeros_like(series)
    acc = np.zeros_like(series)
    if len(series) >= 2:
        vel[0] = (series[1] - series[0]) / dt
        vel[-1] = (series[-1] - series[-2]) / dt
    if len(series) >= 3:
        vel[1:-1] = (series[2:] - series[:-2]) / (2 * dt)
        acc[1:-1] = (series[2:] - 2 * series[1:-1] + series[:-2]) / (dt * dt)
        acc[0] = (series[2] - 2 * series[1] + series[0]) / (dt * dt)
        acc[-1] = (series[-1] - 2 * series[-2] + series[-3]) / (dt * dt)
    return vel, acc


def plot_wrist_y_with_accel(pose_csv: Path, out_dir: Path, mass: Optional[float], fps: float, drop_first_n: int) -> Path:
    df = pd.read_csv(pose_csv)
    frames = df["frame"].to_numpy()
    y = df["joint_16_y"].to_numpy(float)
    dt = 1.0 / fps

    vel, acc = central_diff(y, dt)
    acc_clean, acc_out = mad_clip(acc, threshold=4.0, force_nan_first=True, drop_first_n=drop_first_n)

    mask = np.arange(len(frames)) >= drop_first_n
    frames_p = frames[mask]
    acc_p = acc[mask]
    acc_clean_p = acc_clean[mask]
    acc_out_p = acc_out[mask]

    plt.figure(figsize=(12, 4))
    plt.plot(frames_p, acc_p, label="acc_y raw", alpha=0.45)
    plt.plot(frames_p, acc_clean_p, label="acc_y clipped", linewidth=2)
    if acc_out_p.any():
        plt.scatter(frames_p[acc_out_p], acc_p[acc_out_p], color="red", s=10)
    title_mass = f" (mass={mass} kg)" if mass is not None else ""
    plt.title(f"Right wrist y acceleration{title_mass}")
    plt.xlabel("frame")
    plt.ylabel("acc (m/s^2)")
    ylim = percentile_ylim(acc_clean_p)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"{pose_csv.stem}_wrist_y_acc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run(paths: Iterable[Path], masses: Optional[List[float]], out_dir: Path, fps: float, drop_first_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_list = list(paths)

    masses_iter: List[Optional[float]]
    if masses is None:
        masses_iter = [None] * len(pose_list)
    else:
        masses_iter = [masses[i] if i < len(masses) else None for i in range(len(pose_list))]

    for idx, pose_csv in enumerate(pose_list):
        mass = masses_iter[idx] if idx < len(masses_iter) else None
        torque_csv = pose_csv.parent.parent / "torque" / f"{pose_csv.stem}_torque.csv"
        wrist_png = plot_wrist_y(pose_csv, out_dir, mass, drop_first_n)
        wrist_acc_png = plot_wrist_y_with_accel(pose_csv, out_dir, mass, fps, drop_first_n)
        torque_png = plot_torque_local_y(torque_csv, out_dir, mass, drop_first_n)
        torque_xyz_png = plot_torque_local_xyz(torque_csv, out_dir, mass, drop_first_n)
        print(f"[OUT] {pose_csv.name} wrist plot -> {wrist_png}")
        print(f"[OUT] {pose_csv.name} wrist accel plot -> {wrist_acc_png}")
        print(f"[OUT] {pose_csv.name} torque plot -> {torque_png}")
        print(f"[OUT] {pose_csv.name} torque xyz plot -> {torque_xyz_png}")


def main():
    ap = argparse.ArgumentParser(description="Plot wrist y with outlier clipping and torque local y components")
    ap.add_argument("--pose", nargs="+", required=True, help="Pose CSVs (joint_* columns, e.g., *_joint_gcvspl.csv)")
    ap.add_argument("--mass", nargs="*", type=float, help="Optional masses (kg) aligned with pose files")
    ap.add_argument("--out-dir", default="output_data/plots", help="Directory for output PNGs")
    ap.add_argument("--fps", type=float, default=30.0, help="Sampling rate of pose CSV")
    ap.add_argument("--drop-first", type=int, default=0, help="Drop (set NaN) for the first N frames before clipping")
    args = ap.parse_args()

    pose_paths = [Path(p) for p in args.pose]
    masses = args.mass if args.mass else None
    out_dir = Path(args.out_dir)
    run(pose_paths, masses, out_dir, args.fps, args.drop_first)


if __name__ == "__main__":
    main()