import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mad_clip(series: np.ndarray, threshold: float = 3.5, drop_first: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    series = series.astype(float)
    mask = np.zeros_like(series, dtype=bool)
    n = min(drop_first, len(series))
    if n:
        mask[:n] = True
        series[:n] = np.nan

    med = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - med))
    if mad < 1e-12:
        return series, mask

    z = np.abs(series - med) / (1.4826 * mad)
    mask |= z > threshold
    cleaned = series.copy()
    cleaned[mask] = np.nan
    if mask.any():
        idx = np.arange(len(series))
        ok = np.isfinite(cleaned)
        if ok.any():
            cleaned = np.interp(idx, idx[ok], cleaned[ok])
    return cleaned, mask


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


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dot = np.einsum("ij,ij->i", u, v)
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    denom = np.clip(nu * nv, 1e-12, None)
    cos_th = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def percentile_ylim(data: np.ndarray, low: float = 1.0, high: float = 99.0, pad: float = 0.05):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    p_low, p_high = np.percentile(finite, [low, high])
    span = p_high - p_low
    if span < 1e-9:
        span = max(1e-3, abs(p_high) * 0.1)
    extra = span * pad
    return p_low - extra, p_high + extra


def plot_elbow_torque_xyz(torque_csv: Path, out_dir: Path, drop_first: int):
    df = pd.read_csv(torque_csv)
    frames = df["frame"].to_numpy()
    comps = df[["elbow_R_local_x", "elbow_R_local_y", "elbow_R_local_z"]].to_numpy(float)

    clean = np.empty_like(comps)
    masks = []
    for c in range(3):
        clean[:, c], m = mad_clip(comps[:, c], drop_first=drop_first)
        masks.append(m)

    mask = np.arange(len(frames)) >= drop_first
    f = frames[mask]
    raw = comps[mask]
    cln = clean[mask]
    mks = [m[mask] for m in masks]

    plt.figure(figsize=(12, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["x", "y", "z"]
    for i in range(3):
        plt.plot(f, raw[:, i], label=f"elbow_{labels[i]} raw", alpha=0.35, color=colors[i])
        plt.plot(f, cln[:, i], label=f"elbow_{labels[i]} clipped", linewidth=2, color=colors[i])
        if mks[i].any():
            plt.scatter(f[mks[i]], raw[mks[i], i], color=colors[i], s=10)
    ylim = percentile_ylim(cln)
    if ylim:
        plt.ylim(*ylim)
    plt.title("Elbow local torque (xyz)")
    plt.xlabel("frame")
    plt.ylabel("torque (Nm)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    out = out_dir / f"{torque_csv.stem}_elbow_local_xyz.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_elbow_torque_xyz_global(torque_csv: Path, out_dir: Path, drop_first: int):
    df = pd.read_csv(torque_csv)
    frames = df["frame"].to_numpy()
    comps = df[["elbow_R_x", "elbow_R_y", "elbow_R_z"]].to_numpy(float)

    clean = np.empty_like(comps)
    masks = []
    for c in range(3):
        clean[:, c], m = mad_clip(comps[:, c], drop_first=drop_first)
        masks.append(m)

    mask = np.arange(len(frames)) >= drop_first
    f = frames[mask]
    raw = comps[mask]
    cln = clean[mask]
    mks = [m[mask] for m in masks]

    plt.figure(figsize=(12, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["x", "y", "z"]
    for i in range(3):
        plt.plot(f, raw[:, i], label=f"elbow_global_{labels[i]} raw", alpha=0.35, color=colors[i])
        plt.plot(f, cln[:, i], label=f"elbow_global_{labels[i]} clipped", linewidth=2, color=colors[i])
        if mks[i].any():
            plt.scatter(f[mks[i]], raw[mks[i], i], color=colors[i], s=10)
    ylim = percentile_ylim(cln)
    if ylim:
        plt.ylim(*ylim)
    plt.title("Elbow global torque (xyz)")
    plt.xlabel("frame")
    plt.ylabel("torque (Nm)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    out = out_dir / f"{torque_csv.stem}_elbow_global_xyz.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_forearm_angle_kinematics(pose_csv: Path, out_dir: Path, fps: float, drop_first: int):
    df = pd.read_csv(pose_csv)
    frames = df["frame"].to_numpy()
    s = df[["joint_12_x", "joint_12_y", "joint_12_z"]].to_numpy(float)
    e = df[["joint_14_x", "joint_14_y", "joint_14_z"]].to_numpy(float)
    w = df[["joint_16_x", "joint_16_y", "joint_16_z"]].to_numpy(float)

    upper = e - s  # shoulder->elbow
    fore = w - e   # elbow->wrist
    angle_deg = angle_between(upper, fore)
    dt = 1.0 / fps
    ang_vel, ang_acc = central_diff(angle_deg, dt)

    mask = np.arange(len(frames)) >= drop_first
    f = frames[mask]
    ang = angle_deg[mask]
    vel = ang_vel[mask]
    acc = ang_acc[mask]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(f, ang, label="angle (deg)")
    ylim = percentile_ylim(ang)
    if ylim:
        axes[0].set_ylim(*ylim)
    axes[0].legend()

    axes[1].plot(f, vel, label="angular vel (deg/s)")
    ylim = percentile_ylim(vel)
    if ylim:
        axes[1].set_ylim(*ylim)
    axes[1].legend()

    axes[2].plot(f, acc, label="angular acc (deg/s^2)")
    ylim = percentile_ylim(acc)
    if ylim:
        axes[2].set_ylim(*ylim)
    axes[2].legend()

    for ax in axes:
        ax.grid(False)
    axes[2].set_xlabel("frame")
    plt.tight_layout()

    out = out_dir / f"{pose_csv.stem}_forearm_angle_kinematics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot elbow local torque and forearm angle/velocity/acceleration")
    ap.add_argument("--pose", required=True, help="Pose CSV with joint_* columns")
    ap.add_argument("--torque", required=True, help="Torque CSV with elbow_R_local_* columns")
    ap.add_argument("--fps", type=float, default=30.0, help="Pose sampling rate")
    ap.add_argument("--drop-first", type=int, default=0, help="Frames to drop at start")
    ap.add_argument("--out-dir", default="output_data/plots", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torque_local_png = plot_elbow_torque_xyz(Path(args.torque), out_dir, args.drop_first)
    torque_global_png = plot_elbow_torque_xyz_global(Path(args.torque), out_dir, args.drop_first)
    kinematics_png = plot_forearm_angle_kinematics(Path(args.pose), out_dir, args.fps, args.drop_first)
    print(f"[OUT] elbow torque xyz (local) -> {torque_local_png}")
    print(f"[OUT] elbow torque xyz (global) -> {torque_global_png}")
    print(f"[OUT] forearm angle/vel/acc -> {kinematics_png}")


if __name__ == "__main__":
    main()