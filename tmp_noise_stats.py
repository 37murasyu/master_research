"""Compute noise-related statistics (PSD, HF ratio, work error, uncertainty) for pose/torque.

Outputs:
- output_data/noise_stats_summary.csv
- output_data/noise_stats_summary.json
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Iterable, Sequence

import json
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, find_peaks, welch, coherence
from scipy.integrate import trapezoid


FS = 30.0

POSE_FILES: Sequence[str] = (
    "Adjusted 3D Pose/2_stereo_pose.csv",
    "Adjusted 3D Pose/3_0stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/4_0stereo_pose_scaled_with2d.csv",
    "c:/Users/villa/Desktop/master_Research/cameras_raw/5_20250925_133228/5_1stereo_pose_scaled.csv",
    "Adjusted 3D Pose/6_stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/7_stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/8_stereo_pose_scaled_with2d.csv",
    "Adjusted 3D Pose/kpts3d_9_20250925_201442.csv",
)

TORQUE_FILES: Sequence[str] = (
    "torque/2_stereo_pose_torque.csv",
    "torque/3_0stereo_pose_scaled_with2d_torque.csv",
    "torque/4_0stereo_pose_scaled_with2d_torque.csv",
    "torque/5_1stereo_pose_scaled_torque.csv",
    "torque/6_stereo_pose_scaled_with2d_torque.csv",
    "torque/7_stereo_pose_scaled_with2d_torque.csv",
    "torque/8_stereo_pose_scaled_with2d_torque.csv",
    "torque/kpts3d_9_20250925_201442_torque.csv",
)

POSE_FILES_LPF: Sequence[str] = (
    "output_data/filtered_pose_lpf/2_stereo_pose_lpf.csv",
    "output_data/filtered_pose_lpf/3_0stereo_pose_scaled_with2d_lpf.csv",
    "output_data/filtered_pose_lpf/4_0stereo_pose_scaled_with2d_lpf.csv",
    "output_data/filtered_pose_lpf/5_1stereo_pose_scaled_lpf.csv",
    "output_data/filtered_pose_lpf/6_stereo_pose_scaled_with2d_lpf.csv",
    "output_data/filtered_pose_lpf/7_stereo_pose_scaled_with2d_lpf.csv",
    "output_data/filtered_pose_lpf/8_stereo_pose_scaled_with2d_lpf.csv",
    "output_data/filtered_pose_lpf/kpts3d_9_20250925_201442_lpf.csv",
)

TORQUE_FILES_LPF: Sequence[str] = (
    "output_data/filtered_torque_lpf/2_stereo_pose_torque_lpf.csv",
    "output_data/filtered_torque_lpf/3_0stereo_pose_scaled_with2d_torque_lpf.csv",
    "output_data/filtered_torque_lpf/4_0stereo_pose_scaled_with2d_torque_lpf.csv",
    "output_data/filtered_torque_lpf/5_1stereo_pose_scaled_torque_lpf.csv",
    "output_data/filtered_torque_lpf/6_stereo_pose_scaled_with2d_torque_lpf.csv",
    "output_data/filtered_torque_lpf/7_stereo_pose_scaled_with2d_torque_lpf.csv",
    "output_data/filtered_torque_lpf/8_stereo_pose_scaled_with2d_torque_lpf.csv",
    "output_data/filtered_torque_lpf/kpts3d_9_20250925_201442_torque_lpf.csv",
)

SUBJECT_IDS: Sequence[str] = ("2", "3", "4", "5", "6", "7", "8", "9")

POSE_TRIPLE_CANDIDATES: Sequence[Sequence[str]] = (
    ("joint_16_x", "joint_16_y", "joint_16_z"),
    ("wrist_R_x", "wrist_R_y", "wrist_R_z"),
    ("joint_0_x", "joint_0_y", "joint_0_z"),
)

ELBOW_TRIPLE_CANDIDATES: Sequence[tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]] = (
    (("joint_12_x", "joint_12_y", "joint_12_z"), ("joint_14_x", "joint_14_y", "joint_14_z"), ("joint_16_x", "joint_16_y", "joint_16_z")),
    (("shoulder_R_x", "shoulder_R_y", "shoulder_R_z"), ("elbow_R_x", "elbow_R_y", "elbow_R_z"), ("wrist_R_x", "wrist_R_y", "wrist_R_z")),
)

TORQUE_COLUMN_CANDIDATES: Sequence[str] = (
    "elbow_R_local_y",
    "elbow_R_y",
    "wrist_R_local_y",
    "wrist_R_y",
)


@dataclass
class WelchConfig:
    nperseg: int
    noverlap: int
    window: str = "hann"


@dataclass
class SubjectResult:
    subject: str
    pose_file: str
    torque_file: str
    fs: float
    df: float
    nperseg: int
    noverlap: int
    f0_pose: float
    f0_torque: float
    fc_pose: float
    fc_torque: float
    fc_common: float
    rho_hf_pose: float
    rho_hf_torque: float
    ratio_a: float
    snr_pose: float
    snr_torque: float
    harmonics_pose: str
    harmonics_torque: str
    W_raw: float
    W_lp: float
    delta_W: float
    epsilon: float
    Wp_raw: float
    Wp_lp: float
    delta_Wp: float
    epsilon_p: float
    epsilon_p_median: float
    epsilon_p_iqr: float
    epsilon_p_cv: float
    epsilon_p_ci_low: float
    epsilon_p_ci_high: float
    epsilon_median: float
    epsilon_iqr: float
    epsilon_cv: float
    epsilon_ci_low: float
    epsilon_ci_high: float
    coherence_hf_mean: float


def _select_triplet(df: pd.DataFrame, candidates: Iterable[Sequence[str]]) -> np.ndarray:
    for triplet in candidates:
        if all(c in df.columns for c in triplet):
            return df[list(triplet)].to_numpy(float)
    raise KeyError("no xyz triplet found")


def _select_elbow_triplet(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for s_cols, e_cols, w_cols in ELBOW_TRIPLE_CANDIDATES:
        if all(c in df.columns for c in (*s_cols, *e_cols, *w_cols)):
            p_s = df[list(s_cols)].to_numpy(float)
            p_e = df[list(e_cols)].to_numpy(float)
            p_w = df[list(w_cols)].to_numpy(float)
            return p_s, p_e, p_w
    raise KeyError("no elbow triplet found")


def _select_column(df: pd.DataFrame, candidates: Iterable[str]) -> np.ndarray:
    for col in candidates:
        if col in df.columns:
            return df[col].to_numpy(float)
    raise KeyError("no torque column found")


def _check_files(paths: Sequence[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing files:\n" + "\n".join(missing))


def _welch_config(n: int) -> WelchConfig:
    nperseg = min(256, max(64, n // 4))
    noverlap = nperseg // 2
    return WelchConfig(nperseg=nperseg, noverlap=noverlap)


def _psd(sig: np.ndarray, fs: float, cfg: WelchConfig) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.nanmean(sig)
    sig = np.nan_to_num(sig, nan=np.nanmean(sig))
    freq, psd = welch(sig, fs=fs, window=cfg.window, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    return freq, psd


def _cumulative_power(freq: np.ndarray, psd: np.ndarray) -> np.ndarray:
    df = freq[1] - freq[0] if len(freq) > 1 else 0.0
    return np.cumsum(psd) * df


def _find_fc(freq: np.ndarray, psd: np.ndarray, target: float = 0.95) -> float:
    if len(freq) < 2:
        return 0.0
    cum = _cumulative_power(freq, psd)
    total = float(cum[-1]) if cum.size else 0.0
    if total <= 0:
        return float(freq[-1])
    idx = int(np.searchsorted(cum / total, target))
    idx = min(max(idx, 1), len(freq) - 1)
    return float(freq[idx])


def _rho_hf(freq: np.ndarray, psd: np.ndarray, fc: float) -> float:
    df = freq[1] - freq[0] if len(freq) > 1 else 0.0
    total = float(np.sum(psd) * df)
    if total <= 0:
        return 0.0
    mask = freq >= fc
    hf = float(np.sum(psd[mask]) * df)
    return hf / total


def _f0(freq: np.ndarray, psd: np.ndarray, fmin: float = 0.3) -> float:
    if len(freq) <= 1:
        return 0.0
    mask = freq >= fmin
    if not np.any(mask):
        idx = int(np.argmax(psd[1:])) + 1
    else:
        idx_rel = int(np.argmax(psd[mask]))
        idx = np.where(mask)[0][idx_rel]
    return float(freq[idx])


def _harmonics(f0: float, f_n: float) -> str:
    hs = []
    for k in (2, 3):
        fk = f0 * k
        if fk <= f_n:
            hs.append(f"{k}f0={fk:.4f}")
    return ";".join(hs)


def _lowpass(sig: np.ndarray, fs: float, fc: float) -> np.ndarray:
    fc_use = max(0.1, min(fc, fs / 2.0 - 1e-6))
    b, a = butter(4, fc_use / (fs / 2.0), btype="low")
    return filtfilt(b, a, sig)


def _elbow_angle(p_s: np.ndarray, p_e: np.ndarray, p_w: np.ndarray) -> np.ndarray:
    v1 = p_s - p_e
    v2 = p_w - p_e
    n1 = np.linalg.norm(v1, axis=1) + 1e-12
    n2 = np.linalg.norm(v2, axis=1) + 1e-12
    cos = np.sum(v1 * v2, axis=1) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def _work(tau: np.ndarray, theta: np.ndarray, fs: float) -> float:
    dt = 1.0 / fs
    tau = np.nan_to_num(tau, nan=np.nanmean(tau))
    theta = np.nan_to_num(theta, nan=np.nanmean(theta))
    dtheta = np.gradient(theta, dt)
    power = tau * dtheta
    return float(trapezoid(power, dx=dt))


def _work_positive(tau: np.ndarray, theta: np.ndarray, fs: float) -> float:
    dt = 1.0 / fs
    tau = np.nan_to_num(tau, nan=np.nanmean(tau))
    theta = np.nan_to_num(theta, nan=np.nanmean(theta))
    dtheta = np.gradient(theta, dt)
    power = tau * dtheta
    power = np.maximum(power, 0.0)
    return float(trapezoid(power, dx=dt))


def _segment_epsilons(tau: np.ndarray, theta: np.ndarray, fs: float, fc: float) -> list[float]:
    tau_lp = _lowpass(tau, fs, fc)
    theta_lp = _lowpass(theta, fs, fc)

    min_dist = int(fs * 0.5)
    peaks, _ = find_peaks(theta_lp, distance=min_dist)
    eps = []
    if len(peaks) >= 3:
        for i in range(len(peaks) - 1):
            a, b = peaks[i], peaks[i + 1]
            if b - a < 5:
                continue
            w_raw = _work(tau[a:b], theta[a:b], fs)
            w_lp = _work(tau_lp[a:b], theta_lp[a:b], fs)
            if abs(w_lp) < 1e-9:
                continue
            eps.append((w_raw - w_lp) / w_lp)
    else:
        win = int(fs * 2.0)
        for a in range(0, len(tau) - win + 1, win):
            b = a + win
            w_raw = _work(tau[a:b], theta[a:b], fs)
            w_lp = _work(tau_lp[a:b], theta_lp[a:b], fs)
            if abs(w_lp) < 1e-9:
                continue
            eps.append((w_raw - w_lp) / w_lp)
    return eps


def _segment_epsilons_positive(tau: np.ndarray, theta: np.ndarray, fs: float, fc: float) -> list[float]:
    tau_lp = _lowpass(tau, fs, fc)
    theta_lp = _lowpass(theta, fs, fc)

    min_dist = int(fs * 0.5)
    peaks, _ = find_peaks(theta_lp, distance=min_dist)
    eps = []
    if len(peaks) >= 3:
        for i in range(len(peaks) - 1):
            a, b = peaks[i], peaks[i + 1]
            if b - a < 5:
                continue
            w_raw = _work_positive(tau[a:b], theta[a:b], fs)
            w_lp = _work_positive(tau_lp[a:b], theta_lp[a:b], fs)
            if w_lp <= 1e-9:
                continue
            eps.append((w_raw - w_lp) / w_lp)
    else:
        win = int(fs * 2.0)
        for a in range(0, len(tau) - win + 1, win):
            b = a + win
            w_raw = _work_positive(tau[a:b], theta[a:b], fs)
            w_lp = _work_positive(tau_lp[a:b], theta_lp[a:b], fs)
            if w_lp <= 1e-9:
                continue
            eps.append((w_raw - w_lp) / w_lp)
    return eps


def _snr_harmonics(freq: np.ndarray, psd: np.ndarray, f0: float, fmin: float = 0.3, bw: float = 0.15) -> float:
    if len(freq) <= 1 or f0 <= 0:
        return float("nan")
    sig_mask = np.zeros_like(freq, dtype=bool)
    for k in (1, 2, 3):
        fk = f0 * k
        if fk < fmin or fk > freq[-1]:
            continue
        sig_mask |= (freq >= fk - bw) & (freq <= fk + bw)
    if not np.any(sig_mask):
        return float("nan")
    df = freq[1] - freq[0]
    p_sig = float(np.sum(psd[sig_mask]) * df)
    p_rest = float(np.sum(psd[~sig_mask]) * df)
    if p_rest <= 0:
        return float("nan")
    return p_sig / p_rest


def _bootstrap_ci(values: Sequence[float], n_boot: int = 1000) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    vals = np.array(values, dtype=np.float64)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot.append(np.mean(sample))
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def _coherence_hf(pose_sig: np.ndarray, torque_sig: np.ndarray, fs: float, cfg: WelchConfig, fc: float) -> float:
    f, cxy = coherence(pose_sig, torque_sig, fs=fs, window=cfg.window, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    mask = f >= fc
    if not np.any(mask):
        return float(np.nan)
    return float(np.nanmean(cxy[mask]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute noise stats from pose/torque CSVs")
    ap.add_argument("--use-lpf", action="store_true", help="Use LPF outputs in output_data/filtered_*_lpf")
    args = ap.parse_args()

    pose_files = POSE_FILES_LPF if args.use_lpf else POSE_FILES
    torque_files = TORQUE_FILES_LPF if args.use_lpf else TORQUE_FILES

    _check_files(pose_files)
    _check_files(torque_files)

    results: list[SubjectResult] = []

    for subj, pose_path, torque_path in zip(SUBJECT_IDS, pose_files, torque_files):
        pose_df = pd.read_csv(pose_path).interpolate(limit_direction="both")
        torque_df = pd.read_csv(torque_path).interpolate(limit_direction="both")

        xyz = _select_triplet(pose_df, POSE_TRIPLE_CANDIDATES)
        pose_mag = np.linalg.norm(xyz, axis=1)

        p_s, p_e, p_w = _select_elbow_triplet(pose_df)
        theta = _elbow_angle(p_s, p_e, p_w)

        torque_sig = _select_column(torque_df, TORQUE_COLUMN_CANDIDATES)

        n_min = min(len(pose_mag), len(theta), len(torque_sig))
        pose_mag = pose_mag[:n_min]
        theta = theta[:n_min]
        torque_sig = torque_sig[:n_min]
        pose_mag = np.nan_to_num(pose_mag, nan=np.nanmean(pose_mag))
        theta = np.nan_to_num(theta, nan=np.nanmean(theta))
        torque_sig = np.nan_to_num(torque_sig, nan=np.nanmean(torque_sig))

        cfg = _welch_config(n_min)

        f_pose, psd_pose = _psd(pose_mag, FS, cfg)
        f_torque, psd_torque = _psd(torque_sig, FS, cfg)

        df_val = float(f_pose[1] - f_pose[0]) if len(f_pose) > 1 else 0.0
        f_n = FS / 2.0

        f0_pose = _f0(f_pose, psd_pose, fmin=0.3)
        f0_torque = _f0(f_torque, psd_torque, fmin=0.3)
        fc_pose = _find_fc(f_pose, psd_pose, target=0.95)
        fc_torque = _find_fc(f_torque, psd_torque, target=0.95)
        fc_common = float(min(6.0 * f0_pose if f0_pose > 0 else fc_pose, f_n))
        rho_pose = _rho_hf(f_pose, psd_pose, fc_common)
        rho_torque = _rho_hf(f_torque, psd_torque, fc_common)
        ratio_a = float(rho_torque / rho_pose) if rho_pose > 0 else float("nan")

        snr_pose = _snr_harmonics(f_pose, psd_pose, f0_pose, fmin=0.3)
        snr_torque = _snr_harmonics(f_torque, psd_torque, f0_pose, fmin=0.3)

        harmonics_pose = _harmonics(f0_pose, f_n)
        harmonics_torque = _harmonics(f0_torque, f_n)

        w_raw = _work(torque_sig, theta, FS)
        tau_lp = _lowpass(torque_sig, FS, fc_common)
        theta_lp = _lowpass(theta, FS, fc_common)
        w_lp = _work(tau_lp, theta_lp, FS)
        delta_w = w_raw - w_lp
        eps = delta_w / w_lp if abs(w_lp) > 1e-9 else float("nan")

        eps_list = _segment_epsilons(torque_sig, theta, FS, fc_common)
        eps_median = float(np.median(eps_list)) if eps_list else float("nan")
        eps_iqr = float(np.percentile(eps_list, 75) - np.percentile(eps_list, 25)) if eps_list else float("nan")
        eps_cv = float(np.std(eps_list, ddof=1) / np.mean(eps_list)) if len(eps_list) > 1 and abs(np.mean(eps_list)) > 1e-9 else float("nan")
        ci_low, ci_high = _bootstrap_ci(eps_list, n_boot=1000)

        w_raw_p = _work_positive(torque_sig, theta, FS)
        w_lp_p = _work_positive(tau_lp, theta_lp, FS)
        delta_w_p = w_raw_p - w_lp_p
        eps_p = delta_w_p / w_lp_p if w_lp_p > 1e-9 else float("nan")

        eps_list_p = _segment_epsilons_positive(torque_sig, theta, FS, fc_common)
        eps_p_median = float(np.median(eps_list_p)) if eps_list_p else float("nan")
        eps_p_iqr = float(np.percentile(eps_list_p, 75) - np.percentile(eps_list_p, 25)) if eps_list_p else float("nan")
        eps_p_cv = float(np.std(eps_list_p, ddof=1) / np.mean(eps_list_p)) if len(eps_list_p) > 1 and abs(np.mean(eps_list_p)) > 1e-9 else float("nan")
        ci_low_p, ci_high_p = _bootstrap_ci(eps_list_p, n_boot=1000)

        coh_hf = _coherence_hf(pose_mag, torque_sig, FS, cfg, fc_common)

        results.append(
            SubjectResult(
                subject=subj,
                pose_file=pose_path,
                torque_file=torque_path,
                fs=FS,
                df=df_val,
                nperseg=cfg.nperseg,
                noverlap=cfg.noverlap,
                f0_pose=f0_pose,
                f0_torque=f0_torque,
                fc_pose=fc_pose,
                fc_torque=fc_torque,
                fc_common=fc_common,
                rho_hf_pose=rho_pose,
                rho_hf_torque=rho_torque,
                ratio_a=ratio_a,
                snr_pose=snr_pose,
                snr_torque=snr_torque,
                harmonics_pose=harmonics_pose,
                harmonics_torque=harmonics_torque,
                W_raw=w_raw,
                W_lp=w_lp,
                delta_W=delta_w,
                epsilon=eps,
                Wp_raw=w_raw_p,
                Wp_lp=w_lp_p,
                delta_Wp=delta_w_p,
                epsilon_p=eps_p,
                epsilon_p_median=eps_p_median,
                epsilon_p_iqr=eps_p_iqr,
                epsilon_p_cv=eps_p_cv,
                epsilon_p_ci_low=ci_low_p,
                epsilon_p_ci_high=ci_high_p,
                epsilon_median=eps_median,
                epsilon_iqr=eps_iqr,
                epsilon_cv=eps_cv,
                epsilon_ci_low=ci_low,
                epsilon_ci_high=ci_high,
                coherence_hf_mean=coh_hf,
            )
        )

    out_dir = Path("output_data")
    out_dir.mkdir(exist_ok=True)
    suffix = "_lpf" if args.use_lpf else ""
    out_csv = out_dir / f"noise_stats_summary{suffix}.csv"
    out_json = out_dir / f"noise_stats_summary{suffix}.json"

    df_out = pd.DataFrame([r.__dict__ for r in results])
    df_out.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(df_out.to_dict(orient="records"), ensure_ascii=False, indent=2))

    print(f"[OUT] {out_csv}")
    print(f"[OUT] {out_json}")


if __name__ == "__main__":
    main()
