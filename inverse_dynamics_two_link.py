from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Mass fraction defaults (upper/forearm) from config.py
DEFAULT_UPPER_FRAC = 0.0227
DEFAULT_FOREARM_FRAC = 0.0160
DEFAULT_FOREARM_COM_FRAC = 0.430  # from elbow toward wrist
DEFAULT_UPPER_COM_FRAC = 0.436    # shoulder toward elbow


def butter_bandpass(data: np.ndarray, fs: float, low: float, high: float, order: int = 2) -> np.ndarray:
    try:
        from scipy.signal import butter, filtfilt  # type: ignore
    except Exception:
        # fallback: simple moving average (acts as low-pass only)
        if data.ndim == 1:
            kernel = np.ones(5) / 5
            return np.convolve(data, kernel, mode="same")
        out = np.empty_like(data)
        for i in range(data.shape[1]):
            out[:, i] = np.convolve(data[:, i], np.ones(5) / 5, mode="same")
        return out
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")
    if data.ndim == 1:
        return filtfilt(b, a, data)
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        out[:, i] = filtfilt(b, a, data[:, i])
    return out


def signed_angle(u: np.ndarray, v: np.ndarray, normal: np.ndarray) -> float:
    # signed angle from u to v in plane with normal
    u_n = u / (np.linalg.norm(u) + 1e-12)
    v_n = v / (np.linalg.norm(v) + 1e-12)
    dot = np.clip(np.dot(u_n, v_n), -1.0, 1.0)
    cross = np.cross(u_n, v_n)
    sign = np.sign(np.dot(cross, normal)) or 1.0
    return float(sign * np.arccos(dot))


def compute_angles(p_w: np.ndarray, p_e: np.ndarray, p_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Forearm vector (wrist->elbow), upper arm (elbow->shoulder)
    v1 = p_e - p_w
    v2 = p_s - p_e
    # Plane normal per frame
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n_unit = n / n_norm
    # q1: absolute forearm orientation in y-z plane (atan2(z, y))
    q1 = np.arctan2(v1[:, 2], v1[:, 1])
    # q2: signed angle between forearm and upper arm in their plane
    q2 = np.array([signed_angle(v1[i], v2[i], n_unit[i]) for i in range(len(v1))])
    return q1, q2


def finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x, dt)


def two_link_torques(q1, q2, dq1, dq2, ddq1, ddq2, l1, l2, m1, m2, payload_mass, l1c, l2c, g=9.81):
    # Standard planar RR dynamics with payload at end of link2
    I1 = m1 * (l1 ** 2) / 12.0
    I2 = (m2 + payload_mass) * (l2 ** 2) / 12.0
    m2e = m2 + payload_mass
    # Inertia terms
    h = m2e * l1 * l2c * np.cos(q2)
    M11 = I1 + I2 + m1 * (l1c ** 2) + m2e * (l1 ** 2 + l2c ** 2 + 2 * h)
    M12 = I2 + m2e * (l2c ** 2 + h)
    M22 = I2 + m2e * (l2c ** 2)
    # Coriolis/centrifugal
    C1 = -m2e * l1 * l2c * (2 * dq1 * dq2 + dq2 ** 2) * np.sin(q2)
    C2 = m2e * l1 * l2c * (dq1 ** 2) * np.sin(q2)
    # Gravity (assuming gravity along -y, angles measured from +y)
    G1 = (m1 * g * l1c + m2e * g * l1) * np.sin(q1) + m2e * g * l2c * np.sin(q1 + q2)
    G2 = m2e * g * l2c * np.sin(q1 + q2)
    tau1 = M11 * ddq1 + M12 * ddq2 + C1 + G1
    tau2 = M12 * ddq1 + M22 * ddq2 + C2 + G2
    return tau1, tau2


def process(args: argparse.Namespace):
    df = pd.read_csv(args.pose_csv)
    for c in ("joint_12_x","joint_12_y","joint_12_z","joint_14_x","joint_14_y","joint_14_z","joint_16_x","joint_16_y","joint_16_z"):
        if c not in df.columns:
            raise SystemExit(f"missing column {c} in pose csv")
    p12 = df[["joint_12_x","joint_12_y","joint_12_z"]].to_numpy(float)
    p14 = df[["joint_14_x","joint_14_y","joint_14_z"]].to_numpy(float)
    p16 = df[["joint_16_x","joint_16_y","joint_16_z"]].to_numpy(float)

    # downsample
    stride = max(1, int(args.stride))
    p12 = p12[::stride]
    p14 = p14[::stride]
    p16 = p16[::stride]
    frames = (df["frame"].to_numpy(int) if "frame" in df.columns else np.arange(len(df)))[::stride]

    fs = args.fps / stride
    if args.bpf_low is not None and args.bpf_high is not None:
        p12 = butter_bandpass(p12, fs, args.bpf_low, args.bpf_high)
        p14 = butter_bandpass(p14, fs, args.bpf_low, args.bpf_high)
        p16 = butter_bandpass(p16, fs, args.bpf_low, args.bpf_high)

    # angles
    q1, q2 = compute_angles(p16, p14, p12)
    dt = 1.0 / fs
    dq1 = finite_diff(q1, dt)
    dq2 = finite_diff(q2, dt)
    ddq1 = finite_diff(dq1, dt)
    ddq2 = finite_diff(dq2, dt)

    # lengths & masses
    l1_arr = np.linalg.norm(p14 - p16, axis=1)
    l2_arr = np.linalg.norm(p12 - p14, axis=1)
    l1 = float(np.nanmedian(l1_arr))
    l2 = float(np.nanmedian(l2_arr))
    m1 = args.body_mass * (args.forearm_frac if args.forearm_frac is not None else DEFAULT_FOREARM_FRAC)
    m2 = args.body_mass * (args.upper_frac if args.upper_frac is not None else DEFAULT_UPPER_FRAC)
    payload = args.payload_mass if args.payload_mass is not None else 0.0
    l1c = l1 * (args.forearm_com_frac if args.forearm_com_frac is not None else DEFAULT_FOREARM_COM_FRAC)
    l2c = l2 * (args.upper_com_frac if args.upper_com_frac is not None else DEFAULT_UPPER_COM_FRAC)

    tau1, tau2 = two_link_torques(q1, q2, dq1, dq2, ddq1, ddq2, l1, l2, m1, m2, payload, l1c, l2c, g=args.gravity)

    out = pd.DataFrame({
        "frame": frames,
        "q1": q1,
        "q2": q2,
        "dq1": dq1,
        "dq2": dq2,
        "ddq1": ddq1,
        "ddq2": ddq2,
        "tau_wrist": tau1,
        "tau_elbow": tau2,
    })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OUT] {out_path} (rows={len(out)})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Downsample + BPF + simple 2-link inverse dynamics (wrist-elbow-shoulder)")
    ap.add_argument("--pose-csv", required=True, help="3D pose CSV (joint_12/14/16) in meters or scaled")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--fps", type=float, default=30.0, help="source fps before downsample")
    ap.add_argument("--stride", type=int, default=2, help="downsample stride (1=none)")
    ap.add_argument("--bpf-low", type=float, default=0.5, help="bandpass low-cut [Hz]")
    ap.add_argument("--bpf-high", type=float, default=6.0, help="bandpass high-cut [Hz]")
    ap.add_argument("--body-mass", type=float, required=True, help="body mass [kg]")
    ap.add_argument("--payload-mass", type=float, default=0.0, help="payload mass at shoulder (torso load) [kg]")
    ap.add_argument("--upper-frac", type=float, default=None, help="override upper-arm mass fraction (default 0.0227)")
    ap.add_argument("--forearm-frac", type=float, default=None, help="override forearm mass fraction (default 0.0160)")
    ap.add_argument("--upper-com-frac", type=float, default=None, help="COM fraction upper-arm (default 0.436)")
    ap.add_argument("--forearm-com-frac", type=float, default=None, help="COM fraction forearm (default 0.430)")
    ap.add_argument("--gravity", type=float, default=9.81, help="gravity magnitude [m/s^2], along -y")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    process(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
