from __future__ import annotations

import argparse
import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    dot = float(np.dot(a, b)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    crossn = np.linalg.norm(np.cross(a/na, b/nb))
    import math
    return math.atan2(crossn, dot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forearm-npy', required=True)
    ap.add_argument('--tau-npy', required=True)
    ap.add_argument('--clip-dtheta', type=float, default=0.35)
    args = ap.parse_args()

    vecs = np.load(args.forearm_npy)
    tau = np.load(args.tau_npy)
    N = min(len(vecs), len(tau))
    vecs = vecs[:N]; tau = tau[:N]
    # sanitize
    tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
    u = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    # theta_ref and theta_diff
    ref = u[0]
    theta_ref = np.array([angle_between(ref, ui) for ui in u])
    clip = args.clip_dtheta
    th = np.zeros(N)
    for i in range(1, N):
        d = angle_between(u[i-1], u[i])
        d = max(-clip, min(clip, d))
        th[i] = th[i-1] + d
    # stats
    def rng(x):
        return float(np.nanmax(x) - np.nanmin(x))
    print(f'N={N}')
    print('tau stats: min/median/max/95pct/99pct/ptp', float(np.min(tau)), float(np.median(tau)), float(np.max(tau)), float(np.percentile(tau,95)), float(np.percentile(tau,99)), float(np.ptp(tau)))
    print('abs(tau) stats: median/95pct/99pct/max', float(np.median(np.abs(tau))), float(np.percentile(np.abs(tau),95)), float(np.percentile(np.abs(tau),99)), float(np.max(np.abs(tau))))
    print('theta_ref range (rad):', rng(theta_ref))
    print('theta_diff range (rad):', rng(th))


if __name__ == '__main__':
    main()
