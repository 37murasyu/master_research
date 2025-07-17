from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def find_valleys(z: np.ndarray):
    if len(z) < 3:
        return []
    t = np.asarray(z)
    # light smoothing for display
    if len(t) >= 11:
        k = np.ones(11)/11.0
        t = np.convolve(t, k, mode='same')
    dt = np.diff(t)
    sign = np.sign(dt)
    valleys = []
    for i in range(1, len(sign)):
        if sign[i-1] < 0 and sign[i] >= 0:
            valleys.append(i)
    return valleys, t


def main():
    ap = argparse.ArgumentParser(description='Plot filtered shoulder Z series (R/L).')
    ap.add_argument('--shoulder-z-npy', type=str, help='Right shoulder Z .npy')
    ap.add_argument('--shoulder-z-npy-left', type=str, help='Left shoulder Z .npy')
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--roi-start', type=int, default=None)
    ap.add_argument('--roi-end', type=int, default=None)
    ap.add_argument('--mark-valleys', action='store_true')
    ap.add_argument('--unit', choices=['m','cm','mm'], default='m', help='Unit of loaded arrays; will be converted to meters for display')
    ap.add_argument('--max-right', type=float, default=None, help='Remove from rising crossing for right side above this (m).')
    ap.add_argument('--max-left', type=float, default=None, help='Remove from rising crossing for left side above this (m).')
    ap.add_argument('--remove-from-rise', action='store_true', help='If set, cut series from the first rising crossing above max thresholds.')
    ap.add_argument('--rise-confirm', type=int, default=3, help='Consecutive frames above threshold to confirm crossing.')
    ap.add_argument('--smooth-peaks', action='store_true', help='Smooth only spike segments above threshold by linear interpolation to boundary points.')
    args = ap.parse_args()

    series = []
    labels = []
    smoothed = []
    scale = 1.0
    if args.unit == 'cm':
        scale = 0.01
    elif args.unit == 'mm':
        scale = 0.001
    if args.shoulder_z_npy and os.path.exists(args.shoulder_z_npy):
        zR = np.load(args.shoulder_z_npy) * scale
        series.append(zR)
        labels.append('R')
    if args.shoulder_z_npy_left and os.path.exists(args.shoulder_z_npy_left):
        zL = np.load(args.shoulder_z_npy_left) * scale
        series.append(zL)
        labels.append('L')
    if not series:
        raise SystemExit('No shoulder Z npy found.')

    # apply ROI
    s = args.roi_start if args.roi_start is not None else 0
    e = args.roi_end
    if e is not None:
        e = max(e, s+1)

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['tab:blue', 'tab:orange']
    def _cut_from_rise(x: np.ndarray, thr: float | None, k: int) -> tuple[np.ndarray, int | None]:
        if thr is None:
            return x, None
        kk = max(1, int(k))
        for i in range(1, len(x)):
            if x[i] > thr and x[i-1] <= thr:
                if i + kk - 1 < len(x) and np.all(x[i:i+kk] > thr):
                    return x[:i], i
        return x, None

    for idx, z in enumerate(series):
        zroi = z[s:e] if e is not None else z[s:]
        thr = args.max_right if labels[idx] == 'R' else args.max_left
        # optional spike smoothing (above threshold)
        if args.smooth_peaks and thr is not None:
            y = zroi.copy()
            above = y > thr
            i = 0
            n = len(y)
            while i < n:
                if above[i]:
                    j = i
                    while j < n and above[j]:
                        j += 1
                    left = i - 1
                    right = j
                    if left >= 0 and right < n:
                        v0 = y[left]; v1 = y[right]
                        seg_len = right - left - 1
                        if seg_len > 0:
                            y[left+1:right] = np.linspace(v0, v1, seg_len+2)[1:-1]
                    elif left >= 0 and right >= n:
                        y[left+1:] = y[left]
                    elif left < 0 and right < n:
                        y[:right] = y[right]
                    i = j
                else:
                    i += 1
            zroi = y
        # optional noise removal from rise
        if args.remove_from_rise:
            zroi, cut_idx = _cut_from_rise(zroi, thr, args.rise_confirm)
        else:
            cut_idx = None
        zc = zroi
        valleys, zsm = find_valleys(zc)
        ax.plot(zc, label=f'shoulderZ_{labels[idx]}', color=colors[idx % len(colors)])
        if args.mark_valleys:
            for v in valleys:
                ax.axvline(v, color=colors[idx % len(colors)], alpha=0.25, linestyle=':')
        if cut_idx is not None:
            ax.axvline(cut_idx, color=colors[idx % len(colors)], alpha=0.6, linestyle='--')
    ax.set_xlabel('frame')
    ax.set_ylabel('Z (m)')
    ax.grid(True, ls=':')
    ax.legend(loc='best')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'[OUT] plot -> {args.out}')


if __name__ == '__main__':
    main()
