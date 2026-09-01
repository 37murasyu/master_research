from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from detect_cycles_joint11y_threshold import find_cycles, plot_cycles, _unit_scale


@dataclass
class Cycle:
    start_idx: int
    peak_idx: int
    end_idx: int


def _pick_shoulder_col(df: pd.DataFrame) -> Tuple[int, str]:
    if "joint_12_y" in df.columns:
        return 12, "joint_12_y"
    if "joint_12_y_f" in df.columns:
        return 12, "joint_12_y_f"
    if "joint_0_y" in df.columns:
        return 0, "joint_0_y"
    if "joint_0_y_f" in df.columns:
        return 0, "joint_0_y_f"
    raise ValueError("right shoulder y column not found (joint_12_y or joint_0_y)")


def _auto_thresholds(y: np.ndarray, frac: float) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 0.0
    p10 = float(np.percentile(y, 10))
    p50 = float(np.percentile(y, 50))
    p90 = float(np.percentile(y, 90))
    amp = max(p90 - p10, 1e-6)
    peak_min = p50 + frac * amp
    valley_max = p50 - frac * amp
    return peak_min, valley_max


def _tune_thresholds(y: np.ndarray, target_count: int | None) -> Tuple[float, float, int, float]:
    fracs = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4]
    best = None
    for frac in fracs:
        peak_min, valley_max = _auto_thresholds(y, frac)
        cycles = find_cycles(y, peak_min, valley_max, order="peak-valley-peak")
        count = len(cycles)
        if target_count is None:
            return peak_min, valley_max, count, frac
        diff = abs(count - target_count)
        if best is None or diff < best[0]:
            best = (diff, peak_min, valley_max, count, frac)
        if diff == 0:
            break
    if best is None:
        return 0.0, 0.0, 0, 0.2
    _, peak_min, valley_max, count, frac = best
    return peak_min, valley_max, count, frac


def _local_maxima(y: np.ndarray) -> np.ndarray:
    idx = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            idx.append(i)
    return np.array(idx, dtype=int)


def _select_peaks_by_min_dist(candidates: np.ndarray, y: np.ndarray, min_dist: int) -> np.ndarray:
    if candidates.size == 0:
        return candidates
    # greedy by height
    order = candidates[np.argsort(y[candidates])[::-1]]
    picked: List[int] = []
    for idx in order:
        if all(abs(idx - p) >= min_dist for p in picked):
            picked.append(int(idx))
    picked.sort()
    return np.array(picked, dtype=int)


def _detect_cycles_target(y: np.ndarray, target_cycles: int) -> List[Cycle]:
    if target_cycles <= 0:
        return []
    candidates_all = _local_maxima(y)
    if candidates_all.size == 0:
        return []

    peak_target = target_cycles + 1
    best = None
    n = len(y)
    for pct in (50, 60, 70, 80):
        thr = float(np.percentile(y, pct))
        candidates = candidates_all[y[candidates_all] >= thr]
        if candidates.size == 0:
            continue
        min_dist_options = [max(1, int(n / max(peak_target * k, 1))) for k in (1.5, 2, 3, 4)]
        for md in min_dist_options:
            peaks = _select_peaks_by_min_dist(candidates, y, md)
            diff = abs(len(peaks) - peak_target)
            if best is None or diff < best[0]:
                best = (diff, peaks)
            if diff == 0:
                break
        if best is not None and best[0] == 0:
            break

    if best is None:
        return []

    peaks = best[1]
    if len(peaks) < 2:
        return []

    cycles: List[Cycle] = []
    for p0, p1 in zip(peaks[:-1], peaks[1:]):
        if p1 <= p0 + 1:
            continue
        valley_idx = int(p0 + np.argmin(y[p0:p1 + 1]))
        cycles.append(Cycle(start_idx=int(p0), peak_idx=valley_idx, end_idx=int(p1)))
    if len(cycles) > target_cycles:
        scored = []
        for c in cycles:
            amp = float(y[c.start_idx] - y[c.peak_idx])
            scored.append((amp, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        cycles = [c for _amp, c in scored[:target_cycles]]
        cycles.sort(key=lambda c: c.start_idx)
    return cycles


def _unit_auto(y_raw: np.ndarray) -> str:
    med = np.nanmedian(np.abs(y_raw))
    if med > 5:
        unit = "cm"
        if med > 50:
            unit = "mm"
    else:
        unit = "m"
    return unit


def _detect_in_range(
    frames: np.ndarray,
    y_m: np.ndarray,
    start_frame: int,
    end_frame: int | None,
    target_count: int | None,
    fixed_thresh: Tuple[float, float] | None,
    outlier_clip: bool,
) -> Tuple[List[Cycle], np.ndarray, int, float, float]:
    if end_frame is None:
        mask = frames >= start_frame
    else:
        mask = (frames >= start_frame) & (frames <= end_frame)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return [], idx

    y_sel = y_m[idx]
    # robust clip to reduce outliers
    if outlier_clip and np.isfinite(y_sel).any():
        lo, hi = np.percentile(y_sel[np.isfinite(y_sel)], [1, 99])
        y_sel = np.clip(y_sel, lo, hi)

    if fixed_thresh is not None:
        peak_min, valley_max = fixed_thresh
        cycles_rel = find_cycles(y_sel, peak_min, valley_max, order="peak-valley-peak")
        count, frac = len(cycles_rel), 0.0
    elif target_count is not None:
        cycles_rel_abs = _detect_cycles_target(y_sel, target_count)
        cycles_rel = [
            type("C", (), {
                "start_valley_idx": c.start_idx,
                "peak_idx": c.peak_idx,
                "end_valley_idx": c.end_idx,
            })()
            for c in cycles_rel_abs
        ]
        peak_min, valley_max, count, frac = (0.0, 0.0, len(cycles_rel), 0.0)
    else:
        peak_min, valley_max, count, frac = _tune_thresholds(y_sel, target_count)
        cycles_rel = find_cycles(y_sel, peak_min, valley_max, order="peak-valley-peak")
    cycles_abs: List[Cycle] = []
    for c in cycles_rel:
        cycles_abs.append(
            Cycle(
                start_idx=int(idx[c.start_valley_idx]),
                peak_idx=int(idx[c.peak_idx]),
                end_idx=int(idx[c.end_valley_idx]),
            )
        )
    return cycles_abs, idx, count, frac, peak_min


def _annotate_cycles(df: pd.DataFrame, cycles: List[Cycle], out_csv: Path) -> None:
    T = len(df)
    cycle_index = np.full(T, -1, dtype=int)
    for ci, c in enumerate(cycles, start=1):
        s, e = c.start_idx, c.end_idx
        if 0 <= s < T and 0 <= e < T and s < e:
            cycle_index[s : e + 1] = ci
    df_out = df.copy()
    df_out["cycle_index"] = cycle_index
    df_out.to_csv(out_csv, index=False)
    print(f"[OUT] saved -> {out_csv}")


def process_file(
    path: Path,
    start_frame: int,
    end_frame: int | None,
    target_count: int | None,
    fixed_thresh_raw: Tuple[float, float] | None,
    outlier_clip: bool,
) -> None:
    df = pd.read_csv(path)
    jid, col = _pick_shoulder_col(df)
    frames = df["frame"].to_numpy(int) if "frame" in df.columns else np.arange(len(df))
    y_raw = df[col].to_numpy(float)

    unit = _unit_auto(y_raw)
    scale = _unit_scale(unit)
    y_m = y_raw * scale

    fixed_thresh = None
    if fixed_thresh_raw is not None:
        peak_min_raw, valley_max_raw = fixed_thresh_raw
        fixed_thresh = (peak_min_raw * scale, valley_max_raw * scale)
    cycles, idx, count, frac, peak_min = _detect_in_range(
        frames,
        y_m,
        start_frame,
        end_frame,
        target_count,
        fixed_thresh,
        outlier_clip,
    )
    tgt = "" if target_count is None else f" target={target_count}"
    print(
        f"[INFO] {path.name} range={start_frame}~{'' if end_frame is None else end_frame} cycles={len(cycles)}{tgt} tuned={count} frac={frac}"
    )

    base, ext = os.path.splitext(str(path))
    out_csv = Path(base + "_with_cycles" + ext)
    out_png = Path(base + "_cycles.png")

    # plot only selected range
    if idx.size > 0:
        inv = 1.0 / max(scale, 1e-12)
        y_disp = y_m[idx] * inv
        frames_disp = frames[idx]
        # convert abs cycles to relative indices for plotting
        rel_cycles = []
        idx_map = {int(i): k for k, i in enumerate(idx.tolist())}
        for c in cycles:
            if c.start_idx in idx_map and c.peak_idx in idx_map and c.end_idx in idx_map:
                rel_cycles.append(
                    type("C", (), {
                        "start_valley_idx": idx_map[c.start_idx],
                        "peak_idx": idx_map[c.peak_idx],
                        "end_valley_idx": idx_map[c.end_idx],
                    })()
                )
        plot_cycles(
            frames_disp,
            y_disp,
            rel_cycles,
            str(out_png),
            thr_high=None,
            thr_low=None,
            y_unit_label=unit,
            joint_id=jid,
        )

    _annotate_cycles(df, cycles, out_csv)


def main() -> int:
    base = Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE\output_data\filtered_pose_lpf")
    ranges: Dict[str, Tuple[int, int | None]] = {
        "2_stereo_pose_lpf.csv": (400, 1000),
        "3_0stereo_pose_scaled_with2d_lpf.csv": (250, 1500),
        "5_1stereo_pose_scaled_lpf.csv": (100, 1050),
        "6_stereo_pose_scaled_with2d_lpf.csv": (100, 1200),
        "7_stereo_pose_scaled_with2d_lpf.csv": (250, 1625),
        "8_stereo_pose_scaled_with2d_lpf.csv": (400, None),
        "kpts3d_9_20250925_201442_lpf.csv": (500, None),
    }
    target_counts: Dict[str, int] = {
        "3_0stereo_pose_scaled_with2d_lpf.csv": 10,
        "6_stereo_pose_scaled_with2d_lpf.csv": 14,
        "kpts3d_9_20250925_201442_lpf.csv": 8,
    }
    fixed_thresholds_raw: Dict[str, Tuple[float, float]] = {
        "8_stereo_pose_scaled_with2d_lpf.csv": (-0.4, -0.6),
        "5_1stereo_pose_scaled_lpf.csv": (-0.225, -0.31),
    }
    outlier_clip_files = {
        "3_0stereo_pose_scaled_with2d_lpf.csv",
    }

    for name, (start, end) in ranges.items():
        path = base / name
        if not path.exists():
            print(f"[SKIP] missing: {path}")
            continue
        target = target_counts.get(name)
        fixed_thresh_raw = fixed_thresholds_raw.get(name)
        outlier_clip = name in outlier_clip_files
        process_file(path, start, end, target, fixed_thresh_raw, outlier_clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
