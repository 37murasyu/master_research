"""Compute link length stats from kpts3d CSV."""
from __future__ import annotations

import csv
import math
from pathlib import Path


def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except ValueError:
        return None


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = max(0, min(len(sorted_vals) - 1, int(q * len(sorted_vals))))
    return sorted_vals[idx]


def main() -> None:
    path = Path("output_data/kpts3d_9_20250925_201442.csv")
    rows = list(csv.DictReader(path.open()))

    links = {
        "shoulder_width": ("joint_11", "joint_12"),
        "torso_L": ("joint_11", "joint_23"),
        "torso_R": ("joint_12", "joint_24"),
        "upper_arm_L": ("joint_11", "joint_13"),
        "upper_arm_R": ("joint_12", "joint_14"),
        "forearm_L": ("joint_13", "joint_15"),
        "forearm_R": ("joint_14", "joint_16"),
        "hip_width": ("joint_23", "joint_24"),
        "thigh_L": ("joint_23", "joint_25"),
        "thigh_R": ("joint_24", "joint_26"),
        "shank_L": ("joint_25", "joint_27"),
        "shank_R": ("joint_26", "joint_28"),
    }

    link_vals: dict[str, list[float]] = {k: [] for k in links}

    for r in rows:
        for name, (a, b) in links.items():
            ax, ay, az = safe_float(r[f"{a}_x"]), safe_float(r[f"{a}_y"]), safe_float(r[f"{a}_z"])
            bx, by, bz = safe_float(r[f"{b}_x"]), safe_float(r[f"{b}_y"]), safe_float(r[f"{b}_z"])
            if None in (ax, ay, az, bx, by, bz):
                continue
            d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            link_vals[name].append(d)

    for name, vals in link_vals.items():
        if not vals:
            print(name, "no data")
            continue
        vals.sort()
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        print(
            name,
            "count", n,
            "min", vals[0],
            "max", vals[-1],
            "mean", mean,
            "std", var ** 0.5,
            "p01", percentile(vals, 0.01),
            "p99", percentile(vals, 0.99),
        )


if __name__ == "__main__":
    main()