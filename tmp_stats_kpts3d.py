"""Quick stats for kpts3d CSV (raw pose)."""
from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    path = Path("output_data/kpts3d_9_20250925_201442.csv")
    rows = list(csv.DictReader(path.open()))
    # collect all coordinate columns (exclude frame)
    coord_cols = [c for c in rows[0].keys() if c != "frame"]

    def stats(keys: list[str]):
        vals: list[float] = []
        for r in rows:
            for k in keys:
                try:
                    vals.append(float(r[k]))
                except ValueError:
                    pass
        vals.sort()
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        p1 = vals[int(0.01 * n)]
        p99 = vals[int(0.99 * n)]
        return min(vals), max(vals), mean, var**0.5, p1, p99

    all_vals = []
    for r in rows:
        for k in coord_cols:
            all_vals.append(float(r[k]))
    print("overall", min(all_vals), max(all_vals))

    # per-joint stats
    joints = {}
    for c in coord_cols:
        name = c.rsplit("_", 1)[0]
        joints.setdefault(name, []).append(c)

    for name, keys in sorted(joints.items()):
        mn, mx, mean, std, p1, p99 = stats(keys)
        print(name, mn, mx, mean, std, p1, p99)


if __name__ == "__main__":
    main()