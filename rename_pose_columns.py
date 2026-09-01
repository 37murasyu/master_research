from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

# kpt_0..11 -> mediapipe ids: 16,14,12,11,13,15,24,23,25,26,27,28
MAP_IDS = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]


def rename_csv(src: Path, dst: Path) -> None:
    df = pd.read_csv(src, comment="#")
    if "frame" not in df.columns:
        raise SystemExit("frame column is required")
    rename = {}
    for idx, mp_id in enumerate(MAP_IDS):
        for ax in ("x", "y", "z"):
            old = f"kpt_{idx}_{ax}"
            if old not in df.columns:
                raise SystemExit(f"missing column: {old}")
            rename[old] = f"joint_{mp_id}_{ax}"
    df = df.rename(columns=rename)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"[RENAMED] {src.name} -> {dst.name}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args(argv)
    rename_csv(args.src, args.dst)


if __name__ == "__main__":
    main()
