import argparse
import os
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Scale torque CSV columns by factor")
    ap.add_argument("--csv", required=True, help="input torque csv")
    ap.add_argument("--out", required=True, help="output csv")
    ap.add_argument("--factor", type=float, required=True, help="scale factor (e.g., 0.01)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    torque_cols = [c for c in df.columns if "torque" not in c and (c.endswith("_x") or c.endswith("_y") or c.endswith("_z"))]
    # heuristic: assume global/local torque columns (endswith _x/_y/_z) after frame
    for c in torque_cols:
        if c != "frame":
            df[c] = df[c] * args.factor
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OUT] saved -> {args.out} (factor={args.factor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
