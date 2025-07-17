from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="*_f 列のみを抽出して新しいCSVを出力します（v,a などの他列を除外）")
    ap.add_argument("--input", required=True, help="入力CSVパス")
    ap.add_argument("--output", required=False, help="出力CSVパス（未指定なら *_only_f.csv を自動生成）")
    ap.add_argument("--keep", nargs="*", default=None, help="必ず残す列名（例: frame time）")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output

    if not os.path.isfile(in_path):
        print(f"[ERR] input CSV not found: {in_path}")
        sys.exit(1)

    df = pd.read_csv(in_path)
    cols: List[str] = list(df.columns)

    # 残す列: 末尾が "_f" の列 + keep 指定列
    f_cols = [c for c in cols if c.endswith("_f")]

    keep_set = set(args.keep) if args.keep else set()
    extra_keep = [c for c in cols if c in keep_set]

    out_cols = extra_keep + f_cols
    if not out_cols:
        print("[WARN] 抽出対象の列が見つかりません（*_f や keep 指定が存在しません）")
        # それでも空のCSVは出力する

    out_df = df[out_cols]

    if not out_path:
        base, ext = os.path.splitext(in_path)
        out_path = base + "_only_f.csv"

    out_df.to_csv(out_path, index=False)
    print(f"[OUT] saved -> {out_path}  (cols={len(out_cols)})")


if __name__ == "__main__":
    main()
