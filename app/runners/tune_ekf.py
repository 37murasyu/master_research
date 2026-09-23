"""収録（EKF の手前で書いた生 CSV）から、EKF の較正プロファイルを作る。

::

    python -m app --role script --module app.runners.tune_ekf output_data/kpts3d_raw_0913_120000.csv

GUI の解析タスク（S11）もこのコマンドを呼ぶ。

出力は既定で収録の隣に ``ekf_profile_{dt}.json``。ファイル名に dt を入れるのは、
実行時の探索が **dt でファイルを選ぶ**ため（決定 6）。間引き設定ごとに録った
プロファイルを同じフォルダに並べておける。``--out`` にフォルダを渡すと、その中に同じ名前で書く。

混成（Mac＋Pixel）の収録（サイドカーの ``source`` が ``hybrid``）は、既定で ``hybrid/ekf_profiles/`` に書き、
設定 ``HYBRID_EKF_PROFILE`` に入れる値を案内する。混成の計測の格子は 1/30 s で、実行時の探索は dt の相対差
5% 以内しか選ばないので、それを外れた収録で作ったときは警告する。

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の S7。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from app.hybrid.paths import ekf_profile_root
from app.tuning.ekf_estimate import fit_capture, format_report
from app.tuning.ekf_profile import MIN_N_EFF, build_profile, write_profile
from app.tuning.raw_capture import read_raw_capture


# 混成の計測の格子の dt [s] と、実行時の探索が同じ dt とみなす相対差（ekf_profile.resolve_profile）
HYBRID_GRID_DT = 1.0 / 30.0
DT_TOLERANCE = 0.05


def profile_name(dt: float) -> str:
    return f"ekf_profile_{dt:.5f}.json"


def default_destination(csv_path: str | Path, dt: float) -> Path:
    return Path(csv_path).parent / profile_name(dt)


def resolve_destination(out: str | None, csv_path: str | Path, dt: float, *, hybrid: bool = False) -> Path:
    """書き出し先。``--out`` が既存のフォルダか拡張子の無いパスなら、その中に ``ekf_profile_{dt}.json``。"""
    if out:
        target = Path(out)
        if target.is_dir() or (not target.exists() and target.suffix == ""):
            return target / profile_name(dt)
        return target
    if hybrid:
        return ekf_profile_root() / profile_name(dt)
    return default_destination(csv_path, dt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生 CSV から EKF の較正プロファイルを作る")
    parser.add_argument("csv", help="計測時に書き出した kpts3d_raw_*.csv（同じ名前のサイドカー JSON が要る）")
    parser.add_argument("--out", help="書き出し先（既定: 収録の隣の ekf_profile_{dt}.json）")
    parser.add_argument(
        "--min-n-eff",
        type=int,
        default=MIN_N_EFF,
        help=f"推定を採用する最少の有効サンプル数（既定 {MIN_N_EFF}）",
    )
    args = parser.parse_args(argv)

    try:
        capture = read_raw_capture(args.csv)
    except ValueError as error:
        # EKF を通った値で較正しても、実行時の EKF に入れる値としては意味をなさない
        print(f"[tune_ekf] 較正に使えない入力です（stage が pre_ekf の生 CSV が要ります）: {error}", file=sys.stderr)
        return 2

    dt = float(capture.provenance["dt"])
    fits = fit_capture(capture)
    print(format_report(fits, dt))

    profile = build_profile(capture, fits, min_n_eff=args.min_n_eff)
    hybrid = capture.provenance.get("source") == "hybrid"
    destination = resolve_destination(args.out, args.csv, dt, hybrid=hybrid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_profile(destination, profile)

    sources = Counter(entry["source"] for per_axis in profile["series"].values() for entry in per_axis.values())
    print(f"[tune_ekf] 書き出し: {destination}")
    print("[tune_ekf] 採用元の内訳: " + ", ".join(f"{name}={count}" for name, count in sorted(sources.items())))
    if hybrid:
        if abs(dt - HYBRID_GRID_DT) / HYBRID_GRID_DT > DT_TOLERANCE:
            print(f"[tune_ekf][警告] この収録の dt {dt:.5f} s は混成の計測の格子 1/30 s（{HYBRID_GRID_DT:.5f} s）から "
                  f"{DT_TOLERANCE:.0%} を超えて外れている。混成の計測はこのプロファイルを選ばない（計測フォルダの "
                  "kpts3d_raw_<stamp>.csv から作り直す）")
        print(f"[tune_ekf] 混成の計測で使うには、設定 HYBRID_EKF_PROFILE（GUI のカルマンフィルタの欄）に "
              f"{destination}（またはフォルダ {destination.parent}）を入れる")
    return 0


if __name__ == "__main__":
    sys.exit(main())
