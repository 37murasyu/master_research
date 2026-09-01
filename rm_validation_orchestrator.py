"""
RM compliance validation orchestrator.

This script wires together the RM適合性検証パイプライン:
- 入力: GCVSPL/カルマン/バターワーク済み3D姿勢CSV + 設定JSON
- 手順: 平滑化選択→手首y谷-峰-谷検出→逆動力学→ローカルトルク→台形則積分→理論値比較
- 出力: サイクル検出CSV・プロット、ローカルトルクCSV、仕事量比較CSV/PNG

既存の各モジュールを呼び出すための最小限の統合コードです。詳細処理は既存関数に委譲してください。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import compute_local_torque


# ===== データクラス／設定読み込み =====
@dataclass
class PipelineConfig:
    body_mass: float
    dumbbell_ratio: float  # e.g., 0.7 * elbow_R_outer
    thresholds: Dict[str, Dict[str, float]]  # {video_id: {min:..., max:...}}
    smoothing: str  # "gcv", "kalman", "butter", "none"
    joint_map: List[int]  # expected 8 ids: [shoulderL, shoulderR, elbowL, elbowR, wristL, wristR, hipL, hipR]


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    try:
        return PipelineConfig(
            body_mass=float(raw["body_mass"]),
            dumbbell_ratio=float(raw["dumbbell_ratio"]),
            thresholds=dict(raw.get("thresholds", {})),
            smoothing=str(raw.get("smoothing", "gcv")),
            joint_map=list(raw["joint_map"]),
        )
    except KeyError as exc:  # pragma: no cover - simple validation
        raise SystemExit(f"config missing key: {exc}") from exc


# ===== 入力ローダ =====
def load_pose_csv(csv_path: Path, joint_map: Sequence[int]) -> pd.DataFrame:
    """Load 3D pose CSV and rename columns per joint_map order.

    Accepts either:
      - x{jid}, y{jid}, z{jid}
      - joint_{jid}_x, joint_{jid}_y, joint_{jid}_z
    Returns a DataFrame with normalized column names: x{jid}, y{jid}, z{jid} in the order of joint_map.
    """
    df = pd.read_csv(csv_path)
    if len(joint_map) != 8:
        raise SystemExit("joint_map must have 8 entries (shoulderL/R, elbowL/R, wristL/R, hipL/R)")

    cols_out = []
    data = {}
    for jid in joint_map:
        cand_sets = [
            (f"x{jid}", f"y{jid}", f"z{jid}"),
            (f"joint_{jid}_x", f"joint_{jid}_y", f"joint_{jid}_z"),
        ]
        use = None
        for cx, cy, cz in cand_sets:
            if cx in df.columns and cy in df.columns and cz in df.columns:
                use = (cx, cy, cz)
                break
        if use is None:
            raise SystemExit(f"CSV missing columns for joint {jid} (tried x{jid}/joint_{jid}_x patterns)")
        cx, cy, cz = use
        data[f"x{jid}"] = df[cx].to_numpy()
        data[f"y{jid}"] = df[cy].to_numpy()
        data[f"z{jid}"] = df[cz].to_numpy()
        cols_out.extend([f"x{jid}", f"y{jid}", f"z{jid}"])
    return pd.DataFrame(data, columns=cols_out)


# ===== 平滑化切替（既存処理へのフック用） =====
def apply_smoothing(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = mode.lower()
    if mode in {"", "none"}:
        return df
    # ここで GCVSPL / バターワーク / カルマンを呼び分ける
    # 実装が未接続の場合はそのまま返す
    return df


# ===== サイクル検出（スタブ、外部実装で差し替え推奨） =====
def detect_cycles_wrist_y(y_series: np.ndarray, thresholds: Dict[str, float]) -> List[Tuple[int, int]]:
    """Return list of (start_frame, end_frame) cycles. Placeholder uses empty list."""
    _ = thresholds
    return []


# ===== 逆動力学→ローカルトルク =====
def compute_local_elbow_torque(
    forearm_vec: np.ndarray,
    upperarm_vec: np.ndarray,
    tau_global: np.ndarray,
) -> np.ndarray:
    """Map global elbow torque to forearm-local coords (y = upper×forearm)."""
    out = np.zeros_like(tau_global)
    for i in range(len(tau_global)):
        out[i] = compute_local_torque(tau_global[i], forearm_vec[i], parent_vec=upperarm_vec[i])
    return out


# ===== 仕事量（台形則） =====
def integrate_work(theta: np.ndarray, tau_local_y: np.ndarray) -> float:
    if len(theta) < 2 or len(tau_local_y) < 2:
        return 0.0
    dtheta = np.diff(theta)
    tau_mid = 0.5 * (tau_local_y[1:] + tau_local_y[:-1])
    return float(np.sum(tau_mid * dtheta))


# ===== CLI =====
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RM compliance validation pipeline.",
    )
    parser.add_argument("--config", type=Path, required=True, help="設定JSONパス")
    parser.add_argument("--input-csv", type=Path, nargs="+", help="GCVSPL/カルマン済み3D姿勢CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("output_data"), help="成果物出力ディレクトリ")
    parser.add_argument("--dry-run", action="store_true", help="計画のみ表示")
    return parser.parse_args(list(argv))


def plan_pipeline(input_csvs: Sequence[Path], output_dir: Path) -> list[str]:
    if not input_csvs:
        return ["No input CSVs supplied; nothing to schedule yet."]
    return [
        "Load config JSON",
        "Load pose CSVs with joint mapping",
        "Apply smoothing (GCVSPL/Butter/Kalman/none)",
        "Detect wrist-y cycles with thresholds",
        "Inverse dynamics to elbow torque",
        "Compute local torque (forearm frame) and save CSV",
        "Trapezoidal tau*dtheta per cycle and compare theoretical work",
        f"Write artifacts under {output_dir}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    steps = plan_pipeline(args.input_csv or [], args.output_dir)

    print("RM compliance validation orchestrator")
    print("Inputs:")
    for p in args.input_csv or []:
        print(f"  - {p}")
    print(f"Config: {args.config}")
    print(f"Outputs: {args.output_dir}")
    print("Planned steps:")
    for step in steps:
        print(f"  - {step}")

    if args.dry_run:
        return 0

    cfg = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 入力CSVごとに処理（複数動画をまとめて走らせる）
    for csv_path in args.input_csv:
        print(f"[PROC] {csv_path}")
        df_raw = load_pose_csv(csv_path, cfg.joint_map)
        df_smoothed = apply_smoothing(df_raw, cfg.smoothing)

        # 手首R (joint_map index 5) の y列を抽出
        wrist_r_idx = cfg.joint_map.index(5) if 5 in cfg.joint_map else 5
        col_y = f"y{cfg.joint_map[wrist_r_idx]}"
        y_series = df_smoothed[col_y].to_numpy()

        video_id = csv_path.stem
        thresholds = cfg.thresholds.get(video_id, {})
        cycles = detect_cycles_wrist_y(y_series, thresholds)

        # 逆動力学・トルク計算は既存関数に委譲する前提のスタブ
        # 下記は枠だけ: forearm_vec, upperarm_vec, tau_global は実データに差し替えてください
        forearm_vec = np.zeros((len(df_smoothed), 3))
        upperarm_vec = np.zeros((len(df_smoothed), 3))
        tau_global = np.zeros((len(df_smoothed), 3))
        tau_local = compute_local_elbow_torque(forearm_vec, upperarm_vec, tau_global)

        # 台形則で仕事量（y成分）
        theta = np.zeros(len(df_smoothed))  # 実際は前腕角度系列に置換
        work_total = integrate_work(theta, tau_local[:, 1])

        out_prefix = args.output_dir / video_id
        np.save(Path(f"{out_prefix}_tau_local.npy"), tau_local)
        with Path(f"{out_prefix}_cycles.csv").open("w", encoding="utf-8") as f:
            f.write("start,end\n")
            for s, e in cycles:
                f.write(f"{s},{e}\n")
        with Path(f"{out_prefix}_work.txt").open("w", encoding="utf-8") as f:
            f.write(f"work_total={work_total}\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
