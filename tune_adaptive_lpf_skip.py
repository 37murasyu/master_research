import argparse
import csv
import glob
import itertools
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd


LOOP_DT_RE = re.compile(r"\[LOOP\]\s*#\d+\s*dt=([0-9]*\.?[0-9]+)s")


def _timestamp_with_offset(offset_sec: int) -> str:
    t = datetime.now() + timedelta(seconds=offset_sec)
    return t.strftime("%m%d_%H%M%S")


def _find_output(path_pattern: str) -> str | None:
    cand = sorted(glob.glob(path_pattern), reverse=True)
    return cand[0] if cand else None


def _parse_loop_dt(log_text: str) -> tuple[float | None, float | None, int]:
    vals = [float(m.group(1)) for m in LOOP_DT_RE.finditer(log_text)]
    if not vals:
        return None, None, 0
    mean_v = statistics.mean(vals)
    std_v = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
    return mean_v, std_v, len(vals)


def _evaluate_cycle_csv(path: str | None) -> dict:
    out = {
        "cycles_total": 0,
        "cycles_elbow_R": 0,
        "cycles_elbow_L": 0,
        "e_pos_mean": None,
        "e_pos_std": None,
        "e_pos_cv": None,
    }
    if path is None or (not os.path.exists(path)):
        return out

    df = pd.read_csv(path)
    if df.empty or "part" not in df.columns or "e_pos" not in df.columns:
        return out

    elbow = df[df["part"].isin(["elbow_R", "elbow_L"])].copy()
    out["cycles_total"] = int(len(elbow))
    out["cycles_elbow_R"] = int((elbow["part"] == "elbow_R").sum())
    out["cycles_elbow_L"] = int((elbow["part"] == "elbow_L").sum())
    if len(elbow) >= 1:
        m = float(elbow["e_pos"].mean())
        s = float(elbow["e_pos"].std(ddof=0)) if len(elbow) >= 2 else 0.0
        out["e_pos_mean"] = m
        out["e_pos_std"] = s
        out["e_pos_cv"] = (s / max(abs(m), 1e-9)) if m is not None else None
    return out


def _score(row: dict) -> float:
    cv = row.get("e_pos_cv")
    dt = row.get("loop_dt_mean")
    cycles = row.get("cycles_total", 0) or 0

    if cv is None or dt is None or cycles <= 0:
        return 1e9

    cycle_penalty = 2.0 / max(cycles, 1)
    return float(cv) + 0.2 * float(dt) + cycle_penalty


def run_one(master_script: str, env_common: dict, cfg: dict, run_idx: int, out_dir: str, timeout_sec: int) -> dict:
    ts = _timestamp_with_offset(run_idx)
    env = os.environ.copy()
    env.update(env_common)
    env.update({
        "TIMESTAMP_OVERRIDE": ts,
        "E_FC_ADAPTIVE_ON": "1",
        "E_FC_MIN": str(cfg["E_FC_MIN"]),
        "E_FC_MAX": str(cfg["E_FC_MAX"]),
        "E_FC_K": str(cfg["E_FC_K"]),
        "E_FC_EMA_BETA": str(cfg["E_FC_EMA_BETA"]),
        "E_F0_WIN_SEC": str(cfg["E_F0_WIN_SEC"]),
    })

    returncode = 0
    try:
        proc = subprocess.run(
            [sys.executable, master_script],
            cwd=os.path.dirname(master_script),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        returncode = int(proc.returncode)
        full_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        returncode = 124
        out = e.stdout.decode("utf-8", errors="ignore") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
        err = e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")
        full_log = (out or "") + "\n" + (err or "") + f"\n[TUNE] timeout after {timeout_sec}s\n"

    log_path = os.path.join(out_dir, f"tune_run_{ts}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(full_log)

    loop_mean, loop_std, loop_n = _parse_loop_dt(full_log)
    cyc_path = _find_output(os.path.join(out_dir, f"cycle_energy_debug_{ts}*.csv"))
    cyc_eval = _evaluate_cycle_csv(cyc_path)

    row = {
        "timestamp": ts,
        "returncode": returncode,
        "log_path": log_path,
        "cycle_csv": cyc_path,
        "loop_dt_mean": loop_mean,
        "loop_dt_std": loop_std,
        "loop_dt_n": loop_n,
        **cfg,
        **cyc_eval,
    }
    row["score"] = _score(row)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Adaptive LPF tuning with real-time delay skip model")
    p.add_argument("--cam0", required=True)
    p.add_argument("--cam1", required=True)
    p.add_argument("--calib-dir", required=True, help="Directory containing c0.dat/c1.dat/rot_trans_c0.dat/rot_trans_c1.dat")
    p.add_argument("--master-script", default="master_research_code.py")
    p.add_argument("--max-frames", type=int, default=180)
    p.add_argument("--timeout-sec", type=int, default=1200)
    p.add_argument("--max-cases", type=int, default=0, help="0:全ケース実行 / n:先頭nケースのみ")
    p.add_argument("--out-csv", default="output_data/lpf_tuning_skip_results.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    grid = list(itertools.product(
        [1.2, 1.8, 2.4],
        [4.0, 5.5],
        [3.0, 4.5],
        [0.10, 0.20],
        [3.0],
    ))

    configs = [
        {
            "E_FC_MIN": a,
            "E_FC_MAX": b,
            "E_FC_K": c,
            "E_FC_EMA_BETA": d,
            "E_F0_WIN_SEC": e,
        }
        for a, b, c, d, e in grid
        if b > a
    ]
    if args.max_cases and args.max_cases > 0:
        configs = configs[: args.max_cases]

    env_common = {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "SUBJECT_ID": "5",
        "CAM0": args.cam0,
        "CAM1": args.cam1,
        "CALIB_BASE_DIR": args.calib_dir,
        "USE_SAMPLE_VIDEOS": "0",
        "AUTO_FALLBACK_TO_FILES": "0",
        "HEADLESS": "1",
        "DISABLE_IMSHOW": "1",
        "DISABLE_MPL": "1",
        "DISABLE_WRITE": "1",
        "LOOP_TRACE": "0",
        "PERF_LOG": "0",
        "POSE_DEBUG": "0",
        "E_DEBUG": "0",
        "USE_NATIVE_POSE_MODE": "off",
        "MP_INPUT_SCALE": "0.35",
        "POSE_ROI_ON": "1",
        "TRIANG_NATIVE_ON": "1",
        "E_LPF_NATIVE_ON": "1",
        "USE_NATIVE_DYNAMICS": "1",
        "HX711_ENABLE": "0",
        "RT_DELAY_SKIP_ON": "1",
        "RT_DELAY_SKIP_GAIN": "1.0",
        "RT_DELAY_SKIP_MIN": "0",
        "RT_DELAY_SKIP_MAX": "300",
        "MAX_FRAMES": str(args.max_frames),
        "GRAVITY_FROM_CHECKERBOARD_SHORT": "0",
        "GRAVITY_AUTO_DETECT": "1",
    }

    rows = []
    for i, cfg in enumerate(configs):
        print(f"[RUN {i+1}/{len(configs)}] {cfg}")
        row = run_one(
            master_script=os.path.abspath(args.master_script),
            env_common=env_common,
            cfg=cfg,
            run_idx=i,
            out_dir=os.path.dirname(args.out_csv),
            timeout_sec=args.timeout_sec,
        )
        print(
            f"  -> rc={row['returncode']} score={row['score']:.6f} "
            f"cycles={row['cycles_total']} dt_mean={row['loop_dt_mean']}"
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["score", "returncode"], ascending=[True, True])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    best = df.iloc[0].to_dict() if not df.empty else None
    if best:
        print("\n=== BEST ===")
        print(best)
        best_path = os.path.join(os.path.dirname(args.out_csv), "lpf_tuning_skip_best.json")
        import json
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        print(f"saved: {args.out_csv}")
        print(f"saved: {best_path}")


if __name__ == "__main__":
    main()
