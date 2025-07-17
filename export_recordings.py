"""Batch export 3D keypoints & torque CSV from recorded dual-camera mp4 pairs.

It searches for cam0_output_*.mp4 & cam1_output_*.mp4 pairs sharing the same timestamp suffix
and invokes master_research_code.py in file-processing mode (assumes script auto-detects videos
when USE_SAMPLE_VIDEOS or camera open fails is configured).

Strategy:
 1. Detect pairs: (cam0_output_<stamp>.mp4, cam1_output_<stamp>.mp4)
 2. For each pair run: TIMESTAMP_OVERRIDE=<MMDD_HHMMSS> python master_research_code.py --no-live
    (You may need an argument/flag to force file mode; if not present, ensure env forces sample usage.)
 3. Output CSV appear in output_data/ with names kpts3d_<stamp>.csv and aim_torque_vec_<stamp>.csv

This script currently just lists pairs (dry run) unless --run is specified.
"""
from __future__ import annotations
import os, re, subprocess, sys, argparse, shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CAM0_PATTERN = re.compile(r"cam0_output_(\d{4}_\d{6})\.mp4$")
CAM1_NAME_FMT = "cam1_output_{stamp}.mp4"


def find_pairs(directory: Path):
    cam0_files = []
    for f in directory.glob('cam0_output_*.mp4'):
        m = CAM0_PATTERN.search(f.name)
        if not m:
            continue
        stamp = m.group(1)
        cam1 = directory / CAM1_NAME_FMT.format(stamp=stamp)
        if cam1.is_file():
            cam0_files.append((stamp, f, cam1))
    cam0_files.sort(key=lambda x: x[0])
    return cam0_files


def build_command(stamp: str, extra_args: list[str]) -> list[str]:
    env_ts = stamp  # already in MMDD_HHMMSS format
    # We rely on TIMESTAMP_OVERRIDE to control output file naming
    cmd = [sys.executable, str(ROOT / 'master_research_code.py')]
    cmd.extend(extra_args)
    return cmd, {**os.environ, 'TIMESTAMP_OVERRIDE': env_ts, 'USE_SAMPLE_VIDEOS': '0'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.', help='Directory containing cam*_output_*.mp4')
    ap.add_argument('--run', action='store_true', help='Actually run processing (default: dry list)')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of pairs (0=all)')
    ap.add_argument('--extra', nargs='*', default=[], help='Extra args passed to master_research_code.py')
    args = ap.parse_args()

    target_dir = Path(args.dir).resolve()
    pairs = find_pairs(target_dir)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    if not pairs:
        print('[INFO] No recording pairs found (cam0 & cam1).')
        return

    print(f"[INFO] Found {len(pairs)} pairs:")
    for idx, (stamp, c0, c1) in enumerate(pairs, 1):
        print(f"  {idx:02d}: stamp={stamp} cam0={c0.name} cam1={c1.name}")

    if not args.run:
        print('\n[DRY] Use --run to process. Example:')
        if pairs:
            ex_stamp = pairs[0][0]
            print(f"python export_recordings.py --run --limit 1 --extra --headless --dir {shlex.quote(str(target_dir))}")
        return

    # Execute sequentially
    for idx, (stamp, c0, c1) in enumerate(pairs, 1):
        print(f"\n[RUN {idx}/{len(pairs)}] Processing stamp={stamp}")
        cmd, env = build_command(stamp, args.extra)
        print('[CMD]', ' '.join(shlex.quote(c) for c in cmd), f"TIMESTAMP_OVERRIDE={stamp}")
        # Indicate which videos should be used (set env so code selects them if logic checks latest pair)
        env['USE_SAMPLE_VIDEOS'] = '1'  # if logic uses latest output pair when enabled
        # Optionally tell code to prefer recording pairs
        env['PREFER_RECORDING_PAIRS'] = '1'
        # HEADLESS for speed
        env['HEADLESS'] = '1'
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, check=False)
            print('[RET]', proc.returncode)
            if proc.stdout:
                print('[STDOUT]\n' + proc.stdout[-4000:])  # tail
            if proc.stderr:
                print('[STDERR]\n' + proc.stderr[-4000:])
            if proc.returncode != 0:
                print('[WARN] Non-zero exit, continuing to next.')
        except KeyboardInterrupt:
            print('[INTERRUPT] Aborting batch.')
            break
        except Exception as e:
            print(f'[ERROR] Failed to run pair {stamp}: {e}')

    print('\n[DONE] Batch export complete.')

if __name__ == '__main__':
    main()
