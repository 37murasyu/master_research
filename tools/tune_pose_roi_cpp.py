import argparse
import itertools
import os
import pathlib
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class RunResult:
    margin: float
    min_side: float
    native_mode: str
    ok: bool
    loop_total: float
    triang: float
    mp0: float
    mp1: float
    run_specs: float
    returncode: int
    blocks: int
    log_path: str


def parse_perf_blocks(text: str):
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('[PERF] avg over'):
            d = {}
            i += 1
            while i < len(lines) and lines[i].startswith('  '):
                m = re.match(r'\s*([^:]+):\s*([0-9.]+)', lines[i])
                if m:
                    d[m.group(1).strip()] = float(m.group(2))
                i += 1
            blocks.append(d)
            continue
        i += 1
    if not blocks:
        return None, 0
    keys = set().union(*[set(b.keys()) for b in blocks])
    means = {k: statistics.mean([b.get(k, 0.0) for b in blocks]) for k in keys}
    return means, len(blocks)


def run_one(python_exe: str, script_path: str, margin: float, min_side: float, native_mode: str, max_frames: int, out_dir: pathlib.Path):
    env = os.environ.copy()
    env.update({
        'PYTHONIOENCODING': 'utf-8',
        'SUBJECT_ID': env.get('SUBJECT_ID', '5'),
        'USE_SAMPLE_VIDEOS': env.get('USE_SAMPLE_VIDEOS', '1'),
        'AUTO_FALLBACK_TO_FILES': env.get('AUTO_FALLBACK_TO_FILES', '1'),
        'MAX_FRAMES': str(max_frames),
        'PERF_LOG': '1',
        'PERF_INT': '30',
        'PERF_TRACE': '0',
        'DISABLE_IMSHOW': env.get('DISABLE_IMSHOW', '1'),
        'DISABLE_WRITE': env.get('DISABLE_WRITE', '0'),
        'POSE_ROI_ON': '1',
        'POSE_ROI_MARGIN_RATIO': f'{margin:.4f}',
        'POSE_ROI_MIN_SIDE_RATIO': f'{min_side:.4f}',
        'USE_NATIVE_POSE_MODE': native_mode,
    })

    proc = subprocess.run(
        [python_exe, script_path],
        cwd=str(pathlib.Path(script_path).resolve().parent),
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
    )
    text = (proc.stdout or '') + '\n' + (proc.stderr or '')

    tag = f'm{margin:.3f}_s{min_side:.3f}_n{native_mode}'
    log_path = out_dir / f'tune_{tag}.log'
    log_path.write_text(text, encoding='utf-8')

    means, blocks = parse_perf_blocks(text)
    if means is None:
        return RunResult(margin, min_side, native_mode, False, 0.0, 0.0, 0.0, 0.0, 0.0, proc.returncode, 0, str(log_path))

    return RunResult(
        margin=margin,
        min_side=min_side,
        native_mode=native_mode,
        ok=True,
        loop_total=float(means.get('loop_total', 0.0)),
        triang=float(means.get('triang+transform', 0.0)),
        mp0=float(means.get('mediapipe0', 0.0)),
        mp1=float(means.get('mediapipe1', 0.0)),
        run_specs=float(means.get('run_specs', 0.0)),
        returncode=proc.returncode,
        blocks=blocks,
        log_path=str(log_path),
    )


def main():
    parser = argparse.ArgumentParser(description='Tune POSE_ROI parameters and native pose mode by PERF logs.')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--script', default='master_research_code.py')
    parser.add_argument('--margins', default='0.14,0.18,0.22')
    parser.add_argument('--minsides', default='0.25,0.33,0.40')
    parser.add_argument('--native-modes', default='off,auto')
    parser.add_argument('--max-frames', type=int, default=60)
    parser.add_argument('--out-csv', default='output_data/pose_roi_tuning_results.csv')
    args = parser.parse_args()

    margins = [float(v.strip()) for v in args.margins.split(',') if v.strip()]
    minsides = [float(v.strip()) for v in args.minsides.split(',') if v.strip()]
    native_modes = [v.strip().lower() for v in args.native_modes.split(',') if v.strip()]

    out_csv = pathlib.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_csv.parent / 'pose_roi_tuning_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(margins) * len(minsides) * len(native_modes)
    idx = 0
    for margin, min_side, native_mode in itertools.product(margins, minsides, native_modes):
        idx += 1
        print(f'[TUNE] ({idx}/{total}) margin={margin:.3f} min_side={min_side:.3f} native={native_mode}')
        rr = run_one(args.python, args.script, margin, min_side, native_mode, args.max_frames, log_dir)
        results.append(rr)
        if rr.ok:
            print(f'       loop={rr.loop_total:.2f}ms tri={rr.triang:.2f} mp={rr.mp0 + rr.mp1:.2f} blocks={rr.blocks} rc={rr.returncode}')
        else:
            print(f'       failed blocks=0 rc={rr.returncode} log={rr.log_path}')

    with out_csv.open('w', encoding='utf-8', newline='') as f:
        f.write('margin,min_side,native_mode,ok,loop_total_ms,triang_ms,mediapipe0_ms,mediapipe1_ms,mediapipe_sum_ms,run_specs_ms,returncode,blocks,log_path\n')
        for r in results:
            f.write(
                f'{r.margin:.4f},{r.min_side:.4f},{r.native_mode},{int(r.ok)},{r.loop_total:.3f},{r.triang:.3f},'
                f'{r.mp0:.3f},{r.mp1:.3f},{(r.mp0+r.mp1):.3f},{r.run_specs:.3f},{r.returncode},{r.blocks},{r.log_path}\n'
            )

    valid = [r for r in results if r.ok and r.loop_total > 0]
    if not valid:
        print('[TUNE] no valid result.')
        return 1

    best = sorted(valid, key=lambda r: (r.loop_total, r.triang + r.mp0 + r.mp1))[0]
    print('\n[TUNE] BEST (by loop_total)')
    print(f'  margin={best.margin:.4f} min_side={best.min_side:.4f} native={best.native_mode}')
    print(f'  loop_total={best.loop_total:.2f}ms  triang={best.triang:.2f}ms  mediapipe_sum={(best.mp0 + best.mp1):.2f}ms')
    print(f'  csv={out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
