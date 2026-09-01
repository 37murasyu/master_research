import re
import sys
import math
from pathlib import Path

pat = re.compile(r"\[LOOP\]\s*#\d+\s*dt=([0-9.]+)s\s*fps=([0-9.]+)")


def stats(xs):
    n = len(xs)
    if n == 0:
        return None
    mean = sum(xs) / n
    # population variance
    var_pop = sum((x - mean) ** 2 for x in xs) / n
    # sample variance (unbiased)
    var_samp = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else float('nan')
    return {
        'n': n,
        'mean': mean,
        'var_pop': var_pop,
        'var_samp': var_samp,
        'min': min(xs),
        'max': max(xs),
    }


def parse_file(path: Path):
    fps_values = []
    dt_values = []
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = pat.search(line)
                if m:
                    dt = float(m.group(1))
                    fps = float(m.group(2))
                    dt_values.append(dt)
                    fps_values.append(fps)
    except FileNotFoundError:
        print(f"[WARN] file not found: {path}")
        return None
    s_fps = stats(fps_values)
    s_dt = stats(dt_values)
    return fps_values, dt_values, s_fps, s_dt


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/parse_fps_stats.py <logfile> [<logfile2> ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        path = Path(p)
        res = parse_file(path)
        print(f"=== {path} ===")
        if res is None:
            continue
        fps_values, dt_values, s_fps, s_dt = res
        if not s_fps:
            print("No [LOOP] fps lines found.")
            continue
        print(f"LOOP lines: {s_fps['n']}")
        print(f"fps mean={s_fps['mean']:.4f}, var_pop={s_fps['var_pop']:.6f}, var_samp={s_fps['var_samp']:.6f}, min={s_fps['min']:.4f}, max={s_fps['max']:.4f}")
        if s_dt:
            print(f"dt  mean={s_dt['mean']:.4f}s, var_pop={s_dt['var_pop']:.6f}, var_samp={s_dt['var_samp']:.6f}, min={s_dt['min']:.4f}s, max={s_dt['max']:.4f}s")

if __name__ == '__main__':
    main()
