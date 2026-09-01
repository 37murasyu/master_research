from __future__ import annotations

"""
Apply Kalman smoothing (position/velocity/acceleration) to 3D keypoint CSV.

Assumptions:
- Columns contain 3D positions grouped as name_x, name_y, name_z (or name.x, name.y, name.z).
- Timestep dt is constant (default 0.25s). Real-time systems should measure dt per loop.
- First row for each keypoint has valid numeric positions; missing leading data is not handled.

Outputs:
- Original columns are preserved.
- Added per-keypoint columns: {name}_x_filt, {name}_y_filt, {name}_z_filt, {name}_vx, {name}_vy,
  {name}_vz, {name}_ax, {name}_ay, {name}_az.

Usage example:
    pwsh scripts/apply_kalman_filter.py \
        --input data/kpts3d.csv \
        --output out/kpts3d_kalman.csv \
        --dt 0.25 \
        --process-noise 0.1 \
        --measurement-noise 0.01
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

# Allow importing from src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kalman_filter import KalmanFilterND  # noqa: E402


SuffixTriplet = Tuple[str, str, str]


def _detect_triplets(columns: Iterable[str]) -> Dict[str, SuffixTriplet]:
    """Detect base names that have x/y/z suffixes."""
    cols = list(columns)
    triplets: Dict[str, SuffixTriplet] = {}
    suffix_sets = [("_x", "_y", "_z"), (".x", ".y", ".z")]
    for base_candidate in cols:
        for sx, sy, sz in suffix_sets:
            if base_candidate.endswith(sx):
                base = base_candidate[: -len(sx)]
                cx, cy, cz = base + sx, base + sy, base + sz
                if cx in cols and cy in cols and cz in cols:
                    triplets[base] = (cx, cy, cz)
    return triplets


def _apply_filter_to_keypoint(
    df: pd.DataFrame,
    base: str,
    cols: SuffixTriplet,
    dt: float,
    process_noise: float,
    measurement_noise: float,
    init_vel: float,
    init_acc: float,
    init_pos_var: float,
    init_vel_var: float,
    init_acc_var: float,
) -> pd.DataFrame:
    cx, cy, cz = cols
    series = df[[cx, cy, cz]].to_numpy(dtype=float)

    first_row = series[0]
    if not np.all(np.isfinite(first_row)):
        raise ValueError(f"First row for columns {cols} must be finite; got {first_row}")

    kf = KalmanFilterND(
        process_noise_intensity=process_noise,
        measurement_noise_variance=measurement_noise,
        initial_position=first_row,
        initial_velocity=init_vel,
        initial_acceleration=init_acc,
        initial_position_variance=init_pos_var,
        initial_velocity_variance=init_vel_var,
        initial_acceleration_variance=init_acc_var,
    )

    positions = []
    velocities = []
    accelerations = []

    for row in series:
        kf.predict(dt)
        measurement = row if np.all(np.isfinite(row)) else None
        kf.update(measurement)
        p, v, a = kf.state
        positions.append(p)
        velocities.append(v)
        accelerations.append(a)

    positions = np.vstack(positions)
    velocities = np.vstack(velocities)
    accelerations = np.vstack(accelerations)

    return pd.DataFrame(
        {
            f"{base}_x_filt": positions[:, 0],
            f"{base}_y_filt": positions[:, 1],
            f"{base}_z_filt": positions[:, 2],
            f"{base}_vx": velocities[:, 0],
            f"{base}_vy": velocities[:, 1],
            f"{base}_vz": velocities[:, 2],
            f"{base}_ax": accelerations[:, 0],
            f"{base}_ay": accelerations[:, 1],
            f"{base}_az": accelerations[:, 2],
        }
    )


def run(args: argparse.Namespace) -> Path:
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    triplets = _detect_triplets(df.columns)
    if not triplets:
        raise ValueError("No keypoint column triplets were detected (expected name_x/name_y/name_z or name.x/name.y/name.z)")

    outputs = []
    for base, cols in triplets.items():
        filtered = _apply_filter_to_keypoint(
            df,
            base,
            cols,
            dt=args.dt,
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            init_vel=args.initial_velocity,
            init_acc=args.initial_acceleration,
            init_pos_var=args.initial_position_variance,
            init_vel_var=args.initial_velocity_variance,
            init_acc_var=args.initial_acceleration_variance,
        )
        outputs.append(filtered)

    merged = pd.concat([df] + outputs, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Kalman filter to 3D keypoint CSV")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--dt", type=float, default=0.25, help="Time step in seconds (default: 0.25)")
    parser.add_argument("--process-noise", type=float, default=0.1, help="Process noise intensity q")
    parser.add_argument("--measurement-noise", type=float, default=0.01, help="Measurement noise variance R")
    parser.add_argument("--initial-velocity", type=float, default=0.0, help="Initial velocity for all axes")
    parser.add_argument("--initial-acceleration", type=float, default=0.0, help="Initial acceleration for all axes")
    parser.add_argument("--initial-position-variance", type=float, default=1.0, help="Initial position variance")
    parser.add_argument("--initial-velocity-variance", type=float, default=1.0, help="Initial velocity variance")
    parser.add_argument("--initial-acceleration-variance", type=float, default=1.0, help="Initial acceleration variance")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
