import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description="Plot joint_3 x/y/z time series for raw, gcvspl, kalman (3 Hz)")
    ap.add_argument("--raw", default="filtered_3hz_raw.csv", help="Raw 3 Hz CSV (rtfilt columns)")
    ap.add_argument("--gcvspl", default="filtered_3hz_gcvspl.csv", help="gcvspl 3 Hz CSV")
    ap.add_argument("--kalman", default="kalman_3hz_raw.csv", help="Kalman 3 Hz CSV")
    ap.add_argument("--out", default="joint3_3hz_timeseries.png", help="Output PNG path")
    args = ap.parse_args()

    df_raw = load_df(args.raw)
    df_gc = load_df(args.gcvspl)
    df_kf = load_df(args.kalman)

    # Time columns
    t_raw = df_raw["time_s"]
    t_gc = df_gc["time_s"]
    t_kf = df_kf["time_s"]

    # Column sets
    raw_cols = ["joint_3_x_rtfilt", "joint_3_y_rtfilt", "joint_3_z_rtfilt"]
    gc_cols = ["joint_3_x_rtfilt", "joint_3_y_rtfilt", "joint_3_z_rtfilt"]
    kf_cols = ["joint_3_x_kalman", "joint_3_y_kalman", "joint_3_z_kalman"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes_map = {0: "x", 1: "y", 2: "z"}
    for idx in range(3):
        ax = axes[idx]
        ax.plot(t_raw, df_raw[raw_cols[idx]], label="raw rtfilt", color="#1f77b4", alpha=0.9)
        ax.plot(t_gc, df_gc[gc_cols[idx]], label="gcvspl", color="#2ca02c", alpha=0.9)
        ax.plot(t_kf, df_kf[kf_cols[idx]], label="kalman", color="#d62728", alpha=0.9)
        ax.set_ylabel(f"{axes_map[idx]}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
