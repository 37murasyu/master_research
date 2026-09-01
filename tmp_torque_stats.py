import pandas as pd
import json
import os

FILES = [
    "torque/kpts3d_9_20250925_201442_torque.csv",
    "torque/2_stereo_pose_torque.csv",
    "torque/3_0stereo_pose_scaled_with2d_torque.csv",
    "torque/3_1stereo_pose_scaled_with2d_torque.csv",
    "torque/4_0stereo_pose_scaled_with2d_torque.csv",
    "torque/5_1stereo_pose_scaled_torque.csv",
    "torque/5_stereo_pose_scaled_with2d_torque.csv",
    "torque/6_stereo_pose_scaled_with2d_torque.csv",
    "torque/7_stereo_pose_scaled_with2d_torque.csv",
    "torque/8_stereo_pose_scaled_with2d_torque.csv",
]
COLS = [
    "wrist_R_y", "elbow_R_y", "wrist_L_y", "elbow_L_y",
    "wrist_R_local_y", "elbow_R_local_y", "wrist_L_local_y", "elbow_L_local_y",
]

def main():
    for path in FILES:
        if not os.path.exists(path):
            print(f"[MISS] {path}")
            continue
        df = pd.read_csv(path)
        stats = {}
        for c in COLS:
            if c not in df.columns:
                continue
            s = df[c].dropna()
            if s.empty:
                continue
            stats[c] = {
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p50": float(s.median()),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
            }
        print(f"[FILE] {path} rows={len(df)} cols_found={len(stats)}")
        print(json.dumps(stats, ensure_ascii=False))

if __name__ == "__main__":
    main()
