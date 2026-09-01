from pathlib import Path
import subprocess

ROOT = Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE")
POSE_DIR = ROOT / "output_data" / "filtered_pose_lpf"
OUT_DIR = ROOT / "output_data" / "torque_wristbase"

POSE_FILES = [
    "2_stereo_pose_lpf.csv",
    "3_0stereo_pose_scaled_with2d_lpf.csv",
    "3_1stereo_pose_scaled_with2d_lpf.csv",
    "5_1stereo_pose_scaled_lpf.csv",
    "6_stereo_pose_scaled_with2d_lpf.csv",
    "7_stereo_pose_scaled_with2d_lpf.csv",
    "8_stereo_pose_scaled_with2d_lpf.csv",
    "kpts3d_9_20250925_201442_lpf.csv",
]

python = r"C:/Users/villa/venv312/Scripts/python.exe"
script = str(ROOT / "compute_torque_from_pose.py")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for name in POSE_FILES:
    pose_csv = POSE_DIR / name
    if not pose_csv.exists():
        print(f"[SKIP] missing: {pose_csv}")
        continue
    prefix = pose_csv.stem
    cmd = [
        python,
        script,
        "--pose-csv",
        str(pose_csv),
        "--out-dir",
        str(OUT_DIR),
        "--prefix",
        prefix,
        "--wrist-base",
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
