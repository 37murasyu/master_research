from pathlib import Path
import subprocess

ROOT = Path(r"c:\Users\villa\Desktop\master_Research\MAINCODE")
POSE_DIR = ROOT / "output_data" / "filtered_pose_lpf"
TORQUE_DIR = ROOT / "output_data" / "filtered_torque_lpf_recalc"

pairs = [
    "2_stereo_pose_lpf",
    "3_0stereo_pose_scaled_with2d_lpf",
    "3_1stereo_pose_scaled_with2d_lpf",
    "4_0stereo_pose_scaled_with2d_lpf",
    "5_1stereo_pose_scaled_lpf",
    "6_stereo_pose_scaled_with2d_lpf",
    "7_stereo_pose_scaled_with2d_lpf",
    "8_stereo_pose_scaled_with2d_lpf",
    "kpts3d_9_20250925_201442_lpf",
]

python = r"C:/Users/villa/venv312/Scripts/python.exe"
script = str(ROOT / "recalc_elbow_local_torque.py")

for stem in pairs:
    pose_csv = POSE_DIR / f"{stem}.csv"
    torque_csv = TORQUE_DIR / f"{stem.replace('_lpf','')}_torque_lpf.csv"
    if not pose_csv.exists():
        print(f"[SKIP] missing pose: {pose_csv}")
        continue
    if not torque_csv.exists():
        print(f"[SKIP] missing torque: {torque_csv}")
        continue
    cmd = [python, script, "--pose-csv", str(pose_csv), "--torque-csv", str(torque_csv)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
