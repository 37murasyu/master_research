# Mean ± SD FFT Summary (Pose & Torque)

This script computes the mean FFT spectrum across subjects 2–9 and plots mean ± SD bands.

- **Pose**: magnitude of XYZ (prefers `joint_16_*`, then `wrist_R_*`, then `joint_0_*`).
- **Torque**: single column (prefers `wrist_R_local_y`, then `wrist_R_y`).
- **Frequency axis**: log-x, linear y, up to Nyquist (15 Hz at 30 Hz sampling).
- **Center & band**: geometric mean with P5–P95 band (log-domain).
- **Output**: `fft_plots/mean_sd_pose_torque.png`.

## Run

```bat
cd /d C:\Users\villa\Desktop\master_Research\MAINCODE
C:\Users\villa\venv312\Scripts\python.exe tmp_fft_mean_sd.py
```

## Notes
- Subject 5 pose uses:
  `c:\Users\villa\Desktop\master_Research\cameras_raw\5_20250925_133228\5_1stereo_pose_scaled.csv`
- Subject 3 uses `3_0stereo_pose_scaled_with2d.csv`.
