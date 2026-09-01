# Noise Statistics Summary

This script computes the minimum statistics set for assessing high‑frequency noise impact on energy/work metrics.

## Outputs
- `output_data/noise_stats_summary.csv`
- `output_data/noise_stats_summary.json`

## What it computes (per subject)
- PSD via Welch (with `nperseg` and `noverlap` reported)
- Frequency resolution `df`
- Noise boundary `f_c` via cumulative power 95%
- High‑frequency power ratio `rho_hf` for pose & torque
- Ratio `A = rho_hf_torque / rho_hf_pose`
- Dominant peak `f0` and harmonics (2f0/3f0)
- Work/energy error (`W_raw`, `W_lp`, `ΔW`, `ε`)
- Per‑cycle (or 2s windows) epsilon stats: median/IQR/CV
- Bootstrap CI (95%) for mean epsilon
- Mean coherence in HF band (pose↔torque)

## Run
```bat
cd /d C:\Users\villa\Desktop\master_Research\MAINCODE
C:\Users\villa\venv312\Scripts\python.exe tmp_noise_stats.py
```

## Notes
- Subject 5 pose file uses:
  `c:\Users\villa\Desktop\master_Research\cameras_raw\5_20250925_133228\5_1stereo_pose_scaled.csv`
- Subject 3 uses `3_0stereo_pose_scaled_with2d.csv`.
