# FFT Stats Plots and Mixed-Effects Tables

Creates:
- Figure A: ρHF (pose/torque) box + scatter
- Figure B: f₀ distribution (pose/torque) box + scatter
- Mixed-effects tables (random intercept per subject)

## Outputs
- `fft_plots/fft_stats_figA_figB.png`
- `fft_plots/mixedlm_rho_hf.csv`
- `fft_plots/mixedlm_f0.csv`
- `fft_plots/mixedlm_summary.md`

## Run
```bat
cd /d C:\Users\villa\Desktop\master_Research\MAINCODE
C:\Users\villa\venv312\Scripts\python.exe tmp_fft_stats_plots.py
```
