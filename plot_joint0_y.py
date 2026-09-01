import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv = Path('output_data/poses/kpts3d_subject8_20250925_192700_bpf04.csv')
df = pd.read_csv(csv)
col = 'joint_0_y_f' if 'joint_0_y_f' in df.columns else 'joint_0_y'
if col not in df.columns:
    raise SystemExit('joint_0_y(_f) not found')
plt.figure(figsize=(10,3))
plt.plot(df[col], label=col, color='#1f77b4')
plt.xlabel('frame')
plt.ylabel(col)
plt.grid(alpha=0.3)
plt.legend()
out = csv.parent / 'joint0_y_bpf04.png'
plt.tight_layout()
plt.savefig(out, dpi=200)
print(f'Saved {out.as_posix()}')
