import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--fs', type=float, required=True)
parser.add_argument('--columns', required=True)
args = parser.parse_args()

cols = [c.strip() for c in args.columns.split(',') if c.strip()]
df = pd.read_csv(args.input, usecols=cols)
data = df.to_numpy(float)
if data.shape[1] == 1:
    sig = data[:,0]
else:
    sig = np.linalg.norm(data[:,:3], axis=1)

sig = sig - np.nanmean(sig)
n = sig.size
spec = rfft(sig)
freq = rfftfreq(n, d=1.0/args.fs)
amp = np.abs(spec) * 2.0 / n
# top 5 excluding DC
idx = np.argsort(amp[1:])[::-1][:5] + 1
print("Dominant peaks (Hz, amp):")
for i in idx:
    print(f"  {freq[i]:.4f} Hz\t{amp[i]:.4f}")
