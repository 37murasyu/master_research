import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq

path = r'output_data/poses/kpts3d_subject8_20250925_192700_filtpos_gcvspl_angles.csv'
fs = 30.0
cols = ['elbow_R_angle_raw_deg', 'elbow_R_angle_smooth_deg']
bands = {'sig': (0.4, 0.7), 'noise': (2.0, 15.0)}
res = {}
for c in cols:
    x = pd.read_csv(path)[c].to_numpy(float)
    x = x - np.nanmean(x)
    X = rfft(x)
    f = rfftfreq(x.size, 1/fs)
    res[c] = {}
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f <= hi)
        res[c][name] = float(np.sum(np.abs(X[mask])**2))
    k = int(np.argmax(np.abs(X)))
    res[c]['peak_freq'] = float(f[k])
    res[c]['peak_amp'] = float(2*np.abs(X[k]) / x.size)

snr_raw = res['elbow_R_angle_raw_deg']['sig'] / res['elbow_R_angle_raw_deg']['noise']
snr_s = res['elbow_R_angle_smooth_deg']['sig'] / res['elbow_R_angle_smooth_deg']['noise']
res['snr_improve_dB'] = float(10*np.log10(snr_s / snr_raw))
print(res)
