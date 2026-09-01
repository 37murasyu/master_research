import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq

FILES = {
    'raw3hz_fc1p6': 'filtered_3hz_raw_fc1p6.csv',
    'raw3hz_fc2p0': 'filtered_3hz_raw.csv',
    'gcvspl3hz': 'filtered_3hz_gcvspl.csv',
}
FS = 3.0
COLS = ['joint_3_x_rtfilt', 'joint_3_y_rtfilt', 'joint_3_z_rtfilt']

def peaks(path, fs, k=5):
    df = pd.read_csv(path)
    data = df[COLS].to_numpy(float)
    sig = np.linalg.norm(data[:, :3], axis=1) if data.shape[1] > 1 else data[:, 0]
    sig = sig - np.nanmean(sig)
    n = sig.size
    spec = rfft(sig)
    freq = rfftfreq(n, 1/fs)
    amp = np.abs(spec) * 2.0 / n
    idx = np.argsort(amp[1:])[::-1][:k] + 1
    return [(float(freq[i]), float(amp[i])) for i in idx]

if __name__ == '__main__':
    for name, path in FILES.items():
        p = peaks(path, FS)
        print(name, p)
