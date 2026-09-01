import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq

FILES = {
    'raw3hz_fc2p0': 'filtered_3hz_raw.csv',
    'gcvspl3hz': 'filtered_3hz_gcvspl.csv',
    'kalman3hz': 'kalman_3hz_raw.csv',
}
FS = 3.0
COLS = ['joint_3_x_rtfilt', 'joint_3_y_rtfilt', 'joint_3_z_rtfilt']
BANDS = {
    'main_0.4_0.8': (0.4, 0.8),
    'noise_1_1.5': (1.0, 1.5),
}

def pick_cols(df):
    # prefer rtfilt, then kalman, then first matching prefix per axis
    for suffix in ['_rtfilt', '_kalman']:
        cand = [f'joint_3_{axis}{suffix}' for axis in ['x', 'y', 'z']]
        if all(c in df.columns for c in cand):
            return cand
    fallback = []
    for axis in ['x', 'y', 'z']:
        match = next((c for c in df.columns if c.startswith(f'joint_3_{axis}')), None)
        if match:
            fallback.append(match)
    if len(fallback) == 3:
        return fallback
    raise KeyError('Could not find joint_3 x/y/z columns')


def analyze(path, fs):
    df = pd.read_csv(path)
    cols = pick_cols(df)
    sig = np.linalg.norm(df[cols].to_numpy(float), axis=1)
    sig = sig - np.nanmean(sig)
    n = sig.size
    spec = rfft(sig)
    freq = rfftfreq(n, 1/fs)
    amp = np.abs(spec) * 2.0 / n
    res = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freq >= lo) & (freq <= hi)
        res[name + '_power'] = float(np.sum(np.abs(spec[mask])**2))
    main = res['main_0.4_0.8_power']
    noise = res['noise_1_1.5_power']
    res['snr_main_over_noise'] = float(main / max(1e-12, noise))
    res['snr_main_over_noise_dB'] = float(10 * np.log10(res['snr_main_over_noise']))
    idx = np.argsort(amp[1:])[::-1][:5] + 1
    res['peaks'] = [(float(freq[i]), float(amp[i])) for i in idx]
    return res

def main():
    out = {}
    for k, path in FILES.items():
        out[k] = analyze(path, FS)
    snr = {k: v['snr_main_over_noise'] for k, v in out.items()}
    out['snr_improvements_dB'] = {
        'gcvspl_vs_raw': float(10 * np.log10(snr['gcvspl3hz'] / snr['raw3hz_fc2p0'])),
        'kalman_vs_raw': float(10 * np.log10(snr['kalman3hz'] / snr['raw3hz_fc2p0'])),
        'kalman_vs_gcvspl': float(10 * np.log10(snr['kalman3hz'] / snr['gcvspl3hz'])),
    }
    print(out)

if __name__ == '__main__':
    main()
