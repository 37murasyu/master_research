import pandas as pd
import numpy as np

def main():
    src = 'output_data/poses/kpts3d_subject8_20250925_192700_gcvspl.csv'
    dst = 'output_data/poses/kpts3d_subject8_20250925_192700_gcvspl_joint3_3hz.csv'
    df = pd.read_csv(src)
    time = df['frame'].to_numpy(float) / 30.0
    cols = ['joint_3_x', 'joint_3_y', 'joint_3_z']
    t_out = np.arange(time[0], time[-1] + 1e-9, 1/3)
    out = {'time_s': t_out}
    for c in cols:
        out[c] = np.interp(t_out, time, df[c].to_numpy(float))
    pd.DataFrame(out).to_csv(dst, index=False)
    print('written', dst)

if __name__ == '__main__':
    main()
