import numpy as np, pandas as pd

def compute(pose_csv, tau_csv, cyc_csv, fps=30.0):
    dt = 1.0 / fps
    dp = pd.read_csv(pose_csv)
    dtau = pd.read_csv(tau_csv)
    cyc = pd.read_csv(cyc_csv)
    cols = ['joint_12_x','joint_12_y','joint_12_z','joint_14_x','joint_14_y','joint_14_z','joint_16_x','joint_16_y','joint_16_z']
    for c in cols:
        if c not in dp.columns:
            raise SystemExit(f"missing {c} in {pose_csv}")
    p12 = dp[['joint_12_x','joint_12_y','joint_12_z']].to_numpy(float)
    p14 = dp[['joint_14_x','joint_14_y','joint_14_z']].to_numpy(float)
    p16 = dp[['joint_16_x','joint_16_y','joint_16_z']].to_numpy(float)
    v1 = p12 - p14
    v2 = p16 - p14
    num = (v1 * v2).sum(axis=1)
    den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    den = np.where(den < 1e-9, np.nan, den)
    ang = np.arccos(np.clip(num / den, -1.0, 1.0))
    ang_vel = np.gradient(ang, dt)
    tau = dtau['elbow_R_local_y'].to_numpy(float)
    n = min(len(ang_vel), len(tau), len(cyc))
    ang_vel = ang_vel[:n]
    tau = tau[:n]
    cyc_idx = cyc['cycle_index'].to_numpy(int)[:n]
    power = tau * ang_vel
    out = []
    for c in np.unique(cyc_idx[cyc_idx >= 1]):
        m = cyc_idx == c
        w = float((power[m] * dt).sum())
        wpos = float(np.clip(power[m], 0, None).sum() * dt)
        wneg = float(np.clip(power[m], None, 0).sum() * dt)
        out.append((int(c), w, wpos, wneg))
    return out

def summarize(label, pose, tau, cyc):
    rows = compute(pose, tau, cyc)
    if not rows:
        print(f"{label}: no cycles")
        return
    import pandas as pd
    df = pd.DataFrame(rows, columns=['cycle','work_signed_J','work_pos_J','work_neg_J'])
    print(f"== {label} ==")
    print(df)
    print({
        'cycles': len(df),
        'sum_pos_J': float(df['work_pos_J'].sum()),
        'mean_pos_J': float(df['work_pos_J'].mean()),
        'sum_signed_J': float(df['work_signed_J'].sum()),
        'mean_signed_J': float(df['work_signed_J'].mean()),
    })

if __name__ == '__main__':
    summarize('3kg', 'output_data/poses/kpts3d_00_3_20251215_232033_m_joint_gcvspl.csv', 'output_data/torque/kpts3d_00_3_20251215_232033_m_joint_gcvspl_torque.csv', 'output_data/poses/kpts3d_00_3_20251215_232033_m_joint_gcvspl_with_cycles.csv')
    summarize('6kg', 'output_data/poses/kpts3d_00_6_20251215_232147_m_joint_gcvspl.csv', 'output_data/torque/kpts3d_00_6_20251215_232147_m_joint_gcvspl_torque.csv', 'output_data/poses/kpts3d_00_6_20251215_232147_m_joint_gcvspl_with_cycles.csv')
