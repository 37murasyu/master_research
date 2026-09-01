import os
import pandas as pd
import plotly.graph_objects as go

CSV = 'output_data/poses/kpts3d_subject9_20250925_201442_gcvspl.csv'
OUT = 'output_data/plots/subject9_pose_frame100_interactive.html'
FRAME = 100  # 0-based


def main():
    df = pd.read_csv(CSV)
    if FRAME >= len(df):
        raise SystemExit(f'frame {FRAME} out of range (len={len(df)})')

    joint_ids = sorted({int(c.split('_')[1]) for c in df.columns if c.startswith('joint_') and c.endswith('_x')})
    xs, ys, zs, texts, traces = [], [], [], [], []
    for jid in joint_ids:
        x = df.loc[FRAME, f'joint_{jid}_x']
        y = df.loc[FRAME, f'joint_{jid}_y']
        z = df.loc[FRAME, f'joint_{jid}_z']
        xs.append(x); ys.append(y); zs.append(z); texts.append(str(jid))
    traces.append(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers+text',
            marker=dict(size=5, color='royalblue'),
            text=texts,
            textposition='top center',
            name='joints',
        )
    )

    mx = (min(xs) + max(xs)) / 2
    my = (min(ys) + max(ys)) / 2
    mz = (min(zs) + max(zs)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) * 0.6 or 1.0

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X (lateral)',
            yaxis_title='Y (height)',
            zaxis_title='Z (depth)',
            xaxis=dict(range=[mx - span, mx + span]),
            yaxis=dict(range=[my - span, my + span]),
            zaxis=dict(range=[mz - span, mz + span]),
            aspectmode='cube',
        ),
        title=f'Subject9 pose frame {FRAME}',
        showlegend=False,
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.write_html(OUT, include_plotlyjs='cdn', full_html=True)
    print('[OUT]', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
