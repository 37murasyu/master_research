import os
import pandas as pd
import plotly.graph_objects as go

CSV = 'output_data/torque/kpts3d_subject9_20250925_201442_gcvspl_trimmed_wristfix_torque.csv'
OUT = 'output_data/plots/subject9_torque_frame100_interactive.html'
FRAME = 100  # 0-based index
PARTS = [
    ('wrist_R', 'orange'),
    ('elbow_R', 'blue'),
    ('shoulder_R', 'green'),
]


def main():
    df = pd.read_csv(CSV)
    if FRAME >= len(df):
        raise SystemExit(f'frame {FRAME} out of range (len={len(df)})')

    traces = []
    xs, ys, zs = [], [], []
    for part, color in PARTS:
        x = df.loc[FRAME, f'{part}_x']
        y = df.loc[FRAME, f'{part}_y']
        z = df.loc[FRAME, f'{part}_z']
        xs.append(x); ys.append(y); zs.append(z)
        traces.append(
            go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                marker=dict(size=6, color=color),
                text=[part],
                textposition='top center',
                name=part,
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
        title=f'Subject9 torque frame {FRAME}',
        showlegend=True,
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.write_html(OUT, include_plotlyjs='cdn', full_html=True)
    print('[OUT]', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
