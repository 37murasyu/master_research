import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def load_pose_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    frames = df["frame"].to_numpy()
    # remaining columns are joint_{j}_x/y/z (keep order as-is)
    cols = [c for c in df.columns if c != "frame"]
    num_cols = len(cols)
    if num_cols % 3 != 0:
        raise ValueError(f"column count {num_cols} is not multiple of 3")

    # extract joint ids from column names to keep original MediaPipe indices for labeling
    joint_ids: list[int] = []
    for i in range(0, num_cols, 3):
        triplet = cols[i : i + 3]
        try:
            parts = [p.split("_") for p in triplet]
            jid = int(parts[0][1])
            if not all(int(p[1]) == jid for p in parts):
                raise ValueError
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"column naming does not follow joint_<id>_(x|y|z): {triplet}") from exc
        joint_ids.append(jid)

    J = num_cols // 3
    data = df[cols].to_numpy(dtype=float).reshape(len(df), J, 3)
    return frames, data, np.array(joint_ids, dtype=int)


POSE_EDGES = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
    (23, 24), (11, 23), (12, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (17, 19), (19, 21), (16, 18), (18, 20), (20, 22),
]


def make_figure(
    frames: np.ndarray,
    pts: np.ndarray,
    start_idx: int = 0,
    margin: float = 0.1,
    clip_pct: float = 0.0,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    show_links: bool = False,
    show_labels: bool = False,
    joint_ids: np.ndarray | None = None,
) -> go.Figure:
    # pts shape: (T, J, 3)
    T, J, _ = pts.shape
    # initial frame index (clamped)
    t0 = int(max(0, min(start_idx, T - 1)))
    x0, y0, z0 = pts[t0].T

    fig = go.Figure()
    scatter = go.Scatter3d(
        x=x0,
        y=y0,
        z=z0,
        mode="markers",
        marker=dict(size=5, color=np.linspace(0, 1, J), colorscale="Viridis"),
    )
    fig.add_trace(scatter)

    def label_trace(frame_pts):
        if not show_labels:
            return []
        labels = [str(j) for j in joint_ids] if joint_ids is not None else [str(i) for i in range(frame_pts.shape[0])]
        return [
            go.Scatter3d(
                x=frame_pts[:, 0],
                y=frame_pts[:, 1],
                z=frame_pts[:, 2],
                mode="text",
                text=labels,
                textposition="top center",
                showlegend=False,
                hoverinfo="none",
                textfont=dict(color="#222", size=10),
            )
        ]

    def edge_lines(frame_pts):
        if not show_links:
            return []
        Jcur = frame_pts.shape[0]
        # prefer MediaPipe topology if joint_ids are provided; otherwise fallback
        if joint_ids is not None and len(joint_ids) == Jcur:
            idx_map = {jid: idx for idx, jid in enumerate(joint_ids)}
            edges = [(idx_map[a], idx_map[b]) for a, b in POSE_EDGES if a in idx_map and b in idx_map]
        else:
            # fallback: simple chain when ids are unknown
            edges = [(i, i + 1) for i in range(Jcur - 1)]
        xs = []
        ys = []
        zs = []
        for a, b in edges:
            xs.extend([frame_pts[a, 0], frame_pts[b, 0], None])
            ys.extend([frame_pts[a, 1], frame_pts[b, 1], None])
            zs.extend([frame_pts[a, 2], frame_pts[b, 2], None])
        return [
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="#888", width=2),
                showlegend=False,
            )
        ]

    if show_links:
        fig.add_traces(edge_lines(pts[t0]))
    if show_labels:
        fig.add_traces(label_trace(pts[t0]))

    frames_plotly = []
    for t in range(T):
        xt, yt, zt = pts[t].T
        frame_traces = [go.Scatter3d(x=xt, y=yt, z=zt, mode="markers", marker=scatter.marker)]
        frame_traces.extend(edge_lines(pts[t]))
        frame_traces.extend(label_trace(pts[t]))
        frames_plotly.append(go.Frame(data=frame_traces, name=str(t)))

    steps = [
        {
            "args": [[f"{k}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": str(int(frames[k])),
            "method": "animate",
        }
        for k in range(T)
    ]

    fig.frames = frames_plotly

    # set axis ranges to fit points with optional percentile clipping
    finite_pts = pts[np.isfinite(pts).all(axis=2)]
    if finite_pts.size > 0:
        x_all = finite_pts[:, 0]
        y_all = finite_pts[:, 1]
        z_all = finite_pts[:, 2]

        def lim(v):
            if clip_pct > 0:
                lo, hi = np.percentile(v, [clip_pct, 100 - clip_pct])
            else:
                lo, hi = float(np.min(v)), float(np.max(v))
            lo, hi = float(lo), float(hi)
            if lo == hi:
                return (lo - 1, hi + 1)
            span = hi - lo
            return (lo - span * margin, hi + span * margin)

        xr, yr, zr = lim(x_all), lim(y_all), lim(z_all)
        if x_range is not None:
            xr = x_range
        if y_range is not None:
            yr = y_range
        if z_range is not None:
            zr = z_range
        fig.update_layout(scene=dict(xaxis=dict(range=xr), yaxis=dict(range=yr), zaxis=dict(range=zr)))

    fig.update_layout(
        title="Stereo Pose 3D",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": steps,
                "currentvalue": {"prefix": "frame: "},
                "pad": {"t": 30},
            }
        ],
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description="Interactive 3D plot for stereo_pose.csv")
    ap.add_argument("--csv", required=True, help="path to stereo_pose.csv")
    ap.add_argument("--frame", type=int, default=0, help="initial frame to show (0-based)")
    ap.add_argument("--margin", type=float, default=0.1, help="axis margin ratio (default 0.1)")
    ap.add_argument("--clip-pct", type=float, default=0.0, help="percentile clip (e.g., 1.0 for 1-99%%)")
    ap.add_argument("--x-range", type=float, nargs=2, default=None, help="force X axis range [min max]")
    ap.add_argument("--y-range", type=float, nargs=2, default=None, help="force Y axis range [min max]")
    ap.add_argument("--z-range", type=float, nargs=2, default=None, help="force Z axis range [min max]")
    ap.add_argument("--exclude-joints", type=int, nargs="*", default=None, help="joint IDs to exclude from plotting")
    ap.add_argument("--links", action="store_true", help="draw skeletal links")
    ap.add_argument("--labels", action="store_true", help="draw joint index labels")
    args = ap.parse_args()

    frames, pts, joint_ids = load_pose_csv(args.csv)
    if args.exclude_joints:
        ex = set(args.exclude_joints)
        if joint_ids is not None and len(joint_ids) == pts.shape[1]:
            mask = [jid not in ex for jid in joint_ids]
            pts = pts[:, mask, :]
            joint_ids = joint_ids[mask]
    x_rng = tuple(args.x_range) if args.x_range is not None else None
    y_rng = tuple(args.y_range) if args.y_range is not None else None
    z_rng = tuple(args.z_range) if args.z_range is not None else None
    fig = make_figure(
        frames,
        pts,
        start_idx=args.frame,
        margin=args.margin,
        clip_pct=args.clip_pct,
        x_range=x_rng,
        y_range=y_rng,
        z_range=z_rng,
        show_links=args.links,
        show_labels=args.labels,
        joint_ids=joint_ids,
    )
    fig.show()


if __name__ == "__main__":
    main()
