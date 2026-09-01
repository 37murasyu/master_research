import csv
from pathlib import Path

import plotly.graph_objects as go


def load_rm_method(csv_path: Path):
    reps = []
    multipliers = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reps.append(int(float(row["ダンベル反復回数"])))
            multipliers.append(float(row["推定最大重量倍率(倍)"]))
    return reps, multipliers


def plot_rm_method(csv_path: Path, out_path: Path) -> None:
    reps, multipliers = load_rm_method(csv_path)

    primary = "#1D3178"
    accent = "#B08D2E"
    bg = "#0B0B0B"
    grid = "#2A2A2A"
    text_light = "#E9ECF5"
    gloss = "#B08D2E"

    text_labels = [f"{value:.2f}x" for value in multipliers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=reps,
            y=multipliers,
            marker=dict(color=primary, line=dict(color=accent, width=1.6)),
            text=text_labels,
            textposition="outside",
            textfont=dict(color=text_light, size=12),
            hovertemplate="%{x}回: %{y:.2f}x<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reps,
            y=multipliers,
            mode="lines",
            line=dict(color=gloss, width=2.6, shape="hv"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="RM法: 反復回数と推定最大重量倍率",
        title_font=dict(size=22, color=text_light, family="Yu Gothic, Meiryo, sans-serif"),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=text_light, family="Yu Gothic, Meiryo, sans-serif", size=14),
        xaxis=dict(
            title=dict(text="ダンベル反復回数 (回)", font=dict(color=text_light)),
            tickmode="array",
            tickvals=reps,
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=text_light),
        ),
        yaxis=dict(
            title=dict(text="推定最大重量倍率 (倍)", font=dict(color=text_light)),
            range=[0.9, max(multipliers) + 0.2],
            gridcolor=grid,
            griddash="dash",
            zeroline=False,
            tickfont=dict(color=text_light),
        ),
        margin=dict(l=70, r=40, t=90, b=70),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_path, width=1280, height=720, scale=2)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    csv_path = base / "rm_method.csv"
    out_path = base / "output_data" / "rm_method_bar.png"
    plot_rm_method(csv_path, out_path)
