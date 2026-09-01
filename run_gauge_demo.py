"""Run gauge demo fill with visible window (blocking for ~16s)."""

import os
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt.QtCore import QTimer  # pylint: disable=no-name-in-module,no-member
from Gauge_display import GaugeDisplay

CFG_PATH = os.path.join(os.getcwd(), "gauge_layout.json")
IMG_PATH = "wheelchair_user.png"

if os.path.exists("supervision_stats.csv"):
    df = pd.read_csv("supervision_stats.csv")
    STATS = {row["part"]: (row["mean"], row["std"]) for _, row in df.iterrows()}
else:
    STATS = {k: (0.0, 1.0) for k in ["wrist_R", "elbow_R", "wrist_L", "elbow_L"]}

def main() -> None:
    g = GaugeDisplay(CFG_PATH, STATS, image_path=IMG_PATH, debug=True, warmup_frames=0)
    try:
        g.filter_parts(["wrist_R", "elbow_R", "wrist_L", "elbow_L"])
    except Exception:
        pass
    g.show(on_top=True, x=100, y=100, w=800, h=600, title="Gauge Demo")

    def _start_demo():
        print("[Demo] starting scripted fill...")
        g.play_demo_fill(step_seconds=2.0, steps=8, low_steps=(2, 6), high_pct=0.80, low_pct=0.15, chunk_seconds=0.25)
        print("[Demo] done; closing in 1s...")
        QTimer.singleShot(1000, pg.mkQApp().quit)

    # Start fill after 4 seconds so the window can be moved/resized before animation
    QTimer.singleShot(4000, _start_demo)
    pg.exec()

if __name__ == "__main__":
    main()
