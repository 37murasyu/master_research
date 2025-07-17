# gauge_display_module.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
import numpy as np
import json


class GaugeDisplay:
    def __init__(self, config_path, get_angles_func, image_path="wheelchair_user.png"):
        self.config_path = config_path
        self.get_angles = get_angles_func
        self.image_path = image_path
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.radius = 1.8
        self.gauges = []
        self._load_config()
        self._init_figure()

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def _init_figure(self):
        img = mpimg.imread(self.image_path)
        scale = self.config.get("image_scale", 3.0)
        extent = self.config["image_extent"]
        cx = (extent[0] + extent[1]) / 2
        cy = (extent[2] + extent[3]) / 2
        w = (extent[1] - extent[0]) * scale
        h = (extent[3] - extent[2]) * scale
        new_extent = [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]
        self.ax.imshow(img, extent=new_extent)

        cmap = plt.get_cmap("YlOrRd")
        for gdata in self.config["gauges"]:
            c = gdata["center"]
            label = gdata["label"]
            patches = []
            for j in range(180, 120, -2):
                w = Wedge(c, self.radius, j, j + 2, facecolor="lightgray", lw=0)
                self.ax.add_patch(w)
                patches.append(w)
            for j in range(120, -1, -2):
                color = cmap(j / 120)
                w = Wedge(c, self.radius, j, j + 2, facecolor=color, lw=0)
                self.ax.add_patch(w)
                patches.append(w)
            needle = Line2D([c[0], c[0]], [c[1], c[1]], color="black", lw=2)
            self.ax.add_line(needle)
            label_val = self.ax.text(c[0], c[1] - 0.4, "", ha="center", color="black")
            label_name = self.ax.text(
                c[0], c[1] + 2.1, label, ha="center", fontsize=10, color="black"
            )
            outline = Wedge(
                c, self.radius, 0, 180, facecolor="none", edgecolor="red", lw=2
            )
            outline.set_visible(False)
            self.ax.add_patch(outline)
            self.gauges.append(
                {
                    "center": c,
                    "needle": needle,
                    "label_val": label_val,
                    "label_name": label_name,
                    "outline": outline,
                }
            )

        self.ax.set_xlim(-3, 17)
        self.ax.set_ylim(8, 22)

    def update(self):
        angles = self.get_angles()
        for g, val in zip(self.gauges, angles):
            c = g["center"]
            angle_rad = np.radians(180 - val)
            x_end = c[0] + self.radius * 0.9 * np.cos(angle_rad)
            y_end = c[1] + self.radius * 0.9 * np.sin(angle_rad)
            g["needle"].set_data([c[0], x_end], [c[1], y_end])
            g["label_val"].set_text(f"{int(val)}°")
            color = "red" if val <= 60 else "black"
            g["label_val"].set_color(color)
            g["label_name"].set_color(color)
            g["outline"].set_visible(val <= 60)
        self.fig.canvas.draw_idle()

    def run(self, frames=100, interval=0.1):
        for _ in range(frames):
            self.update()
            plt.pause(interval)
        plt.show()


# 使用例（別スクリプトで呼び出し）
if __name__ == "__main__":
    import time

    def mock_angles():
        t = time.time() % 10
        return [
            max(0, min(180, a + 10 * np.sin(t + i)))
            for i, a in enumerate([160, 130, 90, 70, 110, 50])
        ]

    gd = GaugeDisplay("positions.json", mock_angles)
    gd.run()
