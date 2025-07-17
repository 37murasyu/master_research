import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
import numpy as np
import json


class GaugeDisplay:
    """
    背景画像上にゲージメーターを複数表示し、リアルタイムで針や色を更新する可視化UIクラス。

    Parameters
    ----------
    config_path : str
        ゲージの配置やラベル、スケールなどを記述したJSON設定ファイルのパス。
    stats_dict : dict[str, tuple[float, float]]
        各部位の平均値と標準偏差を持つ辞書。キーは 'wrist_R', 'elbow_R', ... の形式。
    image_path : str, optional
        背景画像のパス。デフォルトは "wheelchair_user.png"。
    """

    def __init__(self, config_path, stats_dict, image_path="wheelchair_user.png"):
        self.config_path = config_path
        self.stats = stats_dict
        self.image_path = image_path
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.radius = 1.8
        self.part_keys = []
        self.current_impulses = {}
        self.gauges = []

        # 設定読み込みと部位キーのマッピング
        self._load_config()
        self._map_labels_to_keys()
        # 初期インパルス値を0に設定
        self.current_impulses = {k: 0.0 for k in self.part_keys}
        # 図の初期化
        self._init_figure()

    def _load_config(self):
        """
        JSONファイルから設定（ゲージの位置やラベルなど）を読み込む。
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def _map_labels_to_keys(self):
        """
        JSONラベルを内部の部位キーに変換する。
        """
        json_labels = [g["label"] for g in self.config["gauges"]]

        def map_label(lbl: str) -> str:
            parts = lbl.split()
            side = "R" if parts[0].lower().startswith("right") else "L"
            area = parts[1].lower()
            if "upper" in area:
                joint = "elbow"
            elif "forearm" in area:
                joint = "wrist"
            else:
                joint = "shoulder"
            return f"{joint}_{side}"

        self.part_keys = [map_label(lbl) for lbl in json_labels]

    def _init_figure(self):
        """
        背景画像とゲージメーターの描画を初期化する。
        """
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
            # 灰色ベースとグラデーションパッチ
            for j in range(180, 120, -2):
                self.ax.add_patch(
                    Wedge(c, self.radius, j, j + 2, facecolor="lightgray", lw=0)
                )
            for j in range(120, -1, -2):
                color = cmap(j / 120)
                self.ax.add_patch(
                    Wedge(c, self.radius, j, j + 2, facecolor=color, lw=0)
                )
            # 針とラベル
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
        # 軸範囲設定
        xlim = self.config.get("xlim", (-3, 17))
        ylim = self.config.get("ylim", (8, 22))
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

    def update_impulses(self, impulses: dict[str, float]):
        """
        最新のインパルス値を更新する。
        """
        for k, v in impulses.items():
            if k in self.current_impulses:
                self.current_impulses[k] = v

    def get_angles(self) -> list[float]:
        """
        現在のインパルス値と統計値からゲージ角度を計算して返す。
        """
        angles = []
        for pk in self.part_keys:
            mu, sigma = self.stats[pk]
            imp = self.current_impulses[pk]
            angle = 60 * (imp / sigma) + 120 - 60 * (mu / sigma)
            angles.append(angle)
        return angles

    def update(self):
        """
        ゲージ針・数値ラベル・色・強調枠を更新する。
        """
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
