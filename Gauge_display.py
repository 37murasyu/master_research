from __future__ import annotations

# pylint: disable=broad-exception-caught, no-member, import-error

import json
import os
from typing import Any
import time as _time

import numpy as np
import cv2 as cv

# Runtime deps: pyqtgraph + Qt (via pyqtgraph.Qt)
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except Exception as _e:  # pragma: no cover
    raise ImportError(
        "PyQtGraph/Qt のインポートに失敗しました。`pip install pyqtgraph PyQt5` などで導入してください."
    ) from _e

Qt = getattr(QtCore, "Qt", None)  # type: ignore[attr-defined]
QApplication = getattr(QtWidgets, "QApplication", None)  # type: ignore[attr-defined]


def _to_rgba_tuple(v: Any, default=(0.2, 0.8, 0.2, 0.9)) -> tuple[float, float, float, float]:
    if isinstance(v, (list, tuple)) and len(v) in (3, 4):
        if len(v) == 3:
            return float(v[0]), float(v[1]), float(v[2]), 0.9
        return float(v[0]), float(v[1]), float(v[2]), float(v[3])
    return tuple(default)  # type: ignore[return-value]


class GaugeDisplay:
    """PyQtGraph ベースの高速ゲージ UI。背景は初期化時に一度だけ描画し、毎フレームは更新しない。"""

    def __init__(
        self,
        config_path: str,
        stats_dict: dict[str, tuple[float, float]],
        image_path: str = "wheelchair_user.png",
        debug: bool = False,
        warmup_frames: int = 0,
    ) -> None:
        self.config_path = config_path
        self.stats = stats_dict
        self.image_path = image_path
        # 環境変数でデバッグ出力制御（既定: 有効）。GAUGE_DEBUG=0 で抑制可能。
        try:
            _env_dbg = os.getenv("GAUGE_DEBUG", "1")
            env_debug = _env_dbg not in ("0", "false", "False")
        except Exception:
            env_debug = True
        self.debug = bool(debug or env_debug)
        self.warmup_frames = max(int(warmup_frames), 0)

        self.radius: float = 1.8
        self.ring_width: float = self.radius * 0.25  # 幾何半径調整用（描画の太さではない）
        # 画面上の線の太さ（ピクセル指定）: 既定は12px。config読込後に上書き。
        self.stroke_px: int = 12
        self.band_stroke_px: int = max(2, self.stroke_px - 4)

        self.part_keys: list[str] = []
        self.current_impulses: dict[str, float] = {}
        self.energy_thresholds: dict[str, tuple[float, float]] = {}
        self._frame_index: int = 0
        self._last_angles: list[float] = []
        self._direct_ratios: dict[str, float] | None = None
        self._direct_fill_rgba: tuple[float, float, float, float] = (0.20, 0.80, 0.20, 0.90)
        try:
            self._event_every = max(1, int(os.getenv('GAUGE_EVENT_EVERY', '1')))
        except Exception:
            self._event_every = 1
        self._fill_colors: dict[str, tuple[float, float, float, float]] = {}

        # UI containers
        self.gauges: list[dict[str, Any]] = []
        self.fill_items: list[Any] = []
        self.band_items: list[Any] = []

        # Qt window
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(show=True, title="GaugeDisplay")
        self.view: pg.ViewBox = self.win.addViewBox(lockAspect=True, enableMenu=False)
        self.view.invertY(False)
        try:
            if hasattr(self.win, "setBackground"):
                self.win.setBackground((0, 0, 0))
            else:
                raise AttributeError("no setBackground on win")
        except Exception:
            try:
                pg.setConfigOptions(background=(0, 0, 0))
            except Exception:
                try:
                    self.view.setBackgroundBrush(pg.mkBrush((0, 0, 0)))
                except Exception:
                    pass

        if self.debug:
            print(f"[GaugeDebug] init start config={config_path} image={image_path}")
        self._load_config()
        self._map_labels_to_keys()
        self._resolve_fill_colors()

        # 線幅の構成（configがあれば上書き）
        try:
            self.stroke_px = int(self.config.get("gauge_stroke_px", self.stroke_px))
        except Exception:
            pass
        try:
            self.band_stroke_px = int(self.config.get("gauge_band_stroke_px", max(2, self.stroke_px - 4)))
        except Exception:
            self.band_stroke_px = max(2, self.stroke_px - 4)

        eth = self.config.get("energy_thresholds") if hasattr(self, "config") else None
        if isinstance(eth, dict):
            for k, v in eth.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    try:
                        self.energy_thresholds[k] = (float(v[0]), float(v[1]))
                    except (TypeError, ValueError):
                        pass

        self.current_impulses = {k: 0.0 for k in self.part_keys}

        self._init_scene()
        if self.debug:
            print(f"[GaugeDebug] part_keys mapped: {self.part_keys}")
            print(f"[GaugeDebug] initial impulses: {self.current_impulses}")

        angles_map = self.config.get("threshold_angles") if hasattr(self, "config") else None
        if isinstance(angles_map, dict):
            self.set_band_angles(angles_map)

    # ----- config helpers -----
    def _load_config(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def _map_labels_to_keys(self) -> None:
        part_keys: list[str] = []
        for g in self.config.get("gauges", []):
            key = g.get("key")
            if isinstance(key, str) and key:
                part_keys.append(key)
            else:
                lbl = str(g.get("label", "")).strip()
                parts = lbl.split()
                if len(parts) >= 2:
                    side = "R" if parts[0].lower().startswith("right") else "L"
                    area = parts[1].lower()
                    if "upper" in area:
                        joint = "elbow"
                    elif "forearm" in area:
                        joint = "wrist"
                    else:
                        joint = "shoulder"
                    part_keys.append(f"{joint}_{side}")
        self.part_keys = part_keys

    def _resolve_fill_colors(self) -> None:
        default_joint_colors: dict[str, tuple[float, float, float, float]] = {
            "shoulder": (0.20, 0.80, 0.20, 0.90),
            "elbow": (0.95, 0.55, 0.10, 0.90),
            "wrist": (0.15, 0.25, 0.95, 0.90),
        }
        for pk in self.part_keys:
            joint = (
                "shoulder"
                if pk.startswith("shoulder")
                else ("elbow" if pk.startswith("elbow") else ("wrist" if pk.startswith("wrist") else "shoulder"))
            )
            self._fill_colors[pk] = default_joint_colors[joint]
        try:
            conf_colors = self.config.get("fill_colors", {}) if hasattr(self, "config") else {}
            if isinstance(conf_colors, dict):
                for k, v in conf_colors.items():
                    if k in self.part_keys:
                        self._fill_colors[k] = _to_rgba_tuple(v, self._fill_colors.get(k, (0.2, 0.8, 0.2, 0.9)))
        except Exception:
            pass

    # ----- scene construction -----
    def _init_scene(self) -> None:
        angles = self.get_angles()
        if not self._last_angles or len(self._last_angles) != len(angles):
            self._last_angles = [float('nan')] * len(angles)
        xlim = self.config.get("xlim", (-3, 17))
        ylim = self.config.get("ylim", (8, 22))
        self.view.setRange(xRange=xlim, yRange=ylim, padding=0.0)
        if self.debug:
            print(f"[GaugeDebug] view setRange xlim={xlim} ylim={ylim}")

        img = None
        if os.path.exists(self.image_path):
            try:
                bgr = cv.imread(self.image_path, cv.IMREAD_COLOR)
                if bgr is not None:
                    if self.debug:
                        try:
                            print(f"[GaugeDebug] imread ok path={self.image_path} shape={bgr.shape}")
                        except Exception:
                            pass
                    img = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                    env_rot = os.getenv("GAUGE_IMAGE_ROTATE", "").strip()
                    rotate_deg = None
                    if env_rot:
                        try:
                            rotate_deg = float(env_rot)
                        except Exception:
                            rotate_deg = None
                    if rotate_deg is None:
                        try:
                            rotate_deg = float(self.config.get("image_rotate_deg", 0.0))
                        except Exception:
                            rotate_deg = 0.0
                    if rotate_deg is None or rotate_deg == 0.0:
                        # heuristic: known sideways asset
                        if os.path.basename(self.image_path).lower().startswith("wheelchair"):
                            rotate_deg = 90.0
                    if rotate_deg:
                        if self.debug:
                            print(f"[GaugeDebug] rotate image {rotate_deg} deg")
                        if rotate_deg == 90:
                            img = np.ascontiguousarray(np.rot90(img, k=3))
                        elif rotate_deg == 180:
                            img = np.ascontiguousarray(np.rot90(img, k=2))
                        elif rotate_deg == 270:
                            img = np.ascontiguousarray(np.rot90(img, k=1))
                        else:
                            (h0, w0) = img.shape[:2]
                            M = cv.getRotationMatrix2D((w0 / 2, h0 / 2), float(rotate_deg), 1.0)
                            img = cv.warpAffine(
                                img,
                                M,
                                (w0, h0),
                                flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_CONSTANT,
                                borderValue=(0, 0, 0),
                            )
            except Exception:
                img = None
        else:
            if self.debug:
                print(f"[GaugeDebug] image path not found: {self.image_path}")
        if img is None:
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            if self.debug:
                print("[GaugeDebug] fallback white image (100x100) used")
        image_item = pg.ImageItem(img)
        image_item.setZValue(-10)
        if self.debug:
            try:
                print(f"[GaugeDebug] ImageItem created, z={image_item.zValue()}, shape={getattr(img, 'shape', None)}")
            except Exception:
                print("[GaugeDebug] ImageItem created")

        # 画像の既定配置: ビュー範囲の中心に、範囲の90%で配置
        xlim = self.config.get("xlim", (-3, 17))
        ylim = self.config.get("ylim", (8, 22))
        extent = self.config.get("image_extent", [xlim[0], xlim[1], ylim[0], ylim[1]])
        scale = float(self.config.get("image_scale", 0.9))
        cx0 = (extent[0] + extent[1]) / 2.0
        cy0 = (extent[2] + extent[3]) / 2.0
        w0 = (extent[1] - extent[0]) * scale
        h0 = (extent[3] - extent[2]) * scale
        x0 = cx0 - w0 / 2.0
        y0 = cy0 - h0 / 2.0
        try:
            image_item.resetTransform()
            try:
                image_item.setImage(img)
            except Exception:
                pass
            rect = QtCore.QRectF(float(x0), float(y0), float(w0), float(h0))
            image_item.setRect(rect)
            if self.debug:
                print(f"[GaugeDebug] image placed via setRect rect=({x0:.2f},{y0:.2f},{w0:.2f},{h0:.2f}) extent={extent} scaleF={scale}")
        except Exception:
            try:
                sx = w0 / float(img.shape[1])
                sy = h0 / float(img.shape[0])
                image_item.resetTransform()
                image_item.scale(sx, sy)
                image_item.setPos(x0, y0)
                if self.debug:
                    print(f"[GaugeDebug] image placed via scale/setPos at ({x0:.2f},{y0:.2f}) scale=({sx:.3f},{sy:.3f}) extent={extent} scaleF={scale}")
            except Exception:
                try:
                    image_item.setPos(x0, y0)
                    if self.debug:
                        print(f"[GaugeDebug] image setPos only at ({x0:.2f},{y0:.2f}) due to transform error, extent={extent}, scaleF={scale}")
                except Exception:
                    pass
        if self.debug:
            print("[GaugeDebug] add ImageItem to view")
        self.view.addItem(image_item)
        if self.debug:
            try:
                vis = bool(self.win.isVisible())
            except Exception:
                vis = True
            print(f"[GaugeDebug] ImageItem added. windowVisible={vis}")

        self.gauges.clear()
        self.fill_items.clear()
        self.band_items.clear()

        gray = (200, 200, 200)
        for g_conf in self.config.get("gauges", []):
            c = g_conf["center"]
            label = g_conf.get("label", "")
            base = self._make_arc_item(c, self.radius - self.ring_width / 2.0, self.stroke_px, 180.0, 360.0, pen_color=gray)
            base.setZValue(1)
            self.view.addItem(base)
            if self.debug:
                print(f"[GaugeDebug] base arc added at center={c}, radius={self.radius}, stroke_px={self.stroke_px}")

            txt = pg.TextItem(text=label, color=(255, 255, 255), anchor=(0.5, 0.5))
            txt.setZValue(8)
            txt.setPos(c[0], c[1] + 2.1)
            self.view.addItem(txt)

            outline = self._make_arc_item(
                c, self.radius - self.ring_width / 2.0, self.stroke_px, 180.0, 360.0, pen_color=(255, 0, 0)
            )
            outline.setZValue(6)
            outline.setVisible(False)
            self.view.addItem(outline)

            pk_index = len(self.gauges)
            pk = self.part_keys[pk_index] if pk_index < len(self.part_keys) else None
            r, g, b, a = self._fill_colors.get(pk, (0.2, 0.8, 0.2, 0.9))
            color = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))
            fill = self._make_arc_item(c, self.radius - self.ring_width / 2.0, self.stroke_px, 180.0, 180.0, pen_color=color)
            fill.setZValue(5)
            self.view.addItem(fill)
            if self.debug:
                print(f"[GaugeDebug] fill arc added label='{label}' color={color}")

            self.gauges.append({"center": c, "label": txt, "outline": outline})
            self.fill_items.append(fill)
            self.band_items.append(None)

            # 初期シーン構築後にイベントをフラッシュしてウィンドウを確実に表示
            try:
                QApplication.processEvents()
            except Exception:
                pass

    # ----- geometry helpers -----
    def _arc_path(self, c: tuple[float, float], radius: float, theta1_deg: float, theta2_deg: float):
        cx, cy = float(c[0]), float(c[1])
        th1 = float(theta1_deg)
        th2 = float(theta2_deg)
        if th2 < th1:
            th1, th2 = th2, th1
        x = cx - radius
        y = cy - radius
        w = radius * 2.0
        h = radius * 2.0
        path = QtGui.QPainterPath()  # type: ignore[attr-defined]
        path.arcMoveTo(x, y, w, h, th1)
        path.arcTo(x, y, w, h, th1, th2 - th1)
        return path

    def _make_arc_item(
        self,
        c: tuple[float, float],
        radius: float,
        thickness: float,
        theta1_deg: float,
        theta2_deg: float,
        pen_color: Any,
    ) -> Any:
        path = self._arc_path(c, radius, theta1_deg, theta2_deg)
        item = QtWidgets.QGraphicsPathItem(path)  # type: ignore[attr-defined]
        pen = pg.mkPen(pen_color, width=float(thickness))
        try:
            if Qt is not None and hasattr(Qt, "RoundCap"):
                pen.setCapStyle(Qt.RoundCap)  # type: ignore[attr-defined]
                pen.setJoinStyle(Qt.RoundJoin)  # type: ignore[attr-defined]
            # ピクセル幅で描画
            if hasattr(pen, "setCosmetic"):
                pen.setCosmetic(True)
        except Exception:
            pass
        item.setPen(pen)
        return item

    # ----- public API -----
    def update_impulses(self, impulses: dict[str, float]) -> None:
        for k, v in impulses.items():
            if k in self.current_impulses:
                self.current_impulses[k] = float(v)

    def set_direct_ratios(
        self,
        ratios: dict[str, float] | None,
        fill_rgba: tuple[float, float, float, float] | None = None,
    ) -> None:
        if ratios is None:
            self._direct_ratios = None
            return
        self._direct_ratios = {k: float(np.clip(v, 0.0, 1.0)) for k, v in ratios.items() if k in self.part_keys}
        if fill_rgba is not None:
            self._direct_fill_rgba = _to_rgba_tuple(fill_rgba, self._direct_fill_rgba)

    def get_angles(self) -> list[float]:
        if self._direct_ratios is not None:
            return [float(np.clip(180.0 - 120.0 * self._direct_ratios.get(pk, 0.0), 0.0, 180.0)) for pk in self.part_keys]
        if self.warmup_frames > 0 and self._frame_index < self.warmup_frames:
            if self.debug and self._frame_index == 0:
                print(f"[GaugeDebug] warmup active ({self.warmup_frames} frames) -> all 180")
            return [180.0] * len(self.part_keys)
        angles: list[float] = []
        for pk in self.part_keys:
            E = float(self.current_impulses.get(pk, 0.0))
            if pk in self.energy_thresholds:
                e_low, e_high = self.energy_thresholds[pk]
                angle_raw = self._angle_from_energy(E, e_low, e_high)
            else:
                mu, sigma = self.stats.get(pk, (0.0, 1.0))
                sigma_safe = sigma if abs(sigma) > 1e-9 else 1.0
                angle_raw = 60.0 * (E / sigma_safe) + 120.0 - 60.0 * (mu / sigma_safe)
            angles.append(float(np.clip(angle_raw, 0, 180)))
        return angles

    def tick(self) -> None:
        self._frame_index += 1

    @staticmethod
    def _angle_from_energy(E: float, E_low: float, E_high: float) -> float:
        if E_high <= 0 or E_low <= 0 or E_high <= E_low:
            x = 0.0 if E <= 0 else (1.0 if E >= E_high else (E / E_high))
            return 180.0 - 120.0 * x
        if E <= 0.0:
            return 180.0
        if E <= E_low:
            return 180.0 - 60.0 * (E / E_low)
        if E <= E_high:
            return 120.0 - 60.0 * ((E - E_low) / (E_high - E_low))
        return 60.0

    def update(self) -> None:
        angles = self.get_angles()
        for idx, val in enumerate(angles):
            fill = self.fill_items[idx] if idx < len(self.fill_items) else None
            if fill is None:
                continue
            c = self.gauges[idx]["center"]
            pk = self.part_keys[idx]
            E = float(self.current_impulses.get(pk, 0.0))
            if self._direct_ratios is not None:
                color = self._direct_fill_rgba
            elif pk in self.energy_thresholds:
                e_low, e_high = self.energy_thresholds[pk]
                if E <= e_low:
                    color = (0.15, 0.25, 0.95, 0.90)
                elif E <= e_high:
                    color = (0.20, 0.80, 0.20, 0.90)
                else:
                    color = (0.95, 0.20, 0.20, 0.90)
            else:
                if val > 120:
                    color = (0.15, 0.25, 0.95, 0.90)
                elif val > 60:
                    color = (0.20, 0.80, 0.20, 0.90)
                else:
                    color = (0.95, 0.20, 0.20, 0.90)
            r, g, b, a = color
            col = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

            # 差分描画: 角度変化が小さいときはパス更新をスキップ（<0.5度）
            prev = self._last_angles[idx]
            if not (isinstance(prev, float) and abs(prev - float(val)) < 0.5):
                pen = pg.mkPen(col, width=float(self.stroke_px))
                try:
                    if Qt is not None and hasattr(Qt, "RoundCap"):
                        pen.setCapStyle(Qt.RoundCap)  # type: ignore[attr-defined]
                        pen.setJoinStyle(Qt.RoundJoin)  # type: ignore[attr-defined]
                    if hasattr(pen, "setCosmetic"):
                        pen.setCosmetic(True)
                except Exception:
                    pass
                fill.setPen(pen)
                end_deg = 360.0 - float(np.clip(val, 0.0, 180.0))
                path = self._arc_path(c, self.radius - self.ring_width / 2.0, 180.0, end_deg)
                fill.setPath(path)
                self._last_angles[idx] = float(val)

            outline = self.gauges[idx]["outline"]
            warn = val <= 60.0
            outline.setVisible(bool(warn))

        # Qt イベント処理を間引き
        if (self._frame_index % self._event_every) == 0:
            QApplication.processEvents()

    def play_demo_fill(
        self,
        step_seconds: float = 2.0,
        steps: int = 8,
        low_steps: tuple[int, int] = (2, 6),
        high_pct: float = 0.80,
        low_pct: float = 0.15,
        chunk_seconds: float = 0.25,
    ) -> None:
        """Fill gauges in a scripted pattern for demo videos.

        2秒周期×8ステップ=16秒。指定ステップのみ低負荷(15%)、それ以外は高負荷(80%)へ0.25秒ごとの段階的インクリメント。
        """

        def _apply_percent(p: float) -> None:
            ang = float(np.clip(180.0 - 120.0 * np.clip(p, 0.0, 1.0), 0.0, 180.0))
            # 色を割合で切り替え: 20%以下は青、超えると緑
            if p <= 0.20:
                col_rgba = (0.15, 0.25, 0.95, 0.90)
            else:
                col_rgba = (0.20, 0.80, 0.20, 0.90)
            pen_col = (
                int(col_rgba[0] * 255),
                int(col_rgba[1] * 255),
                int(col_rgba[2] * 255),
                int(col_rgba[3] * 255),
            )
            for idx, fill in enumerate(self.fill_items):
                if fill is None:
                    continue
                c = self.gauges[idx]["center"]
                pen = pg.mkPen(pen_col, width=float(self.stroke_px))
                try:
                    if Qt is not None and hasattr(Qt, "RoundCap"):
                        pen.setCapStyle(Qt.RoundCap)  # type: ignore[attr-defined]
                        pen.setJoinStyle(Qt.RoundJoin)  # type: ignore[attr-defined]
                    if hasattr(pen, "setCosmetic"):
                        pen.setCosmetic(True)
                except Exception:
                    pass
                fill.setPen(pen)
                end_deg = 360.0 - ang
                path = self._arc_path(c, self.radius - self.ring_width / 2.0, 180.0, end_deg)
                fill.setPath(path)
            try:
                QApplication.processEvents()
            except Exception:
                pass

        pct_seq = [low_pct if (i + 1) in low_steps else high_pct for i in range(steps)]
        dt_chunk = max(0.01, float(chunk_seconds))
        chunks = max(1, int(round(step_seconds / dt_chunk)))
        for target_pct in pct_seq:
            # 明示リセットを入れて0%を確実に表示
            _apply_percent(0.0)
            _time.sleep(dt_chunk)
            for i in range(chunks):
                alpha = float(i + 1) / float(chunks)
                _apply_percent(target_pct * alpha)
                _time.sleep(dt_chunk)
            _apply_percent(target_pct)

    def set_band_angles(self, angle_map: dict[str, list | tuple]) -> None:
        for idx, pk in enumerate(self.part_keys):
            if pk not in angle_map:
                continue
            val = angle_map[pk]
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                continue
            low = float(np.clip(float(val[0]), 0, 180))
            high = float(np.clip(float(val[1]), 0, 180))
            if high < low:
                low, high = high, low
            th1 = 360.0 - high
            th2 = 360.0 - low
            c = self.gauges[idx]["center"]
            band = self._make_arc_item(
                c, self.radius - self.ring_width / 2.0, self.band_stroke_px, th1, th2, pen_color=(51, 204, 51, 180)
            )
            band.setZValue(3)
            if self.band_items[idx] is not None:
                try:
                    self.view.removeItem(self.band_items[idx])
                except Exception:
                    pass
            self.view.addItem(band)
            self.band_items[idx] = band

    def filter_parts(self, allowed_keys: list[str]) -> None:
        if not allowed_keys:
            return
        new_part_keys: list[str] = []
        new_gauges: list[dict[str, Any]] = []
        new_fill_items: list[Any] = []
        new_band_items: list[Any] = []
        for idx, pk in enumerate(self.part_keys):
            if pk in allowed_keys:
                new_part_keys.append(pk)
                new_gauges.append(self.gauges[idx])
                new_fill_items.append(self.fill_items[idx])
                new_band_items.append(self.band_items[idx])
            else:
                try:
                    self.fill_items[idx].setVisible(False)
                except Exception:
                    pass
                try:
                    self.gauges[idx]["label"].setVisible(False)
                    self.gauges[idx]["outline"].setVisible(False)
                except Exception:
                    pass
                if self.band_items[idx] is not None:
                    try:
                        self.band_items[idx].setVisible(False)
                    except Exception:
                        pass
        self.part_keys = new_part_keys
        self.gauges = new_gauges
        self.fill_items = new_fill_items
        self.band_items = new_band_items
        self.current_impulses = {k: self.current_impulses.get(k, 0.0) for k in self.part_keys}
        self.energy_thresholds = {k: v for k, v in self.energy_thresholds.items() if k in self.part_keys}

    def run(self, frames: int = 100, interval: float = 0.05) -> None:
        import time as _t

        for _ in range(frames):
            self.update()
            _t.sleep(float(interval))
        self.win.show()

    def show(
        self,
        x: int | None = None,
        y: int | None = None,
        w: int | None = None,
        h: int | None = None,
        on_top: bool = True,
        title: str | None = None,
    ) -> None:
        try:
            if title:
                try:
                    self.win.setWindowTitle(str(title))
                except Exception:
                    pass
            if on_top:
                try:
                    if Qt is not None and hasattr(Qt, "WindowStaysOnTopHint"):
                        self.win.setWindowFlags(self.win.windowFlags() | Qt.WindowStaysOnTopHint)  # type: ignore[attr-defined]
                except Exception:
                    pass

            # env overrides for window placement
            try:
                env_x = os.getenv("GAUGE_WIN_X")
                env_y = os.getenv("GAUGE_WIN_Y")
                env_w = os.getenv("GAUGE_WIN_W")
                env_h = os.getenv("GAUGE_WIN_H")
                env_screen = os.getenv("GAUGE_WIN_SCREEN")
                if env_x is not None:
                    x = int(env_x)
                if env_y is not None:
                    y = int(env_y)
                if env_w is not None:
                    w = int(env_w)
                if env_h is not None:
                    h = int(env_h)
            except Exception:
                pass

            # auto-center on chosen screen if any coordinate/size missing
            if not all(v is not None for v in (x, y, w, h)) and QtWidgets is not None:
                try:
                    screens = QtWidgets.QApplication.screens() if hasattr(QtWidgets.QApplication, "screens") else []  # type: ignore[attr-defined]
                    screen_idx = 0
                    try:
                        if env_screen is not None:
                            screen_idx = max(0, min(int(env_screen), len(screens) - 1))
                    except Exception:
                        screen_idx = 0
                    screen = screens[screen_idx] if screens else (QtWidgets.QApplication.primaryScreen() if hasattr(QtWidgets.QApplication, "primaryScreen") else None)  # type: ignore[attr-defined]
                    geo = screen.availableGeometry() if screen is not None else None
                    if geo is not None:
                        gw = geo.width()
                        gh = geo.height()
                        gx = geo.x()
                        gy = geo.y()
                        if w is None or h is None:
                            w = int(gw * 0.6)
                            h = int(gh * 0.6)
                        if x is None:
                            x = int(gx + (gw - w) / 2)
                        if y is None:
                            y = int(gy + (gh - h) / 2)
                        if self.debug:
                            print(f"[GaugeDebug] auto place screen={screen_idx} geom=({gx},{gy},{gw},{gh}) -> ({x},{y},{w},{h})")
                except Exception:
                    pass

            if all(v is not None for v in (x, y, w, h)):
                try:
                    self.win.setGeometry(int(x), int(y), int(w), int(h))
                except Exception:
                    pass
            self.win.show()
            try:
                self.win.raise_()
                self.win.activateWindow()
            except Exception:
                pass
            try:
                QApplication.processEvents()
            except Exception:
                pass
        except Exception:
            pass
