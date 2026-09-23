"""被験者ゲージの描画本体。``app.gauge.scene`` の ``Scene`` を ``QPainter`` で描く。

``constraints.md`` の「速さ」の節（30Hz に余裕を持って追いつく・1 回の描画は
数 ms を目標にする）を満たすため、窓の大きさが変わらない限り描き直さない層
（地・見出し帯とアイコン・人物・溝・部位名・凡例・見出しのタイトル）を、
窓の大きさと devicePixelRatio ごとに ``QPixmap`` へ 1 度だけ描いてキャッシュ
する（``GaugeWidget._ensure_static_pixmap``）。毎フレーム変わる層（帯の色・
値の弧・縁・前回・値の文字・状態・帯の数字・見出しの右端・スピナー）だけを
その上に重ねて描く。

動かない要素と動く要素の区別は ``Scene`` の ``role``（``app.gauge.scene`` の
``Arc``・``Line``・``Label`` が持つ）で決める。``_STATIC_ROLES`` に無い role は
すべて毎フレーム描く。``"part"``（部位名の文字）は controller の「判断済みの
こと」の列挙には無いが、``build_scene`` はどの局面でも部位名を無条件に描く
（``scene.py`` の docstring）ため値や状態に一切依存しない。窓の大きさが
変わらない限り描き直す理由が無い、溝や凡例と同じ条件（窓の大きさ・DPR）で
決まる要素なので、ここでは静止層に含めた（widget タスクの report に理由を
書く）。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.qt import QtCore, QtGui, QtSvg, QtWidgets
from app.core.resources import japanese_font_path
from app.gauge import model as gm
from app.gauge import scene as sc
from app.gauge.protocol import GaugeFrame
from app.shell import pictograms, theme

__all__ = ["paint_scene", "render_image", "GaugeWidget"]


# ---------------------------------------------------------------------------
# 設計座標・レイアウト
# ---------------------------------------------------------------------------

# ゲージの設計座標（scene.py・constraints.md）。
DESIGN_W = 800.0
DESIGN_H = 450.0

# 見出し帯の高さと、見出しアイコンを描く矩形（mk_subject.py の header()・ICON。
# ICON は自分の viewBox=48×48 の中に translate(18,10) scale(0.92) を持っているので、
# widget 側は (0,0,48,48) にそのまま置くだけでよい）。
HEADER_HEIGHT = 64.0
HEADER_ICON_RECT = QtCore.QRectF(0.0, 0.0, 48.0, 48.0)
FIGURE_RECT = QtCore.QRectF(0.0, 0.0, DESIGN_W, DESIGN_H)

# 窓の大きさが変わらない限り描き直さない role（constraints.md「速さ」の節。
# "part" を足した理由はモジュール docstring を参照）。
_STATIC_ROLES = frozenset(
    {
        "groove",
        "part",
        "legend_band",
        "legend_prev",
        "legend_band_label",
        "legend_prev_label",
        "header_title",
    }
)


def _is_static_role(role: str) -> bool:
    return role in _STATIC_ROLES


def _is_dynamic_role(role: str) -> bool:
    return role not in _STATIC_ROLES


def _layout(w: float, h: float) -> tuple[float, float, float]:
    """``(倍率, 横オフセット, 縦オフセット)``。800×450 の設計座標を中央に収める。

    ``s = min(w/800, h/450)`` で拡大し、余白ができる側を地の色で埋めて中央に
    寄せる（task-7-brief.md）。``w``・``h`` が 0 以下（窓がまだ描けない大きさ）
    のときは ``s=0`` を返し、呼び出し側はそれ以上描かない。
    """
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0
    s = min(w / DESIGN_W, h / DESIGN_H)
    ox = (w - DESIGN_W * s) / 2.0
    oy = (h - DESIGN_H * s) / 2.0
    return s, ox, oy


# ---------------------------------------------------------------------------
# 字体
# ---------------------------------------------------------------------------

# 候補の並び（Hiragino Sans → Hiragino Kaku Gothic ProN → 同梱 IPAex ゴシック）。
# 初回だけ addApplicationFont を呼び、以後はキャッシュを返す
# （呼ぶたびにフォントを登録し直すのは無駄なため）。
_FONT_FAMILIES: tuple[str, ...] | None = None


def _font_family_candidates() -> tuple[str, ...]:
    global _FONT_FAMILIES  # pylint: disable=global-statement
    if _FONT_FAMILIES is not None:
        return _FONT_FAMILIES
    families = ["Hiragino Sans", "Hiragino Kaku Gothic ProN"]
    font_id = QtGui.QFontDatabase.addApplicationFont(str(japanese_font_path()))
    if font_id != -1:
        families.extend(QtGui.QFontDatabase.applicationFontFamilies(font_id))
    _FONT_FAMILIES = tuple(families)
    return _FONT_FAMILIES


def _run_font(run: sc.Run) -> QtGui.QFont:
    """``Run`` から ``QFont`` を作る。大きさは ``setPixelSize``（task-7-brief.md）。"""
    font = QtGui.QFont()
    font.setFamilies(list(_font_family_candidates()))
    font.setPixelSize(max(1, round(run.size)))
    font.setWeight(QtGui.QFont.Weight(run.weight))
    return font


# ---------------------------------------------------------------------------
# 人物・見出しアイコンの QSvgRenderer（色の組ごとにキャッシュ）
# ---------------------------------------------------------------------------


class _RendererCache:
    """``pictograms`` の SVG から作った ``QSvgRenderer`` を色の組ごとに持つ。

    被験者ゲージの色は ``app/shell/theme.py`` の定数で固定なので、実際には
    1 通りしか使わないが、呼び出し側が色を変えて呼んでも壊れないようにしておく。
    """

    def __init__(self) -> None:
        self._figures: dict[tuple[str, ...], QtSvg.QSvgRenderer] = {}
        self._icons: dict[tuple[str, ...], QtSvg.QSvgRenderer] = {}

    def figure(
        self,
        *,
        figure: str = theme.TEXT,
        chair: str = theme.CHAIR,
        plate: str = theme.PLATE,
        plate_dark: str = theme.PLATE_DARK,
        sleeve: str = theme.SLEEVE,
        background: str = theme.FIELD,
    ) -> QtSvg.QSvgRenderer:
        key = (figure, chair, plate, plate_dark, sleeve, background)
        renderer = self._figures.get(key)
        if renderer is None:
            svg = pictograms.figure_svg(
                figure=figure, chair=chair, plate=plate, plate_dark=plate_dark, sleeve=sleeve, background=background
            )
            renderer = QtSvg.QSvgRenderer(svg.encode("utf-8"))
            self._figures[key] = renderer
        return renderer

    def icon(self, *, figure: str = theme.TEXT, plate: str = theme.PLATE) -> QtSvg.QSvgRenderer:
        key = (figure, plate)
        renderer = self._icons.get(key)
        if renderer is None:
            svg = pictograms.header_icon_svg(figure=figure, plate=plate)
            renderer = QtSvg.QSvgRenderer(svg.encode("utf-8"))
            self._icons[key] = renderer
        return renderer


_RENDERERS = _RendererCache()


# ---------------------------------------------------------------------------
# 場面の 1 要素を描く
# ---------------------------------------------------------------------------


def _draw_arc(painter: QtGui.QPainter, arc: sc.Arc) -> None:
    """弧を 1 本描く。``QPainterPath.arcMoveTo``/``arcTo`` と FlatCap のペン。"""
    rect = QtCore.QRectF(arc.cx - arc.radius, arc.cy - arc.radius, arc.radius * 2.0, arc.radius * 2.0)
    start_deg, sweep_deg = sc.qt_arc_angles(arc.f0, arc.f1)
    path = QtGui.QPainterPath()
    path.arcMoveTo(rect, start_deg)
    path.arcTo(rect, start_deg, sweep_deg)

    pen = QtGui.QPen(QtGui.QColor(arc.color))
    pen.setWidthF(arc.width)
    pen.setCapStyle(QtCore.Qt.FlatCap)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawPath(path)


def _draw_line(painter: QtGui.QPainter, line: sc.Line) -> None:
    color = QtGui.QColor(line.color)
    if line.alpha < 1.0:
        color.setAlphaF(line.alpha)
    pen = QtGui.QPen(color)
    pen.setWidthF(line.width)
    pen.setCapStyle(QtCore.Qt.RoundCap if line.round_cap else QtCore.Qt.FlatCap)
    painter.setPen(pen)
    painter.drawLine(QtCore.QPointF(line.x0, line.y0), QtCore.QPointF(line.x1, line.y1))


def _draw_label(painter: QtGui.QPainter, label: sc.Label) -> None:
    """``Run`` を左から順に並べる。大きさ違いの幅は ``QFontMetricsF`` で測る。"""
    fonts = [_run_font(run) for run in label.runs]
    widths = [QtGui.QFontMetricsF(font).horizontalAdvance(run.text) for run, font in zip(label.runs, fonts)]
    total_width = sum(widths)

    if label.align == "left":
        x = label.x
    elif label.align == "right":
        x = label.x - total_width
    else:
        x = label.x - total_width / 2.0

    for run, font, width in zip(label.runs, fonts, widths):
        painter.setFont(font)
        painter.setPen(QtGui.QColor(run.color))
        painter.drawText(QtCore.QPointF(x, label.y), run.text)
        x += width


def _draw_spinner(painter: QtGui.QPainter, spinner: sc.Spinner) -> None:
    """Pixel 接続待ちのくるくる。薄い輪（固定）＋回る短い弧（``spinner.phase``）。"""
    painter.save()
    painter.translate(spinner.cx, spinner.cy)

    ring_color = QtGui.QColor(theme.CHAIR)
    ring_color.setAlphaF(0.35)
    ring_pen = QtGui.QPen(ring_color)
    ring_pen.setWidthF(3.0)
    painter.setPen(ring_pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawEllipse(QtCore.QPointF(0.0, 0.0), spinner.radius, spinner.radius)

    painter.rotate(spinner.phase)
    rect = QtCore.QRectF(-spinner.radius, -spinner.radius, spinner.radius * 2.0, spinner.radius * 2.0)
    path = QtGui.QPainterPath()
    path.arcMoveTo(rect, 90.0)
    path.arcTo(rect, 90.0, -90.0)
    arc_pen = QtGui.QPen(QtGui.QColor(theme.TEXT))
    arc_pen.setWidthF(3.0)
    arc_pen.setCapStyle(QtCore.Qt.RoundCap)
    painter.setPen(arc_pen)
    painter.drawPath(path)

    painter.restore()


def _draw_backdrop(
    painter: QtGui.QPainter,
    scene: sc.Scene,
    figure_renderer: QtSvg.QSvgRenderer | None,
    icon_renderer: QtSvg.QSvgRenderer | None,
) -> None:
    """見出し帯・見出しアイコン・人物（``Scene`` に絵そのものを持たない層）。"""
    if scene.header_band:
        painter.fillRect(QtCore.QRectF(0.0, 0.0, DESIGN_W, HEADER_HEIGHT), QtGui.QColor(theme.HEADER))
        if icon_renderer is not None and icon_renderer.isValid():
            icon_renderer.render(painter, HEADER_ICON_RECT)
    if scene.show_figure and figure_renderer is not None and figure_renderer.isValid():
        figure_renderer.render(painter, FIGURE_RECT)


def _draw_scene_elements(painter: QtGui.QPainter, scene: sc.Scene, role_ok) -> None:
    """``role_ok(role)`` が真の要素だけを、弧→線→文字の順で描く。

    弧→線→文字の順は ``mk_subject.py`` の描く順（形が先、文字は最後に重ねる）
    と、``build_scene`` が同じ部位の中で積む順（溝→帯→値の弧→…→前回の目盛り）
    に合わせてある。線（前回の目盛り）は弧の輪の上に重ねて見せる印なので弧の
    後、文字はどの図形より手前に来る必要があるので最後に描く。
    """
    for arc in scene.arcs:
        if role_ok(arc.role):
            _draw_arc(painter, arc)
    for line in scene.lines:
        if role_ok(line.role):
            _draw_line(painter, line)
    for label in scene.labels:
        if role_ok(label.role):
            _draw_label(painter, label)


# ---------------------------------------------------------------------------
# 公開の描画関数
# ---------------------------------------------------------------------------


def paint_scene(
    painter: QtGui.QPainter,
    scene: sc.Scene,
    w: float,
    h: float,
    figure_renderer: QtSvg.QSvgRenderer,
    icon_renderer: QtSvg.QSvgRenderer,
) -> None:
    """``scene`` を 1 回で全部描く（キャッシュしない）。試験・``render_image`` 用。

    ``GaugeWidget`` はこれをそのまま使わず、動かない層と動く層を分けて描く
    （モジュール docstring）。地の色は余白（レターボックス）も含めた ``w×h``
    全体を塗る（``constraints.md`` の色は地 1 色しか無く、余白も地の色で
    埋めるという brief の指示に合わせる）。
    """
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
    painter.fillRect(QtCore.QRectF(0.0, 0.0, float(w), float(h)), QtGui.QColor(theme.FIELD))

    s, ox, oy = _layout(w, h)
    if s <= 0:
        return

    painter.save()
    painter.translate(ox, oy)
    painter.scale(s, s)
    _draw_backdrop(painter, scene, figure_renderer, icon_renderer)
    _draw_scene_elements(painter, scene, lambda _role: True)
    if scene.spinner is not None:
        _draw_spinner(painter, scene.spinner)
    painter.restore()


def render_image(state: gm.GaugeState, w: int, h: int) -> QtGui.QImage:
    """1 フレームぶんをオフスクリーンの ``QImage`` に描く。スナップショット・試験用。"""
    scene = sc.build_scene(state, spinner_phase=0.0)
    image = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor(theme.FIELD))
    painter = QtGui.QPainter(image)
    try:
        paint_scene(painter, scene, w, h, _RENDERERS.figure(), _RENDERERS.icon())
    finally:
        painter.end()
    return image


# ---------------------------------------------------------------------------
# GaugeWidget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StaticKey:
    w: int
    h: int
    dpr: float


class GaugeWidget(QtWidgets.QWidget):
    """被験者ゲージの窓の中身。動かない層を ``QPixmap`` にキャッシュして描く。

    公開の口は ``set_frame``・``set_show_joules``・``finish``・``reset``・
    ``state`` プロパティ（controller の「判断済みのこと」）。
    """

    _SPINNER_INTERVAL_MS = 60
    _SPINNER_STEP_DEG = 24.0

    def __init__(self, *, show_joules: bool = True, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = gm.reset(show_joules=show_joules)
        self._spinner_phase = 0.0
        self._scene = sc.build_scene(self._state, spinner_phase=self._spinner_phase)

        self._static_pixmap: QtGui.QPixmap | None = None
        self._static_key: _StaticKey | None = None
        # 試験（test_static_layer_is_cached）が、動かない層を作り直した回数を
        # 数えられるようにするための内部カウンタ。公開の口ではない。
        self._static_builds = 0

        self._spinner_timer = QtCore.QTimer(self)
        self._spinner_timer.setInterval(self._SPINNER_INTERVAL_MS)
        self._spinner_timer.timeout.connect(self._advance_spinner)
        self._sync_spinner_timer()

    # -- 公開の口 ------------------------------------------------------------

    @property
    def state(self) -> gm.GaugeState:
        return self._state

    def set_frame(self, frame: GaugeFrame) -> None:
        self._apply_state(gm.apply_frame(self._state, frame))

    def set_show_joules(self, show_joules: bool) -> None:
        self._apply_state(gm.with_joules(self._state, show_joules))

    def finish(self, exit_code: int) -> None:
        self._apply_state(gm.finish(self._state, exit_code))

    def reset(self) -> None:
        """WAITING へ戻る。J 表示の有無は今の設定を保つ（呼び出し側が明示的に
        変えたいときは、続けて ``set_show_joules`` を呼ぶ想定。``GaugeWindow.begin``
        は ``reset`` の直後に ``set_show_joules`` を呼ぶので、ここでの値は
        上書きされる前提でも構わない）。
        """
        self._apply_state(gm.reset(self._state.show_joules))

    # -- 内部: 状態・場面 ------------------------------------------------------

    def _apply_state(self, state: gm.GaugeState) -> None:
        self._state = state
        self._rebuild_scene()
        self._sync_spinner_timer()
        self.update()

    def _rebuild_scene(self) -> None:
        self._scene = sc.build_scene(self._state, spinner_phase=self._spinner_phase)

    def _advance_spinner(self) -> None:
        self._spinner_phase = (self._spinner_phase + self._SPINNER_STEP_DEG) % 360.0
        self._rebuild_scene()
        self.update()

    def _sync_spinner_timer(self) -> None:
        """スピナーの QTimer は、WAITING で表示中のときだけ動かす（brief）。"""
        should_run = self._state.phase is gm.Phase.WAITING and self.isVisible()
        if should_run and not self._spinner_timer.isActive():
            self._spinner_timer.start()
        elif not should_run and self._spinner_timer.isActive():
            self._spinner_timer.stop()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 (Qt の命名)
        super().showEvent(event)
        self._sync_spinner_timer()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802 (Qt の命名)
        super().hideEvent(event)
        self._sync_spinner_timer()

    # -- 内部: 描画 ------------------------------------------------------------

    def _ensure_static_pixmap(self, w: int, h: int, dpr: float) -> None:
        key = _StaticKey(w, h, dpr)
        if self._static_pixmap is not None and self._static_key == key:
            return
        self._static_pixmap = self._build_static_pixmap(w, h, dpr)
        self._static_key = key
        self._static_builds += 1

    def _build_static_pixmap(self, w: int, h: int, dpr: float) -> QtGui.QPixmap:
        physical = QtCore.QSize(max(1, round(w * dpr)), max(1, round(h * dpr)))
        pixmap = QtGui.QPixmap(physical)
        pixmap.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(pixmap)
        try:
            painter.scale(dpr, dpr)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
            painter.fillRect(QtCore.QRectF(0.0, 0.0, float(w), float(h)), QtGui.QColor(theme.FIELD))

            s, ox, oy = _layout(w, h)
            if s > 0:
                painter.save()
                painter.translate(ox, oy)
                painter.scale(s, s)
                _draw_backdrop(painter, self._scene, _RENDERERS.figure(), _RENDERERS.icon())
                _draw_scene_elements(painter, self._scene, _is_static_role)
                painter.restore()
        finally:
            painter.end()

        pixmap.setDevicePixelRatio(dpr)
        return pixmap

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt の命名)
        w, h = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        self._ensure_static_pixmap(w, h, dpr)

        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self._static_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

        s, ox, oy = _layout(w, h)
        if s > 0:
            painter.save()
            painter.translate(ox, oy)
            painter.scale(s, s)
            _draw_scene_elements(painter, self._scene, _is_dynamic_role)
            if self._scene.spinner is not None:
                _draw_spinner(painter, self._scene.spinner)
            painter.restore()
