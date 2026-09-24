"""被験者ゲージの描画本体（``app.gauge.widget``）の試験。

``app/gauge/scene.py`` が組み立てた ``Scene``（Qt に依存しない場面の値）を
``QPainter``／``QtSvg`` で実際に描く層を検証する。見た目は
``render_image`` が返す ``QImage`` から特定の座標の色を拾って確かめる
（座標は ``app/gauge/scene.py`` の ``CENTERS``・``R``・``RB`` など、正本の
数値をそのまま使う）。``render_image`` は 800×450（設計座標そのもの）で
呼ぶことが多く、そのときは倍率 ``s=1`` で座標が 1:1 に対応するので、
``_layout`` の式を試験側で再現しなくて済む（レターボックスの試験だけ
別の大きさで呼ぶ）。
"""

from __future__ import annotations

import math
import time

import pytest

from app.gauge import model as gm
from app.gauge import scene as sc
from app.gauge.protocol import GaugeFrame, PartReading
from app.shell import theme

pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")


@pytest.fixture(scope="module")
def qt_app():
    from app.core.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


BAND = (80.0, 100.0)


def _frame(parts: dict[str, PartReading], *, link: str = "connected", rep: int = 0, source: str = "measure") -> GaugeFrame:
    return GaugeFrame(link=link, rep=rep, source=source, parts=parts)


def _running_state(parts: dict[str, PartReading], *, show_joules: bool = True, rep: int = 3) -> gm.GaugeState:
    state = gm.reset(show_joules=show_joules)
    return gm.apply_frame(state, _frame(parts, rep=rep))


def _waiting_state(parts: dict[str, PartReading] | None = None, *, show_joules: bool = True) -> gm.GaugeState:
    state = gm.reset(show_joules=show_joules)
    if parts is None:
        return state
    return gm.apply_frame(state, _frame(parts, link="waiting"))


def _pt(cx: float, cy: float, r: float, f: float) -> tuple[int, int]:
    """``scene._point_on_circle`` と同じ式。割合 ``f`` の位置の画素座標。"""
    angle = math.radians(180.0 + 180.0 * f)
    return round(cx + r * math.cos(angle)), round(cy + r * math.sin(angle))


class TestPaintScenePixels:
    """``render_image`` を 800×450（s=1）で呼び、設計座標のまま画素を拾う。"""

    def test_renders_offscreen(self, qt_app):
        from app.core.qt import QtGui
        from app.gauge.widget import render_image

        parts = {"elbow_L": PartReading(now=42.0, prev=30.0, band=BAND)}
        state = _running_state(parts)
        image = render_image(state, 800, 450)

        assert isinstance(image, QtGui.QImage)
        assert image.width() == 800
        assert image.height() == 450
        # 地の色 1 色だけで終わっていないこと（何かしら描けている）の粗い確認。
        colors = {image.pixelColor(x, 225).name() for x in range(0, 800, 20)}
        assert len(colors) > 1

    def test_over_has_red_pixels_in_that_dial(self, qt_app):
        from app.gauge.widget import render_image

        parts = {"elbow_R": PartReading(now=120.0, prev=None, band=BAND)}  # >=100 で過負荷
        state = _running_state(parts)
        image = render_image(state, 800, 450)

        cx, cy = sc.CENTERS["elbow_R"]
        x, y = _pt(cx, cy, sc.R, 0.5)  # 値の弧の範囲内（f=0.96 まで描かれる）
        assert image.pixelColor(x, y).name() == theme.OVER

    def test_in_band_has_bright_blue_pixels(self, qt_app):
        from app.gauge.widget import render_image

        parts = {"elbow_L": PartReading(now=90.0, prev=None, band=BAND)}  # 80<=90<100
        state = _running_state(parts)
        image = render_image(state, 800, 450)

        lo, hi = BAND
        fl = max(0.0, min(1.0, lo / (1.25 * hi)))
        fh = max(0.0, min(1.0, hi / (1.25 * hi)))
        cx, cy = sc.CENTERS["elbow_L"]
        x, y = _pt(cx, cy, sc.RB, (fl + fh) / 2.0)
        assert image.pixelColor(x, y).name() == theme.BAND_ON

    def test_waiting_has_no_white_arc_pixels(self, qt_app):
        from app.gauge.widget import render_image

        parts = {"elbow_L": PartReading(now=None, prev=None, band=BAND)}
        state = _waiting_state(parts)
        image = render_image(state, 800, 450)

        cx, cy = sc.CENTERS["elbow_L"]
        for f in (0.1, 0.3, 0.5, 0.7, 0.9):
            x, y = _pt(cx, cy, sc.R, f)
            color = image.pixelColor(x, y).name()
            assert color != theme.VALUE, f"f={f} で値の弧の白が出ている"
            assert color != theme.OVER, f"f={f} で過負荷の赤が出ている"

    def test_letterbox_is_field_color(self, qt_app):
        from app.gauge.widget import render_image

        state = _running_state({"elbow_L": PartReading(now=10.0, prev=None, band=BAND)})
        image = render_image(state, 1920, 1200)

        # s = min(1920/800, 1200/450) = 2.4（幅基準）。縦に余白ができる。
        assert image.pixelColor(960, 5).name() == theme.FIELD
        assert image.pixelColor(960, 1195).name() == theme.FIELD

    def test_figure_is_drawn(self, qt_app):
        from app.gauge.widget import render_image

        state = _waiting_state()
        image = render_image(state, 800, 450)

        # 頭（figure_svg の head, cx=400 cy=118 r=17）の中心は人物の色（白）。
        assert image.pixelColor(400, 118).name() == theme.TEXT


class TestGaugeWidgetPublicApi:
    def test_setters_update_state(self, qt_app):
        from app.gauge.widget import GaugeWidget

        widget = GaugeWidget(show_joules=True)
        assert widget.state.phase is gm.Phase.WAITING

        frame = _frame({"elbow_L": PartReading(now=1.0)}, link="connected", rep=2)
        widget.set_frame(frame)
        assert widget.state.phase is gm.Phase.RUNNING
        assert widget.state.frame is frame

        widget.set_show_joules(False)
        assert widget.state.show_joules is False

        widget.finish(0)
        assert widget.state.phase is gm.Phase.DONE

        widget.reset()
        assert widget.state.phase is gm.Phase.WAITING
        assert widget.state.frame is None
        # J 表示の有無は reset の前の値を保つ。
        assert widget.state.show_joules is False

    def test_spinner_timer_runs_only_while_waiting(self, qt_app):
        from app.gauge.widget import GaugeWidget

        widget = GaugeWidget(show_joules=True)
        try:
            assert widget.state.phase is gm.Phase.WAITING
            assert widget._spinner_timer.isActive() is False  # まだ表示していない

            widget.show()
            qt_app.processEvents()
            assert widget.isVisible() is True
            assert widget._spinner_timer.isActive() is True

            widget.set_frame(_frame({}, link="connected", rep=0))
            assert widget.state.phase is gm.Phase.RUNNING
            assert widget._spinner_timer.isActive() is False

            widget.set_frame(_frame({}, link="waiting", rep=0))
            assert widget.state.phase is gm.Phase.WAITING
            assert widget._spinner_timer.isActive() is True

            widget.hide()
            qt_app.processEvents()
            assert widget._spinner_timer.isActive() is False
        finally:
            widget.deleteLater()

    def test_check_and_cross_glyphs_are_available(self, qt_app):
        from app.core.qt import QtGui
        from app.gauge import fonts as gauge_fonts

        font = gauge_fonts.font_set().font("text", 16, 700)  # 状態の文字（役 text）の書体
        metrics = QtGui.QFontMetrics(font)
        # model.status_label が返す "✓ 目標帯"・"✕ 過負荷" の記号。
        assert metrics.inFontUcs4(0x2713) is True  # ✓
        assert metrics.inFontUcs4(0x2715) is True  # ✕


class TestGaugeWidgetPaintEvent:
    """``GaugeWidget.paintEvent`` そのもの（静止層の ``QPixmap`` キャッシュの上に
    動く層を重ねる、実際に画面へ出る経路）の画素を確かめる。

    ``TestPaintScenePixels`` は ``render_image``／``paint_scene``（キャッシュを
    使わない別経路。``paint_scene`` は毎回 role を問わず全部描く）だけを見て
    おり、``paintEvent`` の経路（``_ensure_static_pixmap`` のキャッシュ→
    ``drawPixmap``→動く層を重ねる）を実際に描いて確かめる試験が無かった
    （レビュー指摘）。``widget.repaint()`` で同期的に描かせたあと
    ``widget.grab().toImage()`` で画素を拾う。800×450 にしているのは、
    ``TestPaintScenePixels`` と同じく ``s=1`` で設計座標と画素が 1:1に
    対応し、``_layout`` の式を試験側で再現しなくて済むため。
    """

    def test_paint_event_draws_expected_pixels(self, qt_app):
        from app.gauge.widget import GaugeWidget

        parts = {
            "elbow_L": PartReading(now=90.0, prev=None, band=BAND),  # 80<=90<100 → 帯が点灯
            "elbow_R": PartReading(now=120.0, prev=None, band=BAND),  # >=100 → 過負荷
        }
        frame = _frame(parts, rep=3)

        widget = GaugeWidget(show_joules=True)
        try:
            widget.resize(800, 450)
            widget.show()
            qt_app.processEvents()
            widget.set_frame(frame)
            widget.repaint()
            image = widget.grab().toImage()

            lo, hi = BAND
            fl = max(0.0, min(1.0, lo / (1.25 * hi)))
            fh = max(0.0, min(1.0, hi / (1.25 * hi)))
            cx, cy = sc.CENTERS["elbow_L"]
            x, y = _pt(cx, cy, sc.RB, (fl + fh) / 2.0)
            assert image.pixelColor(x, y).name() == theme.BAND_ON  # 動く層（帯の色）

            cx, cy = sc.CENTERS["elbow_R"]
            x, y = _pt(cx, cy, sc.R, 0.5)
            assert image.pixelColor(x, y).name() == theme.OVER  # 動く層（値の弧）

            # 静止層（QPixmap キャッシュ）側の画素も見ておく: 人物の頭。
            assert image.pixelColor(400, 118).name() == theme.TEXT
        finally:
            widget.deleteLater()

    def test_paint_event_matches_render_image(self, qt_app):
        from app.core.qt import QtGui
        from app.gauge.widget import GaugeWidget, render_image

        parts = {
            "elbow_L": PartReading(now=90.0, prev=70.0, band=BAND),
            "elbow_R": PartReading(now=120.0, prev=None, band=BAND),
        }
        frame = _frame(parts, rep=3)

        widget = GaugeWidget(show_joules=True)
        try:
            widget.resize(800, 450)
            widget.show()
            qt_app.processEvents()
            widget.set_frame(frame)
            widget.repaint()
            painted = widget.grab().toImage().convertToFormat(QtGui.QImage.Format_ARGB32)

            expected = render_image(widget.state, 800, 450).convertToFormat(QtGui.QImage.Format_ARGB32)

            # 4px おきに標本を取って比較する（全画素の突き合わせは遅いだけで
            # 得るものが無い）。静止層（キャッシュ経由）と動く層（毎回重ねる）
            # の両方を通る経路が、キャッシュを使わない paint_scene と同じ画を
            # 描けているかどうかがここでの関心事。
            mismatches = 0
            for x in range(0, 800, 4):
                for y in range(0, 450, 4):
                    if painted.pixelColor(x, y) != expected.pixelColor(x, y):
                        mismatches += 1
            assert mismatches == 0, f"paintEvent と render_image で {mismatches} 点ずれた（4px おきの標本）"
        finally:
            widget.deleteLater()


class TestGaugeWidgetPerformance:
    def test_static_layer_is_cached(self, qt_app):
        from app.gauge.widget import GaugeWidget

        widget = GaugeWidget(show_joules=True)
        try:
            widget.resize(1920, 1080)
            widget.show()
            qt_app.processEvents()

            widget.repaint()
            widget.repaint()
            assert widget._static_builds == 1

            widget.resize(1000, 700)
            qt_app.processEvents()
            widget.repaint()
            assert widget._static_builds == 2

            widget.repaint()
            assert widget._static_builds == 2
        finally:
            widget.deleteLater()

    def test_static_layer_is_rebuilt_when_the_font_preset_changes(self, qt_app):
        from app.gauge.widget import GaugeWidget

        widget = GaugeWidget(show_joules=True, font_preset="system")
        try:
            widget.resize(800, 450)
            widget.show()
            qt_app.processEvents()
            widget.repaint()
            assert widget._static_builds == 1

            widget.set_font_preset(" System ")
            widget.repaint()
            assert widget._static_builds == 1, "同じ組なら動かない層を作り直さない"

            widget.set_font_preset("tsukushi")
            widget.repaint()
            assert widget._static_builds == 2
            assert widget.font_set.name == "tsukushi"
        finally:
            widget.deleteLater()

    def test_repaint_is_fast_enough(self, qt_app):
        from app.gauge.widget import GaugeWidget

        widget = GaugeWidget(show_joules=True)
        try:
            widget.resize(1920, 1080)
            widget.show()
            qt_app.processEvents()

            frame = _frame(
                {
                    "elbow_L": PartReading(now=190.0, prev=175.0, band=(175.0, 212.5)),
                    "wrist_L": PartReading(now=60.0, prev=55.0, band=(112.0, 136.0)),
                    "elbow_R": PartReading(now=240.0, prev=200.0, band=(175.0, 212.5)),
                    "wrist_R": PartReading(now=120.0, prev=100.0, band=(112.0, 136.0)),
                },
                rep=6,
            )
            widget.set_frame(frame)
            widget.repaint()  # 1 回分は静止層の初回構築を含むので測定から外す

            iterations = 60
            start = time.perf_counter()
            for _ in range(iterations):
                widget.repaint()
            elapsed = time.perf_counter() - start

            average_ms = (elapsed / iterations) * 1000.0
            assert average_ms < 15.0, f"1 回あたり平均 {average_ms:.3f}ms（上限 15ms）"
        finally:
            widget.deleteLater()
