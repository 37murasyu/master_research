"""人物ピクトグラム（``app.shell.pictograms``）の試験。

正本は画面案の生成スクリプト ``mk_subject.py`` の ``FIG``・``ICON``（最新版）。
そこでは CSS の class と ``var(--x)`` で色を持つが、QtSvg（SVG Tiny 1.2 相当）は
CSS 変数を解さないので、ここで作る SVG は色を引数の属性に直書きする
（class や var は 1 つも出てこない）。

``figure_svg`` は椅子・プレート・人物をひと続きに描く。胴と脚を切り離さないのは
脊髄損傷のある人への配慮で、constraints.md の「人物」の節が絶対に守ると定める
ところなので、描く順（id の出現順）と「肩から足先まで地の色の画素が無い」こと
（見た目のつながり）の両方を試験にする。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from app.shell import pictograms
from app.shell import theme

SVG_NS = "{http://www.w3.org/2000/svg}"

FIGURE_COLORS = dict(
    figure=theme.TEXT,
    chair=theme.CHAIR,
    plate=theme.PLATE,
    plate_dark=theme.PLATE_DARK,
    sleeve=theme.SLEEVE,
    background=theme.FIELD,
)

DRAW_ORDER = [
    "chair-pads",
    "plate-l",
    "plate-r",
    "sleeves",
    "leg-knockout",
    "seat",
    "head",
    "shoulders",
    "torso",
    "arms",
    "knees",
    "shins",
]


def _ids_in_order(svg: str) -> list[str]:
    root = ET.fromstring(svg)
    return [el.get("id") for el in root.iter() if el.get("id") is not None]


def _find(svg: str, element_id: str) -> ET.Element:
    root = ET.fromstring(svg)
    for el in root.iter():
        if el.get("id") == element_id:
            return el
    raise AssertionError(f"id={element_id!r} の要素が見つからない: {svg}")


# ---------------------------------------------------------------------------
# 形の骨格（Qt に依存しない）
# ---------------------------------------------------------------------------


def test_svgs_are_well_formed():
    figure = pictograms.figure_svg(**FIGURE_COLORS)
    icon = pictograms.header_icon_svg(figure=theme.TEXT, plate=theme.PLATE)

    figure_root = ET.fromstring(figure)
    icon_root = ET.fromstring(icon)

    assert figure_root.tag == f"{SVG_NS}svg"
    assert figure_root.get("viewBox") == "0 0 800 450"
    assert icon_root.tag == f"{SVG_NS}svg"
    assert icon_root.get("viewBox") == "0 0 48 48"


def test_draw_order_is_chair_plates_knockout_seat_body():
    svg = pictograms.figure_svg(**FIGURE_COLORS)
    assert _ids_in_order(svg) == DRAW_ORDER


def test_colors_come_from_arguments():
    colors = dict(FIGURE_COLORS)
    colors.update(
        figure="#111111",
        chair="#222222",
        plate="#333333",
        plate_dark="#444444",
        sleeve="#555555",
        background="#666666",
    )
    svg = pictograms.figure_svg(**colors)

    assert _find(svg, "leg-knockout").get("stroke") == colors["background"]
    assert _find(svg, "chair-pads").get("stroke") == colors["chair"]
    assert _find(svg, "seat").get("stroke") == colors["chair"]
    assert _find(svg, "sleeves").get("stroke") == colors["sleeve"]

    plate_l = _find(svg, "plate-l")
    plate_r = _find(svg, "plate-r")
    for plate_el in (plate_l, plate_r):
        rects = list(plate_el)
        assert rects[0].get("fill") == colors["plate"]
        assert rects[1].get("fill") == colors["plate_dark"]

    assert _find(svg, "head").get("fill") == colors["figure"]
    for element_id in ("shoulders", "arms", "knees", "shins"):
        assert _find(svg, element_id).get("stroke") == colors["figure"]
    torso = _find(svg, "torso")
    assert torso.get("fill") == colors["figure"]
    assert torso.get("stroke") == colors["figure"]

    icon = pictograms.header_icon_svg(figure="#111111", plate="#333333")
    assert 'stroke="#111111"' in icon
    assert 'fill="#333333"' in icon


def test_no_css_classes_or_vars():
    figure = pictograms.figure_svg(**FIGURE_COLORS)
    icon = pictograms.header_icon_svg(figure=theme.TEXT, plate=theme.PLATE)
    for svg in (figure, icon):
        assert "class=" not in svg
        assert "var(" not in svg


# ---------------------------------------------------------------------------
# 見た目のつながり（QtSvg で実際に描いて確かめる）
#
# importorskip と PySide6 の import は、Qt が要る試験だけが実行する
# qt_app フィクスチャの中に閉じる（tests/test_hybrid_gui.py の
# test_input_switch_disables_during_run が前例）。モジュール直下に置くと、
# PySide6 の無い環境でファイル全体が collection の時点でスキップになり、
# その手前にある Qt に依存しない 4 件（test_svgs_are_well_formed 等）まで
# 消えてしまう。
# ---------------------------------------------------------------------------

SCALE = 4  # 800x450 を 3200x1800 で描く

# FIG の値そのもの（constraints.md の「人物」の節が絶対に守る一続きを検査する）。
SHOULDER_Y = 150.0  # id="shoulders" の d="M362 150H438"
TORSO_CENTER_X = 400.0  # id="torso" の d="M372 150H428L416 256H384Z" の中心
KNEE_Y = 266.0  # id="knees" の d="M386 266H414"
LEFT_SHIN_X = 387.0  # id="shins" の d="...M386 266L387 392..."（線の中心寄り）
RIGHT_SHIN_X = 413.0  # id="shins" の d="...M414 266L413 392"（線の中心寄り）
FOOT_Y = 392.0  # id="shins" の終点の y
SEAT_Y = 302.0  # id="seat" の d="M370 302H430"


def _render(svg: str, background: str):
    from PySide6 import QtSvg
    from app.core.qt import QtCore, QtGui

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    assert renderer.isValid(), "SVG を解釈できない"
    width, height = int(800 * SCALE), int(450 * SCALE)
    image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor(background))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()
    return image


def _column(image, x: float, y0: float, y1: float, background) -> None:
    """列（x 固定）を y0 から y1 まで走査し、地の色の画素が無いことを確かめる。"""
    from app.core.qt import QtGui

    px = round(x * SCALE)
    field = QtGui.QColor(background)
    y = y0
    while y <= y1:
        py = round(y * SCALE)
        color = image.pixelColor(px, py)
        assert color != field, f"x={x}, y={y} が地の色になっている"
        y += 1.0


@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")
    from app.core.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_figure_is_continuous_from_shoulders_to_feet(qt_app):
    svg = pictograms.figure_svg(**FIGURE_COLORS)
    image = _render(svg, theme.FIELD)

    # 胴の中心の列: 肩から膝まで。
    _column(image, TORSO_CENTER_X, SHOULDER_Y, KNEE_Y, theme.FIELD)
    # 両すねの中心の列: 膝から足先まで。
    _column(image, LEFT_SHIN_X, KNEE_Y, FOOT_Y, theme.FIELD)
    _column(image, RIGHT_SHIN_X, KNEE_Y, FOOT_Y, theme.FIELD)


def test_seat_passes_behind_the_legs(qt_app):
    from app.core.qt import QtGui

    svg = pictograms.figure_svg(**FIGURE_COLORS)
    image = _render(svg, theme.FIELD)

    figure_color = QtGui.QColor(theme.TEXT)
    for x in (LEFT_SHIN_X, RIGHT_SHIN_X):
        px, py = round(x * SCALE), round(SEAT_Y * SCALE)
        assert image.pixelColor(px, py) == figure_color
