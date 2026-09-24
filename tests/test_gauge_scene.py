"""場面の計算（``app.gauge.scene``）を検証する。

``app/gauge/scene.py`` は Qt に依存しない純粋なモジュールで、``model.GaugeState``
から「どこに何を描くか」（``Arc``・``Line``・``Label``・``Spinner``）を組み立てる。
次の widget タスクがこれを QPainter で描く。数値・色は
``.superpowers/sdd/2026-09-24-subject-gauge/task-6-brief.md`` と、その正本である
``mk_subject.py``（画面案の生成スクリプト）に合わせる。
"""

from __future__ import annotations

import math

import pytest

from app.gauge import model as gm
from app.gauge import scene as sc
from app.gauge.protocol import GaugeFrame, PartReading
from app.shell import theme


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


# ---------------------------------------------------------------------------
# 位置（鏡の対応）
# ---------------------------------------------------------------------------


def test_dials_are_mirrored():
    state = _waiting_state()
    scene = sc.build_scene(state)

    grooves = {arc.part: arc for arc in scene.find("groove")}
    assert grooves["elbow_L"].cx == 165.0
    assert grooves["elbow_R"].cx == 635.0
    assert grooves["wrist_L"].cx == 165.0
    assert grooves["wrist_R"].cx == 635.0
    # 左右で中心 x を足すと画面幅 800 になる（鏡の対応）。
    assert grooves["elbow_L"].cx + grooves["elbow_R"].cx == 800.0
    assert grooves["wrist_L"].cx + grooves["wrist_R"].cx == 800.0
    # 同じ関節どうしは y が同じ。
    assert grooves["elbow_L"].cy == grooves["elbow_R"].cy == 196.0
    assert grooves["wrist_L"].cy == grooves["wrist_R"].cy == 348.0
    # 本体の弧は R=62 で共通。
    assert grooves["elbow_L"].radius == grooves["wrist_R"].radius == 62.0


# ---------------------------------------------------------------------------
# 局面ごとの描き分け
# ---------------------------------------------------------------------------


def test_waiting_shows_groove_and_band_only():
    parts = {"elbow_L": PartReading(now=None, prev=None, band=BAND)}
    state = _waiting_state(parts, show_joules=True)
    scene = sc.build_scene(state)

    assert len(scene.find("groove", "elbow_L")) == 1
    assert len(scene.find("band", "elbow_L")) == 1
    assert len(scene.find("band_label", "elbow_L")) == 2  # J オンなので帯の数字も
    assert scene.find("value", "elbow_L") == []
    assert scene.find("value_rim", "elbow_L") == []
    assert scene.find("prev", "elbow_L") == []
    assert scene.find("state", "elbow_L") == []


def test_first_rep_has_value_but_no_prev_tick():
    parts = {"elbow_L": PartReading(now=42.0, prev=None, band=BAND)}
    state = _running_state(parts)
    scene = sc.build_scene(state)

    assert len(scene.find("value", "elbow_L")) == 1
    assert len(scene.find("value_rim", "elbow_L")) == 1
    assert scene.find("prev", "elbow_L") == []


def test_nth_rep_has_value_and_prev_tick():
    parts = {"elbow_L": PartReading(now=42.0, prev=30.0, band=BAND)}
    state = _running_state(parts)
    scene = sc.build_scene(state)

    assert len(scene.find("value", "elbow_L")) == 1
    assert len(scene.find("prev", "elbow_L")) == 1


def test_in_band_lights_the_band_and_keeps_white_arc():
    parts = {"elbow_L": PartReading(now=90.0, prev=None, band=BAND)}  # 80<=90<100
    state = _running_state(parts)
    scene = sc.build_scene(state)

    band_arc = scene.find("band", "elbow_L")[0]
    value_arc = scene.find("value", "elbow_L")[0]
    state_label = scene.find("state", "elbow_L")[0]

    assert band_arc.color == theme.BAND_ON
    assert value_arc.color == theme.VALUE
    assert state_label.runs[0].text == "✓ 目標帯"
    assert state_label.runs[0].color == theme.TEXT


def test_over_draws_red_arc_and_label_with_band_unlit():
    parts = {"elbow_L": PartReading(now=120.0, prev=None, band=BAND)}  # >=100
    state = _running_state(parts)
    scene = sc.build_scene(state)

    band_arc = scene.find("band", "elbow_L")[0]
    value_arc = scene.find("value", "elbow_L")[0]
    state_label = scene.find("state", "elbow_L")[0]

    assert band_arc.color == theme.BAND  # 帯は点灯しない
    assert value_arc.color == theme.OVER
    assert state_label.runs[0].text == "✕ 過負荷"
    assert state_label.runs[0].color == theme.OVER


def test_short_has_no_state_label():
    parts = {"elbow_L": PartReading(now=50.0, prev=None, band=BAND)}  # <80
    state = _running_state(parts)
    scene = sc.build_scene(state)

    assert scene.find("state", "elbow_L") == []
    # 値の弧そのものは描く（不足でも数字は見える）。
    assert len(scene.find("value", "elbow_L")) == 1


def test_value_arc_has_field_rim_of_2():
    parts = {"elbow_L": PartReading(now=90.0, prev=None, band=BAND)}
    state = _running_state(parts)
    scene = sc.build_scene(state)

    rim = scene.find("value_rim", "elbow_L")[0]
    value = scene.find("value", "elbow_L")[0]

    assert rim.color == theme.FIELD
    # 縁は幅 W、値の弧は幅 W-4（片側 2px）。
    assert rim.width - value.width == 4.0
    assert rim.f0 == value.f0
    assert rim.f1 == value.f1


def test_arc_ends_are_flat():
    parts = {"elbow_L": PartReading(now=90.0, prev=30.0, band=BAND)}
    state = _running_state(parts)
    scene = sc.build_scene(state)

    # Arc には丸め (round_cap) の概念自体が無い。Line と違って端は常に平ら。
    for arc in scene.arcs:
        assert not hasattr(arc, "round_cap")


def test_joules_off_hides_numbers_and_raises_state():
    parts = {"elbow_L": PartReading(now=90.0, prev=None, band=BAND)}
    state = _running_state(parts, show_joules=False)
    scene = sc.build_scene(state)

    assert scene.find("band_label", "elbow_L") == []
    assert scene.find("value_text", "elbow_L") == []
    # 値の弧そのものは J のオン/オフに関係なく描く（消えるのは数字だけ）。
    assert len(scene.find("value", "elbow_L")) == 1

    state_label = scene.find("state", "elbow_L")[0]
    assert state_label.x == 165.0
    assert state_label.y == 196.0  # cy（J オフは cy+14 ではなく cy）
    assert state_label.runs[0].size == 16


def test_null_band_draws_groove_only():
    parts = {"elbow_L": PartReading(now=None, prev=None, band=None)}
    state = _running_state(parts)
    scene = sc.build_scene(state)

    assert len(scene.find("groove", "elbow_L")) == 1
    assert len(scene.find("part", "elbow_L")) == 1
    assert scene.find("band", "elbow_L") == []
    assert scene.find("value", "elbow_L") == []
    assert scene.find("value_rim", "elbow_L") == []
    assert scene.find("prev", "elbow_L") == []
    assert scene.find("state", "elbow_L") == []


def test_null_band_with_value_still_shows_number_when_joules_on():
    parts = {"elbow_L": PartReading(now=12.0, prev=None, band=None)}
    state = _running_state(parts, show_joules=True)
    scene = sc.build_scene(state)

    value_labels = scene.find("value_text", "elbow_L")
    assert len(value_labels) == 1
    assert value_labels[0].runs[0].text == gm.joule_text(12.0)
    # band が無いので弧そのものは描かない。
    assert scene.find("value_rim", "elbow_L") == []


def test_null_now_draws_no_value():
    parts = {"elbow_L": PartReading(now=None, prev=None, band=BAND)}
    state = _running_state(parts)
    scene = sc.build_scene(state)

    assert len(scene.find("groove", "elbow_L")) == 1
    assert len(scene.find("band", "elbow_L")) == 1
    assert scene.find("value", "elbow_L") == []
    assert scene.find("value_rim", "elbow_L") == []
    assert scene.find("prev", "elbow_L") == []
    assert scene.find("state", "elbow_L") == []


def test_done_shows_prev_ticks_without_values():
    parts = {"elbow_L": PartReading(now=90.0, prev=70.0, band=BAND)}
    state = _running_state(parts)
    state = gm.finish(state, exit_code=0)
    scene = sc.build_scene(state)

    assert len(scene.find("groove", "elbow_L")) == 1
    assert len(scene.find("band", "elbow_L")) == 1
    assert len(scene.find("prev", "elbow_L")) == 1
    assert scene.find("value", "elbow_L") == []
    assert scene.find("value_rim", "elbow_L") == []
    assert scene.find("state", "elbow_L") == []


def test_done_keeps_prev_tick_when_last_now_is_null():
    # 最後のフレームで今回値が NaN（null）でも、終了後の前回の目盛りは残す。
    parts = {"elbow_L": PartReading(now=None, prev=70.0, band=BAND)}
    state = gm.finish(_running_state(parts), exit_code=0)
    scene = sc.build_scene(state)

    assert len(scene.find("prev", "elbow_L")) == 1
    assert scene.find("value", "elbow_L") == []


def test_failed_keeps_previous_dials():
    parts = {"elbow_L": PartReading(now=90.0, prev=70.0, band=BAND)}
    running = _running_state(parts)
    running_scene = sc.build_scene(running)

    failed = gm.finish(running, exit_code=1)
    failed_scene = sc.build_scene(failed)

    # ダイヤルは「直前の表示のまま」（見出しだけ異常終了に変わる）。
    assert failed_scene.find("value", "elbow_L") == running_scene.find("value", "elbow_L")
    assert failed_scene.find("value_rim", "elbow_L") == running_scene.find("value_rim", "elbow_L")
    assert failed_scene.find("prev", "elbow_L") == running_scene.find("prev", "elbow_L")
    assert failed_scene.find("band", "elbow_L") == running_scene.find("band", "elbow_L")


def test_value_clamps_at_right_end():
    parts = {"elbow_L": PartReading(now=1000.0, prev=None, band=BAND)}
    state = _running_state(parts)
    scene = sc.build_scene(state)

    value_arc = scene.find("value", "elbow_L")[0]
    assert value_arc.f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 凡例
# ---------------------------------------------------------------------------


def test_legend_has_only_band_and_prev():
    state = _waiting_state()
    scene = sc.build_scene(state)

    band_swatches = scene.find("legend_band")
    prev_swatches = scene.find("legend_prev")
    band_labels = scene.find("legend_band_label")
    prev_labels = scene.find("legend_prev_label")

    assert len(band_swatches) == 1
    assert len(prev_swatches) == 1
    assert len(band_labels) == 1
    assert len(prev_labels) == 1

    swatch = band_swatches[0]
    assert (swatch.x0, swatch.y0, swatch.x1, swatch.y1) == (322.0, 424.0, 350.0, 424.0)
    assert swatch.width == 7.0

    tick = prev_swatches[0]
    assert (tick.x0, tick.y0, tick.x1, tick.y1) == (414.0, 416.0, 414.0, 432.0)

    assert band_labels[0].runs[0].text == "目標帯"
    assert prev_labels[0].runs[0].text == "前回"


# ---------------------------------------------------------------------------
# 帯の数字が重ならない（帯が右端に寄って狭いとき）
# ---------------------------------------------------------------------------


def _label_box(label: sc.Label) -> tuple[float, float, float, float]:
    """文字の箱の概算（幅は「文字数 × 大きさ × 0.6」、高さは大きさそのもの）。"""
    text = "".join(run.text for run in label.runs)
    size = max(run.size for run in label.runs)
    width = max(len(text) * size * 0.6, 1.0)
    height = size
    if label.align == "left":
        x0 = label.x
    elif label.align == "right":
        x0 = label.x - width
    else:
        x0 = label.x - width / 2
    y0 = label.y - height
    return x0, y0, x0 + width, label.y


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


@pytest.mark.parametrize("band", [(175.0, 212.5), (112.0, 136.0)])
@pytest.mark.parametrize("part", ["elbow_L", "elbow_R", "wrist_L", "wrist_R"])
def test_band_labels_do_not_collide_for_narrow_right_band(band, part):
    parts = {part: PartReading(now=None, prev=None, band=band)}
    state = _waiting_state(parts, show_joules=True)
    scene = sc.build_scene(state)

    labels = scene.find("band_label", part)
    assert len(labels) == 2

    boxes = [_label_box(label) for label in labels]
    assert not _boxes_overlap(*boxes)

    for x0, y0, x1, y1 in boxes:
        assert 0.0 <= x0 and x1 <= 800.0
        assert 0.0 <= y0 and y1 <= 450.0
        # 人物の範囲（x 300〜500）に入らない。
        assert x1 <= 300.0 or x0 >= 500.0


# ---------------------------------------------------------------------------
# qt_arc_angles
# ---------------------------------------------------------------------------


def test_qt_arc_angles():
    assert sc.qt_arc_angles(0.0, 1.0) == pytest.approx((180.0, -180.0))
    assert sc.qt_arc_angles(0.0, 0.0) == pytest.approx((180.0, 0.0))
    assert sc.qt_arc_angles(0.25, 0.75) == pytest.approx((135.0, -90.0))


# ---------------------------------------------------------------------------
# 見出し（brief の「判断済みのこと」に合わせる）
# ---------------------------------------------------------------------------


def test_header_waiting_shows_spinner_and_wait_text():
    state = _waiting_state()
    scene = sc.build_scene(state, spinner_phase=0.3)

    assert scene.spinner is not None
    assert (scene.spinner.cx, scene.spinner.cy, scene.spinner.radius) == (620.0, 32.0, 10.0)
    assert scene.spinner.phase == 0.3

    wait_label = scene.find("header_wait")[0]
    assert (wait_label.x, wait_label.y) == (640.0, 38.0)
    assert wait_label.runs[0].text == "Pixel 接続待ち"


def test_header_running_shows_rep_number_right_aligned():
    parts = {"elbow_L": PartReading(now=1.0, band=BAND)}
    state = _running_state(parts, rep=6)
    scene = sc.build_scene(state)

    assert scene.spinner is None
    main = scene.find("header_rep")[0]
    sub = scene.find("header_rep_sub")[0]
    assert (main.x, main.y) == (742.0, 46.0)
    assert main.align == "right"
    assert main.runs[0].text == "7"
    assert (sub.x, sub.y) == (782.0, 46.0)
    assert sub.runs[0].text == "回目"


def test_header_replay_pill_shown_for_replay_source():
    state = gm.reset(show_joules=True)
    state = gm.apply_frame(state, _frame({}, rep=0, source="replay"))
    scene = sc.build_scene(state)

    replay = scene.find("header_replay")
    assert len(replay) == 1
    assert (replay[0].x, replay[0].y) == (230.0, 40.0)
    assert replay[0].runs[0].text == "▶ 再生"
    assert replay[0].runs[0].color == theme.HEADER_SUB


@pytest.mark.parametrize("part", ["elbow_L", "elbow_R", "wrist_L", "wrist_R"])
def test_band_label_near_top_is_centered_and_lifted(part, monkeypatch):
    # lo=62.5・hi=100 なら下端の割合は 62.5/125 = 0.5 で、弧の頂上（中央揃えの範囲）に来る
    parts = {part: PartReading(now=None, prev=None, band=(62.5, 100.0))}
    state = _waiting_state(parts, show_joules=True)

    def lo_label():
        labels = sc.build_scene(state).find("band_label", part)
        (label,) = [lb for lb in labels if lb.runs[0].text == "62"]
        return label

    lifted = lo_label()
    assert lifted.align == "center"
    monkeypatch.setattr(sc, "CENTER_LABEL_LIFT", 0.0)
    assert lo_label().y - lifted.y == pytest.approx(10.0), "中央揃えの数字は弧に重ならないよう持ち上げる"
