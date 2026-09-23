"""被験者ゲージの「状態→場面」の段階。800×450 の設計座標。

``app/gauge/model.py`` の ``GaugeState``（フレーム・局面・J 表示の有無）を受け取り、
画面に描く場面そのもの（弧・直線・文字・スピナー）を Qt に依存しない値として返す、
純粋なモジュール。次の widget タスクがこれを ``QPainter`` で描く。

数値・色・描き分けの正本は 2 つ:

1. ``.superpowers/sdd/2026-09-24-subject-gauge/`` の ``constraints.md``・
   ``task-6-brief.md``（座標・色・状態ごとの描き分け）。
2. 画面案の生成スクリプト ``mk_subject.py``（``gauge()``・``header()``・``LEGEND``）
   と、そこから作った ``1_被験者ゲージ_計測中.png``・``2_被験者ゲージ_状態別.png``。
   文章の表（constraints.md の「描き分け」）は要約であり、``mk_subject.py`` の
   ``gauge()`` が部位名の文字（``.t`` クラス）を ``if`` の外で無条件に描いている
   （wait の画面案でも部位名は出ている）ことに合わせ、部位名はどの局面でも
   常に描く。文章の表の「溝と帯だけ」「溝と部位名だけ」は、この「常に描く
   部位名」を土台にした「値まわり（弧・前回・状態）を足すかどうか」の話として読む。

人物（FIG）と見出しのアイコンは、窓の大きさが変わらない限り描き直さない層
なので widget 側が別に描く（``Scene.show_figure``・``Scene.header_band`` は
「描いてよいか」の旗だけを持ち、実際の絵はここに入れない）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.gauge import model as gm
from app.gauge.protocol import PART_NAMES
from app.shell import theme

__all__ = [
    "Run",
    "Arc",
    "Line",
    "Label",
    "Spinner",
    "Scene",
    "build_scene",
    "qt_arc_angles",
]


# ---------------------------------------------------------------------------
# 純粋な場面の型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arc:
    """本体・帯・値の弧の 1 本。角度は割合 (``f0``→``f1``、0〜1) で持つ。

    Qt の角度（``QPainter.drawArc`` が使う度数）には ``qt_arc_angles`` で変える。
    端は常に平ら（``round_cap`` のような属性を持たない）。丸めるのは
    ``Line``（前回の目盛り）だけで、太く描く弧を丸めると帯どうしの境が
    にじんで見えるため、ここでは選べないようにしてある。
    """

    cx: float
    cy: float
    radius: float
    f0: float
    f1: float
    width: float
    color: str
    role: str
    part: str | None = None


@dataclass(frozen=True)
class Line:
    """1 本の直線。前回の目盛りと、凡例の見本に使う。"""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    color: str
    alpha: float = 1.0
    round_cap: bool = False
    role: str = ""
    part: str | None = None


@dataclass(frozen=True)
class Run:
    """1 続きの文字（大きさ・太さ・色が揃った区間）。値の文字の「数字」と
    「 J」のように、1 個の ``Label`` の中で書体を変えたい区間を表す。
    """

    text: str
    size: float
    weight: int
    color: str


@dataclass(frozen=True)
class Label:
    """``Run`` を 1 つ以上つないだ、1 個の文字列。

    ``align`` は ``"left"``・``"center"``・``"right"``（SVG の
    ``text-anchor`` の ``start``・``middle``・``end`` に対応）。
    """

    x: float
    y: float
    runs: tuple[Run, ...]
    align: str
    role: str
    part: str | None = None


@dataclass(frozen=True)
class Spinner:
    """Pixel 接続待ちのくるくる。``phase`` は回転の位相（widget 側が進める）。"""

    cx: float
    cy: float
    radius: float
    phase: float


@dataclass(frozen=True)
class Scene:
    """1 フレーム分の場面そのもの。

    ``show_figure``・``header_band`` は、人物の絵と見出しの帯（アイコン込み）を
    widget が描いてよいかの旗。どの局面でも常に True で、値そのものは
    ``build_scene`` が変えることはない（窓の大きさが変わらない限り widget 側が
    1 度だけ描いてキャッシュする層なので、Scene の責務は「描け」の合図だけ）。
    """

    arcs: tuple[Arc, ...]
    lines: tuple[Line, ...]
    labels: tuple[Label, ...]
    spinner: Spinner | None
    show_figure: bool = True
    header_band: bool = True

    def find(self, role: str, part: str | None = None) -> list:
        """``role``（と、指定すれば ``part``）が一致する要素を弧・線・文字から集める。"""
        items: list = [*self.arcs, *self.lines, *self.labels]
        return [item for item in items if item.role == role and (part is None or item.part == part)]


def qt_arc_angles(f0: float, f1: float) -> tuple[float, float]:
    """割合 (f0, f1) を Qt の (開始角, 弧の角度)（度）に変える。

    ``(180 − 180·f0, −180·(f1 − f0))``（task-6-brief.md）。f=0 が本体弧の左端
    （水平・180°）、f=1 が右端（水平・0°）に対応し、その間を弧の下側（時計回り
    に見て下向き）で結ぶための式。widget 側は Qt の 1/16 度単位へさらに掛ける。
    """
    return 180.0 - 180.0 * f0, -180.0 * (f1 - f0)


# ---------------------------------------------------------------------------
# 座標・寸法（mk_subject.py の gauge() と同じ値。task-6-brief.md「数値」）
# ---------------------------------------------------------------------------

CENTERS: dict[str, tuple[float, float]] = {
    "elbow_L": (165.0, 196.0),
    "wrist_L": (165.0, 348.0),
    "elbow_R": (635.0, 196.0),
    "wrist_R": (635.0, 348.0),
}

R = 62.0  # 本体の弧の半径
W = 22.0  # 本体の弧の太さ
RB = R + W / 2 + 7  # 帯の半径 = 80
WB = 7.0  # 帯の太さ
LABEL_RADIUS = RB + 14  # 帯の数字の半径 = 94
PREV_R0 = R - W / 2 - 3  # 前回の目盛りの内側半径 = 48
PREV_R1 = R + W / 2 + 3  # 前回の目盛りの外側半径 = 76
PREV_WIDTH = 2.5
PREV_ALPHA = 0.7
VALUE_RIM_INSET = 4.0  # 縁の幅 − 値の弧の幅（片側 2px）
BAND_LABEL_SIZE = 11.0
CENTER_LABEL_LIFT = 10.0  # 帯の数字が中央揃えのとき、弧に重ならないよう上へずらす量

LEGEND_BAND_LINE = (322.0, 424.0, 350.0, 424.0)
LEGEND_PREV_LINE = (414.0, 416.0, 414.0, 432.0)
LEGEND_BAND_LABEL_POS = (356.0, 428.0)
LEGEND_PREV_LABEL_POS = (422.0, 428.0)
LEGEND_LABEL_SIZE = 11.0

HEADER_TITLE_POS = (70.0, 40.0)
HEADER_REP_MAIN_POS = (742.0, 46.0)
HEADER_REP_SUB_POS = (782.0, 46.0)
HEADER_STATUS_POS = (782.0, 40.0)
HEADER_WAIT_TEXT_POS = (640.0, 38.0)
HEADER_SPINNER = (620.0, 32.0, 10.0)
HEADER_REPLAY_POS = (230.0, 40.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _point_on_circle(cx: float, cy: float, radius: float, f: float) -> tuple[float, float]:
    """割合 f（0〜1）の位置。``mk_subject.py`` の ``pt()`` と同じ式。"""
    angle = math.radians(180.0 + 180.0 * f)
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


# ---------------------------------------------------------------------------
# 部品を 1 個組み立てる小さな関数（build_scene から使う）
# ---------------------------------------------------------------------------


def _part_label(cx: float, cy: float, part: str) -> Label:
    return Label(cx, cy + 38, (Run(gm.PART_LABELS[part], 15, 700, theme.TEXT),), "center", "part", part)


def _value_label(cx: float, cy: float, now: float, part: str) -> Label:
    # role は弧（"value"）と区別する。両方に "value" を使うと Scene.find("value")
    # が弧と文字を両方拾ってしまい、呼び出し側が個数を数え違える。
    runs = (Run(gm.joule_text(now), 22, 800, theme.TEXT), Run(" J", 12, 600, theme.TEXT))
    return Label(cx, cy - 6, runs, "center", "value_text", part)


def _state_label(cx: float, cy: float, status: gm.Status, text: str, show_joules: bool, part: str) -> Label:
    if show_joules:
        y, size = cy + 14, 13
    else:
        y, size = cy, 16
    over = status is gm.Status.OVER
    weight = 800 if over else 700
    color = theme.OVER if over else theme.TEXT
    return Label(cx, y, (Run(text, size, weight, color),), "center", "state", part)


def _band_label_align(f: float) -> str:
    if f < 0.45:
        return "right"
    if f > 0.55:
        return "left"
    return "center"


def _band_label(cx: float, cy: float, f: float, text: str, part: str) -> Label:
    x, y = _point_on_circle(cx, cy, LABEL_RADIUS, f)
    y += 4.0
    align = _band_label_align(f)
    if align == "center":
        y -= CENTER_LABEL_LIFT
    return Label(x, y, (Run(text, BAND_LABEL_SIZE, 400, theme.SUBTEXT),), align, "band_label", part)


def _prev_line(cx: float, cy: float, band: tuple[float, float], prev: float, part: str) -> Line:
    fp = gm.fraction(prev, band)
    x0, y0 = _point_on_circle(cx, cy, PREV_R0, fp)
    x1, y1 = _point_on_circle(cx, cy, PREV_R1, fp)
    return Line(x0, y0, x1, y1, PREV_WIDTH, theme.TEXT, alpha=PREV_ALPHA, round_cap=True, role="prev", part=part)


def _legend_lines() -> tuple[Line, Line]:
    bx0, by0, bx1, by1 = LEGEND_BAND_LINE
    band = Line(bx0, by0, bx1, by1, WB, theme.BAND, alpha=1.0, round_cap=False, role="legend_band", part=None)
    px0, py0, px1, py1 = LEGEND_PREV_LINE
    prev = Line(px0, py0, px1, py1, PREV_WIDTH, theme.TEXT, alpha=PREV_ALPHA, round_cap=True, role="legend_prev", part=None)
    return band, prev


def _legend_labels() -> tuple[Label, Label]:
    bx, by = LEGEND_BAND_LABEL_POS
    px, py = LEGEND_PREV_LABEL_POS
    band = Label(bx, by, (Run("目標帯", LEGEND_LABEL_SIZE, 400, theme.SUBTEXT),), "left", "legend_band_label", None)
    prev = Label(px, py, (Run("前回", LEGEND_LABEL_SIZE, 400, theme.SUBTEXT),), "left", "legend_prev_label", None)
    return band, prev


def _header_labels(info: gm.Header) -> tuple[Label, ...]:
    labels: list[Label] = [
        Label(*HEADER_TITLE_POS, (Run("上肢の仕事量", 22, 800, theme.TEXT),), "left", "header_title", None)
    ]
    if info.spinner:
        labels.append(
            Label(*HEADER_WAIT_TEXT_POS, (Run(info.main, 15, 700, theme.TEXT),), "left", "header_wait", None)
        )
    elif info.sub:
        # RUNNING: main は回数の数字、sub は「回目」。
        labels.append(
            Label(*HEADER_REP_MAIN_POS, (Run(info.main, 36, 800, theme.TEXT),), "right", "header_rep", None)
        )
        labels.append(
            Label(*HEADER_REP_SUB_POS, (Run(info.sub, 14, 400, theme.HEADER_SUB),), "right", "header_rep_sub", None)
        )
    else:
        # DONE / FAILED: main が「✓ 終了 N 回」または「✕ 異常終了」そのもの。
        labels.append(
            Label(*HEADER_STATUS_POS, (Run(info.main, 16, 700, theme.TEXT),), "right", "header_status", None)
        )
    if info.replay:
        labels.append(
            Label(*HEADER_REPLAY_POS, (Run("▶ 再生", 13, 700, theme.HEADER_SUB),), "left", "header_replay", None)
        )
    return tuple(labels)


def _effective_phase(state: gm.GaugeState) -> gm.Phase:
    """ダイヤルを描くための「実質の局面」。FAILED は「直前の表示のまま」
    （constraints.md）なので、最後に受け取ったフレームの ``link`` から
    「落ちる直前は WAITING だったか RUNNING だったか」を復元する
    （``model.apply_frame`` が局面を決めた基準と同じ式）。見出しの文字
    （``model.header``）は局面そのもの（FAILED なら「異常終了」）を見るので、
    ここでの読み替えは見出しには影響しない。
    """
    if state.phase is not gm.Phase.FAILED:
        return state.phase
    if state.frame is None or state.frame.link == "waiting":
        return gm.Phase.WAITING
    return gm.Phase.RUNNING


# ---------------------------------------------------------------------------
# build_scene
# ---------------------------------------------------------------------------


def build_scene(state: gm.GaugeState, *, spinner_phase: float = 0.0) -> Scene:
    """``GaugeState`` から 1 フレーム分の場面を組み立てる。

    描き分け（task-6-brief.md）:

    * 溝と部位名は、どの局面でも部位ごとに常に描く（``mk_subject.py`` の
      ``gauge()`` が無条件に描いているのに合わせる）。
    * 帯が無い（``band is None``）部位は、溝と部位名だけ。J がオンで値
      （``now``）があれば、弧なしで中央の数字だけ足す。
    * 帯はあるが値（``now``）が無い部位は、溝・帯（J オンなら帯の数字）まで。
    * 値・状態・前回の目盛りは、局面が実質 RUNNING（FAILED は直前の局面を
      復元して判定）のときだけ描く。DONE は前回の目盛りだけ残す。
    * 帯の中（IN_BAND）なら帯を点灯させ、過負荷（OVER）なら値の弧と状態の
      文字を過負荷の色にする（帯そのものは点灯させない）。
    """
    frame = state.frame
    show_joules = state.show_joules
    ep = _effective_phase(state)

    arcs: list[Arc] = []
    lines: list[Line] = []
    labels: list[Label] = []

    for part in PART_NAMES:
        cx, cy = CENTERS[part]
        reading = frame.parts.get(part) if frame is not None else None
        now = reading.now if reading is not None else None
        prev = reading.prev if reading is not None else None
        band = reading.band if reading is not None else None

        arcs.append(Arc(cx, cy, R, 0.0, 1.0, W, theme.TRACK, "groove", part))
        labels.append(_part_label(cx, cy, part))

        if band is None:
            if show_joules and now is not None and ep is gm.Phase.RUNNING:
                labels.append(_value_label(cx, cy, now, part))
            continue

        lo, hi = band
        st = gm.status(now, band)
        fl = _clamp01(lo / (1.25 * hi))
        fh = _clamp01(hi / (1.25 * hi))
        band_color = theme.BAND_ON if st is gm.Status.IN_BAND else theme.BAND
        arcs.append(Arc(cx, cy, RB, fl, fh, WB, band_color, "band", part))

        if show_joules:
            lo_text, hi_text = gm.band_labels(band)
            labels.append(_band_label(cx, cy, fl, lo_text, part))
            labels.append(_band_label(cx, cy, fh, hi_text, part))

        if now is None:
            continue

        if ep is gm.Phase.RUNNING:
            f = gm.fraction(now, band)
            value_color = theme.OVER if st is gm.Status.OVER else theme.VALUE
            arcs.append(Arc(cx, cy, R, 0.0, f, W, theme.FIELD, "value_rim", part))
            arcs.append(Arc(cx, cy, R, 0.0, f, W - VALUE_RIM_INSET, value_color, "value", part))

            if show_joules:
                labels.append(_value_label(cx, cy, now, part))

            state_text = gm.status_label(st)
            if state_text:
                labels.append(_state_label(cx, cy, st, state_text, show_joules, part))

            if prev is not None:
                lines.append(_prev_line(cx, cy, band, prev, part))
        elif ep is gm.Phase.DONE:
            if prev is not None:
                lines.append(_prev_line(cx, cy, band, prev, part))
        # WAITING: 溝・帯（・帯の数字）だけで、これ以上は何も描かない。

    lines.extend(_legend_lines())
    labels.extend(_legend_labels())

    header_info = gm.header(state)
    spinner = Spinner(*HEADER_SPINNER, spinner_phase) if header_info.spinner else None
    labels.extend(_header_labels(header_info))

    return Scene(arcs=tuple(arcs), lines=tuple(lines), labels=tuple(labels), spinner=spinner)
