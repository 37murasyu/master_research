"""被験者ゲージの「値→割合・状態・表示」の段階。

``app/gauge/protocol.py`` の ``GaugeFrame``・``PartReading`` を受け取り、
画面に描く直前の値（弧の割合・状態・文字列）へ変換する、Qt に依存しない
純粋なモジュール。ここでの決定はすべて日本語の関数名・値のまま次の scene
タスク（Qt の場面を組み立てる側）へ渡す。Qt に依存しないのは protocol.py と
同じ理由で、計測の子プロセス側の試験からも import できるようにするため
（``app.gauge`` パッケージの他モジュールから独立して読み込めることを
``test_protocol_does_not_import_qt`` に倣って将来も確かめられるようにする）。

``GaugeState`` は frozen dataclass で、遷移はすべて「新しい状態を返す関数」
として書く（``dataclasses.replace`` を使い、その場で書き換えない）。Qt の
シグナル/スロットから ``self._state = apply_frame(self._state, frame)`` の
ように呼べる形にするためで、途中の状態を誰かが保持し続けて壊れる心配が無い。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum, auto

from app.gauge.protocol import GaugeFrame

__all__ = [
    "Status",
    "Phase",
    "Header",
    "GaugeState",
    "PART_LABELS",
    "fraction",
    "status",
    "status_label",
    "joule_text",
    "band_labels",
    "apply_frame",
    "finish",
    "reset",
    "with_joules",
    "header",
]


# ---------------------------------------------------------------------------
# 部位名の日本語ラベル
# ---------------------------------------------------------------------------

PART_LABELS: dict[str, str] = {
    "elbow_L": "左 上腕",
    "wrist_L": "左 前腕",
    "elbow_R": "右 上腕",
    "wrist_R": "右 前腕",
}


# ---------------------------------------------------------------------------
# 値→割合・状態（純粋関数）
# ---------------------------------------------------------------------------


def fraction(now: float | None, band: tuple[float, float] | None) -> float:
    """弧の割合 ``f = clamp(now / (1.25·hi), 0, 1)``（constraints.md「弧の割合」）。

    ``now`` が None・NaN・負なら 0（Global Constraints の式の定義域外）。
    ``band`` が None のとき（分母の ``hi`` が無い）も同じく 0 にする。
    protocol.decode を通った値なら band は常に lo<hi の組か None しか来ないが、
    この関数は呼び出し側の組み合わせを信用せず、壊れた入力で例外を投げない
    （描画ループが 30Hz で回り続けるので、1 部位のために止まってはいけない）。
    """
    if band is None or now is None:
        return 0.0
    if math.isnan(now) or now < 0:
        return 0.0
    lo, hi = band
    f = now / (1.25 * hi)
    return max(0.0, min(1.0, f))


class Status(Enum):
    """帯に対する今の値の状態（constraints.md「状態の判定」）。"""

    NONE = auto()
    SHORT = auto()
    IN_BAND = auto()
    OVER = auto()


def status(now: float | None, band: tuple[float, float] | None) -> Status:
    """``now < lo`` は SHORT、``lo ≤ now < hi`` は IN_BAND、``now ≥ hi`` は OVER。

    band か now が None なら NONE（帯が無い、または値がまだ無い部位）。
    """
    if band is None or now is None:
        return Status.NONE
    lo, hi = band
    if now < lo:
        return Status.SHORT
    if now < hi:
        return Status.IN_BAND
    return Status.OVER


def status_label(value: Status) -> str:
    """状態の文字。SHORT・NONE は「不足」を文字で示さない（constraints.md どおり）。"""
    if value is Status.IN_BAND:
        return "✓ 目標帯"
    if value is Status.OVER:
        return "✕ 過負荷"
    return ""


def joule_text(value: float | None) -> str:
    """整数に丸めた文字。値が無い（None・非有限）なら空文字。"""
    if value is None or not math.isfinite(value):
        return ""
    return str(round(value))


def band_labels(band: tuple[float, float] | None) -> tuple[str, str]:
    """帯の両端の文字。下端は数字だけ、上端は単位つき（例: ``("175", "213 J")``）。

    band が None の部位は両方とも空文字（constraints.md「描かないもの」）。
    """
    if band is None:
        return ("", "")
    lo, hi = band
    return (str(round(lo)), f"{round(hi)} J")


# ---------------------------------------------------------------------------
# 状態機械
# ---------------------------------------------------------------------------


class Phase(Enum):
    """見出しの局面。Pixel 接続待ち／計測中／正常終了／異常終了。"""

    WAITING = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class GaugeState:
    """画面が持つ状態そのもの。局面・最後に受け取ったフレーム・J 表示の有無。"""

    phase: Phase
    frame: GaugeFrame | None
    show_joules: bool


def apply_frame(state: GaugeState, frame: GaugeFrame) -> GaugeState:
    """フレームを 1 つ受け取って次の状態を返す。

    DONE / FAILED の後に届いたフレームは捨てる（プロセスは終わっているのに、
    パイプに残っていた最後の数行が遅れて届いて表示を巻き戻すのを防ぐ）。
    それ以外は ``link`` だけで局面を決める。RUNNING 中に link=waiting が来たら
    WAITING に戻す（計測中に Pixel との接続が切れた扱い）。
    """
    if state.phase in (Phase.DONE, Phase.FAILED):
        return state
    phase = Phase.WAITING if frame.link == "waiting" else Phase.RUNNING
    return replace(state, phase=phase, frame=frame)


def finish(state: GaugeState, exit_code: int) -> GaugeState:
    """子プロセスの終了コードを反映する。0 なら DONE、それ以外は FAILED。

    最後に受け取ったフレームはそのまま保つ（見出しの「終了 N 回」がこの
    フレームの rep を使う）。
    """
    phase = Phase.DONE if exit_code == 0 else Phase.FAILED
    return replace(state, phase=phase)


def reset(show_joules: bool) -> GaugeState:
    """フレームを捨てて WAITING に戻った、まっさらな状態を作る（次の計測の前）。"""
    return GaugeState(phase=Phase.WAITING, frame=None, show_joules=show_joules)


def with_joules(state: GaugeState, show_joules: bool) -> GaugeState:
    """J 表示の有無だけを切り替える。局面・フレームは保つ。"""
    return replace(state, show_joules=show_joules)


# ---------------------------------------------------------------------------
# 見出し
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Header:
    """見出しの表示に要るものだけ（文字の組み立てそのものは scene 側で描く）。"""

    spinner: bool
    main: str
    sub: str
    replay: bool


def header(state: GaugeState) -> Header:
    """局面から見出しを組み立てる。

    ``replay`` は局面と独立に「最後に受け取ったフレームの source が replay か」
    だけで決める（constraints.md「source」: replay のときは局面を問わず
    「▶ 再生」の印を出す）。
    """
    replay = state.frame is not None and state.frame.source == "replay"
    last_rep = state.frame.rep if state.frame is not None else 0

    if state.phase is Phase.WAITING:
        return Header(spinner=True, main="Pixel 接続待ち", sub="", replay=replay)
    if state.phase is Phase.RUNNING:
        return Header(spinner=False, main=str(last_rep + 1), sub="回目", replay=replay)
    if state.phase is Phase.DONE:
        return Header(spinner=False, main=f"✓ 終了 {last_rep} 回", sub="", replay=replay)
    # Phase.FAILED
    return Header(spinner=False, main="✕ 異常終了", sub="", replay=replay)
