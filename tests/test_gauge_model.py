"""値→割合・状態・表示の段階（``app.gauge.model``）を検証する。

``app/gauge/model.py`` は Qt に依存しない純粋なモジュール。次の scene タスクが
これを使って場面を組み立てる。protocol.py の ``GaugeFrame``・``PartReading`` を
そのまま受け取り、弧の割合（``fraction``）・状態（``status``）・見出し
（``header``）などの「値→表示」の変換だけを担う。
"""

from __future__ import annotations

import math

import pytest

from app.gauge import model as m
from app.gauge.protocol import GaugeFrame, PartReading


BAND = (80.0, 100.0)


# ---------------------------------------------------------------------------
# fraction
# ---------------------------------------------------------------------------


def test_fraction_zero_at_zero():
    assert m.fraction(0.0, BAND) == 0.0


def test_fraction_at_band_edges():
    # f = now / (1.25 * hi)。帯の下端・上端でも特別扱いせず同じ式で計算する。
    lo, hi = BAND
    assert m.fraction(lo, BAND) == pytest.approx(lo / (1.25 * hi))
    assert m.fraction(hi, BAND) == pytest.approx(hi / (1.25 * hi))  # = 0.8


def test_fraction_caps_at_one_beyond_right_edge():
    lo, hi = BAND
    assert m.fraction(1.25 * hi, BAND) == pytest.approx(1.0)
    assert m.fraction(1.25 * hi + 50.0, BAND) == pytest.approx(1.0)


def test_fraction_nan_and_negative_are_zero():
    assert m.fraction(float("nan"), BAND) == 0.0
    assert m.fraction(-5.0, BAND) == 0.0
    assert m.fraction(None, BAND) == 0.0
    # band が無ければ計算のしようが無いので 0（呼び出し側の想定外の組み合わせに備える）。
    assert m.fraction(50.0, None) == 0.0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_boundaries():
    lo, hi = BAND
    eps = 1e-6
    assert m.status(lo - eps, BAND) is m.Status.SHORT
    assert m.status(lo, BAND) is m.Status.IN_BAND
    assert m.status(hi - eps, BAND) is m.Status.IN_BAND
    assert m.status(hi, BAND) is m.Status.OVER


def test_status_without_band_is_none():
    assert m.status(50.0, None) is m.Status.NONE
    assert m.status(None, BAND) is m.Status.NONE
    assert m.status(None, None) is m.Status.NONE


def test_status_labels_carry_symbols():
    assert m.status_label(m.Status.IN_BAND) == "✓ 目標帯"
    assert m.status_label(m.Status.OVER) == "✕ 過負荷"
    assert m.status_label(m.Status.SHORT) == ""
    assert m.status_label(m.Status.NONE) == ""


# ---------------------------------------------------------------------------
# joule_text / band_labels / PART_LABELS
# ---------------------------------------------------------------------------


def test_joule_text_rounds_to_integer():
    assert m.joule_text(12.4) == "12"
    assert m.joule_text(12.6) == "13"
    assert m.joule_text(0.0) == "0"
    # 値が無ければ空文字（呼び出し側は「値なし」の描画に落ちる）。
    assert m.joule_text(None) == ""
    assert m.joule_text(float("nan")) == ""


def test_joule_text_draws_negative_as_zero():
    """設計書 §7「値が NaN・負: 0 として描く」。弧（fraction）は 0 なのに数字だけ「-5 J」と出ていた。"""
    assert m.joule_text(-5.0) == "0"
    assert m.joule_text(-0.4) == "0"


def test_fraction_without_positive_upper_edge_is_zero():
    """分母の上端が 0 以下の帯でも例外を投げない（描画ループを止めない）。"""
    assert m.fraction(1.0, (-1.0, 0.0)) == 0.0
    assert m.fraction(1.0, (-10.0, -5.0)) == 0.0


def test_band_labels():
    assert m.band_labels((174.6, 212.6)) == ("175", "213 J")
    # band が無い部位は両方とも空文字。
    assert m.band_labels(None) == ("", "")


def test_part_labels():
    assert m.PART_LABELS == {
        "elbow_L": "左 上腕",
        "wrist_L": "左 前腕",
        "elbow_R": "右 上腕",
        "wrist_R": "右 前腕",
    }


# ---------------------------------------------------------------------------
# GaugeState の遷移
# ---------------------------------------------------------------------------


def _frame(link: str, rep: int = 0, source: str = "measure") -> GaugeFrame:
    return GaugeFrame(link=link, rep=rep, source=source, parts={"elbow_L": PartReading(now=12.3)})


def test_phase_follows_link_and_finish():
    state = m.reset(show_joules=False)
    assert state.phase is m.Phase.WAITING

    state = m.apply_frame(state, _frame("connected", rep=0))
    assert state.phase is m.Phase.RUNNING

    # 計測中に接続が切れた扱い: link=waiting のフレームが来たら WAITING に戻る。
    state = m.apply_frame(state, _frame("waiting", rep=1))
    assert state.phase is m.Phase.WAITING

    state = m.apply_frame(state, _frame("connected", rep=1))
    assert state.phase is m.Phase.RUNNING

    done = m.finish(state, exit_code=0)
    assert done.phase is m.Phase.DONE

    failed = m.finish(state, exit_code=1)
    assert failed.phase is m.Phase.FAILED


def test_frames_after_finish_are_ignored():
    state = m.reset(show_joules=False)
    state = m.apply_frame(state, _frame("connected", rep=3))
    done = m.finish(state, exit_code=0)

    # DONE の後に届いたフレームは捨てる（表示は「終了 3 回」のまま動かない）。
    ignored = m.apply_frame(done, _frame("connected", rep=9))
    assert ignored.phase is m.Phase.DONE
    assert ignored.frame.rep == 3

    failed = m.finish(state, exit_code=1)
    ignored_failed = m.apply_frame(failed, _frame("connected", rep=9))
    assert ignored_failed.phase is m.Phase.FAILED
    assert ignored_failed.frame.rep == 3


def test_reset_returns_to_waiting_without_frame():
    state = m.reset(show_joules=False)
    state = m.apply_frame(state, _frame("connected", rep=5))
    state = m.finish(state, exit_code=0)

    fresh = m.reset(show_joules=True)
    assert fresh.phase is m.Phase.WAITING
    assert fresh.frame is None
    assert fresh.show_joules is True


def test_with_joules_keeps_phase_and_frame():
    state = m.reset(show_joules=False)
    state = m.apply_frame(state, _frame("connected", rep=2))

    toggled = m.with_joules(state, True)
    assert toggled.show_joules is True
    assert toggled.phase is state.phase
    assert toggled.frame is state.frame


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------


def test_header_texts():
    waiting = m.reset(show_joules=False)
    h = m.header(waiting)
    assert h.spinner is True
    assert h.main == "Pixel 接続待ち"
    assert h.sub == ""

    running = m.apply_frame(waiting, _frame("connected", rep=6))
    h = m.header(running)
    assert h.spinner is False
    assert h.main == "7"
    assert h.sub == "回目"

    done = m.finish(running, exit_code=0)
    h = m.header(done)
    assert h.main == "✓ 終了 6 回"
    assert h.sub == ""

    # フレームが 1 度も来ないまま終了した場合は 0 回。
    done_without_frame = m.finish(m.reset(show_joules=False), exit_code=0)
    assert m.header(done_without_frame).main == "✓ 終了 0 回"

    failed = m.finish(running, exit_code=1)
    h = m.header(failed)
    assert h.main == "✕ 異常終了"
    assert h.sub == ""


def test_replay_marks_header():
    state = m.reset(show_joules=False)
    live = m.apply_frame(state, _frame("connected", rep=0, source="measure"))
    assert m.header(live).replay is False

    replay = m.apply_frame(state, _frame("connected", rep=0, source="replay"))
    assert m.header(replay).replay is True


def test_status_nan_is_none_not_over():
    # NaN は比較がすべて偽になるので、そのままだと OVER に落ちる。fraction（NaN は 0）とそろえる
    assert m.status(math.nan, (80.0, 100.0)) is m.Status.NONE
    assert m.status_label(m.status(math.nan, (80.0, 100.0))) == ""
