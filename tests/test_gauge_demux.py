"""``LineDemux``（子プロセスの出力からゲージの行を拾う）を検証する。

計測の子プロセスは Qt を持たないことがあるので、``QProcess`` は塊（chunk）単位で
標準出力を渡してくる（行の途中で切れることがある）。``LineDemux`` は Qt に依存せず、
``feed(data: bytes)`` で塊を受け取り、ゲージの行（``@@GAUGE `` + JSON）は
``GaugeFrame`` に、それ以外はログ文字列として返す。

解析スクリプトが ``\\r`` で進捗を上書き表示することがあるので、改行で終わっていない
普通の行はためずにすぐログへ流す（ゲージの行らしい途中だけをためる）。
"""

from __future__ import annotations

import time

from app.gauge import protocol as g
from app.gauge.protocol import LineDemux


def _gauge_line(rep: int = 1) -> str:
    """試験用の、確実に decode できるゲージの行を1本作る。"""
    frame = g.GaugeFrame(
        link="connected",
        rep=rep,
        source="measure",
        parts={"elbow_L": g.PartReading(now=12.3, prev=10.1, band=(10.0, 20.0), w1rm=66.0)},
    )
    return g.encode(frame)


def _gauge_frame(rep: int = 1) -> g.GaugeFrame:
    return g.decode(_gauge_line(rep))


def test_gauge_line_becomes_frame_and_not_log():
    demux = LineDemux()
    log, frames = demux.feed(_gauge_line().encode("utf-8"))
    assert log == ""
    assert frames == [_gauge_frame()]


def test_ordinary_lines_pass_through_unchanged():
    demux = LineDemux()
    text = "start\nprogress: 10%\nprogress: 20%\n"
    log, frames = demux.feed(text.encode("utf-8"))
    assert log == text
    assert frames == []


def test_gauge_line_cut_between_chunks_is_joined():
    # JSON の途中で切れる場合
    line = _gauge_line()
    cut = len(line) // 2
    demux = LineDemux()
    log1, frames1 = demux.feed(line[:cut].encode("utf-8"))
    assert log1 == ""
    assert frames1 == []
    log2, frames2 = demux.feed(line[cut:].encode("utf-8"))
    assert log2 == ""
    assert frames2 == [_gauge_frame()]

    # "@@GA" | "UGE " で PREFIX 自体が切れる場合
    demux2 = LineDemux()
    assert g.PREFIX == "@@GAUGE "
    head, tail = "@@GA", "UGE " + line[len(g.PREFIX) :]
    log1, frames1 = demux2.feed(head.encode("utf-8"))
    assert log1 == ""
    assert frames1 == []
    log2, frames2 = demux2.feed(tail.encode("utf-8"))
    assert log2 == ""
    assert frames2 == [_gauge_frame()]


def test_partial_ordinary_line_is_shown_immediately():
    demux = LineDemux()
    log, frames = demux.feed(b"loading... 42%")
    assert log == "loading... 42%"
    assert frames == []


def test_multibyte_char_cut_between_chunks_is_not_garbled():
    text = "計測中です\n"
    data = text.encode("utf-8")
    # マルチバイト文字（"測"）の途中でバイト列を切る。
    cut = 4  # "計" (3 bytes) + "測" の1バイト目
    demux = LineDemux()
    log1, frames1 = demux.feed(data[:cut])
    log2, frames2 = demux.feed(data[cut:])
    assert log1 + log2 == text
    assert frames1 == frames2 == []


def test_broken_gauge_line_goes_to_log():
    demux = LineDemux()
    broken = g.PREFIX + "{これは JSON ではない\n"
    log, frames = demux.feed(broken.encode("utf-8"))
    assert log == broken
    assert frames == []


def test_flush_returns_held_tail_as_log():
    demux = LineDemux()
    held = "@@GAUGE {\"v\":2"
    log, frames = demux.feed(held.encode("utf-8"))
    assert log == ""
    assert frames == []
    assert demux.flush() == held
    # flush 後はためた分を吐き出し済みなので、もう一度呼んでも空。
    assert demux.flush() == ""


def test_crlf_line_endings():
    demux = LineDemux()
    gauge_line = _gauge_line().rstrip("\n") + "\r\n"
    ordinary_line = "ordinary\r\n"
    log, frames = demux.feed((gauge_line + ordinary_line).encode("utf-8"))
    assert log == ordinary_line
    assert frames == [_gauge_frame()]


def test_reset_forgets_previous_run():
    demux = LineDemux()
    line = _gauge_line()
    cut = len(line) // 2
    log, frames = demux.feed(line[:cut].encode("utf-8"))
    assert log == "" and frames == []

    demux.reset()

    # reset 前にためた前半とはもう繋がらない。後半だけでは PREFIX から始まらないので
    # 普通の行としてすぐログに出る（decode 候補としては扱われない）。
    log, frames = demux.feed(line[cut:].encode("utf-8"))
    assert frames == []
    assert log == line[cut:]
    assert demux.flush() == ""


def test_prefix_appearing_mid_line_is_split_into_log_and_gauge_candidate():
    """任意の要件: 行の途中に PREFIX が現れたら、手前はログへ・PREFIX から先はゲージの行の候補へ。"""
    demux = LineDemux()

    # 改行で終わった行の途中に PREFIX が現れる場合
    line = "stray text before" + _gauge_line()
    log, frames = demux.feed(line.encode("utf-8"))
    assert log == "stray text before"
    assert frames == [_gauge_frame()]

    # 改行で終わっていない途中の行の途中に PREFIX が現れる場合
    demux2 = LineDemux()
    partial = "stray text before" + g.PREFIX + '{"v":2'
    log, frames = demux2.feed(partial.encode("utf-8"))
    assert log == "stray text before"
    assert frames == []
    assert demux2.flush() == g.PREFIX + '{"v":2'


def test_throughput_handles_30hz_easily():
    gauge_lines = [_gauge_line(rep=i) for i in range(3000)]
    ordinary_lines = [f"progress: {i}\n" for i in range(3000)]
    mixed = "".join(
        line for pair in zip(gauge_lines, ordinary_lines) for line in pair
    )
    data = mixed.encode("utf-8")

    demux = LineDemux()
    start = time.perf_counter()
    log, frames = demux.feed(data)
    elapsed = time.perf_counter() - start

    assert len(frames) == 3000
    per_line = elapsed / 6000
    assert per_line < 0.0002, f"1行あたり {per_line * 1000:.4f} ms かかった"
