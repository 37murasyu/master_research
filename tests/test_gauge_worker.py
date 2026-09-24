"""``WorkerRunner`` が子プロセスの出力をゲージの行とログへ振り分けることを確かめる。

子プロセス（計測・demo・script いずれも）は標準出力に ``@@GAUGE `` 行（app.gauge.protocol）
と普通のログ行を混ぜて出す。``WorkerRunner`` はこれを ``LineDemux`` で解き、
ゲージの行は ``gauge_frame`` シグナルへ、それ以外は従来どおり ``output`` シグナルへ流す。

GUI は 30Hz で届くフレームを ``output``（テキストのログ表示）へ混ぜて出してしまうと、
ログの行数上限（``LogView.MAX_BLOCKS``）をあっという間に食い潰すうえ、ゲージの
値としても使えない。振り分けが要になる。
"""

from __future__ import annotations

import sys

import pytest

from app import entry
from app.core.settings import Settings
from app.gauge import protocol as g

pytest.importorskip("PySide6", reason="Qt が無い環境ではスキップ")


@pytest.fixture(scope="module")
def qt_app():
    from app.core.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _gauge_line(rep: int = 0) -> str:
    frame = g.GaugeFrame(link="connected", rep=rep, source="measure", parts={})
    return g.encode(frame)


def _gauge_frame(rep: int = 0) -> g.GaugeFrame:
    return g.decode(_gauge_line(rep))


class TestHandleBytes:
    """``_handle_bytes`` 単体（実プロセスを起動しない）での振り分け。"""

    def test_gauge_lines_reach_gauge_frame_not_output(self, qt_app):
        from app.runners.worker import WorkerRunner

        runner = WorkerRunner("script")
        outputs: list[str] = []
        frames: list[g.GaugeFrame] = []
        runner.output.connect(lambda t: outputs.append(t))
        runner.gauge_frame.connect(lambda f: frames.append(f))

        mixed = _gauge_line(rep=1) + "普通のログ行\n" + _gauge_line(rep=2)
        runner._handle_bytes(mixed.encode("utf-8"))

        assert frames == [_gauge_frame(rep=1), _gauge_frame(rep=2)]
        assert outputs == ["普通のログ行\n"]
        assert all("@@GAUGE" not in text for text in outputs)

    def test_split_line_across_chunks(self, qt_app):
        from app.runners.worker import WorkerRunner

        runner = WorkerRunner("script")
        outputs: list[str] = []
        frames: list[g.GaugeFrame] = []
        runner.output.connect(lambda t: outputs.append(t))
        runner.gauge_frame.connect(lambda f: frames.append(f))

        line = _gauge_line(rep=3)
        cut = len(line) // 2
        data = line.encode("utf-8")

        runner._handle_bytes(data[:cut])
        assert frames == []
        assert outputs == []

        runner._handle_bytes(data[cut:])
        assert frames == [_gauge_frame(rep=3)]
        assert outputs == []

    def test_new_run_starts_with_empty_buffer(self, qt_app, monkeypatch):
        """``start`` のたびに ``LineDemux`` をリセットし、前回の途中の断片を持ち越さない。"""
        from app.runners.worker import WorkerRunner

        runner = WorkerRunner("script")
        # 前回の実行の、改行が来ないまま終わった断片を想定してためておく。
        log, frames = runner._demux.feed(b"@@GA")
        assert log == "" and frames == []

        monkeypatch.setattr(
            entry,
            "worker_command",
            lambda role, passthrough=None, module=None: [sys.executable, "-c", "pass"],
        )
        outputs: list[str] = []
        runner.output.connect(lambda t: outputs.append(t))
        assert runner.start(Settings(), module="dummy")
        runner._process.waitForFinished(5000)

        # start～finished の間の起動・終了メッセージは今回の確認と無関係なので捨てる。
        outputs.clear()
        runner._handle_bytes(b"ordinary\n")
        assert outputs == ["ordinary\n"], "前回の断片 (\"@@GA\") が新しい実行に持ち越された"

    def test_tail_is_flushed_before_finished(self, qt_app):
        """出る順: 残りのログ（flush の分） → stopped → finished。"""
        from app.core.qt import QtCore
        from app.runners.worker import WorkerRunner

        runner = WorkerRunner("script")
        held = '@@GAUGE {"v":2'  # 改行が来ないまま終わる、ゲージの行の途中
        runner._handle_bytes(held.encode("utf-8"))

        events: list[tuple] = []
        runner.output.connect(lambda t: events.append(("output", t)))
        runner.state_changed.connect(lambda s: events.append(("state", s)))
        runner.finished.connect(lambda c: events.append(("finished", c)))

        runner._on_finished(0, QtCore.QProcess.NormalExit)

        output_events = [e for e in events if e[0] == "output" and e[1] == held]
        assert output_events, "flush の残りが output に出ていない"
        tail_index = events.index(output_events[0])
        stopped_index = events.index(("state", "stopped"))
        finished_index = events.index(("finished", 0))
        assert tail_index < stopped_index < finished_index


class TestCrashExit:
    def test_crash_exit_is_reported_as_nonzero(self, qt_app):
        from app.core.qt import QtCore
        from app.runners.worker import WorkerRunner

        runner = WorkerRunner("script")
        finished_codes: list[int] = []
        runner.finished.connect(lambda c: finished_codes.append(c))

        runner._on_finished(0, QtCore.QProcess.CrashExit)

        assert finished_codes == [1], "CrashExit なのに終了コード 0 のまま出た"

    def test_crash_exit_keeps_nonzero_code(self, qt_app):
        """CrashExit でも既に非 0 の終了コードなら、それをそのまま使う。"""
        from app.core.qt import QtCore
        from app.runners.worker import WorkerRunner

        runner = WorkerRunner("script")
        finished_codes: list[int] = []
        runner.finished.connect(lambda c: finished_codes.append(c))

        runner._on_finished(9, QtCore.QProcess.CrashExit)

        assert finished_codes == [9]


class TestRealChildFrames:
    """実プロセスを ``WorkerRunner`` 経由で起動し、フレームが届くことを確かめる。

    デモ（app.gauge.demo）はこの作業ツリーにまだ無いので、代わりに
    ``python -c`` で ``@@GAUGE`` 行 5 つ（間に普通の行 1 つ）を出す小さなスクリプトを
    起動する（tests/test_shell_smoke.py の TestStopDirectory と同じ形で
    entry.worker_command を差し替える）。
    """

    _CHILD_SCRIPT = (
        "import sys\n"
        "from app.gauge.protocol import GaugeFrame, encode\n"
        "frame = GaugeFrame(link='connected', rep=0, source='measure', parts={})\n"
        "lines = [encode(frame)] * 2 + ['ordinary line\\n'] + [encode(frame)] * 3\n"
        "for line in lines:\n"
        "    sys.stdout.write(line)\n"
        "    sys.stdout.flush()\n"
    )

    def test_real_child_frames_arrive(self, qt_app, monkeypatch):
        from app.runners.worker import WorkerRunner

        monkeypatch.setattr(
            entry,
            "worker_command",
            lambda role, passthrough=None, module=None: [sys.executable, "-c", self._CHILD_SCRIPT],
        )

        runner = WorkerRunner("script")
        frames: list[g.GaugeFrame] = []
        outputs: list[str] = []
        runner.gauge_frame.connect(lambda f: frames.append(f))
        runner.output.connect(lambda t: outputs.append(t))

        assert runner.start(Settings(), module="dummy")
        assert runner._process.waitForFinished(5000)

        assert len(frames) == 5
        joined_output = "".join(outputs)
        assert "@@GAUGE" not in joined_output
        assert "ordinary line" in joined_output


class TestChildOutputEncoding:
    """子の標準出力は UTF-8 で、書いたそばから届く（親は UTF-8 で読み、ログを逐次出す）。

    Windows のパイプの既定（cp932）だと、本体の "✅" の print で子が UnicodeEncodeError で落ちる。
    Python はパイプへの出力をためるので、flush しない print は子が終わるまでログに出なかった。
    ここでは親の環境に cp932 を残して前者を、flush しない print で後者を再現する。
    """

    def _run(self, monkeypatch, code: str):
        from app.runners.worker import WorkerRunner

        monkeypatch.setattr(entry, "worker_command",
                            lambda role, passthrough=None, module=None: [sys.executable, "-c", code])
        runner = WorkerRunner("script")
        outputs: list[str] = []
        runner.output.connect(outputs.append)
        assert runner.start(Settings(), module="dummy")
        return runner, outputs

    def test_the_child_writes_utf8_even_if_the_parent_says_otherwise(self, qt_app, monkeypatch):
        monkeypatch.setenv("PYTHONIOENCODING", "cp932")
        runner, outputs = self._run(monkeypatch, "print('✅ Mediapipe・モデル準備 完了')")
        codes: list[int] = []
        runner.finished.connect(codes.append)
        assert runner._process.waitForFinished(10_000)

        assert codes == [0], "".join(outputs)
        assert "✅ Mediapipe・モデル準備 完了" in "".join(outputs)

    def test_lines_arrive_while_the_child_is_still_running(self, qt_app, monkeypatch):
        import time

        monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
        # 目印は子が組み立てる（「[起動]」のログにコマンドの文字がそのまま出るので、それと区別する）
        runner, outputs = self._run(monkeypatch, "import time\nprint('step', 1 + 1)\ntime.sleep(30)\n")
        try:
            deadline = time.monotonic() + 10
            while "step 2" not in "".join(outputs) and time.monotonic() < deadline:
                runner._process.waitForReadyRead(100)
                runner._drain_output()
            assert "step 2" in "".join(outputs), "子が終わるまでログが届かない（出力がためられている）"
            assert runner.is_running
        finally:
            runner._process.kill()
            runner._process.waitForFinished(5000)
