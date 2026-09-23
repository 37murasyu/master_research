"""GUI から停止したとき、計測スクリプトがループを抜けて終了時の書き出しを行うことを固定する。

**なぜこのテストがあるか。**

GUI の停止（``WorkerRunner.stop``）は ``terminate()`` を送っていた。POSIX では SIGTERM だが、
``master_research_code.py`` にハンドラが無く即死し、ループの後で書く ``kpts3d_*.csv``・トルク CSV・
サイクルの診断 CSV が失われていた（KNOWN_ISSUES §3-2）。Windows の ``QProcess.terminate()`` は
WM_CLOSE を送るだけで、コンソールのプロセスには届かない。

いまは停止ファイル（環境変数 ``APP_STOP_FILE``）を置いて知らせ、スクリプトはループの先頭で
それを見て抜ける。OS に依らない。SIGTERM（Windows は SIGBREAK）も同じ停止要求として扱う。
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.core.stop_request import STOP_FILE_ENV, StopRequest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestStopRequest:
    def test_nothing_is_requested_at_first(self, tmp_path):
        assert not StopRequest(tmp_path / "stop").requested()

    def test_the_stop_file_requests_a_stop(self, tmp_path):
        stop = StopRequest(tmp_path / "stop")
        (tmp_path / "stop").touch()
        assert stop.requested()

    def test_the_path_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv(STOP_FILE_ENV, str(tmp_path / "stop"))
        stop = StopRequest.from_environment()
        (tmp_path / "stop").touch()
        assert stop.requested()

    def test_without_a_path_nothing_is_requested(self, monkeypatch):
        monkeypatch.delenv(STOP_FILE_ENV, raising=False)
        assert not StopRequest.from_environment().requested()


def _child(marker: Path) -> str:
    """計測ループを模した子。停止要求を受けたらループを抜け、終了時の書き出しをする。"""
    return textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from app.core.stop_request import StopRequest

        stop = StopRequest.from_environment()
        stop.install_signal_handlers()
        print("ready", flush=True)
        for _ in range(3000):
            if stop.requested():
                break
            time.sleep(0.01)
        # 計測ループを抜けた後の保存処理に相当
        open({str(marker)!r}, "w").write("done")
        """
    )


def _run_child(tmp_path, request_stop):
    marker = tmp_path / "end_of_run.txt"
    stop_file = tmp_path / "stop"
    env = {**__import__("os").environ, STOP_FILE_ENV: str(stop_file)}
    proc = subprocess.Popen([sys.executable, "-c", _child(marker)], stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert proc.stdout.readline().strip() == "ready", "子プロセスがループまで進まなかった"
        request_stop(proc, stop_file)
        code = proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
    return code, marker


class TestTheLoopEndsCleanly:
    def test_the_stop_file_lets_the_end_of_run_code_run(self, tmp_path):
        code, marker = _run_child(tmp_path, lambda proc, stop_file: stop_file.touch())
        assert code == 0
        assert marker.exists(), "停止ファイルを置いても終了時の書き出しが走らなかった"

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM を送れるのは POSIX")
    def test_sigterm_lets_the_end_of_run_code_run(self, tmp_path):
        code, marker = _run_child(tmp_path, lambda proc, stop_file: proc.send_signal(signal.SIGTERM))
        assert code == 0
        assert marker.exists(), "SIGTERM で終了時の書き出しが走らなかった"


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM を送れるのは POSIX")
class TestSecondSignal:
    """停止要求の後にループが固まっても、2 回目の SIGTERM では止まる（端末から kill できる）。"""

    def test_a_second_sigterm_terminates(self, tmp_path):
        child = textwrap.dedent(
            f"""
            import sys, time
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from app.core.stop_request import StopRequest

            stop = StopRequest()
            stop.install_signal_handlers()
            print("ready", flush=True)
            while not stop.requested():
                time.sleep(0.01)
            print("requested", flush=True)
            time.sleep(60)   # 終了時処理が固まったつもり
            """
        )
        proc = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout.readline().strip() == "ready"
            proc.send_signal(signal.SIGTERM)
            assert proc.stdout.readline().strip() == "requested"
            proc.send_signal(signal.SIGTERM)
            code = proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()
        assert code == -signal.SIGTERM, "2 回目の SIGTERM でも止まらない"


class TestWiring:
    """ランナーが停止ファイルを渡し、計測スクリプトがループの先頭でそれを見る。"""

    def test_worker_environment_carries_the_stop_file(self):
        from app import entry
        from app.core.settings import Settings

        env = entry.worker_environment(Settings(), role="realtime", stop_file="/tmp/x/stop")
        assert env[STOP_FILE_ENV] == "/tmp/x/stop"

    def test_no_stop_file_leaves_the_variable_unset(self, monkeypatch):
        from app import entry
        from app.core.settings import Settings

        monkeypatch.delenv(STOP_FILE_ENV, raising=False)
        assert STOP_FILE_ENV not in entry.worker_environment(Settings(), role="realtime")

    def test_the_main_loop_checks_the_request_first(self):
        """ループの本体の先頭で見る。ループの途中には continue が多く、末尾に届かない周回がある。"""
        import ast

        tree = ast.parse((REPO_ROOT / "master_research_code.py").read_text(encoding="utf-8"))
        loops = [n for n in tree.body if isinstance(n, ast.While)]
        assert loops, "master_research_code.py にトップレベルの while ループが無い"
        first = loops[0].body[0]
        assert isinstance(first, ast.If) and "requested" in ast.unparse(first.test), (
            "メインループの最初の文が停止要求の判定になっていない")
