"""既存スクリプトを別プロセスで動かすランナー。

なぜ別プロセスか。``master_research_code.py`` は 4,383 行のトップレベル
直書きスクリプトで、import 時に実行される文が 359 行、メインループは
2,936 行目にあり ``__main__`` ガードも無い。これを GUI と同じプロセスの
スレッドで動かすには 2,900 行の初期化を解きほぐす必要があり、
動いている計測パイプラインを壊すリスクが高い。

プロセスを分ければ:

- 既存コードを 1 行も変えずに済む（設定は環境変数で渡す）
- 計測側がクラッシュしても GUI が生き残る
- OpenCV や matplotlib のウィンドウ管理が GUI と干渉しない

代償は、ゲージと映像が子プロセス側のウィンドウとして出ること。
将来 GUI 側に取り込むなら、子が標準出力に JSON Lines を吐く経路を足す。

キャリブレーションも同じ仕組みで動かす。calib.py の撮影フェーズは対話式の
``cv.imshow`` + ``waitKey`` で、macOS では GUI 操作がメインスレッド必須のため、
QThread から呼ぶと "Unknown C++ exception from OpenCV code" になる（実測済み）。
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from app import entry
from app.core.qt import QtCore
from app.core.settings import Settings

__all__ = ["WorkerRunner"]


class WorkerRunner(QtCore.QObject):
    """ワーカー（計測 / キャリブレーション）の起動・停止と、出力の中継。"""

    output = QtCore.Signal(str)
    state_changed = QtCore.Signal(str)  # "starting" / "running" / "stopped"
    finished = QtCore.Signal(int)  # 終了コード

    # 停止要求からの猶予。これを過ぎたら強制終了する。
    # 計測終了時に CSV を書き出すので、その時間は待つ必要がある。
    GRACE_MS = 10_000

    def __init__(self, role: str = "realtime", parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.role = role
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._drain_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        # 停止要求のファイルを置くディレクトリ。実行ごとに作り直すので前回の残りを考えなくてよい
        self._stop_dir: str | None = None

    # -- 操作 --------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._process.state() != QtCore.QProcess.NotRunning

    def start(
        self,
        settings: Settings,
        passthrough: list[str] | None = None,
        module: str | None = None,
    ) -> bool:
        """ワーカーを起動する。

        ``module`` は ``--role script`` のときに実行するモジュール名。
        インスタンスの状態にせず引数で受けるのは、1 回の実行に属する情報だから。
        状態に置くと呼び出し側が start の直前に代入する形になり、
        代入忘れや「前回の値が残る」が起こりうる。
        """
        if self.is_running:
            self.output.emit("[警告] 既に動いています。\n")
            return False

        command = entry.worker_command(self.role, passthrough, module=module)
        # 穏やかな停止に対応するワーカーへ停止ファイルを渡す。
        self._stop_dir = tempfile.mkdtemp(prefix="wt_stop_") if entry.uses_stop_file(self.role) else None
        environment = entry.worker_environment(settings, role=self.role, stop_file=self._stop_file())

        process_env = QtCore.QProcessEnvironment()
        for key, value in environment.items():
            process_env.insert(key, value)
        self._process.setProcessEnvironment(process_env)

        self.state_changed.emit("starting")
        self.output.emit(f"[起動] {' '.join(command)}\n")
        self._process.start(command[0], command[1:])

        if not self._process.waitForStarted(5000):
            self.output.emit(f"[エラー] 起動できませんでした: {self._process.errorString()}\n")
            # 起動に失敗すると finished が来ないので、ここで片付ける
            self._remove_stop_dir()
            self.state_changed.emit("stopped")
            return False

        # 子の標準入力を閉じる。既存コードには input() で対話入力を待つ箇所があり
        # （master_research_code.py:905 の被験者番号）、GUI から起動すると端末が
        # 無いため**永久にハングする**。閉じておけば EOFError になり、
        # 呼び出し側の try/except が拾って既定の経路に進む。
        # 値そのものは環境変数（SUBJECT_ID）で渡す。
        self._process.closeWriteChannel()

        self.state_changed.emit("running")
        return True

    def stop(self) -> None:
        """穏やかに止める。応じなければ強制終了する。

        計測終了時に 3D 座標とトルクの CSV を書き出すので、
        いきなり kill すると成果物が失われる。

        計測（realtime）には停止ファイルを置いて知らせる。スクリプトはループの先頭でそれを見て抜け、
        終了時の書き出しをする（``app.core.stop_request``）。かつては ``terminate()`` だけで、
        POSIX の SIGTERM ではハンドラが無く即死し、Windows では WM_CLOSE がコンソールに届かず、
        どちらでも CSV が書かれなかった（KNOWN_ISSUES §3-2）。停止ファイルを見ない役割
        （キャリブレーション・解析スクリプト）には従来どおり ``terminate()`` を送る。
        """
        if not self.is_running:
            return

        self.output.emit("[停止] 終了を要求しました。CSV の書き出しを待ちます。\n")
        stop_file = self._stop_file()
        if entry.uses_stop_file(self.role) and stop_file is not None:
            Path(stop_file).touch()
        else:
            self._process.terminate()

        if not self._process.waitForFinished(self.GRACE_MS):
            self.output.emit("[停止] 応答が無いため強制終了します。\n")
            self._process.kill()
            self._process.waitForFinished(2000)

    # -- 内部 --------------------------------------------------------------
    def _stop_file(self) -> str | None:
        return None if self._stop_dir is None else str(Path(self._stop_dir) / "stop")

    def _remove_stop_dir(self) -> None:
        if self._stop_dir is not None:
            shutil.rmtree(self._stop_dir, ignore_errors=True)
            self._stop_dir = None

    def _drain_output(self) -> None:
        data = self._process.readAllStandardOutput()
        text = bytes(data).decode("utf-8", errors="replace")
        if text:
            self.output.emit(text)

    def _on_finished(self, exit_code: int, _status) -> None:
        self._drain_output()
        self._remove_stop_dir()
        self.output.emit(f"[終了] 終了コード {exit_code}\n")
        self.state_changed.emit("stopped")
        self.finished.emit(exit_code)

    def _on_error(self, error) -> None:
        # FailedToStart 以外は finished でも拾えるので、ここでは説明だけ出す。
        self.output.emit(f"[エラー] {self._process.errorString()} ({error})\n")
