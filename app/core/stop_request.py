"""GUI から計測ワーカーへの停止要求。

GUI（``WorkerRunner.stop``）はかつて ``terminate()`` を送っていた。POSIX では SIGTERM だが、
計測スクリプトにハンドラが無く即死し、ループの後で書く CSV が失われていた（KNOWN_ISSUES §3-2）。
Windows の ``QProcess.terminate()`` は WM_CLOSE を送るだけで、コンソールのプロセスには届かない。

そこで停止ファイル（環境変数 ``APP_STOP_FILE`` が指すパス）を置いて知らせる。スクリプトは
ループの先頭で ``requested()`` を見て抜け、終了時の書き出しを行う。OS に依らない。
SIGTERM（Windows は SIGBREAK）も同じ停止要求として扱い、端末から止めたときにも書き出す。
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

__all__ = ["STOP_FILE_ENV", "StopRequest"]

STOP_FILE_ENV = "APP_STOP_FILE"


class StopRequest:
    """停止ファイルの出現か、停止のシグナルで立つフラグ。"""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self._signalled = False

    @classmethod
    def from_environment(cls) -> "StopRequest":
        return cls(os.environ.get(STOP_FILE_ENV, "").strip() or None)

    def install_signal_handlers(self) -> None:
        """SIGTERM・SIGBREAK を停止要求に読み替える。メインスレッドから呼ぶこと。

        ループに入る直前に呼ぶ。初期化中（カメラやモデルの読み込み）のシグナルは従来どおり即死させる。
        """
        for name in ("SIGTERM", "SIGBREAK"):
            number = getattr(signal, name, None)
            if number is not None:
                signal.signal(number, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        self._signalled = True
        # 2 回目は既定の動作（即死）に戻す。終了時処理が固まっても、端末から kill すれば止まる
        signal.signal(signum, signal.SIG_DFL)

    def requested(self) -> bool:
        return self._signalled or (self.path is not None and self.path.exists())
