"""Qt バックエンドを 1 箇所で確定させる。

``Gauge_display.py`` は ``pyqtgraph.Qt`` 経由で Qt を読み込んでおり、
バックエンドは自動選択される。pyqtgraph の既定の探索順は PyQt5 を先に見るため、
環境に PyQt5 が残っていると意図せずそちらが選ばれる。

配布物に PyQt5 を含めるとライセンス上の問題がある（GPL v3。アプリ全体に
ソース開示義務が及ぶ）。PySide6 は LGPL なのでその義務が生じない。
**pyqtgraph を import する前に**環境変数で指定して確定させる。

この module を Qt より先に import すること。
"""

from __future__ import annotations

import os

# pyqtgraph より先に設定する必要がある。import 済みだと効かない。
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # noqa: E402  (順序が重要)
from pyqtgraph.Qt import QT_LIB  # noqa: E402

__all__ = ["QtCore", "QtGui", "QtWidgets", "QT_LIB", "assert_lgpl_backend"]


def assert_lgpl_backend() -> None:
    """配布に適さない Qt バインディングが選ばれていないか確かめる。

    開発機にたまたま PyQt5 が入っていると静かにそちらが使われ、
    ライセンス違反に気づかないまま配布してしまう。起動時に明示的に落とす。
    """
    if QT_LIB not in ("PySide6", "PySide2"):
        raise RuntimeError(
            f"Qt バックエンドが {QT_LIB} になっています。\n"
            f"  配布物には LGPL の PySide6 を使ってください（PyQt5/PyQt6 は GPL v3）。\n"
            f"  PYQTGRAPH_QT_LIB=PySide6 を設定するか、PyQt5 をアンインストールしてください。"
        )
