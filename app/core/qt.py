"""Qt の型を 1 箇所から取り出す。

バックエンドの選択（``PYQTGRAPH_QT_LIB``）は ``app/__init__.py`` が行う。
このモジュールを import する時点で ``app`` パッケージは読み込まれているので、
ここで環境変数を触る必要はない。
"""

from __future__ import annotations

from pyqtgraph.Qt import QT_LIB, QtCore, QtGui, QtSvg, QtWidgets

__all__ = ["QtCore", "QtGui", "QtSvg", "QtWidgets", "QT_LIB", "assert_lgpl_backend"]


def assert_lgpl_backend() -> None:
    """配布に適さない Qt バインディングが選ばれていないか確かめる。

    開発機にたまたま PyQt5 が入っていると静かにそちらが使われ、
    ライセンス違反に気づかないまま配布してしまう。起動時に明示的に落とす。
    """
    if QT_LIB not in ("PySide6", "PySide2"):
        raise RuntimeError(
            f"Qt バックエンドが {QT_LIB} になっています。\n"
            f"  配布物には LGPL の PySide6 を使ってください（PyQt5/PyQt6 は GPL v3）。\n"
            f"  PyQt5 をアンインストールするか、app/__init__.py の設定を確認してください。"
        )
