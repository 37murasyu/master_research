"""テスト全体の前提。

GUI テストを画面なしで走らせるため、Qt のプラットフォームプラグインを
offscreen にする。CI（GitHub Actions）にはディスプレイが無いので、
これが無いと Qt が起動できない。

Qt を import する前に設定する必要があるため、conftest.py で行う。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Qt バックエンドも固定する。開発機に PyQt5 が残っていると
# pyqtgraph がそちらを選び、ライセンス上まずい構成のままテストが通ってしまう。
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

# matplotlib も画面を開かないバックエンドにする。
os.environ.setdefault("MPLBACKEND", "Agg")
