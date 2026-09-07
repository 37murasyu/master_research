"""車椅子駆動の関節トルク計測アプリ。

Qt バックエンドをここで確定させる。``app.core.qt`` の import 副作用に頼ると、
Qt を import しない経路（計測ワーカー）が保証の外に落ちる:

    GUI     : app/__main__.py -> app.shell.main_window -> app.core.qt   （効く）
    ワーカー: app/__main__.py -> entry.run_worker -> master_research_code
              -> Gauge_display -> pyqtgraph                              （通らない）

``app`` パッケージを 1 つでも import すれば効くので、GUI・ワーカー・テスト・
凍結 exe が同じ 1 行でカバーされる。

PySide6 を選ぶ理由は LGPL であること。PyQt5 は GPL v3 で、配布物に含めると
アプリ全体にソース開示義務が及ぶ。
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
