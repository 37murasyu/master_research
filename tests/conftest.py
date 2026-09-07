"""テスト全体の前提。

GUI テストを画面なしで走らせるため、Qt のプラットフォームプラグインを
offscreen にする。CI（GitHub Actions）にはディスプレイが無いので、
これが無いと Qt が起動できない。

Qt バックエンドの選択は app/__init__.py が行うので、ここでは触らない。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# matplotlib も画面を開かないバックエンドにする。
os.environ.setdefault("MPLBACKEND", "Agg")
