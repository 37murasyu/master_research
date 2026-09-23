"""PyInstaller の入口。``python -m app`` と同じ ``main`` を呼ぶ。

凍結後はこの実行ファイル自身が ``--role realtime`` などを付けて呼び直され、
ワーカーとしても動く（``app/entry.py`` の「多重エントリ」）。
"""

from app.__main__ import main

raise SystemExit(main())
