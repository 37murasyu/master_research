"""PC 側の session をこの PC に保存して使い回す。

Pixel のアプリは最後に読んだ接続先（QR の URL）を覚え、切れたら自動でつなぎ直す。
PC 側が起動のたびに session を作り直すと、覚えていた接続先が「古い QR」として断られ、
そのたびに Pixel を手に取って QR を読み直すことになる。

古い端末をわざと締め出したいときだけ ``renew=True``（ランナーの ``--new-session``）で作り直す。
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path

from app.hybrid.paths import session_file

_VALID = re.compile(r"[0-9a-f]{8,32}")


def stable_session(path: Path | None = None, *, renew: bool = False) -> str:
    """保存してある session を返す。無い・壊れている・``renew`` のときは作って保存する。"""
    path = Path(path) if path is not None else session_file()
    if not renew:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        if _VALID.fullmatch(value):
            return value
    value = secrets.token_hex(4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    return value
