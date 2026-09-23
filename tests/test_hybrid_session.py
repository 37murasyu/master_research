"""PC 側の session を保存して使い回す。

Pixel のアプリは最後に読んだ接続先へ自動でつなぎ直す。PC 側が起動のたびに session を
作り直すと「古い QR」として断られ、そのたびに QR を読み直すことになる。
"""

from app.hybrid.link import PhoneLink
from app.hybrid.session import stable_session


def test_session_is_kept_across_runs(tmp_path):
    path = tmp_path / "session.txt"
    first = stable_session(path)
    assert stable_session(path) == first
    assert path.read_text().strip() == first


def test_renew_makes_a_new_session(tmp_path):
    """古い端末をわざと締め出したいとき（--new-session）。"""
    path = tmp_path / "session.txt"
    first = stable_session(path)
    renewed = stable_session(path, renew=True)
    assert renewed != first
    assert stable_session(path) == renewed


def test_broken_file_is_replaced(tmp_path):
    path = tmp_path / "session.txt"
    path.write_text("これは session ではない\n")
    value = stable_session(path)
    assert value != "これは session ではない"
    assert stable_session(path) == value


def test_link_uses_the_given_session():
    link = PhoneLink(host="127.0.0.1", port=0, session="0a1b2c3d")
    assert link.session == "0a1b2c3d"
    assert "session=0a1b2c3d" in link.url
