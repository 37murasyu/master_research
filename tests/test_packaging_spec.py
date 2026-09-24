"""凍結アプリ（``packaging/app.spec``）に同梱するファイルの置き場所を確かめる。

凍結すると、``resources.resource_root()`` と、計測スクリプトの ``os.path.dirname(__file__)`` は
どちらも ``sys._MEIPASS`` の直下（根）を指す。そこから読むファイルは、spec の ``datas`` で根（``"."``）に
置かないと、開発時は動くのに凍結したときだけ起動直後に落ちる。ワークスペースの初期値（``seed/``）に
入れただけでは根には無い（Mac＋Pixel の校正の ``board_defaults`` が FileNotFoundError で落ちていた）。

spec は PyInstaller が与える名前（``SPECPATH``・``Analysis`` など）を前提に書かれているので、
それらを差し替えて実行し、``datas`` の一覧を取り出す。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6", reason="spec は解析画面の TASKS を読むので Qt が要る")

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = REPO_ROOT / "packaging" / "app.spec"

# 凍結時に _MEIPASS の根から読むファイルと、読む場所
ROOT_READS = {
    # app/runners/hybrid_calibrate.py の board_defaults（resources.resource_root()）
    "calibration_settings.yaml": "hybrid_calibrate.board_defaults",
    # app/hybrid/pose_detector.py（resource_root()）と master_research_code.py（dirname(__file__)）
    "pose_landmarker_lite.task": "pose_detector / master_research_code",
    # master_research_code.py の config_path（dirname(__file__)）
    "gauge_layout.json": "master_research_code",
    # master_research_code.py の stats_file（dirname(__file__)）。無いと _MEIPASS に雛形を書こうとする
    "supervision_stats.csv": "master_research_code",
}


def _spec_datas(monkeypatch) -> list[tuple[str, str]]:
    """PyInstaller の名前を差し替えて spec を実行し、``datas`` を返す。"""
    monkeypatch.setattr(sys, "path", list(sys.path))  # spec が sys.path に足すのを試験の後に戻す
    built = SimpleNamespace(scripts=[], binaries=[], datas=[], pure=[])
    namespace = {"SPECPATH": str(SPEC.parent), "__name__": "__main__"}
    for name in ("Analysis", "PYZ", "EXE", "COLLECT", "BUNDLE"):
        namespace[name] = lambda *args, **kwargs: built
    exec(compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec"), namespace)  # noqa: S102
    return [(str(src), str(dest)) for src, dest in namespace["datas"]]


def _bundle(datas: list[tuple[str, str]], meipass: Path) -> Path:
    """``datas`` のとおりに ``_MEIPASS`` を組み立てる（PyInstaller と同じ置き方）。"""
    for src, dest in datas:
        source = Path(src)
        target_dir = meipass / dest
        target_dir.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target_dir / source.name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target_dir / source.name)
    return meipass


class TestRootFiles:
    @pytest.mark.parametrize("name", sorted(ROOT_READS))
    def test_files_read_from_the_bundle_root_are_placed_there(self, name, monkeypatch):
        at_root = {Path(src).name for src, dest in _spec_datas(monkeypatch) if Path(dest) == Path(".")}
        assert name in at_root, f"{name}（{ROOT_READS[name]} が読む）が凍結アプリの根に無い"

    def test_every_root_file_exists_in_the_repository(self, monkeypatch):
        """spec が参照するファイルが無いとビルドが落ちる。"""
        missing = [src for src, _dest in _spec_datas(monkeypatch) if not Path(src).exists()]
        assert missing == []

    def test_the_hybrid_calibration_reads_the_board_from_the_bundle(self, monkeypatch, tmp_path):
        """凍結と同じ置き方の _MEIPASS で、Mac＋Pixel の校正が盤の既定値を読めること（起動直後に落ちない）。"""
        from app.core import resources
        from app.runners import hybrid_calibrate

        meipass = _bundle(_spec_datas(monkeypatch), tmp_path / "meipass")
        monkeypatch.setattr(resources, "is_frozen", lambda: True)
        monkeypatch.setattr(resources, "resource_root", lambda: meipass)

        board = hybrid_calibrate.board_defaults()
        assert set(board) == {"rows", "cols", "square_cm"}
