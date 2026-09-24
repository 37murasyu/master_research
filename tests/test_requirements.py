"""依存の一覧を ``requirements_min.txt`` の 1 か所にそろえることを固定する。

**なぜこのテストがあるか。**

``requirements.txt`` が ``requirements_min.txt`` と別に並べていたので食い違っていた。``opencv-python`` と、mediapipe が
要る ``opencv-contrib-python`` が同じ ``cv2`` に上書きで入る（どちらが使われるか分からない）、mediapipe の版の固定
（Solutions API のある 0.10.14）が無い、配布に入れない PyQt5（GPL）が入り PySide6 が無い。``README_pose_workflow.md``
はその ``requirements.txt`` を入れるよう書き、``pip install --upgrade mediapipe`` も勧めていた（上げると
``video_pose_extractor.py`` が使う ``mp.solutions.pose`` が無くなる）。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _entries(name: str) -> list[str]:
    lines = (REPO_ROOT / name).read_text(encoding="utf-8").splitlines()
    return [entry for entry in (line.split("#", 1)[0].strip() for line in lines) if entry]


def test_requirements_txt_only_points_at_the_minimal_list():
    entries = _entries("requirements.txt")
    assert "-r requirements_min.txt" in entries
    assert all(entry.startswith("-r ") for entry in entries), f"直接並べたパッケージがある（食い違いの元）: {entries}"
    assert "-r requirements_extra.txt" not in entries, "extra は PyQt5（GPL）を入れる。計測に要らない"


def test_the_minimal_list_has_one_opencv_and_a_pinned_mediapipe():
    names = [entry.split(";")[0].split("=")[0].split(">")[0].strip().lower() for entry in _entries("requirements_min.txt")]
    assert [n for n in names if n.startswith("opencv")] == ["opencv-contrib-python"]
    assert "mediapipe==0.10.14" in _entries("requirements_min.txt")
    assert "pyqt5" not in names and "pyside6" in names


def test_the_pose_workflow_readme_installs_the_pinned_list():
    text = (REPO_ROOT / "README_pose_workflow.md").read_text(encoding="utf-8")
    assert "pip install -r requirements_min.txt" in text
    assert "--upgrade mediapipe" not in text
