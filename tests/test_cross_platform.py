"""どの OS でもコードが読み込めることを守る回帰テスト。

このリポジトリは元々 Windows 実機専用だった。特に ``calib.py:13`` の
``import wmi`` がトップレベル無条件だったため、macOS ではキャリブレーション
モジュールを import した時点で落ちていた。同種の退行を防ぐ。

ソースを **実行せずに** ast で調べるのが要点。``twin_video_capture.py`` や
``master_research_code_00.py`` はトップレベルでカメラを開くので、
テストから import すると実際に動き出してしまう。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# import しただけで ImportError になってはいけないモジュール。
# アプリの 3 機能（計測・キャリブレーション・解析）が依存する範囲。
IMPORT_SAFE_MODULES = [
    "config",
    "utils",
    "utils_dynamic",
    "calib",
    "Gauge_display",
    "video_io",
    "link_vector_calculator_module",
    "body_part_storage_module",
    "energy_pipeline",
    "extended_kalman_filter",
    "pose_runtime",
    "app.core.platform_compat",
]

# Windows でしか存在しないモジュール。使うなら関数内 import か try で囲むこと。
WINDOWS_ONLY = {
    "winsound",
    "wmi",
    "comtypes",
    "msvcrt",
    "win32api",
    "win32com",
    "pywin32",
}

# サードパーティのコピーとビルド成果物。自分たちのコードではないので対象外。
EXCLUDED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".mypy_cache",
    "lambda_stereo",
    "lambda_test-python3.9",
    "build",
    ".pio",
    "node_modules",
}


def _own_source_files() -> list[Path]:
    return [
        p
        for p in sorted(REPO_ROOT.rglob("*.py"))
        if not any(part in EXCLUDED_DIRS for part in p.relative_to(REPO_ROOT).parts)
    ]


def _module_level_windows_imports(path: Path) -> list[tuple[int, str]]:
    """モジュールレベルで無条件に import されている Windows 専用モジュールを返す。

    ``tree.body`` だけを見るので、関数定義の中や ``try:`` の中の import は
    対象にならない。それらは意図的なガード付き import として許可する。
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        pytest.skip(f"構文解析できない: {path}")

    found: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in WINDOWS_ONLY:
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in WINDOWS_ONLY:
                found.append((node.lineno, node.module))
    return found


@pytest.mark.parametrize("module_name", IMPORT_SAFE_MODULES)
def test_module_imports_on_this_platform(module_name):
    """OS を問わず import できること。"""
    importlib.import_module(module_name)


def test_no_unconditional_windows_only_imports():
    """Windows 専用モジュールをトップレベルで無条件 import していないこと。"""
    offenders = []
    for path in _own_source_files():
        for lineno, name in _module_level_windows_imports(path):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno} import {name}")

    assert not offenders, (
        "Windows 専用モジュールがモジュールレベルで import されています。\n"
        "関数内 import にするか、app.core.platform_compat 経由にしてください:\n  "
        + "\n  ".join(offenders)
    )


def test_calib_exposes_camera_helpers_without_wmi():
    """calib のカメラ列挙が wmi 非依存で呼べること（Windows 以外では空を返す）。"""
    calib = importlib.import_module("calib")
    names = calib.enumerate_camera_device_names_windows()
    assert isinstance(names, list)
