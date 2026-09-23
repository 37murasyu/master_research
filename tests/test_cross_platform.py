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
    "app.tuning.raw_capture",
    "app.tuning.ekf_likelihood",
    "app.tuning.ekf_estimate",
    "app.tuning.ekf_profile",
    "app.hybrid.replay",
    "app.gauge.thresholds",
    "app.gauge.tracker",
    "app.hybrid.gravity",
    "app.hybrid.rep_detector",
    "app.hybrid.demo_gauge",
    "app.hybrid.gravity_board",
    "app.hybrid.link",
    "app.hybrid.mac_camera",
    "app.hybrid.pose_detector",
    "app.hybrid.display",
    "app.hybrid.live",
    "app.hybrid.checkerboard",
    "app.hybrid.calibration_io",
    "app.hybrid.collector",
    "app.hybrid.recorder",
    "app.hybrid.measurement",
    "app.hybrid.paths",
    "app.hybrid.session",
    "app.runners.hybrid_preview",
    "app.runners.hybrid_calibrate",
    "app.runners.hybrid_measure",

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


def test_camera_enumeration_works_without_wmi():
    """カメラ列挙が wmi 非依存で呼べること（Windows 以外では空を返す）。

    calib.py:13 の無条件 `import wmi` が macOS でモジュール全体を
    import 不能にしていた問題の回帰テスト。
    """
    importlib.import_module("calib")  # import 自体が通ること
    from app.core.platform_compat import enumerate_camera_device_names

    assert isinstance(enumerate_camera_device_names(), list)


# パス区切りにバックスラッシュを含む文字列。POSIX ではファイル名の一部として
# 扱われ、'/path/to/repo\rm_method.csv' のような存在しないパスになる。
_PATH_HINTS = (".csv", ".txt", ".dat", ".json", ".ttc", ".ttf", ".npy", ".mp4")


def _windows_path_literals(path: Path) -> list[tuple[int, str]]:
    """パスらしき文字列リテラルのうち、バックスラッシュ区切りのものを返す。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        text = node.value
        if "\\" not in text or len(text) > 120:
            continue
        looks_like_path = text.endswith(_PATH_HINTS) or "\\\\" in text or text.count("\\") > 1
        if looks_like_path and any(c.isalnum() for c in text):
            found.append((node.lineno, text))
    return found


def test_no_windows_path_separators_in_core_modules():
    """アプリが依存するモジュールで、パスをバックスラッシュで組み立てていないこと。

    ``config.py`` の ``folder_path + "\\\\rm_method.csv"`` は macOS で
    ``/Users/.../master_research\\rm_method.csv`` という存在しないパスになり、
    計測が FileNotFoundError で止まっていた。os.path.join / pathlib を使う。
    """
    modules = [
        "config.py",
        "utils.py",
        "utils_dynamic.py",
        "master_research_code.py",
        "JpText.py",
        "video_io.py",
    ]
    offenders = []
    for name in modules:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for lineno, text in _windows_path_literals(path):
            offenders.append(f"{name}:{lineno}  {text!r}")

    assert not offenders, (
        "パスをバックスラッシュで組み立てている箇所があります。\n"
        "os.path.join か pathlib を使ってください:\n  " + "\n  ".join(offenders)
    )


def test_bundled_font_is_used_instead_of_meiryo():
    """Meiryo はライセンス上同梱できないので、参照が残っていないこと。"""
    offenders = []
    for path in _own_source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "meiryo" not in line.lower():
                continue
            # 説明のためのコメントや docstring での言及は許す
            stripped = line.strip()
            if stripped.startswith("#") or "``" in line or "以前" in line:
                continue
            if "truetype" in line or "font_path" in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}  {stripped}")

    assert not offenders, (
        "Meiryo を読み込んでいる箇所があります。Microsoft の商用フォントなので\n"
        "配布物に同梱できません。app.core.resources.japanese_font_path() を使ってください:\n  "
        + "\n  ".join(offenders)
    )
