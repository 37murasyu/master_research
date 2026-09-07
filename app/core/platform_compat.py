"""OS 差を 1 箇所に閉じ込める層。

既存コードは Windows 前提の記述が各所に散っていた。代表例:

- ``calib.py:13`` の ``import wmi`` がトップレベル無条件 → macOS では
  calib.py を import した時点で ModuleNotFoundError になる
- ``calib.py:180, 390, 1221`` がバックエンドを ``CAP_DSHOW`` 固定 → macOS では
  1 台もカメラが見つからない
- ``config.py`` が ``os.makedirs("output_data")`` を CWD 相対で実行 → 凍結アプリでは
  Program Files 配下になり書き込めない

ここに集約し、呼び出し側は OS を意識しない。
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

# pylint: disable=no-member
import cv2 as cv

__all__ = [
    "is_windows",
    "is_macos",
    "is_linux",
    "camera_backends",
    "enumerate_camera_device_names",
    "enumerate_serial_ports",
    "user_config_dir",
    "user_data_dir",
    "user_output_dir",
    "beep",
]


# ---------------------------------------------------------------------------
# OS 判定
# ---------------------------------------------------------------------------
def is_windows() -> bool:
    return platform.system() == "Windows"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_linux() -> bool:
    return platform.system() == "Linux"


# ---------------------------------------------------------------------------
# カメラ
# ---------------------------------------------------------------------------
def camera_backends() -> list[int]:
    """``cv.VideoCapture(index, backend)`` に渡すバックエンドを優先順に返す。

    末尾は必ず ``CAP_ANY``。OS 固有のバックエンドがすべて失敗しても、
    OpenCV の自動選択に賭ける経路を残しておくため。

    定数は OpenCV のビルド構成によって存在しないことがあるので ``getattr`` で確認する
    （``video_io.py:100`` が既に同じ配慮をしている）。
    """
    if is_windows():
        preferred = ["CAP_DSHOW", "CAP_MSMF"]
    elif is_macos():
        preferred = ["CAP_AVFOUNDATION"]
    else:
        preferred = ["CAP_V4L2"]

    backends: list[int] = []
    for name in preferred:
        value = getattr(cv, name, None)
        if value is not None and value not in backends:
            backends.append(value)
    if cv.CAP_ANY not in backends:
        backends.append(cv.CAP_ANY)
    return backends


def enumerate_camera_device_names() -> list[str]:
    """接続されているカメラの表示名を返す。取得できない場合は空。

    名前は「どのカメラを校正済みか」の対応付け（``calib.load_camera_mapping``）に使う。
    取得できなくても計測自体は index 指定で動くので、**失敗は空リストで返し、例外にしない**。

    結果はキャッシュする（macOS の ``system_profiler`` は 1 秒前後かかるため、
    UI から繰り返し呼ばれても平気なように）。キャッシュ本体は tuple で保持し、
    呼び出し側にはコピーした list を返すので、呼び出し側が破壊してもキャッシュは汚れない。
    カメラを挿し直したときは ``enumerate_camera_device_names.cache_clear()`` を呼ぶ。
    """
    return list(_camera_device_names_cached())


def _cache_clear() -> None:
    _camera_device_names_cached.cache_clear()


enumerate_camera_device_names.cache_clear = _cache_clear  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def _camera_device_names_cached() -> tuple[str, ...]:
    try:
        if is_windows():
            return tuple(_camera_names_windows())
        if is_macos():
            return tuple(_camera_names_macos())
        return tuple(_camera_names_linux())
    except Exception:  # pragma: no cover - 環境依存の失敗は握りつぶす
        return ()


def _camera_names_windows() -> list[str]:
    # wmi は Windows 専用パッケージ。**関数内で import する**のが要点で、
    # これを怠ると macOS/Linux でモジュール全体が import できなくなる。
    try:
        import wmi  # type: ignore[import-not-found]
    except ImportError:
        return []

    conn = wmi.WMI()
    names = []
    for item in conn.Win32_PnPEntity():
        if item.Name and "camera" in item.Name.lower():
            names.append(item.Name)
    return names


def _camera_names_macos() -> list[str]:
    # AVFoundation を直に叩くには pyobjc-framework-AVFoundation が要る。
    # 依存を増やさずに済ませるため system_profiler の JSON 出力を読む。
    proc = subprocess.run(
        ["system_profiler", "-json", "SPCameraDataType"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if proc.returncode != 0:
        return []
    data = json.loads(proc.stdout or "{}")
    return [
        item.get("_name", "")
        for item in data.get("SPCameraDataType", [])
        if item.get("_name")
    ]


def _camera_names_linux() -> list[str]:
    names = []
    for node in sorted(Path("/sys/class/video4linux").glob("*/name")):
        try:
            names.append(node.read_text(encoding="utf-8").strip())
        except OSError:
            continue
    return names


# ---------------------------------------------------------------------------
# シリアル（HX711 ロードセルの M5StampS3 と繋ぐ）
# ---------------------------------------------------------------------------
def enumerate_serial_ports() -> list[str]:
    """シリアルポートのデバイス名を返す。Windows は ``COM*``、macOS は ``/dev/cu.*``。

    pyserial が無い環境でも落ちないよう、import は関数内で行う。
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        return []

    ports = [p.device for p in list_ports.comports()]
    if is_macos():
        # macOS には /dev/tty.* と /dev/cu.* が対で存在する。発信側は cu.* を使う
        # （tty.* は DCD を待ってブロックすることがある）。
        cu_ports = [p for p in ports if "/cu." in p]
        if cu_ports:
            return cu_ports
    return ports


# ---------------------------------------------------------------------------
# ディレクトリ
#
# 凍結アプリでは実行ファイルの隣に書き込めない（Windows の Program Files、
# macOS の .app バンドル内）。用途ごとに OS 標準の場所へ逃がす。
# ---------------------------------------------------------------------------
def user_config_dir(app_name: str) -> Path:
    """設定ファイルの置き場。ユーザには見せない前提。"""
    if is_windows():
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif is_macos():
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / app_name


def user_data_dir(app_name: str) -> Path:
    """キャッシュや内部状態の置き場。ユーザには見せない前提。"""
    if is_windows():
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif is_macos():
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / app_name


def user_output_dir(app_name: str) -> Path:
    """計測結果 CSV など、**ユーザが自分で開く**成果物の置き場。

    設定やキャッシュと違い、ここは見つけやすい場所である必要がある。
    現行の ``config.py:97`` は CWD 相対の ``output_data`` を使っており、
    凍結アプリでは書き込めない場所になり得る。
    """
    documents = Path.home() / "Documents"
    base = documents if documents.is_dir() else Path.home()
    return base / app_name


# ---------------------------------------------------------------------------
# 通知音
# ---------------------------------------------------------------------------
def beep(frequency_hz: int = 500, duration_ms: int = 1000) -> None:
    """計測の区切りを知らせる短い音。**鳴らせなくても絶対に例外を投げない**。

    旧コード（``twin_video_capture.py:6`` など）は ``winsound`` を直接 import
    していたため、macOS では import 時点で失敗していた。
    """
    try:
        if is_windows():
            import winsound  # type: ignore[import-not-found]

            winsound.Beep(int(frequency_hz), int(duration_ms))
            return

        if is_macos():
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Tink.aiff"],
                capture_output=True,
                timeout=max(1.0, duration_ms / 1000 + 1),
                check=False,
            )
            return

        # Linux はターミナルベルに留める（環境差が大きく、確実な手段が無い）
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:  # pragma: no cover - 音は本質でないので握りつぶす
        pass
