"""platform_compat の契約を検証する。

このモジュールの存在意義は「OS ごとの差で落ちないこと」なので、
テストも「どの OS でも例外を投げずに妥当な値を返す」ことを中心に書く。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2 as cv
import pytest

from app.core import platform_compat as pc


class TestPlatformDetection:
    def test_exactly_one_platform_is_true(self):
        """OS 判定は排他的であること。両方 True や全部 False は設定ミスを意味する。"""
        flags = [pc.is_windows(), pc.is_macos(), pc.is_linux()]
        assert sum(flags) == 1, f"OS 判定が排他的でない: {flags}"


class TestCameraBackends:
    def test_returns_non_empty_int_list(self):
        backends = pc.camera_backends()
        assert backends, "バックエンド候補が空"
        assert all(isinstance(b, int) for b in backends)

    def test_ends_with_cap_any(self):
        """最後は必ず CAP_ANY。どの OS でも「とりあえず開く」経路を残すため。"""
        assert pc.camera_backends()[-1] == cv.CAP_ANY

    def test_no_duplicates(self):
        backends = pc.camera_backends()
        assert len(backends) == len(set(backends))

    @pytest.mark.skipif(not pc.is_macos(), reason="macOS 固有")
    def test_macos_prefers_avfoundation_and_excludes_dshow(self):
        """macOS で DirectShow を試すのは無意味。AVFoundation が先頭に来ること。"""
        backends = pc.camera_backends()
        assert backends[0] == cv.CAP_AVFOUNDATION
        assert cv.CAP_DSHOW not in backends

    @pytest.mark.skipif(not pc.is_windows(), reason="Windows 固有")
    def test_windows_prefers_dshow(self):
        assert pc.camera_backends()[0] == cv.CAP_DSHOW


class TestCameraDeviceNames:
    def test_does_not_raise_on_any_platform(self):
        """これが本命。旧 calib.py は import wmi で macOS では import すら通らなかった。"""
        names = pc.enumerate_camera_device_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


class TestSerialPorts:
    def test_does_not_raise_and_returns_strings(self):
        ports = pc.enumerate_serial_ports()
        assert isinstance(ports, list)
        assert all(isinstance(p, str) for p in ports)


class TestUserDirectories:
    def test_config_dir_is_absolute_and_named(self):
        p = pc.user_config_dir("TestApp")
        assert isinstance(p, Path)
        assert p.is_absolute()
        assert "TestApp" in p.parts

    def test_data_dir_is_absolute_and_named(self):
        p = pc.user_data_dir("TestApp")
        assert p.is_absolute()
        assert "TestApp" in p.parts

    def test_dirs_are_not_inside_the_executable_directory(self):
        """凍結アプリでは実行ファイル隣は書き込み不可（Program Files など）。
        ユーザ領域に逃がせていることを確認する。"""
        exe_dir = Path(sys.executable).resolve().parent
        for p in (pc.user_config_dir("TestApp"), pc.user_data_dir("TestApp")):
            assert exe_dir not in p.resolve().parents

    @pytest.mark.skipif(not pc.is_macos(), reason="macOS 固有")
    def test_macos_uses_application_support(self):
        assert "Application Support" in str(pc.user_config_dir("TestApp"))


class TestBeep:
    def test_beep_never_raises(self):
        """音が出ない環境（CI・ヘッドレス）でも例外で計測を止めないこと。"""
        pc.beep(440, 10)
