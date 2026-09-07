"""映像入力の抽象層を検証する。

現状 ``video_io.resolve_input_streams()`` は「int ならカメラ、それ以外はファイル」
という二分法で、`file_mode=True` になると下流が

  - 先頭に巻き戻す（master_research_code.py:1802-1808）
  - カメラ制御をスキップ（同 2809）
  - 終端でループ再生（同 2968）

という**動画ファイル専用の挙動**を発動させる。ネットワークストリームは
「ライブだが文字列」なので、この二分法では表現できない。

さらに ``video_io.py:109`` の ``os.path.normpath()`` は ``rtsp://host`` を
``rtsp:/host`` に潰してしまう。ここを踏まないことを保証する。
"""

from __future__ import annotations

import pytest

from app.core import video_source as vs


class TestSourceKindDetection:
    def test_integer_is_usb_camera(self):
        spec = vs.parse_spec(0)
        assert spec.kind is vs.SourceKind.USB
        assert spec.is_live is True

    def test_numeric_string_is_usb_camera(self):
        """CAM0=1 のように環境変数から来ると文字列になる。"""
        spec = vs.parse_spec("1")
        assert spec.kind is vs.SourceKind.USB
        assert spec.value == 1

    def test_file_path_is_file(self):
        spec = vs.parse_spec("media/cam000_test.mp4")
        assert spec.kind is vs.SourceKind.FILE
        assert spec.is_live is False

    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://192.168.1.10:554/stream",
            "http://192.168.1.10:8080/video",
            "https://example.com/live.m3u8",
            "rtmp://host/live",
            "udp://@:1234",
            "tcp://192.168.1.10:5000",
        ],
    )
    def test_urls_are_network_and_live(self, url):
        spec = vs.parse_spec(url)
        assert spec.kind is vs.SourceKind.NETWORK
        assert spec.is_live is True, "ネットワークはライブ。巻き戻しやループ再生をしてはいけない"

    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://192.168.1.10:554/stream",
            "http://192.168.1.10:8080/video?x=1&y=2",
        ],
    )
    def test_url_is_passed_through_untouched(self, url):
        """``os.path.normpath`` に通すと ``rtsp://`` が ``rtsp:/`` に壊れる。

        既存の video_io.py:109 が踏んでいる地雷。ここでは絶対に踏まない。
        """
        assert vs.parse_spec(url).value == url

    @pytest.mark.parametrize(
        "path",
        [
            r"C:\videos\cam0.mp4",
            r"D:\data\9_20250925\cam1.mp4",
        ],
    )
    def test_windows_drive_letter_is_not_mistaken_for_url(self, path):
        """``C:\\...`` はスキーム ``c:`` に見える。1 文字のスキームは URL ではない。"""
        spec = vs.parse_spec(path)
        assert spec.kind is vs.SourceKind.FILE, f"{path} を URL と誤認した"

    def test_device_name_string_is_usb(self):
        """DirectShow の ``video=HD Pro Webcam C920`` 形式。config.py が想定している。"""
        spec = vs.parse_spec("video=HD Pro Webcam C920")
        assert spec.kind is vs.SourceKind.USB


class TestLiveVsFileSemantics:
    def test_only_file_sources_are_seekable(self):
        """巻き戻し・ループ再生をしてよいのはファイルだけ。"""
        assert vs.parse_spec("media/x.mp4").is_seekable is True
        assert vs.parse_spec(0).is_seekable is False
        assert vs.parse_spec("rtsp://h/s").is_seekable is False

    def test_only_usb_supports_camera_controls(self):
        """露出やフォーカスの固定は UVC のプロパティ経由。ファイルや無線には効かない。"""
        assert vs.parse_spec(0).supports_camera_controls is True
        assert vs.parse_spec("media/x.mp4").supports_camera_controls is False
        assert vs.parse_spec("rtsp://h/s").supports_camera_controls is False


class TestBackendSelection:
    def test_usb_uses_platform_backends(self):
        from app.core.platform_compat import camera_backends

        assert vs.parse_spec(0).backends == camera_backends()

    def test_network_prefers_ffmpeg(self):
        import cv2 as cv

        backends = vs.parse_spec("rtsp://h/s").backends
        assert backends[0] == getattr(cv, "CAP_FFMPEG", cv.CAP_ANY)

    def test_backends_always_end_with_cap_any(self):
        import cv2 as cv

        for spec in (vs.parse_spec(0), vs.parse_spec("x.mp4"), vs.parse_spec("rtsp://h/s")):
            assert spec.backends[-1] == cv.CAP_ANY


class TestOpening:
    def test_open_nonexistent_file_reports_failure_not_exception(self, tmp_path):
        """存在しないファイルで例外を投げず、失敗として返すこと。

        無線化すると接続失敗は日常的に起きる。呼び出し側で扱えるようにする。
        """
        source = vs.open_source(str(tmp_path / "nope.mp4"))
        assert source is None or not source.is_opened
        if source is not None:
            source.release()
