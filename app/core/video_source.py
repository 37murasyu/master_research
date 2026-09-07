"""映像入力の種類を明示的に表現する層。

既存の ``video_io.resolve_input_streams()`` は「int ならカメラ、それ以外はファイル」
という二分法だった（``video_io.py:79-80``）。この分類には**ライブだが文字列**という
第3の入力（ネットワークストリーム）を表す場所が無い。

``file_mode`` が真になると下流が動画ファイル専用の挙動を発動する:

- 先頭に巻き戻す（``master_research_code.py:1802-1808, 1835-1842``）
- カメラ制御をスキップする（同 ``2809``）
- 終端でループ再生する（同 ``2968``）

ネットワーク入力にこれらが適用されると誤動作する。そこで「巻き戻せるか」
「カメラ制御が効くか」を**種類から導出できる属性として持たせる**。

これはスマホ無線化に備えた継ぎ目でもある。ただし実際に無線でステレオ計測を
成立させるには、この層だけでは足りない（2台のフレーム同期が別途必要）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# pylint: disable=no-member
import cv2 as cv
import numpy as np

from app.core.platform_compat import camera_backends

__all__ = [
    "SourceKind",
    "SourceSpec",
    "parse_spec",
    "open_capture",
    "open_source",
    "VideoSource",
]


class SourceKind(Enum):
    USB = "usb"
    FILE = "file"
    NETWORK = "network"


# OpenCV が扱えるストリーム系スキーム。
# 1 文字のスキームを弾くのが要点で、これが無いと Windows のドライブレター
# ``C:\videos\cam0.mp4`` を ``c:`` スキームの URL と誤認する。
_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]{1,}://")
_STREAM_SCHEMES = {"rtsp", "rtmp", "http", "https", "udp", "tcp", "rtp", "srt"}


@dataclass(frozen=True)
class SourceSpec:
    """入力ソースの正規化された指定。"""

    kind: SourceKind
    value: int | str

    @property
    def is_live(self) -> bool:
        """録画済みではなく、その場で流れてくる映像か。"""
        return self.kind in (SourceKind.USB, SourceKind.NETWORK)

    @property
    def is_seekable(self) -> bool:
        """巻き戻し・ループ再生をしてよいか。ファイルだけが該当する。"""
        return self.kind is SourceKind.FILE

    @property
    def supports_camera_controls(self) -> bool:
        """露出・フォーカス固定などの UVC プロパティが効くか。

        ``master_research_code.py:2802-2891`` の ``_apply_camera_controls()`` は
        UVC 前提なので、ファイルにもネットワークにも効かない。
        """
        return self.kind is SourceKind.USB

    @property
    def backends(self) -> list[int]:
        """``cv.VideoCapture(value, backend)`` に渡す候補を優先順に返す。"""
        if self.kind is SourceKind.USB:
            return camera_backends()

        preferred = getattr(cv, "CAP_FFMPEG", None)
        backends = [preferred] if preferred is not None else []
        if cv.CAP_ANY not in backends:
            backends.append(cv.CAP_ANY)
        return backends


def parse_spec(spec: int | str) -> SourceSpec:
    """``config.input_stream1`` 相当の値を種類つきで解釈する。

    **URL は一切加工しない**。``os.path.normpath()`` に通すと ``rtsp://host`` が
    ``rtsp:/host`` に潰れる（``video_io.py:109`` が踏んでいる地雷）。
    """
    if isinstance(spec, int):
        return SourceSpec(SourceKind.USB, spec)

    text = str(spec).strip()

    # "1" のような数字文字列はデバイス番号（環境変数 CAM0 から来る形）
    if text.lstrip("-+").isdigit():
        return SourceSpec(SourceKind.USB, int(text))

    match = _URL_RE.match(text)
    if match:
        scheme = text.split("://", 1)[0].lower()
        if scheme in _STREAM_SCHEMES:
            return SourceSpec(SourceKind.NETWORK, text)  # そのまま渡す

    # DirectShow のデバイス名指定（config.py が想定している形）
    if text.lower().startswith("video="):
        return SourceSpec(SourceKind.USB, text)

    return SourceSpec(SourceKind.FILE, text)


class VideoSource:
    """``cv.VideoCapture`` の薄いラッパ。種類ごとの違いを属性として持ち回る。"""

    def __init__(self, spec: SourceSpec, capture: cv.VideoCapture):
        self.spec = spec
        self._capture = capture

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def fps(self) -> float:
        """ソースが申告する fps。信用できない値は 0 を返す。

        呼び出し側は 0 のとき実測にフォールバックすること。ネットワーク入力は
        でたらめな値を返すことがある。
        """
        try:
            value = float(self._capture.get(cv.CAP_PROP_FPS))
        except Exception:
            return 0.0
        return value if 1.0 <= value <= 1000.0 else 0.0

    @property
    def frame_size(self) -> tuple[int, int]:
        try:
            w = int(self._capture.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(self._capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            return (0, 0)
        return (w, h)

    def grab(self) -> bool:
        return bool(self._capture.grab())

    def retrieve(self) -> tuple[bool, "np.ndarray | None"]:
        ok, frame = self._capture.retrieve()
        return bool(ok), frame if ok else None

    def read(self) -> tuple[bool, "np.ndarray | None"]:
        ok, frame = self._capture.read()
        return bool(ok), frame if ok else None

    def rewind(self) -> bool:
        """先頭に巻き戻す。ファイル以外では何もせず False を返す。"""
        if not self.spec.is_seekable:
            return False
        return bool(self._capture.set(cv.CAP_PROP_POS_FRAMES, 0))

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()


def open_capture(spec: int | str) -> "tuple[SourceSpec, cv.VideoCapture] | None":
    """種類に応じたバックエンドを順に試して開く。開けなければ None。

    「OS ごとの定数」ではなく「開き方」をここに集約するのが要点。
    定数だけを配ると、呼び出し側が全員「順に試す・失敗したら release する」
    ループを書き直すことになる（実際 5 箇所に複製されていた）。

    素の ``cv.VideoCapture`` を返すのは、解像度設定など OpenCV の API を
    直接使いたい呼び出し側（``calib.py``）があるため。
    ラップした形が欲しい場合は :func:`open_source` を使う。
    """
    parsed = parse_spec(spec)

    for backend in parsed.backends:
        try:
            capture = cv.VideoCapture(parsed.value, backend)
        except Exception:
            continue
        if capture is not None and capture.isOpened():
            return parsed, capture
        if capture is not None:
            capture.release()

    return None


def open_source(spec: int | str) -> VideoSource | None:
    """入力を開く。開けなければ **例外ではなく None** を返す。

    無線化すると接続失敗は日常的に起きるので、呼び出し側で扱えるようにする。
    """
    opened = open_capture(spec)
    if opened is None:
        return None
    parsed, capture = opened
    return VideoSource(parsed, capture)
