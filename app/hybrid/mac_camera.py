"""Mac camera acquisition; timestamps use the same analysis-entry convention as Android."""

from __future__ import annotations
import os
import sys
import time
import cv2 as cv
from app.core.video_source import open_capture
from config import _parse_cam_env


def default_camera_index() -> int:
    """設定 CAM0 から Mac のカメラ番号を読む。番号でなければ 0 番。

    USB 経路は CAM0 に動画のパスや ``video=...`` も許す（``config._parse_cam_env``）。
    その設定のまま混成に切り替えても、起動の段階で落とさない。
    """
    value = _parse_cam_env(os.environ.get("CAM0"))
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    print(
        f"CAM0={value!r} はカメラ番号ではないため 0 番を使います（--camera で指定できます）",
        file=sys.stderr,
    )
    return 0


class MacCamera:
    def __init__(
        self, index=0, *, size=(1280, 720), opener=open_capture, clock=time.monotonic_ns
    ):
        self._clock = clock
        opened = opener(index)
        if opened is None:
            raise RuntimeError(
                f"カメラ {index} を開けません。カメラ番号と macOS の権限を確認してください"
            )
        _, self._capture = opened
        try:
            self._capture.set(cv.CAP_PROP_FRAME_WIDTH, size[0])
            self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, size[1])
            for _ in range(10):
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Mac カメラから画像を取得できません")
            self.size = (frame.shape[1], frame.shape[0])
            if self.size != tuple(size):
                raise ValueError(
                    f"Mac カメラの実寸 {self.size} が指定 {size} と異なります"
                )
        except BaseException:
            self.close()
            raise

    def read(self):
        ok = self._capture.grab()
        stamp = self._clock()
        if not ok:
            raise RuntimeError("Mac カメラの取り込みが止まりました")
        ok, frame = self._capture.retrieve()
        if not ok or frame is None:
            raise RuntimeError("Mac カメラの画像を取得できません")
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("Mac カメラの解像度が途中で変わりました")
        return stamp, frame

    def close(self):
        self._capture.release()
