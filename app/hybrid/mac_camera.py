"""Mac camera acquisition; timestamps use the same analysis-entry convention as Android.

起動時に USB と同じカメラ設定（``app.core.camera_controls``：自動露出・自動 WB・自動焦点をオフ、固定値）を
best effort でかけ、``set`` の戻り値・``get`` の値・起動時の実測 fps を 1 行で報告して ``controls`` に残す。
OpenCV の AVFoundation 経路は露出・WB・焦点をほぼ設定できない見込みで、pyobjc は入れない（Camo で番号と装置が
ずれる、.app の同梱が増える）。解像度と FOURCC は校正の寸法に合わせてここで決めるので、設定からは使わない。
"""

from __future__ import annotations
import os
import re
import sys
import time
import cv2 as cv
from app.core.camera_controls import apply_camera_controls
from app.core.video_source import open_capture
from config import _parse_cam_env

# 設定から除く鍵（CAM_WIDTH・CAM0_HEIGHT・CAM1_FOURCC など）。解像度と FOURCC は MacCamera が決める
_SIZE_KEYS = re.compile(r"CAM\d*_(WIDTH|HEIGHT|FOURCC)")
# 起動時の読み捨て（10 枚）のうち、実測 fps に使う枚数の始まり（最初の数枚は遅れがち）
_FPS_FROM = 2


def _prop_names():
    names = {}
    for attr, name in (("CAP_PROP_AUTO_EXPOSURE", "auto_exposure"), ("CAP_PROP_EXPOSURE", "exposure"),
                       ("CAP_PROP_GAIN", "gain"), ("CAP_PROP_FPS", "fps"), ("CAP_PROP_AUTO_WB", "auto_wb"),
                       ("CAP_PROP_WB_TEMPERATURE", "wb_temperature"), ("CAP_PROP_AUTOFOCUS", "autofocus"),
                       ("CAP_PROP_FOCUS", "focus")):
        if hasattr(cv, attr):
            names[getattr(cv, attr)] = name
    return names


class _RecordingCapture:
    """``apply_camera_controls`` に渡す包み。``set`` の戻り値と直後の ``get`` の値を設定の名前ごとに残す。"""

    def __init__(self, capture):
        self._capture = capture
        self._names = _prop_names()
        self.applied: dict[str, dict] = {}

    def set(self, prop, value):
        name = self._names.get(prop, str(prop))
        entry = self.applied.setdefault(name, {"tries": 0})
        entry.update(value=float(value), ok=False, get=None)
        entry["tries"] += 1
        ok = bool(self._capture.set(prop, value))
        entry["ok"] = ok
        return ok

    def get(self, prop):
        value = self._capture.get(prop)
        entry = self.applied.get(self._names.get(prop, str(prop)))
        if entry is not None:
            entry["get"] = float(value)
        return value


def _reported_fps(capture):
    try:
        value = float(capture.get(cv.CAP_PROP_FPS))
    except Exception:  # noqa: BLE001  get を持たない・受け付けないカメラ
        return None
    return value if value > 0 else None


def _summary(index, controls):
    parts = []
    for name, entry in controls["applied"].items():
        got = "" if entry["get"] is None else f"(get {entry['get']:g})"
        parts.append(f"{name}={entry['value']:g} {'成功' if entry['ok'] else '失敗'}{got}")
    fps = controls["reported_fps"]
    measured = controls["measured_fps"]
    return (f"[Mac カメラ] カメラ {index} {controls['size'][0]}x{controls['size'][1]}、設定（best effort）: "
            + ("、".join(parts) or "なし")
            + f" / 報告 fps {'—' if fps is None else f'{fps:.1f}'}"
            + f" / 実測 fps {'—' if measured is None else f'{measured:.1f}'}")


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
        self, index=0, *, size=(1280, 720), opener=open_capture, clock=time.monotonic_ns,
        env=None, fps_clock=time.perf_counter,
    ):
        self._clock = clock
        self.controls = {}
        opened = opener(index)
        if opened is None:
            raise RuntimeError(
                f"カメラ {index} を開けません。カメラ番号と macOS の権限を確認してください"
            )
        _, self._capture = opened
        try:
            self._capture.set(cv.CAP_PROP_FRAME_WIDTH, size[0])
            self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, size[1])
            source = os.environ if env is None else env
            recording = _RecordingCapture(self._capture)
            apply_camera_controls(
                recording, index if isinstance(index, int) else None,
                {k: v for k, v in source.items() if not _SIZE_KEYS.fullmatch(k)},
            )
            stamps = []
            for _ in range(10):
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Mac カメラから画像を取得できません")
                stamps.append(fps_clock())
            self.size = (frame.shape[1], frame.shape[0])
            span = stamps[-1] - stamps[_FPS_FROM]
            self.controls = {
                "index": index,
                "size": list(self.size),
                "applied": recording.applied,
                "reported_fps": _reported_fps(self._capture),
                "measured_fps": (len(stamps) - 1 - _FPS_FROM) / span if span > 0 else None,
            }
            print(_summary(index, self.controls), flush=True)
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
