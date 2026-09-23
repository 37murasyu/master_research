"""Lazy MediaPipe Tasks VIDEO wrapper, safe to import without opening a camera.

Mac 側の姿勢推定の設定は ``HYBRID_POSE_*`` だけを読む（``PoseOptions.from_env``）。GUI は USB 向けの既定
（``POSE_ROI_ON=1``・``MP_INPUT_SCALE=0.5``・``POSE_MIN_DET``）も子プロセスへ渡すので、それを読むと誰も選んで
いないのに推定の条件が変わる。既定は今までと同じ（同梱の lite・VIDEO モード・閾値 0.5・縮小 1.0・ROI なし）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Mapping

import cv2 as cv
from app.core.resources import resource_root

__all__ = ["PoseOptions", "PoseDetector", "landmarker_options"]

LITE_MODEL = "pose_landmarker_lite.task"
MIN_INPUT_SCALE = 0.25


def _env_float(env: Mapping[str, str], key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(env.get(key) or default)
    except ValueError:
        print(f"[姿勢推定][警告] {key}={env.get(key)!r} は数でないので {default:g} を使う")
        return default
    return min(high, max(low, value))


@dataclass(frozen=True)
class PoseOptions:
    """Mac 側の姿勢推定の設定。``model`` が None なら同梱の lite。"""

    model: str | None = None
    min_detection: float = 0.5
    min_presence: float = 0.5
    min_tracking: float = 0.5
    input_scale: float = 1.0
    roi: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "PoseOptions":
        """``HYBRID_POSE_*`` を読む（USB 向けの ``POSE_*``・``MP_*`` は読まない）。壊れた値は既定、範囲外は端に寄せる。"""
        env = os.environ if env is None else env
        return cls(
            model=(env.get("HYBRID_POSE_MODEL") or "").strip() or None,
            min_detection=_env_float(env, "HYBRID_POSE_MIN_DET", 0.5, 0.0, 1.0),
            min_presence=_env_float(env, "HYBRID_POSE_MIN_PRESENCE", 0.5, 0.0, 1.0),
            min_tracking=_env_float(env, "HYBRID_POSE_MIN_TRACK", 0.5, 0.0, 1.0),
            input_scale=_env_float(env, "HYBRID_POSE_INPUT_SCALE", 1.0, MIN_INPUT_SCALE, 1.0),
            roi=(env.get("HYBRID_POSE_ROI") or "0").strip().lower() in ("1", "true", "on", "yes"),
        )

    @property
    def running_mode(self) -> str:
        """ROI で切り出すと画像ごとに位置が変わり追跡が崩れるので IMAGE モード。それ以外は VIDEO モード。"""
        return "IMAGE" if self.roi else "VIDEO"

    def model_path(self) -> str:
        return str(self.model or resource_root() / LITE_MODEL)

    def describe(self) -> str:
        model = self.model or f"{LITE_MODEL}（同梱）"
        return (f"[姿勢推定] Mac: モデル {model}、{self.running_mode} モード、閾値 検出 {self.min_detection:g}・"
                f"存在 {self.min_presence:g}・追跡 {self.min_tracking:g}、縮小 {self.input_scale:g}、"
                f"ROI {'あり' if self.roi else 'なし'}")


def landmarker_options(options: PoseOptions):
    """MediaPipe の ``PoseLandmarkerOptions`` を組み立てる（検出器はまだ作らない）。"""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    return vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=options.model_path()),
        running_mode=getattr(vision.RunningMode, options.running_mode),
        num_poses=1,
        min_pose_detection_confidence=options.min_detection,
        min_pose_presence_confidence=options.min_presence,
        min_tracking_confidence=options.min_tracking,
    )


def _create(options: PoseOptions):
    """本物の検出器と、画像を MediaPipe の形に包む関数。"""
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    detector = vision.PoseLandmarker.create_from_options(landmarker_options(options))
    return detector, (lambda rgb: mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


class PoseDetector:
    def __init__(self, model=None, *, detector=None, image_factory=None, options: PoseOptions | None = None):
        if detector is None:
            options = options or PoseOptions.from_env()
            if model:
                options = replace(options, model=str(model))
            if options.roi:
                # ROI の切り出し（app.hybrid.pose_roi）はまだ推定に配線していない
                print("[姿勢推定][警告] HYBRID_POSE_ROI=1 はこの版では未配線。全画面・VIDEO モードで推定する")
                options = replace(options, roi=False)
            detector, image_factory = _create(options)
            print(options.describe(), flush=True)
        self.options = options or PoseOptions()
        self._detector = detector
        self._image = image_factory
        self._last_ms = -1

    def detect(self, bgr, t_ns):
        self._last_ms = max(self._last_ms + 1, t_ns // 1_000_000)
        scale = self.options.input_scale
        if scale < 1.0:
            # MediaPipe の座標は画像に対する割合なので、縮小しても全体の座標のまま使える
            bgr = cv.resize(bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        result = self._detector.detect_for_video(
            self._image(cv.cvtColor(bgr, cv.COLOR_BGR2RGB)), self._last_ms
        )
        if not result.pose_landmarks:
            return None
        return [
            (p.x, p.y, p.z, 1.0 if p.visibility is None else p.visibility)
            for p in result.pose_landmarks[0]
        ]

    def close(self):
        self._detector.close()
