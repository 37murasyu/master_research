"""Lazy MediaPipe Tasks VIDEO wrapper, safe to import without opening a camera.

Mac 側の姿勢推定の設定は ``HYBRID_POSE_*`` だけを読む（``PoseOptions.from_env``）。GUI は USB 向けの既定
（``POSE_ROI_ON=1``・``MP_INPUT_SCALE=0.5``・``POSE_MIN_DET``）も子プロセスへ渡すので、それを読むと誰も選んで
いないのに推定の条件が変わる。既定は今までと同じ（同梱の lite・VIDEO モード・閾値 0.5・縮小 1.0・ROI なし）。

``HYBRID_POSE_ROI=1`` なら USB の経路と同じ式（``app.hybrid.pose_roi``）で、前のフレームの点から決めた ROI だけを
IMAGE モードで推定し、点を全体の画像に対する正規化座標へ戻して返す。呼ぶ側から見た戻り値の形は変わらない。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Mapping

import cv2 as cv
from app.core.resources import resource_root
from app.hybrid import pose_roi
from config import pose_keypoints

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
            detector, image_factory = _create(options)
            print(options.describe(), flush=True)
        self.options = options or PoseOptions()
        self._detector = detector
        self._image = image_factory
        self._last_ms = -1
        # ROI の追跡の状態（本体の _pose_roi0 と _pose_roi0_miss）。最初は前の点が無いので全画面
        self._roi = None
        self._roi_miss = 0

    def detect(self, bgr, t_ns):
        """1 枚の BGR 画像の 33 点 ``(x, y, z, visibility)``（全体の画像に対する正規化座標）。人がいなければ None。"""
        self._last_ms = max(self._last_ms + 1, t_ns // 1_000_000)
        if self.options.roi:
            return self._detect_roi(bgr)
        result = self._detector.detect_for_video(self._rgb(bgr), self._last_ms)
        return self._points(result)

    def _rgb(self, bgr):
        scale = self.options.input_scale
        if scale < 1.0:
            # MediaPipe の座標は画像に対する割合なので、縮小しても全体の座標のまま使える
            bgr = cv.resize(bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        return self._image(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))

    @staticmethod
    def _points(result):
        if not result.pose_landmarks:
            return None
        return [
            (p.x, p.y, p.z, 1.0 if p.visibility is None else p.visibility)
            for p in result.pose_landmarks[0]
        ]

    def _detect_roi(self, bgr):
        """本体の _pose_process_with_roi と ROI の更新（master_research_code.py の推定の前後）と同じ手順。

        見失った回数だけ ROI を広げ、``MAX_MISS`` 回を超えたら全画面に戻す。点が少なすぎて ROI を決められない
        ときも全画面。IMAGE モードなので切り出す場所が画像ごとに変わっても追跡は崩れない。
        """
        roi = self._roi if self._roi_miss <= pose_roi.MAX_MISS else None
        for _ in range(self._roi_miss if roi is not None else 0):
            roi = pose_roi.expand_roi(roi, bgr.shape, pose_roi.MISS_GROW_RATIO)
            if roi is None:
                break
        crop = None
        if roi is not None:
            x0, y0, x1, y1 = roi
            crop = bgr[y0:y1, x0:x1]
            if crop.size == 0:
                crop = None
        points = self._points(self._detector.detect(self._rgb(bgr if crop is None else crop)))
        if points is not None and crop is not None:
            points = pose_roi.remap_to_fullframe(points, roi, bgr.shape)

        next_roi = pose_roi.roi_from_keypoints(pose_roi.landmarks_to_pixels(points, bgr.shape, pose_keypoints),
                                               bgr.shape)
        if next_roi is None:
            self._roi_miss += 1
            if self._roi_miss > pose_roi.MAX_MISS:
                self._roi = None
        else:
            self._roi, self._roi_miss = next_roi, 0
        return points

    def close(self):
        self._detector.close()
