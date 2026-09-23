"""Lazy MediaPipe Tasks VIDEO wrapper, safe to import without opening a camera."""

import cv2 as cv
from app.core.resources import resource_root


class PoseDetector:
    def __init__(self, model=None, *, detector=None, image_factory=None):
        if detector is None:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(
                        model or resource_root() / "pose_landmarker_lite.task"
                    )
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = vision.PoseLandmarker.create_from_options(options)
            image_factory = lambda rgb: mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb
            )
        self._detector = detector
        self._image = image_factory
        self._last_ms = -1

    def detect(self, bgr, t_ns):
        self._last_ms = max(self._last_ms + 1, t_ns // 1_000_000)
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
