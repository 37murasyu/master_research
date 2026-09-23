import time
import numpy as np
from app.net.mock_sender import synthetic_pose


class FakeCamera:
    size = (1280, 720)

    def read(self):
        return time.monotonic_ns(), np.zeros((720, 1280, 3), np.uint8)

    def close(self):
        pass


class FakeDetector:
    def detect(self, image, t_ns):
        return synthetic_pose(t_ns / 1e9, "cam0")

    def close(self):
        pass
