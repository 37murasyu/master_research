"""Timestamp, stillness and view-diversity gates for calibration capture."""

from collections import deque
import cv2 as cv
import numpy as np
from app.hybrid.checkerboard import detect_board

# 「まだ検出していない」印。「検出したが見つからない」（None）と区別する。
_UNSET = object()


class BoardCollector:
    def __init__(
        self,
        board,
        *,
        detect=detect_board,
        mono_required=15,
        pairs_required=12,
        cached=(False, False),
    ):
        self.board, self.detect = board, detect
        self.mono_required, self.pairs_required = mono_required, pairs_required
        self.cached = cached
        self.ring = deque()
        self.pending = deque(maxlen=12)
        self.mono = [[], []]
        self.pairs = []
        self.pair_images = []
        self.sizes = [None, None]
        self._corners = {}
        self.rejected = 0

    def _gray(self, image):
        return (
            cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        )

    def add_mac(self, t_ns, image, *, corners=_UNSET):
        """``corners`` は表示のために済ませた検出の結果（見つからなければ None）。

        渡されなければ必要になったときに検出する。渡された結果は None も含めて使い回す
        （全解像度の検出をメインスレッドで 2 度走らせない）。
        """
        self.sizes[0] = (image.shape[1], image.shape[0])
        self.ring.append((t_ns, self._gray(image)))
        if corners is not _UNSET:
            self._corners[t_ns] = corners
        while self.ring and t_ns - self.ring[0][0] > 1_500_000_000:
            t, _ = self.ring.popleft()
            self._corners.pop(t, None)
        self._drain()

    def add_remote(self, t_ns, image, *, corners=_UNSET):
        """``corners`` の意味は ``add_mac`` と同じ。"""
        size = (image.shape[1], image.shape[0])
        if self.sizes[1] is not None and size != self.sizes[1]:
            raise ValueError(
                "収集中に Pixel の画像寸法が変わりました。やり直してください"
            )
        self.sizes[1] = size
        self.pending.append((t_ns, self._gray(image), corners))
        self._drain()

    def _get(self, entry):
        t, image = entry
        if t not in self._corners:
            self._corners[t] = self.detect(image, self.board)
        return self._corners[t]

    def _novel(self, points, views, role):
        diagonal = np.linalg.norm(self.sizes[role])
        return all(
            np.mean(np.linalg.norm(points.reshape(-1, 2) - v.reshape(-1, 2), axis=1))
            / diagonal
            > 0.05
            for v in views
        )

    def _drain(self):
        while (
            self.pending
            and self.ring
            and self.ring[-1][0] >= self.pending[0][0] + 150_000_000
        ):
            stamp, image, remote = self.pending.popleft()
            entry = min(self.ring, key=lambda item: abs(item[0] - stamp))
            a = self._get(entry) if abs(entry[0] - stamp) <= 40_000_000 else None
            b = self.detect(image, self.board) if remote is _UNSET else remote
            for role, points in enumerate((a, b)):
                if points is not None and self._novel(points, self.mono[role], role):
                    self.mono[role].append(points)
            if a is None or b is None:
                self.rejected += 1
                continue
            before = min(
                self.ring, key=lambda item: abs(item[0] - (stamp - 150_000_000))
            )
            after = min(
                self.ring, key=lambda item: abs(item[0] - (stamp + 150_000_000))
            )
            if (
                abs(before[0] - (stamp - 150_000_000)) > 40_000_000
                or abs(after[0] - (stamp + 150_000_000)) > 40_000_000
            ):
                self.rejected += 1
                continue
            samples = [
                self._get(item)
                for item in self.ring
                if before[0] <= item[0] <= after[0]
            ]
            if any(
                v is None or np.max(np.linalg.norm(v - a, axis=2)) > 1.0
                for v in samples
            ):
                self.rejected += 1
                continue
            if self._novel(a, [v[0] for v in self.pairs], 0):
                self.pairs.append((a, b))
                self.pair_images.append((entry[1].copy(), image.copy()))

    @property
    def ready(self):
        return len(self.pairs) >= self.pairs_required and all(
            self.cached[i] or len(self.mono[i]) >= self.mono_required for i in (0, 1)
        )
