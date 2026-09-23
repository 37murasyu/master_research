"""Timestamp, stillness and view-diversity gates for calibration capture."""

from collections import Counter, deque
import cv2 as cv
import numpy as np
from app.hybrid.checkerboard import detect_board

# 「まだ検出していない」印。「検出したが見つからない」（None）と区別する。
_UNSET = object()

# 盤が静止しているとみなす、前後 150 ms の Mac の角点の最大の動き [px]。
# 2 台の撮影時刻のずれ（100 ms 程度まで）の間に盤が動くと、対応する角点がずれる。300 ms で
# 2 px なら、そのずれは 0.7 px 以下。手で持って止めたつもりの動きは実測で 0.3〜2.7 px
# （2026-09-23）で、1.0 px ではほとんど採れなかった。最終的な品質は推定後の検査
# （RMS ≤ 1 px、マス寸法の誤差 ≤ 1 mm）が別に守る。
STILL_PX = 2.0

# ペアを見送った理由。画面に数を出し、使う人が何を直せばよいか分かるようにする。
REASONS = {
    "time": "時刻が合う Mac の画像が無い",
    "one_side": "片方のカメラにしか盤が写っていない",
    "window": "前後の Mac の画像が足りない",
    "lost": "前後で盤を見失った",
    "moving": "盤が動いている",
    "similar": "既に採った位置と近い",
}


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
        # 見送った理由ごとの数（REASONS のキー）と、直近に測った盤の動き [px]
        self.reasons = Counter()
        self.last_motion_px = None

    def _reject(self, reason):
        self.rejected += 1
        self.reasons[reason] += 1

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
            matched = abs(entry[0] - stamp) <= 40_000_000
            a = self._get(entry) if matched else None
            b = self.detect(image, self.board) if remote is _UNSET else remote
            for role, points in enumerate((a, b)):
                if points is not None and self._novel(points, self.mono[role], role):
                    self.mono[role].append(points)
            if a is None or b is None:
                self._reject("one_side" if matched else "time")
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
                self._reject("window")
                continue
            samples = [
                self._get(item)
                for item in self.ring
                if before[0] <= item[0] <= after[0]
            ]
            if any(v is None for v in samples):
                self._reject("lost")
                continue
            # 角点の配列の形は OpenCV の版で (N, 2) と (N, 1, 2) が混在するので、揃えてから比べる
            reference = np.reshape(a, (-1, 2))
            motion = max(
                float(np.max(np.linalg.norm(np.reshape(v, (-1, 2)) - reference, axis=1)))
                for v in samples
            )
            self.last_motion_px = motion
            if motion > STILL_PX:
                self._reject("moving")
                continue
            if not self._novel(a, [v[0] for v in self.pairs], 0):
                self._reject("similar")
                continue
            self.pairs.append((a, b))
            self.pair_images.append((entry[1].copy(), image.copy()))

    @property
    def ready(self):
        return len(self.pairs) >= self.pairs_required and all(
            self.cached[i] or len(self.mono[i]) >= self.mono_required for i in (0, 1)
        )
