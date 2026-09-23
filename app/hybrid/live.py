"""One main-thread acquisition / inference / display step, with injectable output."""

from collections import deque
import time
import cv2 as cv
import numpy as np
from config import pose_keypoints
from app.net.protocol import LandmarkFrame
from app.hybrid.checkerboard import detect_board
from app.hybrid.display import compose

# Pixel の画像に重ねる骨格を探す時刻の許容幅。MediaPipe は処理が追いつかないフレームを
# 飛ばすので、画像にしたフレームそのものの点が無いことが多い（実機で 6 割）。そのときは
# 隣のフレーム（30 fps で約 33 ms 先）の点を重ねる。向き合わせの確認にはこれで足りる。
REMOTE_OVERLAY_TOLERANCE_NS = 50_000_000


def decode_capture(capture):
    image = cv.imdecode(np.frombuffer(capture.jpeg, np.uint8), cv.IMREAD_COLOR)
    if image is None or (image.shape[1], image.shape[0]) != (
        capture.width,
        capture.height,
    ):
        raise ValueError("Pixel JPEG の画像寸法が電文と一致しません")
    return image


class LiveSession:
    def __init__(self, camera, detector, link, *, output=None, cam0_offset_ms=0):
        self.camera, self.detector, self.link = camera, detector, link
        self.output = output or (lambda image: cv.imshow("Mac + Pixel", image))
        self.offset_ns = round(cam0_offset_ms * 1e6)
        self.board = None
        self.local_corners = None
        self.remote_corners = None
        self.seq = 0
        self.remote_image = None
        self.remote_capture = None
        self.local_image = None
        self.local_t_ns = 0
        # 復号できず捨てた Pixel の画像の数（壊れた JPEG、電文と寸法が違うもの）。
        self.bad_captures = 0
        self._times = deque(maxlen=120)
        self._pairs = deque(maxlen=120)

    def step(self, *, infer=True, lines=(), left_corners=None, right_corners=None):
        """1 フレーム取り込んで表示する。新しく受け取った Pixel の画像があれば返す。

        返すのは**復号できた**画像だけ。復号した結果は ``remote_image`` に、盤の検出結果は
        ``remote_corners`` に置くので、呼び出し側は復号も検出もやり直さなくてよい。
        """
        stamp, image = self.camera.read()
        stamp += self.offset_ns
        self.local_image, self.local_t_ns = image, stamp
        if self.board is not None:
            self.local_corners = detect_board(image, self.board)
            left_corners = self.local_corners
        points = self.detector.detect(image, stamp) if infer else None
        if points:
            self.link.inject(
                LandmarkFrame(
                    "cam0", self.seq, stamp, image.shape[1], image.shape[0], points
                )
            )
        self.seq += 1
        self._times.append(time.monotonic())
        capture = self.link.take_capture()
        if capture is not None:
            try:
                remote_image = decode_capture(capture)
            except ValueError:
                # 無線の相手からの入力。1 枚壊れていても表示と収集は続ける
                self.bad_captures += 1
                capture = None
            else:
                self.remote_image = remote_image
                self.remote_capture = capture
                if self.board is not None:
                    self.remote_corners = detect_board(remote_image, self.board)
        if self.board is not None:
            right_corners = self.remote_corners
        remote = (
            self.link.nearest_remote(self.remote_capture.t_capture_ns, REMOTE_OVERLAY_TOLERANCE_NS)
            if self.remote_capture
            else None
        )
        status = self.link.status()
        now = self._times[-1]
        self._pairs.append((now, status.pairs))
        while len(self._pairs) > 1 and now - self._pairs[0][0] > 1:
            self._pairs.popleft()
        duration = now - self._pairs[0][0]
        pair_hz = (status.pairs - self._pairs[0][1]) / duration if duration else 0
        fps = sum(t > now - 1 for t in self._times)
        device = status.devices.get("cam1")

        def visible(pts):
            return (
                sum(
                    pts[i][3] >= 0.5 and 0 <= pts[i][0] <= 1 and 0 <= pts[i][1] <= 1
                    for i in pose_keypoints
                )
                if pts
                else 0
            )

        if not lines:
            lines = (
                f"Mac {fps:.0f} fps / Pixel {status.remote_fps:.0f} fps / ペア {pair_hz:.1f}/秒 / 要点 {visible(points)}・{visible(remote.landmarks if remote else None)}",
                f"位相差 平均 {status.mean_skew_ms:.1f} / 最大 {status.max_skew_ms:.1f} ms / 欠測破棄 {status.dropped_gap+status.dropped_late}",
                f'{device.device} / {device.device_id or "ID未取得"}'
                if device
                else "cam1 の QR を Pixel で読み取ってください",
            )
        view = compose(
            image,
            self.remote_image if device else None,
            left_landmarks=points,
            right_landmarks=remote.landmarks if remote and infer else None,
            url=self.link.url,
            lines=lines,
            left_corners=left_corners,
            right_corners=right_corners,
        )
        self.output(view)
        return capture
