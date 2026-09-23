"""PC内蔵カメラの姿勢を、同じPCの受信サーバへ送る。

Pixel 7a を cam0、この送信器を cam1 として使う。撮影時刻は read 直後の
PC単調時計で近似するため、カメラ内部の遅延は実機で評価する必要がある。
推論・撮影は専用スレッドで直列に実行し、ネットワークの受信を止めない。
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
import time
from urllib.parse import urlparse

import numpy as np
from websockets.asyncio.client import connect

from app.net import protocol as p


class CameraLandmarkReader:
    """撮影時刻と元画像サイズを保ったまま、既存の姿勢推定器を接続する。"""

    def __init__(self, source, estimator, role="cam1", clock=time.monotonic_ns):
        if role not in p.ROLES:
            raise ValueError("role は cam0 または cam1 を指定してください")
        self.source = source
        self.estimator = estimator
        self.role = role
        self.clock = clock
        self.seq = 0

    def read(self) -> p.LandmarkFrame | None:
        import cv2

        ok, image = self.source.read()
        captured = self.clock()
        if not ok or image is None:
            raise RuntimeError("PCカメラのフレームを取得できません")
        seq = self.seq
        self.seq += 1
        height, width = image.shape[:2]
        result = self.estimator.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pose = getattr(result, "pose_landmarks", None)
        if pose is None:
            return None
        landmarks = [(float(lm.x), float(lm.y), float(lm.z), float(lm.visibility))
                     for lm in pose.landmark]
        if len(landmarks) != 33 or not np.isfinite(landmarks).all():
            return None
        return p.LandmarkFrame(self.role, seq, captured, width, height, landmarks)

    def close(self):
        try:
            self.estimator.close()
        finally:
            self.source.release()


def open_reader(camera: int, role: str, width: int, height: int) -> CameraLandmarkReader:
    """カメラと推定器の生成・使用・解放を同じ専用スレッドで行う。"""
    import cv2
    from app.core.video_source import open_capture
    from pose_runtime import MP_THREADS, POSE_TASK_MODEL, USE_POSE_LANDMARKER, PoseEstimator

    opened = open_capture(camera)
    if opened is None:
        raise RuntimeError(f"PCカメラ {camera} を開けません")
    _, capture = opened
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        estimator = PoseEstimator(USE_POSE_LANDMARKER, POSE_TASK_MODEL, num_threads=MP_THREADS)
        print(f"PCカメラ {camera}: {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} / {role}")
        return CameraLandmarkReader(capture, estimator, role)
    except BaseException:
        capture.release()
        raise


async def send_camera(url, reader, *, role="cam1", session="local", fps=30.,
                      duration_sec=None, executor=None) -> int:
    """同じPC時計を使う loopback 接続へ送り、送信したフレーム数を返す。"""
    if urlparse(url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("PC時計を共有するため、接続先は同じPCの localhost を指定してください")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps は正の有限値を指定してください")
    if duration_sec is not None and (not math.isfinite(duration_sec) or duration_sec <= 0):
        raise ValueError("duration は正の有限値を指定してください")
    if role not in p.ROLES:
        raise ValueError("role は cam0 または cam1 を指定してください")
    # テストや他の起動器からも、同じスレッドで read を呼び続ける。
    if executor is None:
        with ThreadPoolExecutor(max_workers=1) as worker:
            return await send_camera(url, reader, role=role, session=session, fps=fps,
                                     duration_sec=duration_sec, executor=worker)
    loop = asyncio.get_running_loop()
    sent = 0
    async with connect(url) as ws:
        await ws.send(p.encode(p.Hello(role, "PC camera", session)))
        started = loop.time()
        while duration_sec is None or loop.time() - started < duration_sec:
            frame_started = loop.time()
            frame = await loop.run_in_executor(executor, reader.read)
            if frame is not None:
                if frame.role != role:
                    raise ValueError("カメラの role と送信する role が一致しません")
                await ws.send(p.encode(frame))
                sent += 1
            await asyncio.sleep(max(0., 1. / fps - (loop.time() - frame_started)))
    return sent


async def _main_async(args):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as worker:
        reader = await loop.run_in_executor(worker, open_reader, args.camera, args.role,
                                            args.width, args.height)
        try:
            sent = await send_camera(args.url, reader, role=args.role, session=args.session,
                                     fps=args.fps, duration_sec=args.duration, executor=worker)
            print(f"送信終了: {sent} フレーム")
        finally:
            await loop.run_in_executor(worker, reader.close)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="同じPCで起動した受信サーバ")
    parser.add_argument("--camera", type=int, default=0, help="PC内蔵カメラの番号")
    parser.add_argument("--role", choices=p.ROLES, default="cam1")
    parser.add_argument("--session", default="local", help="受信サーバが表示したセッション名")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30., help="送信の上限fps。実効速度は推論速度に依存")
    parser.add_argument("--duration", type=float, default=None, help="送信する秒数。省略すると停止まで継続")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"PCカメラ送信を終了しました: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
