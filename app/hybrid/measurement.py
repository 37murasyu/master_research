"""Receiver-thread measurement callbacks with identity and resolution checks."""

import sys
import threading
from config import pose_keypoints
from app.hybrid.gravity import read_board_up
from app.hybrid.recorder import Recorder
from app.runners.network_measure import (
    EXIT_IMPLAUSIBLE_SCALE,
    ImplausibleBodyScale,
    MeasurementConfig,
    NetworkMeasurement,
)


class MeasurementSession:
    def __init__(self, calibration, *, root=None, config=None, metadata=None, tracker=None):
        self.calibration = calibration
        self.root = root
        self.metadata = metadata or {}
        self.config = config or MeasurementConfig()
        # ゲージの状態（app.gauge.tracker.GaugeTracker）。受信スレッドが積み、メインスレッドが行を書く
        self.tracker = tracker
        self.measurement = None
        self.recorder = None
        self.failed = threading.Event()
        self.exit_code = 0
        self.error = None
        self.size_drops = 0
        # 何で止まったか（stop_request・key・ctrl_c・failed・error）。記録を閉じるときに meta.json へ残す（§3-2 の確認用）
        self.stop_reason = None
        self.consecutive = {"cam0": 0, "cam1": 0}

    @property
    def directory(self):
        return self.recorder.directory if self.recorder else None

    def _ensure(self):
        if self.recorder is None:
            self.recorder = Recorder(
                self.calibration,
                pose_keypoints,
                root=self.root,
                metadata=dict(
                    self.metadata,
                    body_mass_kg=self.config.body_mass_kg,
                    gravity_mode=self.config.gravity_mode,
                ),
            )
            self.measurement = NetworkMeasurement(
                *self.calibration.projections,
                pose_keypoints,
                self.config,
                lens=dict(zip(("cam0", "cam1"), self.calibration.intrinsics)),
                tracker=self.tracker,
                # 校正の最後に盤を立てた向き（無ければ None で、重力は体幹から決める）
                board_up=read_board_up(self.calibration.meta),
            )
            if self.tracker is not None:
                # 記録を始めた＝Pixel の点が届いた
                self.tracker.set_link("connected")

    def check_hello(self, hello):
        expected = self.calibration.meta["cameras"][1].get("device_id")
        if not expected or hello.device_id != expected:
            return "校正した Pixel と端末 ID が違います。校正した端末で QR を読み直してください"
        return None

    def accept_frame(self, frame):
        # 名乗る前の点と、名乗りを断った端末の点は PhoneLink（require_hello と on_hello）が捨てる
        i = 0 if frame.role == "cam0" else 1
        good = (frame.width, frame.height) == self.calibration.intrinsics[i].size
        self.consecutive[frame.role] = 0 if good else self.consecutive[frame.role] + 1
        if not good:
            self.size_drops += 1
            if self.consecutive[frame.role] >= 30:
                self.exit_code = 3
                self.error = "校正時と異なる解像度が30フレーム続きました"
                self.failed.set()
        return good

    def _failure(self, exc):
        self.error = str(exc)
        self.exit_code = 1
        self.failed.set()
        raise exc

    def on_landmarks(self, frame):
        # 記録は Pixel の点が初めて届いたときに始める。Mac の点は起動直後から流れるが、
        # Pixel が繋がらないまま終えると、中身の無い「完了」の計測フォルダが残る。
        if self.recorder is None and frame.role != "cam1":
            return
        try:
            self._ensure()
            self.recorder.landmarks(frame)
        except Exception as exc:
            self._failure(exc)

    def on_pairs(self, pairs):
        if self.failed.is_set():
            return  # 止めると決めた後に届いた組は捨てる（メインループが抜けるまでの間）
        try:
            self._ensure()
            for pair in pairs:
                result = self.measurement.process(pair)
                if result is not None:
                    self.recorder.record(result)
        except ImplausibleBodyScale as exc:
            # 座標の単位か校正が壊れている。トルクが桁違いになるので止める（終了コード 3、理由は meta.json の error）
            self.error = str(exc)
            self.exit_code = EXIT_IMPLAUSIBLE_SCALE
            self.failed.set()
            print(f"[計測] {exc}。計測を止めます（終了コード {EXIT_IMPLAUSIBLE_SCALE}）", file=sys.stderr)
        except Exception as exc:
            self._failure(exc)

    def flush(self):
        if self.recorder is None:
            return  # まだ記録を始めていない
        try:
            self.recorder.flush()
        except Exception as exc:
            self._failure(exc)

    def close(self):
        if self.recorder is None:
            return  # Pixel の点が一度も届かなかった。残すものは無い
        try:
            self.recorder.close(
                status="failed" if self.exit_code else "complete",
                exit_code=self.exit_code,
                error=self.error,
                size_drops=self.size_drops,
                stop_reason=self.stop_reason,
            )
        except Exception as exc:
            self._failure(exc)
