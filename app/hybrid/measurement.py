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
        # 計算器は作った側のスレッド（計測の子ではメインスレッド）で先に作る。EKF の較正プロファイル
        # （HYBRID_EKF_PROFILE）の読み込みを受信スレッドで走らせず、壊れていれば起動の時点で止める
        # （Pixel の最初の点で落ちると、記録器ができる前なので meta.json に理由が残らない）
        self.measurement = NetworkMeasurement(
            *calibration.projections,
            pose_keypoints,
            self.config,
            lens=dict(zip(("cam0", "cam1"), calibration.intrinsics)),
            tracker=tracker,
            # 校正の最後に盤を立てた向き（無ければ None で、重力は体幹から決める）
            board_up=read_board_up(calibration.meta),
        )
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

    def _raw_provenance(self, measurement):
        """生 3D（kpts3d_raw）のサイドカー。USB の kpts3d_raw と同じ鍵に、混成の出どころを足す。"""
        ekf = self.config.ekf
        grid = self.config.grid
        noise = None if measurement.ekf is None else measurement.ekf.noise
        return {
            "unit": "m",
            "frame": "runtime",
            "dt": grid.period_s,
            "dt_source": f"混成ステレオの同期バッファの {grid.target_hz:g} Hz の格子（抜けた格子は NaN の行）",
            "src_fps": float(grid.target_hz),
            "source": "hybrid",
            "times": "grid",
            "file_mode": False,
            "RT_POSE_FIXED_HZ_ON": False,
            "EKF_ENABLE": bool(ekf.enabled),
            "EKF_GATE_STD": ekf.gate_std,
            "EKF_ROBUST_GATE": bool(ekf.robust_gate),
            "EKF_MAX_GAP_S": ekf.max_gap_s,
            "EKF_BPF_LOW": ekf.bpf_low,
            "EKF_BPF_HIGH": ekf.bpf_high,
            "EKF_BPF_ORDER": ekf.bpf_order,
            "EKF_VECTORIZED": True,
            "HYBRID_EKF_PROFILE": ekf.profile,
            "ekf_noise": None if noise is None else noise.provenance(),
            "coordinates": "(-camera_x, -camera_z, -camera_y)",
            "calibration": str(self.calibration.directory),
        }

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
                    subject_id=self.config.subject_id,
                    one_rm_kg=None if self.config.one_rm is None else dict(self.config.one_rm),
                    dyn_gate=self.config.dyn_gate,
                ),
                raw_provenance=self._raw_provenance(self.measurement),
                offline_wrist=self.config.offline_wrist_capture,
                grid=self.config.grid,
            )
            self.recorder.meta["ekf"] = self.measurement.ekf_provenance()
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
                    if result.window_closed:
                        # 先頭の窓で重力が決まった。窓を閉じたこのフレームから横長のトルクを書く
                        self.recorder.note_raw(**self.measurement.window)
                        label = self.measurement.window.get("gravity_label")
                        if label:
                            self.recorder.open_torque_vectors(label)
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
                **self.measurement.summary(),
                status="failed" if self.exit_code else "complete",
                exit_code=self.exit_code,
                error=self.error,
                size_drops=self.size_drops,
                stop_reason=self.stop_reason,
            )
        except Exception as exc:
            self._failure(exc)
