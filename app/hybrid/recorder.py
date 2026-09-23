"""Incremental measurement writer. Construct and call only on the receiver loop thread."""

import csv
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import shutil
import threading
import time
from typing import TYPE_CHECKING
from config import slot_of
from app.gauge.protocol import PART_NAMES
from app.hybrid.calibration_io import FILES, write_json
from app.hybrid.ekf import GRID_NS as _GRID_NS   # 同期バッファの格子（30 Hz）[ns]。生 3D の frame は格子の番号
from app.hybrid.paths import measurement_root
from app.tuning.raw_capture import RawCaptureWriter

if TYPE_CHECKING:
    from app.runners.network_measure import FrameResult


def _cycle_columns(result, key):
    """cycle_work の W+・W−・W_1RM・スコア。無いものは空欄。"""
    w1rm = result.cycle_w1rm.get(key)
    work = result.cycle_parts.get(key)
    if work is None:
        return ["", "", "", ""]
    score = work.pos / w1rm if w1rm else ""
    return [work.pos, work.neg, "" if w1rm is None else w1rm, score]


class Recorder:
    def __init__(
        self,
        calibration,
        pose_keypoints,
        *,
        root=None,
        metadata=None,
        clock=time.monotonic,
        raw_provenance=None,
        offline_wrist=False,
    ):
        self._owner = threading.get_ident()
        self.clock = clock
        self._flushed = clock()
        self._first_ns = None
        self.frames = 0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.directory = Path(root or measurement_root()) / stamp
        self.directory.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(calibration.directory / name, self.directory / name)
        self.meta = dict(
            metadata or {},
            status="recording",
            started_at=datetime.now(timezone.utc).isoformat(),
            calibration=str(calibration.directory),
            calibration_meta=calibration.meta,
            coordinates="(-camera_x, -camera_z, -camera_y)",
            units="m",
            torque_units="N m",
            work_units="J",
            pose_keypoints=sorted(pose_keypoints),
            writer_thread=threading.current_thread().name,
        )
        self._streams = []
        self.closed = False

        def writer(prefix, header, suffix=""):
            stream = (self.directory / f"{prefix}_{stamp}{suffix}.csv").open(
                "w", newline="", encoding="utf-8"
            )
            self._streams.append(stream)
            result = csv.writer(stream)
            result.writerow(header)
            return result

        self._writer = writer
        self._stamp = stamp
        # 局所トルクの横長（USB の aim_torque_vec と同じ形）。重力が決まってから開く（open_torque_vectors）
        self.torque_vectors = None

        self.points = writer(
            "kpts3d",
            ["frame"]
            + [
                f"joint_{i}_{axis}"
                for i in range(len(pose_keypoints))
                for axis in "xyz"
            ],
        )
        # 既存の列の後ろに: 同期バッファの格子の番号・前の組からの dt・関所・高さ・回の番号・腕の長さの安全策
        self.times = writer("frames", ["frame", "t_ns", "t_s", "cycle_detected", "grid_index", "dt_s",
                                       "dyn_active", "height_m", "rep", "arm_ok_L", "arm_ok_R"])
        self.raw = writer(
            "landmarks2d",
            [
                "role",
                "seq",
                "t_ns",
                "width",
                "height",
                "landmark",
                "x",
                "y",
                "z",
                "visibility",
            ],
        )
        self.torques = writer("local_torque", ["frame", "t_ns", "joint", "x", "y", "z"])
        # work_j は符号付きの W±（既存の列）。W+ = Σmax(P,0)·dt、W− = Σmin(P,0)·dt、score = W+ / W_1RM（論文 4.5.2 節）
        self.work = writer("cycle_work", ["frame", "t_ns", "joint", "work_j", "work_pos_j", "work_neg_j", "w1rm_j", "score"])
        # ゲージに出した値（今の回の W+ [J]）。毎フレーム 1 行。帯と定義は閉じるときに .json へ
        self._gauge_parts = PART_NAMES
        self._gauge_path = self.directory / f"gauge_energy_{stamp}.csv"
        self.gauge = writer("gauge_energy", ["frame", "t_ns", "rep", "dyn_active", *self._gauge_parts])
        # 肘の濾波 E±（USB の cycle_energy_debug_* と同じ量。回の確定ごとに肘の左右で 1 行ずつ）
        self.energy = writer("cycle_energy", ["frame", "t_ns", "part", "e_pos", "e_neg", "fc_current", "dt_sec", "n_u"])
        # OFFLINE_WRIST_CAPTURE: 前腕（肘→手首）(N,3) と手首の局所 τ_y (N,)。閉じるときに npy へ（USB と同じ名前）
        self._wrist = {"R": ([], []), "L": ([], [])} if offline_wrist else None
        # EKF の手前の生 3D（EKF の較正 tune_ekf の入力）。1/30 s の格子で、抜けた格子は NaN の行で埋める
        # （行を詰めると dt 一定の前提が崩れる）。raw_provenance が無ければ書かない（EKF を通さない記録）
        self.raw3d = None
        self._raw_ids = sorted(pose_keypoints)
        self._raw_next = 0
        self._raw_blank = np.full((len(self._raw_ids), 3), np.nan)  # 抜けた格子を埋める NaN の行（append は書き換えない）
        if raw_provenance is not None:
            self.raw3d = RawCaptureWriter(self.directory / f"kpts3d_raw_{stamp}.csv", self._raw_ids,
                                          provenance=raw_provenance)
        write_json(self.directory / "meta.json", self.meta)
        self.flush(force=True)

    def _check(self):
        if self._owner != threading.get_ident():
            raise RuntimeError("Recorder は受信ループのスレッド専用です")
        if self.closed:
            raise RuntimeError("Recorder は既に閉じています")

    def landmarks(self, frame):
        self._check()
        self.raw.writerows(
            [
                frame.role,
                frame.seq,
                frame.t_capture_ns,
                frame.width,
                frame.height,
                i,
                *point,
            ]
            for i, point in enumerate(frame.landmarks)
        )
        self.flush()

    # USB の aim_torque_vec の部位の並び
    TORQUE_VECTOR_PARTS = ("wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L")

    def open_torque_vectors(self, gravity_label):
        """``aim_torque_vec_<stamp>_s<版>_g<重力>.csv`` を開く（先頭の窓で重力が決まったとき）。"""
        self._check()
        if self.torque_vectors is not None:
            return
        from config import OUTPUT_SCHEMA_VERSION

        self.torque_vectors = self._writer(
            "aim_torque_vec",
            ["frame"] + [f"{part}_{axis}" for part in self.TORQUE_VECTOR_PARTS for axis in "xyz"],
            suffix=f"_s{OUTPUT_SCHEMA_VERSION}_g{gravity_label}",
        )

    def _capture_wrist(self, result):
        """USB（``master_research_code.py`` の OFFLINE_WRIST_CAPTURE）と同じく、前腕が有限のフレームだけ積む。"""
        points = result.points_3d
        for side, (vectors, taus) in self._wrist.items():
            forearm = np.asarray(points[slot_of(f"{side}_WRIST")] - points[slot_of(f"{side}_ELBOW")], dtype=float)
            if not np.all(np.isfinite(forearm)):
                continue
            tau = result.local_torques.get(f"wrist_{side}")
            vectors.append(forearm)
            taus.append(float(tau[1]) if tau is not None and np.all(np.isfinite(tau)) else 0.0)

    def _save_wrist(self):
        from config import OUTPUT_SCHEMA_VERSION

        for side, (vectors, taus) in (self._wrist or {}).items():
            if not vectors:
                continue
            suffix = f"{side}_{self._stamp}_s{OUTPUT_SCHEMA_VERSION}.npy"
            np.save(self.directory / f"forearm_{suffix}", np.asarray(vectors, dtype=float))
            np.save(self.directory / f"tau_wrist_{suffix}", np.asarray(taus, dtype=float))

    def note_raw(self, **fields):
        """生 3D のサイドカーに、先頭の窓で決まった値（体格の比・重力など）を書き足す。"""
        self._check()
        if self.raw3d is not None:
            self.raw3d.note(**fields)

    def _append_raw(self, result):
        raw = result.points_raw
        if self.raw3d is None or raw is None:
            return
        grid = int(result.grid_index)
        # flush は Recorder.flush（1 秒に 1 回）に任せる
        while self._raw_next < grid:
            self.raw3d.append(self._raw_next, self._raw_next * _GRID_NS / 1e9, self._raw_blank, flush=False)
            self._raw_next += 1
        if grid < self._raw_next:
            return  # 格子が戻った（起こらないはず）。書かない
        self.raw3d.append(grid, (result.t_ns - self._first_ns) / 1e9, raw, flush=False)
        self._raw_next = grid + 1

    def record(self, result: "FrameResult"):
        self._check()
        if self._first_ns is None:
            self._first_ns = result.t_ns
        self._append_raw(result)
        self.points.writerow([self.frames, *result.points_3d.ravel()])
        arm_ok = result.arm_ok
        self.times.writerow(
            [
                self.frames,
                result.t_ns,
                (result.t_ns - self._first_ns) / 1e9,
                int(result.cycle_detected),
                result.grid_index,
                result.dt_s,
                int(bool(result.dyn_active)),
                result.height_m,
                result.rep,
                int(arm_ok.get("L", True)),
                int(arm_ok.get("R", True)),
            ]
        )
        self.torques.writerows(
            [self.frames, result.t_ns, key, *value]
            for key, value in result.local_torques.items()
        )
        self.work.writerows(
            [self.frames, result.t_ns, key, value, *_cycle_columns(result, key)]
            for key, value in result.cycle_work_j.items()
        )
        gauge = result.gauge_now
        self.gauge.writerow([self.frames, result.t_ns, result.rep,
                             int(bool(result.dyn_active)),
                             *(gauge.get(part, "") for part in self._gauge_parts)])
        if self._wrist is not None and result.local_torques:
            self._capture_wrist(result)
        if self.torque_vectors is not None and result.local_torques:
            torques = result.local_torques
            self.torque_vectors.writerow(
                [self.frames, *(value for part in self.TORQUE_VECTOR_PARTS
                                for value in torques.get(part, (float("nan"),) * 3))])
        self.energy.writerows(
            [self.frames, result.t_ns, part, e["e_pos"], e["e_neg"], e["fc"], 1.0 / 30.0, e["n_u"]]
            for part, e in result.cycle_energy.items()
        )
        self.frames += 1
        self.flush()

    def flush(self, force=False):
        self._check()
        if force or self.clock() - self._flushed >= 1:
            for stream in self._streams:
                stream.flush()
            if self.raw3d is not None:
                self.raw3d.flush()
            self._flushed = self.clock()

    def close(self, **metadata):
        if self.closed:
            return
        self._check()
        self.flush(force=True)
        for stream in self._streams:
            stream.close()
        if self.raw3d is not None:
            self.raw3d.close()
        self._save_wrist()
        self.meta.update(
            status="complete",
            ended_at=datetime.now(timezone.utc).isoformat(),
            frames=self.frames,
        )
        self.meta.update(metadata)
        write_json(self.directory / "meta.json", self.meta)
        write_json(self._gauge_path.with_suffix(".json"), {
            "unit": "J",
            "definition": ("今の回の正の仕事 W+ = Σmax(P, 0)·dt（論文 4.5.2 節）。P = τ_y × ω_rel·y。力学の関所が開いている間"
                           "だけ積み（dyn_active）、回の確定で 0 に戻る。帯は W_0.70・W_0.85 = theoretical_1rm_work("
                           "部位, 体重, 実測の前腕長, c·1RM)。v < W_0.70 不足、W_0.70 ≤ v < W_0.85 目標帯、v ≥ W_0.85 過負荷"),
            "dt": "フレームごとの dt（frames の dt_s）。0.1 s を超えるフレームは積まない",
            **{key: self.meta.get(key) for key in ("subject_id", "one_rm_kg", "body_mass_kg", "forearm_len_m",
                                                   "w1rm_j", "gauge_bands_j", "gauge_band_reasons", "reps",
                                                   "discarded_reps", "dyn_gate", "arm_length_guard")},
        })
        self.closed = True
