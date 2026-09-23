"""Incremental measurement writer. Construct and call only on the receiver loop thread."""

import csv
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import shutil
import threading
import time
from app.hybrid.calibration_io import FILES, write_json
from app.hybrid.paths import measurement_root
from app.tuning.raw_capture import RawCaptureWriter

# 同期バッファの格子（30 Hz）[ns]。生 3D の frame は格子の番号
_GRID_NS = 33_333_333


def _cycle_columns(result, key):
    """cycle_work の W+・W−・W_1RM・スコア。無いものは空欄。"""
    parts = getattr(result, "cycle_parts", None) or {}
    w1rm = (getattr(result, "cycle_w1rm", None) or {}).get(key)
    work = parts.get(key)
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

        def writer(prefix, header):
            stream = (self.directory / f"{prefix}_{stamp}.csv").open(
                "w", newline="", encoding="utf-8"
            )
            self._streams.append(stream)
            result = csv.writer(stream)
            result.writerow(header)
            return result

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
        # 肘の濾波 E±（USB の cycle_energy_debug_* と同じ量。回の確定ごとに肘の左右で 1 行ずつ）
        self.energy = writer("cycle_energy", ["frame", "t_ns", "part", "e_pos", "e_neg", "fc_current", "dt_sec", "n_u"])
        # EKF の手前の生 3D（EKF の較正 tune_ekf の入力）。1/30 s の格子で、抜けた格子は NaN の行で埋める
        # （行を詰めると dt 一定の前提が崩れる）。raw_provenance が無ければ書かない（EKF を通さない記録）
        self.raw3d = None
        self._raw_ids = sorted(pose_keypoints)
        self._raw_next = 0
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

    def note_raw(self, **fields):
        """生 3D のサイドカーに、先頭の窓で決まった値（体格の比・重力など）を書き足す。"""
        self._check()
        if self.raw3d is not None:
            self.raw3d.note(**fields)

    def _append_raw(self, result):
        raw = getattr(result, "points_raw", None)
        if self.raw3d is None or raw is None:
            return
        grid = int(getattr(result, "grid_index", self._raw_next))
        blank = np.full((len(self._raw_ids), 3), np.nan)
        while self._raw_next < grid:
            self.raw3d.append(self._raw_next, self._raw_next * _GRID_NS / 1e9, blank)
            self._raw_next += 1
        if grid < self._raw_next:
            return  # 格子が戻った（起こらないはず）。書かない
        self.raw3d.append(grid, (result.t_ns - self._first_ns) / 1e9, raw)
        self._raw_next = grid + 1

    def record(self, result):
        self._check()
        if self._first_ns is None:
            self._first_ns = result.t_ns
        self._append_raw(result)
        self.points.writerow([self.frames, *result.points_3d.ravel()])
        arm_ok = getattr(result, "arm_ok", None) or {}
        self.times.writerow(
            [
                self.frames,
                result.t_ns,
                (result.t_ns - self._first_ns) / 1e9,
                int(result.cycle_detected),
                getattr(result, "grid_index", ""),
                getattr(result, "dt_s", ""),
                int(bool(getattr(result, "dyn_active", False))),
                getattr(result, "height_m", float("nan")),
                getattr(result, "rep", ""),
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
        self.energy.writerows(
            [self.frames, result.t_ns, part, e["e_pos"], e["e_neg"], e["fc"], 1.0 / 30.0, e["n_u"]]
            for part, e in (getattr(result, "cycle_energy", None) or {}).items()
        )
        self.frames += 1
        self.flush()

    def flush(self, force=False):
        self._check()
        if force or self.clock() - self._flushed >= 1:
            for stream in self._streams:
                stream.flush()
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
        self.meta.update(
            status="complete",
            ended_at=datetime.now(timezone.utc).isoformat(),
            frames=self.frames,
        )
        self.meta.update(metadata)
        write_json(self.directory / "meta.json", self.meta)
        self.closed = True
