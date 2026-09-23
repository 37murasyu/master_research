"""Incremental measurement writer. Construct and call only on the receiver loop thread."""

import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import threading
import time
from app.hybrid.calibration_io import FILES, write_json
from app.hybrid.paths import measurement_root


class Recorder:
    def __init__(
        self,
        calibration,
        pose_keypoints,
        *,
        root=None,
        metadata=None,
        clock=time.monotonic,
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
        self.times = writer("frames", ["frame", "t_ns", "t_s", "cycle_detected"])
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
        self.work = writer("cycle_work", ["frame", "t_ns", "joint", "work_j"])
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

    def record(self, result):
        self._check()
        if self._first_ns is None:
            self._first_ns = result.t_ns
        self.points.writerow([self.frames, *result.points_3d.ravel()])
        self.times.writerow(
            [
                self.frames,
                result.t_ns,
                (result.t_ns - self._first_ns) / 1e9,
                int(result.cycle_detected),
            ]
        )
        self.torques.writerows(
            [self.frames, result.t_ns, key, *value]
            for key, value in result.local_torques.items()
        )
        self.work.writerows(
            [self.frames, result.t_ns, key, value]
            for key, value in result.cycle_work_j.items()
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
        self.meta.update(
            status="complete",
            ended_at=datetime.now(timezone.utc).isoformat(),
            frames=self.frames,
        )
        self.meta.update(metadata)
        write_json(self.directory / "meta.json", self.meta)
        self.closed = True
