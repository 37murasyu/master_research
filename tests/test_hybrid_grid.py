"""混成の計測の同期の格子（``app.net.sync_buffer.GridSpec``）が 1 か所から下流へ届くことを固定する。

**なぜこのテストがあるか。**

同期バッファ（``SyncBuffer``）は格子の間隔（30 Hz）と補間で埋める穴の上限（100 ms）を持つのに、かつては下流が同じ
値を直書きしていた（計測の格子の番号 ``round((t−t0)/33,333,333)``、EKF の dt、回の仕事を積む dt の上限 0.1 s、
肘の濾波 E± の dt、生 3D の格子とサイドカーの dt）。片方だけ変えると黙ってずれる。

- 既定の格子の値は従来の直書きと同じ（1/30 s・33,333,333 ns・100 ms・0.1 s）。出力の CSV の値は変わらない
- 組は同期バッファの格子の番号を持ち、計測は時刻から割り戻さずにそれを使う
- 格子を変える（15 Hz・200 ms）と、EKF の dt・積む dt の上限・サイドカーの dt・生 3D の格子がそれに追随する
- ``verify_run`` は frames の ``grid_index`` と生 CSV の ``frame``（格子の番号）で直接つなぐ。``grid_index`` の無い
  古い記録は時刻を格子に丸めて（round(t/dt)）合わせる
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import app.hybrid.replay as rp
from app.hybrid.ekf import EkfSettings
from app.net.protocol import LandmarkFrame
from app.net.sync_buffer import DEFAULT_GRID, GridSpec, PairedSample, SyncBuffer
from app.runners.network_measure import MeasurementConfig, NetworkMeasurement
from app.tuning.raw_capture import read_raw_capture
from tools import verify_run as vr
from tools.synth_session import write_session

GRID_15 = GridSpec(target_hz=15.0, max_gap_ms=200.0)


def _frame(role, t_ns, seq=0):
    return LandmarkFrame(role, seq, t_ns, 640, 480, [(0.5, 0.5, 0.0, 1.0)] * 33)


class TestGridSpec:
    def test_the_default_grid_has_the_values_that_used_to_be_written_out(self):
        assert DEFAULT_GRID.target_hz == 30.0
        assert DEFAULT_GRID.period_ns == 33_333_333
        assert DEFAULT_GRID.period_s == 1.0 / 30.0   # 厳密に同じ浮動小数（出力の dt の列・サイドカーのバイト）
        assert DEFAULT_GRID.max_gap_ns == 100_000_000
        assert DEFAULT_GRID.max_gap_s == 0.1

    def test_the_sync_buffer_keeps_its_old_arguments(self):
        buffer = SyncBuffer(target_hz=15.0, max_gap_ms=200.0)
        assert buffer.grid == GRID_15
        assert (buffer.period_ns, buffer.max_gap_ns) == (66_666_667, 200_000_000)
        assert SyncBuffer(grid=GRID_15, target_hz=30.0).grid == GRID_15, "grid を渡したら target_hz より優先する"
        assert SyncBuffer().grid == DEFAULT_GRID

    def test_a_non_positive_rate_is_rejected(self):
        with pytest.raises(ValueError):
            GridSpec(target_hz=0.0)
        with pytest.raises(ValueError):
            SyncBuffer(target_hz=-1.0)


class TestPairsCarryTheGridIndex:
    def test_the_index_counts_grid_slots_and_skips_the_ones_that_were_dropped(self):
        buffer = SyncBuffer()
        period = DEFAULT_GRID.period_ns
        t0 = 1_000_000_000
        # cam1 は 0.2 s（100 ms を超える穴）抜ける。その間の格子は組にならない
        for k in range(30):
            buffer.push(_frame("cam0", t0 + k * period, k))
            if not 9 <= k <= 15:
                buffer.push(_frame("cam1", t0 + k * period, k))
        pairs = buffer.drain()
        indices = [p.grid_index for p in pairs]
        assert indices == [k for k in range(30) if not 9 <= k <= 15]
        assert all(p.t_ns == t0 + p.grid_index * period for p in pairs), "番号と時刻が格子の上で一致しない"

    def test_the_measurement_takes_the_index_from_the_pair(self):
        """組が番号を持っていれば、時刻（ここではわざとずらす）から割り戻さない。"""
        from test_network_measure import (POSE_KEYPOINTS, _body_points, _pair_from_pixels, _project,
                                          _stereo_projections)

        P0, P1 = _stereo_projections()
        measurement = NetworkMeasurement(P0, P1, POSE_KEYPOINTS, MeasurementConfig(ekf=EkfSettings(enabled=False)))
        results = []
        for k in (5, 6, 9, 10):
            points = _body_points(k / 30.0)
            pair = _pair_from_pixels(0, _project(P0, points), _project(P1, points))
            # 格子の時刻から 7 ms × k ずらす。時刻から割り戻すと 9 は 5（(4 格子 + 28 ms) / 33.3 ms を丸める）になる
            jittered = PairedSample(t_ns=k * DEFAULT_GRID.period_ns + 7_000_000 * k, frames=pair.frames, grid_index=k)
            results.append(measurement.process(jittered))
        assert [r.grid_index for r in results] == [0, 1, 4, 5]
        assert all(r.t0_ns == results[0].t_ns for r in results), "時刻の原点は最初の組"


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Pixel 13 Hz、0.25 s と 0.6 s の穴がある合成の押し上げ 1 回（抜けた格子・補間・EKF の作り直しを通す）。"""
    root = tmp_path_factory.mktemp("synth")
    return write_session(root, reps=1, still_s=2.0, pixel_hz=13.0, gaps_s=[(2.5, 2.75), (3.5, 4.1)])


def _replay(synth, tmp_path, grid):
    box = {}
    config = MeasurementConfig(body_mass_kg=65.0, grid=grid)
    out = rp.replay(synth, root=tmp_path / "replay", speed=0, config=config,
                    on_session=lambda s: box.setdefault("session", s))
    stamp = next(out.glob("frames_*.csv")).stem.removeprefix("frames_")
    return out, stamp, box["session"]


@pytest.mark.parametrize("grid", [DEFAULT_GRID, GRID_15], ids=["30Hz", "15Hz"])
class TestTheGridReachesDownstream:
    def test_ekf_dt_work_limit_and_sidecar_follow_the_grid(self, synth, tmp_path, grid):
        out, stamp, session = _replay(synth, tmp_path, grid)
        measurement = session.measurement
        assert measurement.ekf is not None and measurement.ekf.dt == grid.period_s
        assert measurement.rep_work.max_step_s == grid.max_gap_s
        assert session.recorder.grid == grid

        sidecar = json.loads(next(out.glob("kpts3d_raw_*.json")).read_text(encoding="utf-8"))
        assert sidecar["dt"] == grid.period_s
        assert sidecar["src_fps"] == grid.target_hz
        assert f"{grid.target_hz:g} Hz の格子" in sidecar["dt_source"]

        frames = pd.read_csv(out / f"frames_{stamp}.csv")
        t = frames["t_ns"].to_numpy(np.int64)
        grid_index = frames["grid_index"].to_numpy(int)
        assert grid_index[0] == 0 and np.all(np.diff(grid_index) >= 1)
        assert np.any(np.diff(grid_index) > 1), "穴で抜けた格子が無い（合成の穴が効いていない）"
        assert np.array_equal(t - t[0], grid_index * grid.period_ns), "格子の番号と時刻が格子の上で一致しない"

        raw = read_raw_capture(out / f"kpts3d_raw_{stamp}.csv")
        assert np.array_equal(raw.frame, np.arange(grid_index[-1] + 1)), "生 3D は抜けた格子も行を持つ"
        blank = ~np.isfinite(raw.points).any(axis=(1, 2))
        assert blank.any()
        assert np.array_equal(raw.t[blank], raw.frame[blank] * grid.period_ns / 1e9)

        energy = pd.read_csv(out / f"cycle_energy_{stamp}.csv", float_precision="round_trip")
        assert (energy["dt_sec"] == grid.period_s).all()

    def test_verify_run_joins_on_the_grid_index(self, synth, tmp_path, grid):
        out, stamp, _ = _replay(synth, tmp_path, grid)
        ekf = vr.check_run(out)["ekf"]
        assert ekf["alignment"] == "grid_index"
        assert ekf["matched_rows"] == ekf["kpts_rows"] > 0


def test_an_old_record_without_grid_index_is_aligned_by_rounding(synth, tmp_path):
    out, stamp, _ = _replay(synth, tmp_path, DEFAULT_GRID)
    joined = vr.check_run(out)["ekf"]
    path = out / f"frames_{stamp}.csv"
    pd.read_csv(path).drop(columns="grid_index").to_csv(path, index=False)
    rounded = vr.check_run(out)["ekf"]
    assert rounded["alignment"] == "round(t/dt)"
    assert rounded["matched_rows"] == joined["matched_rows"]
    assert [row["rms_mm"] for row in rounded["series"]] == [row["rms_mm"] for row in joined["series"]]
