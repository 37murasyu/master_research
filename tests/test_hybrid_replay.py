"""記録した混成の計測（``landmarks2d_*``）を、計測中と同じ道筋で流し直す再生の入口（``app.hybrid.replay``）。

**なぜこのテストがあるか。**

被験者がいない夜のうちに（また朝の練習で）、実機の記録や合成の押し上げを子プロセスの計測の全体（受け付けの検査、
30 Hz 格子への補間、三角測量、EKF、回の区切り、ゲージの値）に通して確かめたい。再生が本番と違う道筋を通ると、
本番でだけ起きる不具合（組の抜け、補間、解像度の検査、記録を始める条件）を見逃す。そこで再生は、受信スレッドと
同じ順（``accept_frame`` → ``SyncBuffer.push`` → ``on_landmarks`` → ``drain`` → ``on_pairs``）で
``MeasurementSession`` を呼び、記録は本物の ``Recorder`` が書く。

Recorder は作ったスレッドからしか書けないので、記録を開くのも閉じるのも再生のスレッドで行う。
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pandas as pd
import pytest

import app.hybrid.replay as rp
from app.hybrid.ekf import EkfSettings
from app.runners.network_measure import MeasurementConfig
from app.hybrid.retriangulate import read_landmarks
from test_hybrid_verification import _expected, cut_the_last_frame, make_body_run


def _outputs(directory):
    stamp = next(directory.glob("frames_*.csv")).stem.removeprefix("frames_")
    return stamp, json.loads((directory / "meta.json").read_text(encoding="utf-8"))


class TestMergedFrames:
    def test_frames_are_in_capture_order_within_the_window(self, tmp_path):
        frames = rp.merged_frames(read_landmarks(make_body_run(tmp_path, seconds=2.0)), start_s=0.5, end_s=1.5)
        times = [f.t_capture_ns for f in frames]
        assert times == sorted(times)
        assert min(times) >= 0.5e9 and max(times) < 1.5e9, "窓の外のフレームが混ざった"
        assert {f.role for f in frames} == {"cam0", "cam1"}

    def test_the_window_is_measured_from_the_first_frame_of_either_camera(self, tmp_path):
        frames = read_landmarks(make_body_run(tmp_path, seconds=2.0))
        first = min(f.t_capture_ns for role in frames.values() for f in role)
        merged = rp.merged_frames(frames, start_s=1.0)
        assert min(f.t_capture_ns for f in merged) >= first + 1.0e9


class TestReplay:
    def test_replay_reproduces_the_measurement_through_the_sync_buffer(self, tmp_path):
        """配管の検査なので EKF は切る（この体は 1.2 Hz で奥行きに速く揺れ、Pixel 12 Hz の補間の点を EKF の門が
        外れ値と見て 20 cm 級の飛びを作る。押し上げの速さでは起きない。2026-09-24 の検証、KNOWN_ISSUES §6-10）。"""
        session = make_body_run(tmp_path, seconds=3.0)
        out = rp.replay(session, root=tmp_path / "replay", speed=0,
                        config=MeasurementConfig(body_mass_kg=65.0, ekf=EkfSettings(enabled=False)))
        stamp, meta = _outputs(out)
        assert out.parent == tmp_path / "replay", "再生の記録は計測の記録と混ざらない場所に書く"
        assert meta["status"] == "complete"
        assert meta["replay_of"] == str(session)
        frames = pd.read_csv(out / f"frames_{stamp}.csv")
        assert len(frames) > 60, "30 Hz 格子の組がほとんど出ていない"
        points = pd.read_csv(out / f"kpts3d_{stamp}.csv").drop(columns="frame").to_numpy().reshape(len(frames), -1, 3)
        for k in (20, 45, 70):
            if k < len(frames):
                t = frames["t_ns"].iloc[k] / 1e9
                assert np.nanmax(np.abs(points[k] - _expected(t))) < 0.02, "再生の 3D が記録の体とずれた"

    def test_a_record_cut_by_a_kill_replays(self, tmp_path):
        """kill で ``landmarks2d`` の最後のフレームが途中で切れた記録も、最後まで流して complete で閉じる。

        以前は点の足りないフレームを流し、計測が IndexError で failed になった。
        """
        session = make_body_run(tmp_path, seconds=2.0)
        cut_the_last_frame(session)
        out = rp.replay(session, root=tmp_path / "replay", speed=0,
                        config=MeasurementConfig(body_mass_kg=65.0, ekf=EkfSettings()))
        meta = _outputs(out)[1]
        assert (meta["status"], meta.get("error"), meta["stop_reason"]) == ("complete", None, "replay_end")

    def test_the_timing_of_the_measurement_is_recorded(self, tmp_path):
        out = rp.replay(make_body_run(tmp_path, seconds=2.0), root=tmp_path / "replay", speed=0)
        timing = _outputs(out)[1]["replay_timing"]
        assert timing["pairs"] > 30
        assert 0 <= timing["on_pairs_ms_median"] <= timing["on_pairs_ms_p95"] <= timing["on_pairs_ms_max"]

    def test_real_time_replay_waits_for_the_capture_times(self, tmp_path):
        """speed=2 なら記録の 2 秒を約 1 秒で流す（実時間の再生で、ゲージの動きを画面で確かめるため）。"""
        now = [0.0]

        def sleep(seconds):
            now[0] += max(0.0, seconds)

        rp.replay(make_body_run(tmp_path, seconds=2.0), root=tmp_path / "replay", speed=2.0,
                  clock=lambda: now[0], sleep=sleep)
        assert now[0] == pytest.approx(1.0, abs=0.1)

    def test_a_stop_request_closes_the_record(self, tmp_path):
        """GUI の停止ボタン（停止ファイル）で止めても、記録は閉じて meta に理由が残る（§3-2 と同じ扱い）。"""
        calls = {"n": 0}

        def should_stop():
            calls["n"] += 1
            return calls["n"] > 60

        out = rp.replay(make_body_run(tmp_path, seconds=3.0), root=tmp_path / "replay", speed=0,
                        should_stop=should_stop)
        meta = _outputs(out)[1]
        assert meta["status"] == "complete"
        assert meta["stop_reason"] == "stop_request"

    def test_a_stop_request_in_a_gap_is_seen_within_a_tenth_of_a_second(self, tmp_path):
        """記録に 2 台とも点の無い区間（人が画面の外）があっても、実時間の再生の停止の要求は 0.1 s 以内に効く。

        以前は次の点の時刻まで 1 回で寝たので、区間の長さだけ停止が効かなかった。GUI の猶予（10 s）を超えると
        kill され、再生の記録の meta が recording のまま残った。
        """
        session = make_body_run(tmp_path, seconds=3.0)
        path = next(session.glob("landmarks2d_*.csv"))
        table = pd.read_csv(path)
        table[(table["t_ns"] < 0.5e9) | (table["t_ns"] >= 2.5e9)].to_csv(path, index=False)
        now = [0.0]
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += max(0.0, seconds)

        out = rp.replay(session, root=tmp_path / "replay", speed=1.0, clock=lambda: now[0], sleep=sleep,
                        should_stop=lambda: now[0] >= 1.0)
        assert max(sleeps) <= 0.1 + 1e-9, "一度に長く寝た"
        assert now[0] <= 1.1 + 1e-9, f"停止の要求（1.0 s）から戻るまでが長い（{now[0]:.2f} s）"
        assert _outputs(out)[1]["stop_reason"] == "stop_request"

    def test_the_record_is_written_from_the_replay_thread(self, tmp_path):
        """Recorder は作ったスレッドからしか書けない。別スレッドで回しても開閉が同じスレッドで起きる。"""
        box = {}
        thread = threading.Thread(target=lambda: box.setdefault(
            "out", rp.replay(make_body_run(tmp_path, seconds=1.5), root=tmp_path / "replay", speed=0)))
        thread.start()
        thread.join(timeout=60)
        assert _outputs(box["out"])[1]["status"] == "complete"
