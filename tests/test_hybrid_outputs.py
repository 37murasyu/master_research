"""混成の計測の記録（``app.hybrid.recorder.Recorder``、計画の T10・T11）。

**なぜこのテストがあるか。**

混成の記録は 3D（EKF なし）・トルク・サイクルの符号付きの仕事しか残しておらず、2026-09-23 の実機の計測では
EKF の較正（``tune_ekf``）にかける生 3D も、回ごとのスコア（W_pos / W_1RM、論文 4.5.2 節）も、関所の状態も
後から確かめられなかった。記録のファイルは足すだけにし（既存の列は変えない）、次を残す:

- ``kpts3d_raw_<stamp>.csv``: EKF の手前の 3D（``RawCaptureWriter`` の形）。1/30 s の格子で、抜けた格子は NaN の行
  （行を詰めると dt 一定の前提が崩れて ``tune_ekf`` の推定が狂う）
- ``cycle_work_<stamp>.csv``: 既存の ``work_j``（符号付き W±）の後ろに W+・W−・W_1RM・スコア・status（確定した回は
  closed、止めたときに開いたままだった回は末尾に unfinished）
- ``frames_<stamp>.csv``: 既存の列の後ろに格子の番号・dt・関所・高さ・回・腕の長さの安全策・回の区切りの基準の高さ
- ``meta.json``: 被験者・1RM・体重・前腕長・帯・重力・EKF の出どころ・関所
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from app.gauge.tracker import GaugeTracker
from app.hybrid.measurement import MeasurementSession
from app.net.protocol import LandmarkFrame
from app.runners.network_measure import MeasurementConfig
from app.tuning.raw_capture import read_raw_capture
from config import pose_keypoints
from hybrid_pushup import PushUp, calibrated_pairs
from test_hybrid_measure import calibration

ONE_RM = {"elbow_L": 20.0, "elbow_R": 22.0, "wrist_L": 8.0, "wrist_R": 9.0}
IDS = sorted(pose_keypoints)


def _session(tmp_path, *, reps=1, drop=(), **config):
    """押し上げの合成を本物の MeasurementSession に流した計測フォルダ。校正は test_hybrid_measure の歪みつき。"""
    cal = calibration(tmp_path)
    tracker = GaugeTracker()
    session = MeasurementSession(
        cal, root=tmp_path / "measure", tracker=tracker,
        config=MeasurementConfig(body_mass_kg=65.0, one_rm=ONE_RM, subject_id="00", **config))
    session.on_landmarks(LandmarkFrame("cam1", 0, 1, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))

    pairs = calibrated_pairs(PushUp(reps=reps), drop=drop)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.on_pairs(pairs)
    assert session.exit_code == 0, session.error
    session.stop_reason = "stop_request"
    session.close()
    return session, tracker


def _file(session, prefix):
    stamp = next(session.directory.glob("frames_*.csv")).stem[len("frames_"):]
    return session.directory / f"{prefix}_{stamp}.csv"


class TestCycleWork:
    def test_the_columns_are_appended(self, tmp_path):
        session, _ = _session(tmp_path)
        table = pd.read_csv(_file(session, "cycle_work"))
        assert list(table.columns) == ["frame", "t_ns", "joint", "work_j", "work_pos_j", "work_neg_j", "w1rm_j", "score",
                                       "status"]
        elbow = table[table.joint == "elbow_R"].iloc[0]
        assert elbow.work_j == pytest.approx(elbow.work_pos_j + elbow.work_neg_j)
        assert elbow.work_pos_j == pytest.approx(22.0, rel=0.25)
        w1rm = session.measurement.bands["elbow_R"].w1rm
        assert elbow.w1rm_j == pytest.approx(w1rm, rel=1e-6)
        assert elbow.score == pytest.approx(elbow.work_pos_j / w1rm, rel=1e-6)
        shoulder = table[table.joint == "shoulder_R"].iloc[0]
        assert np.isnan(shoulder.w1rm_j) and np.isnan(shoulder.score), "帯が無ければ空"


class TestUnfinishedRep:
    """止めたときに開いたままだった回を cycle_work に「未完」（status=unfinished）で残す（2026-09-24 のレビュー）。

    かつては確定した回しか書かず、最後の押し上げの途中で止めるとその回の仕事が記録のどこにも残らなかった。基準が
    ずれて回が閉じなくなったときも（W+ 231.7 J の開いた回）、記録からは何が起きたか分からなかった。
    """

    def _stopped_mid_rep(self, tmp_path):
        """押し上げ 2 回のうち、2 回目の上げの頂点で止めた計測フォルダ。"""
        cal = calibration(tmp_path)
        session = MeasurementSession(
            cal, root=tmp_path / "measure",
            config=MeasurementConfig(body_mass_kg=65.0, one_rm=ONE_RM, subject_id="00"))
        session.on_landmarks(LandmarkFrame("cam1", 0, 1, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))
        motion = PushUp(reps=2)
        top = motion.rest_s + motion.period_s + motion.rise_s + 0.2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.on_pairs(calibrated_pairs(motion)[: int(top * 30)])
        session.stop_reason = "stop_request"
        session.close()
        return session

    def test_the_open_rep_is_kept_as_unfinished_rows(self, tmp_path):
        session = self._stopped_mid_rep(tmp_path)
        table = pd.read_csv(_file(session, "cycle_work"))
        assert list(table.status.unique()) == ["closed", "unfinished"]
        closed, open_ = table[table.status == "closed"], table[table.status == "unfinished"]
        assert sorted(closed.joint) == sorted(open_.joint) and len(open_) == 6, "部位ごとに 1 行"
        elbow = open_[open_.joint == "elbow_R"].iloc[0]
        assert elbow.work_pos_j == pytest.approx(22.0, rel=0.3), "上げの仕事は積んである"
        assert elbow.work_j == pytest.approx(elbow.work_pos_j + elbow.work_neg_j)
        assert elbow.score == pytest.approx(elbow.work_pos_j / elbow.w1rm_j)
        frames = pd.read_csv(_file(session, "frames"))
        assert elbow.frame == frames.frame.iloc[-1] and elbow.t_ns == frames.t_ns.iloc[-1]
        assert frames.cycle_detected.sum() == 1, "未完の回は確定した回に数えない"
        meta = json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))
        assert meta["reps"] == 1
        unfinished = meta["unfinished_rep"]
        assert unfinished["parts"]["elbow_R"]["work_pos_j"] == pytest.approx(elbow.work_pos_j)
        assert unfinished["max_lift_m"] == pytest.approx(0.13, abs=0.01)

    def test_nothing_is_added_when_every_rep_closed(self, tmp_path):
        session, _ = _session(tmp_path)
        table = pd.read_csv(_file(session, "cycle_work"))
        assert (table.status == "closed").all()
        meta = json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))
        assert meta["unfinished_rep"] is None


class TestFrames:
    def test_the_columns_are_appended(self, tmp_path):
        session, _ = _session(tmp_path, drop={50, 51, 52, 53})
        table = pd.read_csv(_file(session, "frames"))
        assert list(table.columns) == ["frame", "t_ns", "t_s", "cycle_detected", "grid_index", "dt_s", "dyn_active",
                                       "height_m", "rep", "arm_ok_L", "arm_ok_R", "baseline_m"]
        meta = json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))
        assert table.baseline_m.dropna().iloc[0] == pytest.approx(meta["baseline_height_m"])
        assert meta["rep_baseline"]["initial_m"] == pytest.approx(meta["baseline_height_m"])
        assert meta["rep_baseline"]["final_m"] == pytest.approx(table.baseline_m.iloc[-1])
        assert 50 not in set(table.grid_index) and 54 in set(table.grid_index)
        assert table.loc[table.grid_index == 54, "dt_s"].item() == pytest.approx(5 / 30, rel=1e-6)
        assert table.dyn_active.sum() > 20 and table.rep.iloc[-1] == 1
        assert table.arm_ok_L.all() and table.arm_ok_R.all()


class TestMeta:
    def test_the_keys(self, tmp_path):
        session, _ = _session(tmp_path)
        meta = json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))
        assert meta["subject_id"] == "00"
        assert meta["one_rm_kg"] == ONE_RM
        assert meta["body_mass_kg"] == 65.0
        assert meta["dyn_gate"] is True
        assert meta["ekf"]["enabled"] is True and meta["ekf"]["origin"] == "builtin"
        assert meta["arm_length_guard"] == {"tolerance": 0.25, "rejected_frames": {"L": 0, "R": 0}}
        assert meta["forearm_len_m"]["R"] == pytest.approx(0.25, abs=0.01)
        bands = session.measurement.bands
        assert meta["w1rm_j"]["elbow_R"] == pytest.approx(bands["elbow_R"].w1rm)
        assert meta["gauge_bands_j"]["elbow_R"] == pytest.approx(list(bands["elbow_R"].band))
        assert meta["gravity"]["label"] == "Z-" and meta["gravity"]["source"] == "trunk"
        assert meta["gravity"]["vector"] == pytest.approx([0.0, 0.0, -9.81])
        assert meta["output_schema_version"] == 2
        assert meta["reps"] == 1


class TestRawCapture:
    """EKF の較正（``tune_ekf``）にそのまま使える生 3D。名前は USB と同じ ``kpts3d_raw_<stamp>.csv``。"""

    DROP = {50, 51, 52, 53}

    def test_the_raw_capture_is_on_the_grid_with_nan_rows(self, tmp_path):
        session, _ = _session(tmp_path, drop=self.DROP)
        capture = read_raw_capture(_file(session, "kpts3d_raw"))
        assert capture.landmark_ids == tuple(IDS)
        n = int(round(PushUp(reps=1).duration_s * 30))
        np.testing.assert_array_equal(capture.frame, np.arange(n)), "抜けた格子も行がある"
        np.testing.assert_allclose(np.diff(capture.t), 1 / 30, rtol=1e-6)
        for k in self.DROP:
            assert np.isnan(capture.points[k]).all()
        assert np.isfinite(capture.points[49]).all() and np.isfinite(capture.points[54]).all()

    def test_the_sidecar_names_the_source_and_the_ekf(self, tmp_path):
        session, _ = _session(tmp_path)
        meta = read_raw_capture(_file(session, "kpts3d_raw")).provenance
        assert (meta["source"], meta["times"], meta["unit"], meta["frame"]) == ("hybrid", "grid", "m", "runtime")
        assert meta["dt"] == pytest.approx(1 / 30)
        assert meta["ekf_noise"]["origin"] == "builtin"
        assert meta["EKF_ENABLE"] is True and meta["EKF_ROBUST_GATE"] is True
        assert meta["HYBRID_EKF_PROFILE"] is None
        # 先頭の窓が閉じたときに書き足す
        assert meta["ekf_run_length_m"] == pytest.approx(0.28, abs=0.01)
        assert meta["gravity_label"] == "Z-"
        assert meta["gravity"] == pytest.approx([0.0, 0.0, -9.81])

    def test_the_raw_values_are_before_the_ekf(self, tmp_path):
        session, _ = _session(tmp_path)
        capture = read_raw_capture(_file(session, "kpts3d_raw"))
        kpts = pd.read_csv(_file(session, "kpts3d")).drop(columns="frame").to_numpy(float)
        raw = capture.points.reshape(len(capture.points), -1)
        assert raw.shape == kpts.shape
        assert not np.allclose(raw, kpts, atol=1e-9), "EKF の後の値と同じになっている"
        assert np.nanmax(np.abs(raw - kpts)) < 0.05


class TestTiming:
    """受信スレッドの ``process`` の時間（計画の「速さの予算」）。再生の検証で 95% < 10 ms を確かめるため meta に残す。"""

    def test_the_process_time_is_kept_in_the_meta(self, tmp_path):
        session, _ = _session(tmp_path)
        timing = json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))["timing"]
        assert timing["frames"] == int(round(PushUp(reps=1).duration_s * 30))
        assert 0.0 < timing["median_ms"] <= timing["p95_ms"] <= timing["max_ms"]
        assert timing["p95_ms"] < 50.0


class TestCheck:
    def test_the_structure_checks_pass(self, tmp_path):
        """朝の手順の ``python -m tools.verify_run check <計測フォルダ>`` が、新しい記録で構造の検査に通る。"""
        from tools import verify_run as vr

        session, _ = _session(tmp_path, reps=2)
        report = vr.check_run(session.directory, expect_stop=True)
        assert report["kind"] == "hybrid"
        failed = [c for c in report["checks"] if not c["ok"] and not c["name"].startswith(("3D:", "配置:", "速さ:"))]
        assert not failed, failed
        assert report["cycles"]["detected"] == 2


class TestCycleEnergy:
    """肘の濾波 E±（USB 経路の ``compute_cycle_energy_filtered``、``cycle_energy_debug_*`` と同じ量）を回ごとに残す。

    混成の回の値は Σ P·dt（関節の仕事）で、USB が肘に使う濾波の経路（LPF → 80 点に再標本化 → τ を分位で切る →
    dθ 制限 → ∫τdθ の正負）と直接比べられなかった。オフラインの検証で比べられるよう、同じ計算を回の確定で行う。
    """

    def test_one_row_per_elbow_per_rep(self, tmp_path):
        session, _ = _session(tmp_path, reps=2)
        table = pd.read_csv(_file(session, "cycle_energy"))
        assert list(table.columns) == ["frame", "t_ns", "part", "e_pos", "e_neg", "fc_current", "dt_sec", "n_u"]
        assert sorted(table.part) == ["elbow_L", "elbow_L", "elbow_R", "elbow_R"]
        assert (table.fc_current == 1.2).all() and (table.n_u == 80).all()
        assert table.dt_sec.iloc[0] == pytest.approx(1 / 30)
        work = pd.read_csv(_file(session, "cycle_work"))
        elbow = work[work.joint == "elbow_R"].work_pos_j.to_numpy()
        energy = table[table.part == "elbow_R"]
        # 向きの取り方（θ は肩→肘と肘→手首のなす角）で E+ と E− のどちらに出るかは変わる。大きさは Σ max(P,0)·dt と同じ桁
        magnitude = np.maximum(energy.e_pos.to_numpy(), energy.e_neg.to_numpy())
        np.testing.assert_allclose(magnitude, elbow, rtol=0.5)

    def test_the_adaptive_cutoff_runs(self, tmp_path):
        from energy_pipeline import EnergyFilterConfig

        session, _ = _session(tmp_path, reps=2, energy_filter=EnergyFilterConfig(fc_adaptive_on=True))
        table = pd.read_csv(_file(session, "cycle_energy"))
        assert len(table) == 4
        # fc は既定の 1.2 Hz から EMA（β=0.15、1 秒ごと）で推定値 k·f0 の側へ動いていく。短い合成では途中の値
        assert (table.fc_current > 1.2).all() and (table.fc_current <= 6.0).all()
        assert np.isfinite(table.e_pos).all()


class TestGaugeEnergy:
    """ゲージに出した値を毎フレーム残す（USB の ``gauge_energy_*`` に当たる。§6-2 の確認と、朝の再生の検証用）。"""

    def test_the_gauge_values_per_frame(self, tmp_path):
        session, tracker = _session(tmp_path)
        table = pd.read_csv(_file(session, "gauge_energy"))
        assert list(table.columns) == ["frame", "t_ns", "rep", "dyn_active", "elbow_L", "elbow_R", "wrist_L", "wrist_R"]
        frames = pd.read_csv(_file(session, "frames"))
        assert len(table) == len(frames)
        closing = frames.index[frames.cycle_detected == 1][0]
        assert table.elbow_R.iloc[closing - 1] == pytest.approx(tracker.snapshot().parts["elbow_R"].prev)
        assert table.elbow_R.iloc[closing] == 0.0, "確定したフレームで 0 に戻る"
        assert (table.loc[table.dyn_active == 0, "elbow_R"].iloc[:30] == 0.0).all()

    def test_the_sidecar_explains_the_bands(self, tmp_path):
        session, _ = _session(tmp_path)
        meta = json.loads(_file(session, "gauge_energy").with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["unit"] == "J"
        assert "Σmax(P, 0)·dt" in meta["definition"]
        assert meta["gauge_bands_j"]["elbow_R"] == pytest.approx(list(session.measurement.bands["elbow_R"].band))
        assert meta["one_rm_kg"] == ONE_RM and meta["body_mass_kg"] == 65.0
        assert meta["forearm_len_m"]["L"] == pytest.approx(0.25, abs=0.01)
        assert meta["reps"] == 1


class TestTorqueVectors:
    """USB と同じ形の局所トルク（``aim_torque_vec_<stamp>_s2_g<重力>.csv``）。オフラインの解析の道具がそのまま読める。"""

    PARTS = ["wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L"]

    def test_the_wide_table_matches_the_local_torques(self, tmp_path):
        session, _ = _session(tmp_path)
        stamp = _file(session, "frames").stem[len("frames_"):]
        path = session.directory / f"aim_torque_vec_{stamp}_s2_gZ-.csv"
        wide = pd.read_csv(path)
        assert list(wide.columns) == ["frame"] + [f"{p}_{a}" for p in self.PARTS for a in "xyz"]
        long = pd.read_csv(_file(session, "local_torque"))
        elbow = long[long.joint == "elbow_R"].set_index("frame")
        assert set(wide.frame) == set(elbow.index), "frame は kpts3d・local_torque と同じ番号"
        row = wide.iloc[10]
        assert row.elbow_R_y == pytest.approx(elbow.loc[int(row.frame), "y"])


class TestOfflineWrist:
    """``OFFLINE_WRIST_CAPTURE=1`` のとき、USB と同じ npy（前腕＝肘→手首 (N,3)、手首の局所 τ_y (N,)）を残す。"""

    def test_the_npy_files(self, tmp_path):
        session, _ = _session(tmp_path, offline_wrist_capture=True)
        stamp = _file(session, "frames").stem[len("frames_"):]
        torques = pd.read_csv(_file(session, "local_torque"))
        for side in ("R", "L"):
            forearm = np.load(session.directory / f"forearm_{side}_{stamp}_s2.npy")
            tau = np.load(session.directory / f"tau_wrist_{side}_{stamp}_s2.npy")
            wrist = torques[torques.joint == f"wrist_{side}"]
            assert forearm.shape == (len(wrist), 3) and tau.shape == (len(wrist),)
            np.testing.assert_allclose(tau, wrist.y.to_numpy(), rtol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(forearm, axis=1), 0.25, atol=0.02)

    def test_off_by_default(self, tmp_path):
        session, _ = _session(tmp_path)
        assert not list(session.directory.glob("*.npy"))
