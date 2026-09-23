"""混成ステレオ（Mac 内蔵カメラ＋同じ Wi-Fi の Pixel 7a）の計測でも、§6-2・§6-3・§3-2 を確かめられることを固定する。

**なぜこのテストがあるか。**

実際の計測構成は Pixel 7a 1 台＋Mac 内蔵カメラ（``app.hybrid``、``app.runners.hybrid_measure``）で、USB カメラ 2 台の
経路（``master_research_code.py``）とは出力の形が違う。Pixel の映像は JPEG で毎秒 3〜4 枚しか届かない
（``app.hybrid.link`` の PREVIEW・CALIBRATION）ので、30 fps で残るのは 2D ランドマーク（``landmarks2d_*``）と
三角測量した 3D（``kpts3d_*``、EKF なし）。検証はこの記録の上に作る。

- §3-2: 停止の要求で止めても記録を正しく閉じる（``meta.json`` の ``status`` が ``complete``）。何で止まったかも
  ``meta.json`` に残す（以前は残しておらず、停止ボタンで止まったのか失敗で止まったのか区別できなかった）
- §6-2: トルクの大きさ、サイクルごとの仕事、Pixel・Mac・組の実際の速さ
- §6-3: 混成の経路は EKF を使っていない。S6（雑音の推定）は、3D を生 CSV の形（``kpts3d_raw_*``）に直せば
  ``app.tuning.ekf_estimate`` / ``app.runners.tune_ekf`` がそのまま使える。4 Hz 間引きは記録を間引いて作る
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.hybrid.measurement import MeasurementSession
from app.hybrid.recorder import Recorder
from app.net.protocol import LandmarkFrame
from app.runners.network_measure import FrameResult
from app.tuning.raw_capture import read_raw_capture
from config import pose_keypoints
from test_hybrid_measure import calibration
from tools import verify_run as vr

REPO_ROOT = Path(__file__).resolve().parents[1]
IDS = sorted(pose_keypoints)
N = 120
DT_NS = 33_333_333


def _pixel_frame(seq: int) -> LandmarkFrame:
    return LandmarkFrame("cam1", seq, seq * DT_NS, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33)


def make_hybrid_run(tmp_path: Path, *, frames=N, close=True, stop_reason="stop_request", step_ns=DT_NS) -> Path:
    """混成の計測 1 回ぶんのフォルダを、本物の記録器（Recorder）で作る。"""
    recorder = Recorder(calibration(tmp_path), pose_keypoints, root=tmp_path / "measure",
                        metadata={"body_mass_kg": 60.0, "gravity_mode": "axis"})
    base = np.array([[0.1 * i, 0.2, 1.0] for i in range(len(IDS))])
    for k in range(frames):
        for role, seq in (("cam0", k), ("cam1", k)):
            recorder.landmarks(LandmarkFrame(role, seq, k * step_ns, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))
        points = base + 0.01 * np.sin(k / 5.0)
        torques = {"wrist_R": np.array([0.0, 12.0, 0.0]), "elbow_R": np.array([0.0, -20.0, 1.0]),
                   "wrist_L": np.array([0.0, 8.0, 0.0]), "elbow_L": np.array([0.0, 15.0, 0.0])}
        detected = k in (40, 80)
        work = {"wrist_R": 3.0 + k / 40, "elbow_R": 5.0} if detected else {}
        recorder.record(FrameResult(t_ns=k * step_ns, points_3d=points, local_torques=torques,
                                    cycle_detected=detected, cycle_work_j=work))
    if close:
        recorder.close(status="complete", exit_code=0, error=None, size_drops=0, stop_reason=stop_reason)
    return recorder.directory


def _structural_failures(report):
    """構造の検査の不合格。make_hybrid_run の 3D は人体の形ではない（点を 10 cm 間隔に並べただけ）ので、配置と 3D の質の検査は除く。"""
    return [c for c in report["checks"] if not c["ok"] and not c["name"].startswith(("3D:", "配置:"))]


def _check(report, name):
    matches = [c for c in report["checks"] if c["name"] == name]
    assert len(matches) == 1, [c["name"] for c in report["checks"]]
    return matches[0]


class TestStopReason:
    def test_the_session_records_why_it_stopped(self, tmp_path):
        session = MeasurementSession(calibration(tmp_path), root=tmp_path / "measure")
        session.on_landmarks(_pixel_frame(0))
        session.stop_reason = "stop_request"
        session.close()
        meta = json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))
        assert (meta["status"], meta["stop_reason"]) == ("complete", "stop_request")

    def test_the_runner_names_the_stop_request(self):
        """停止ファイル・SIGTERM（GUI の停止ボタン）で抜けたときに stop_request と記録する。"""
        tree = ast.parse((REPO_ROOT / "app" / "runners" / "hybrid_measure.py").read_text(encoding="utf-8"))
        assigned = {node.value.value if isinstance(node.value, ast.Constant) else None
                    for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    for target in node.targets if getattr(target, "attr", None) == "stop_reason"}
        values = {c.value for node in ast.walk(tree) if isinstance(node, ast.IfExp)
                  for c in ast.walk(node) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        assert {"key", "ctrl_c"} <= assigned
        assert "stop_request" in assigned | values


class TestCheckHybrid:
    def test_a_complete_run_passes(self, tmp_path):
        report = vr.check_run(make_hybrid_run(tmp_path), expect_stop=True)
        assert report["kind"] == "hybrid"
        failed = _structural_failures(report)
        assert not failed, failed

    def test_the_latest_session_is_found_from_the_measure_folder(self, tmp_path):
        run = make_hybrid_run(tmp_path)
        assert vr.check_run(run.parent)["out_dir"] == str(run)

    def test_a_run_that_was_not_closed_fails(self, tmp_path):
        """kill されると meta.json は recording のまま。停止で記録を閉じられたか（§3-2）の判定に使う。"""
        report = vr.check_run(make_hybrid_run(tmp_path, close=False))
        assert not _check(report, "meta: 記録を正しく閉じた")["ok"]

    def test_a_stop_other_than_the_request_is_named(self, tmp_path):
        report = vr.check_run(make_hybrid_run(tmp_path, stop_reason="key"), expect_stop=True)
        assert not _check(report, "meta: 停止要求で止まった")["ok"]

    def test_torque_cycles_and_rates(self, tmp_path):
        report = vr.check_run(make_hybrid_run(tmp_path))
        assert report["torque"]["wrist_R"]["median_abs"] == pytest.approx(12.0)
        assert report["torque"]["elbow_R"]["median_abs"] == pytest.approx(20.0)
        assert report["cycles"]["detected"] == 2
        assert report["cycles"]["work"]["wrist_R"] == pytest.approx([4.0, 5.0])
        assert report["fps"]["processed_fps"] == pytest.approx(30.0, rel=1e-3)
        assert report["fps"]["role_fps"]["cam1"] == pytest.approx(30.0, rel=1e-3)
        assert "EKF" in report["ekf"]["note"]

    def test_a_short_file_is_named(self, tmp_path):
        run = make_hybrid_run(tmp_path)
        frames = next(run.glob("frames_*.csv"))
        pd.read_csv(frames).iloc[:10].to_csv(frames, index=False)
        assert not _check(vr.check_run(run), "行: kpts3d・frames・meta の frames が一致")["ok"]


class TestHybridRawCapture:
    def test_the_3d_becomes_a_raw_capture(self, tmp_path):
        run = make_hybrid_run(tmp_path)
        path = vr.hybrid_raw_capture(run, grid=True)
        capture = read_raw_capture(path)
        recorded = next(p for p in run.glob("kpts3d_*.csv") if not p.name.startswith("kpts3d_raw_"))
        kpts = pd.read_csv(recorded).drop(columns="frame").to_numpy(float)
        assert capture.landmark_ids == tuple(IDS)
        assert len(capture.points) == N
        assert np.allclose(capture.points.reshape(N, -1), kpts)
        assert capture.provenance["dt"] == pytest.approx(DT_NS / 1e9, rel=1e-6)
        assert capture.provenance["source"] == "hybrid"

    def test_a_stride_emulates_the_fixed_rate(self, tmp_path):
        """混成の経路には間引きの設定が無い。S6 の 2 設定目（4 Hz）は記録を 8 組おきに間引いて作る。"""
        path = vr.hybrid_raw_capture(make_hybrid_run(tmp_path), stride=8, grid=True)
        capture = read_raw_capture(path)
        assert capture.provenance["dt"] == pytest.approx(8 * DT_NS / 1e9, rel=1e-6)
        assert len(capture.points) == len(range(0, N, 8))
        assert path.name.endswith("_s8.csv")

    def test_converting_twice_and_checking_after_conversion(self, tmp_path):
        """変換した kpts3d_raw_* は計測フォルダに並ぶ。kpts3d_* のつもりで拾うと、2 回目の変換も検査も狂う。"""
        run = make_hybrid_run(tmp_path)
        vr.hybrid_raw_capture(run, grid=True)
        assert len(read_raw_capture(vr.hybrid_raw_capture(run, stride=8, grid=True)).points) == len(range(0, N, 8))
        report = vr.check_run(run)
        assert report["kind"] == "hybrid"
        assert not _structural_failures(report)

    def test_the_grid_can_still_be_used_for_comparison(self, tmp_path):
        path = vr.hybrid_raw_capture(make_hybrid_run(tmp_path), grid=True)
        assert read_raw_capture(path).provenance["times"] == "grid"

    def test_uneven_intervals_are_recorded(self, tmp_path):
        """組の間隔はネットワークで揺れる。EKF の推定は dt 一定を前提にするので、揺れの大きさを残す。"""
        path = vr.hybrid_raw_capture(make_hybrid_run(tmp_path), grid=True)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert {"interval_p05", "interval_p95"} <= set(meta)


# ---------------------------------------------------------------------------
# Pixel が 30 fps を出せないとき（実機の Pixel 7a は 10〜15 Hz）
# ---------------------------------------------------------------------------
#
# 同期バッファは 2 台を 30 Hz の格子へ線形補間で並べ直す（app.net.sync_buffer）。Pixel が 12 Hz なら、記録された
# 組の Pixel 側は大半が補間で作った点になる。補間の区間は直線なので、そのまま雑音の推定（S6）にかけると
# 「なめらかで雑音が小さい」系列に見える。S6 には、遅い方のカメラの実際の撮影時刻で三角測量し直した 3D を使う。

from app.hybrid import retriangulate as rt  # noqa: E402
from test_hybrid_measure import geometry  # noqa: E402
from test_network_measure import _body_points  # noqa: E402

import cv2 as cv  # noqa: E402


def wide_stereo():
    """Pixel を Mac から 150 cm 横に置き、被写体（奥行き 250 cm）へ向けた配置。肘での視線のなす角は約 30°。"""
    from app.hybrid.checkerboard import Stereo

    angle = np.arctan2(150.0, 250.0)
    R = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    T = -R @ np.array([150.0, 0.0, 0.0])
    return Stereo(R, T.reshape(3, 1), 0, [], list(range(12)))


def calibration_for(tmp_path: Path, stereo):
    from app.hybrid.calibration_io import load_calibration, save_calibration
    from app.hybrid.checkerboard import Board

    intr, _ = geometry()
    return load_calibration(save_calibration(intr, intr, stereo, Board(), root=tmp_path / "calibration",
                                             cameras=[{"kind": "mac", "device_id": "mac"},
                                                      {"kind": "pixel", "device_id": "pixel-1"}]))


def _marks(role: str, t: float, stereo=None) -> list[tuple[float, float, float, float]]:
    intr, narrow = geometry()
    stereo = stereo or narrow
    truth = _body_points(t)
    truth[:, 1] -= 25
    rvec = np.zeros(3) if role == "cam0" else cv.Rodrigues(np.asarray(stereo.R, dtype=float))[0].ravel()
    tvec = np.zeros(3) if role == "cam0" else np.asarray(stereo.T, dtype=float).ravel()
    pixels = cv.projectPoints(truth, rvec, tvec, intr.K, intr.distortion)[0].reshape(-1, 2)
    marks = [(0.0, 0.0, 0.0, 1.0)] * 33
    for slot, landmark_id in enumerate(IDS):
        marks[landmark_id] = (pixels[slot][0] / 1280, pixels[slot][1] / 720, 0.0, 1.0)
    return marks


def make_body_run(tmp_path: Path, *, seconds=4.0, mac_hz=30.0, pixel_hz=12.0, drop=(), stereo=None) -> Path:
    """Mac を mac_hz、Pixel を pixel_hz で撮った 2D と、30 Hz の格子の 3D（真の値）を記録した計測フォルダ。

    drop は Mac を撮らない時間帯 [s]。stereo を与えると Pixel をその配置にする（既定は基線 35 cm の狭い配置）。
    """
    cal = calibration_for(tmp_path, stereo) if stereo is not None else calibration(tmp_path)
    recorder = Recorder(cal, pose_keypoints, root=tmp_path / "measure")
    events = [("cam0", k / mac_hz) for k in range(int(seconds * mac_hz))
              if not any(a <= k / mac_hz < b for a, b in drop)]
    events += [("cam1", k / pixel_hz) for k in range(int(seconds * pixel_hz))]
    seqs = {"cam0": 0, "cam1": 0}
    for role, t in sorted(events, key=lambda e: e[1]):
        recorder.landmarks(LandmarkFrame(role, seqs[role], round(t * 1e9), 1280, 720, _marks(role, t, stereo)))
        seqs[role] += 1
    for k in range(int(seconds * 30)):   # 30 Hz の格子の組（3D は真の値）
        recorder.record(FrameResult(t_ns=round(k / 30 * 1e9), points_3d=_expected(k / 30)))
    recorder.close(status="complete", exit_code=0, stop_reason="stop_request")
    return recorder.directory


def _expected(t: float) -> np.ndarray:
    truth = _body_points(t)
    truth[:, 1] -= 25
    return truth[:, [0, 2, 1]] * -0.01


class TestRealTimePairs:
    def test_the_slower_camera_sets_the_times(self, tmp_path):
        result = rt.retriangulate(make_body_run(tmp_path))
        assert result.reference == "cam1"
        assert len(result.t_ns) == int(4.0 * 12), "Pixel の撮影時刻ごとに 1 組"
        assert np.median(np.diff(result.t_ns)) == pytest.approx(1e9 / 12, rel=1e-3)

    def test_the_3d_matches_the_body_at_the_real_times(self, tmp_path):
        result = rt.retriangulate(make_body_run(tmp_path))
        for k in (5, 20, 40):
            t = result.t_ns[k] / 1e9
            assert np.max(np.abs(result.points[k] - _expected(t))) < 0.01, "補間か三角測量がずれている"

    def test_a_long_gap_in_the_other_camera_is_not_bridged(self, tmp_path):
        """同期バッファと同じく、100 ms を超える穴は補間で埋めない。"""
        result = rt.retriangulate(make_body_run(tmp_path, drop=[(1.0, 1.3)]))
        times = result.t_ns / 1e9
        assert not np.any((times > 1.0) & (times < 1.3))
        assert result.skipped >= 3


class TestRawCaptureAtRealTimes:
    def test_the_raw_capture_uses_the_pixel_times_by_default(self, tmp_path):
        path = vr.hybrid_raw_capture(make_body_run(tmp_path))
        capture = read_raw_capture(path)
        assert capture.provenance["times"] == "cam1"
        assert capture.provenance["dt"] == pytest.approx(1 / 12, rel=1e-3)
        k = 10
        assert np.max(np.abs(capture.points[k] - _expected(capture.t[k] + capture.provenance["t0_s"]))) < 0.01

    def test_a_target_rate_picks_the_stride(self, tmp_path):
        """4 Hz 間引き（S6 の 2 設定目）に当たる間引き幅は、Pixel の実際の速さから決める（12 Hz なら 3 組おき）。"""
        capture = read_raw_capture(vr.hybrid_raw_capture(make_body_run(tmp_path), hz=4.0))
        assert capture.provenance["stride"] == 3
        assert capture.provenance["dt"] == pytest.approx(3 / 12, rel=1e-3)


class TestRateCheck:
    def test_a_slow_pixel_fails_the_30fps_check(self, tmp_path):
        report = vr.check_run(make_body_run(tmp_path))
        check = _check(report, "速さ: Mac・Pixel とも 24 fps 以上（30 fps の 8 割）")
        assert not check["ok"] and "Pixel" in check["detail"]
        assert report["fps"]["real_share"] == pytest.approx(12 / 30, abs=0.03), "組のうち Pixel の実測に基づく割合"

    def test_both_cameras_at_30fps_pass(self, tmp_path):
        assert _check(vr.check_run(make_hybrid_run(tmp_path)), "速さ: Mac・Pixel とも 24 fps 以上（30 fps の 8 割）")["ok"]


# ---------------------------------------------------------------------------
# 配置と 3D の質（本番の前の試し計測で、置き方を直すべきかをその場で決めるため）
# ---------------------------------------------------------------------------
#
# 2026-09-23 の実機の計測（Mac＋Pixel、4.5 分）では、Mac と Pixel の間（基線）が 36.7 cm しかなく、肘での 2 本の
# 視線のなす角が中央値 8° だった。わずかな 2D の誤差や 2 台の時刻のずれで奥行きが大きく振れ、右上腕の長さが
# 19% の組で 0.12〜0.5 m の外（最大 4,734 m）に出て、手首のトルクが 100 万 N·m に達した。Mac は手首が
# 26〜45% の時間で画面の外だった。いずれも置き方の問題なので、試し計測の check で見つけられるようにする。

QUALITY = {
    "segments": "3D: 肩幅・上腕・前腕の長さが妥当な範囲に入る割合 95% 以上",
    "forearm": "3D: 前腕の長さのばらつき（標準偏差）1.5 cm 未満",
    "angle": "配置: 肘での 2 本の視線のなす角 15° 以上",
    "inside": "配置: 肘・手首が両カメラの画面内にある割合 95% 以上",
}


class TestPlacementAndQuality:
    def test_a_wide_placement_passes(self, tmp_path):
        report = vr.check_run(make_body_run(tmp_path, stereo=wide_stereo(), pixel_hz=30.0))
        for name in QUALITY.values():
            assert _check(report, name)["ok"], _check(report, name)
        assert report["quality"]["angle_deg"]["R"] == pytest.approx(31.0, abs=4.0)

    def test_the_narrow_placement_of_the_real_run_is_named(self, tmp_path):
        report = vr.check_run(make_body_run(tmp_path))
        check = _check(report, QUALITY["angle"])
        assert not check["ok"] and "基線" in check["detail"]
        assert report["quality"]["baseline_cm"] == pytest.approx(35.0, abs=0.5)

    def test_broken_3d_is_named(self, tmp_path):
        """実機のように右腕の 3D が時々飛ぶと、長さの割合とばらつきの検査が落ちる。"""
        run = make_body_run(tmp_path, stereo=wide_stereo(), pixel_hz=30.0)
        kpts = next(p for p in run.glob("kpts3d_*.csv") if not p.name.startswith("kpts3d_raw_"))
        table = pd.read_csv(kpts)
        elbow = IDS.index(14)
        table.loc[::5, f"joint_{elbow}_x"] += 3.0   # 5 組に 1 組、右肘を 3 m 飛ばす
        table.to_csv(kpts, index=False)
        report = vr.check_run(run)
        check = _check(report, QUALITY["segments"])
        assert not check["ok"] and "上腕R" in check["detail"]

    def test_hands_out_of_the_frame_are_named(self, tmp_path):
        run = make_body_run(tmp_path, stereo=wide_stereo(), pixel_hz=30.0)
        marks = next(run.glob("landmarks2d_*.csv"))
        table = pd.read_csv(marks)
        mac_wrist = (table.role == "cam0") & (table.landmark == 15)
        table.loc[mac_wrist, "y"] = 1.2   # Mac で左手首が画面の下に切れている
        table.to_csv(marks, index=False)
        check = _check(vr.check_run(run), QUALITY["inside"])
        assert not check["ok"] and "Mac" in check["detail"] and "左手首" in check["detail"]
