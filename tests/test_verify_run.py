"""計測の出力を確かめる検証スクリプト（``tools/verify_run.py``）の判定と、再生の環境変数を固定する。

**なぜこのテストがあるか。**

§6-2（ゲージの値・30 fps）、§6-3（EKF の S9b の RMS 差と棄却率）、§3-2（停止で CSV が書かれるか）は、
実機でも録画の再生でも、同じ物差しで確かめたい。そこで出力フォルダを読んで確かめるスクリプトを作った。

- 合否を出すのは構造（ファイルの有無、行の対応、実機での処理間隔）だけ。値の期待範囲は S6 の実測で
  決める（設計メモ :364）ので、数値は並べて出すだけにする
- 再生は GUI と同じ設定（``entry.worker_environment``）を土台にし、再生に要るものだけを重ねる。
  とくにデモ表示の 2 つ（コードの既定は 1）を 0 にしないと、力学が回らずトルクが全部 0 になる
- カメラが開けないと別の録画へ黙って切り替わる（``AUTO_FALLBACK_TO_FILES``）ので、再生では切る
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.settings import OUTPUT_DIR_ENV, Settings
from app.core.stop_request import STOP_FILE_ENV
from app.tuning.raw_capture import RawCaptureWriter
from tools import verify_run as vr

IDS = [11, 12, 13, 14, 15, 16]
TS = "0923_200000"
N = 120
DT = 1.0 / 30.0
PARTS = ["wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L"]


def _truth(n: int) -> np.ndarray:
    t = np.arange(n) * DT
    base = np.array([[i * 0.1, 0.2, 1.0 + 0.05 * i] for i in range(len(IDS))])
    wave = 0.05 * np.sin(2 * np.pi * 0.5 * t)
    return base[None, :, :] + wave[:, None, None]


def make_run(tmp_path: Path, *, raw_offset=None, file_mode=True, t_step=DT, skip=(), log=True,
             provenance=None) -> Path:
    """計測 1 回ぶんの出力フォルダ（master_research_code.py が書くものと同じ名前・列）。"""
    out = tmp_path / "run"
    out.mkdir()
    truth = _truth(N)
    raw = truth.copy()
    if raw_offset is not None:
        lid, axis, value = raw_offset
        raw[:, IDS.index(lid), "xyz".index(axis)] += value
    meta = {"unit": "m", "dt": DT, "src_fps": 30.0, "file_mode": file_mode, "EKF_ENABLE": True,
            "EKF_Q_ACC": 0.1, "EKF_R": 1e-4, "EKF_GATE_STD": 3.0, "RT_POSE_FIXED_HZ_ON": False,
            "ekf_noise": {"origin": "env", "path": None, "reason": None, "profile_dt": None, "sources": {}}}
    meta.update(provenance or {})
    writer = RawCaptureWriter(out / f"kpts3d_raw_{TS}.csv", IDS, meta)
    for k in range(N):
        writer.append(k, k * t_step, raw[k])
    writer.note(gravity=[0.0, 0.0, -9.80665], gravity_label="Z-", gravity_set=True)
    writer.close()

    kpts = {"frame": np.arange(N)}
    for i in range(len(IDS)):
        for a, axis in enumerate("xyz"):
            kpts[f"joint_{i}_{axis}"] = np.round(truth[:, i, a], 4)
    if "kpts" not in skip:
        pd.DataFrame(kpts).to_csv(out / f"kpts3d_{TS}_gZ-.csv", index=False)

    n_torque = N - 29
    torque = {"frame": np.arange(n_torque)}
    for part in PARTS:
        for axis in "xyz":
            torque[f"{part}_{axis}"] = np.zeros(n_torque)
    torque["wrist_R_y"] = np.full(n_torque, 12.0)
    torque["elbow_R_y"] = np.linspace(-30.0, 30.0, n_torque)
    if "torque" not in skip:
        pd.DataFrame(torque).to_csv(out / f"aim_torque_vec_{TS}_s2_gZ-.csv", index=False, encoding="utf-8-sig")

    cycles = np.repeat([0, 1, 2], [30, 31, 30])
    gauge = pd.DataFrame({"frame": np.arange(n_torque), "cam_frame": np.arange(n_torque), "t": np.arange(n_torque) * DT,
                          "cycle_index": cycles})
    for part, peak in (("wrist_R", [5.0, 50.0, 150.0]), ("elbow_R", [1.0, 2.0, 3.0]),
                       ("wrist_L", [0.0, 0.0, 0.0]), ("elbow_L", [0.0, 0.0, 0.0])):
        gauge[part] = np.concatenate([np.linspace(0, p, c) for p, c in zip(peak, (30, 31, 30))])
    if "gauge" not in skip:
        gauge.to_csv(out / f"gauge_energy_{TS}_s2_gZ-.csv", index=False, encoding="utf-8-sig")
        thresholds = {p: [40.0, 100.0] for p in ("wrist_R", "elbow_R", "wrist_L", "elbow_L")}
        (out / f"gauge_energy_{TS}_s2_gZ-.json").write_text(json.dumps({"thresholds_auto": thresholds}),
                                                            encoding="utf-8")
    if log:
        (out / "run.log").write_text(
            "[DT] dt=0.03333s (1frame / 30.000fps)\n"
            "[LOOP] #5 dt=0.040s fps=25.0\n[LOOP] #10 dt=0.050s fps=20.0\n"
            "[STOP] 停止要求を受けました。ループを抜けて CSV を書き出します\n"
            "✅ aim_torque（ベクトル形式）を保存しました: x\n", encoding="utf-8")
    return out


def _check(report, name):
    matches = [c for c in report["checks"] if c["name"] == name]
    assert len(matches) == 1, [c["name"] for c in report["checks"]]
    return matches[0]


class TestStructure:
    def test_a_complete_run_passes(self, tmp_path):
        report = vr.check_run(make_run(tmp_path), log=tmp_path / "run" / "run.log")
        failed = [c for c in report["checks"] if not c["ok"]]
        assert not failed, failed
        assert report["timestamp"] == TS

    @pytest.mark.parametrize("missing, name", [("gauge", "gauge_energy"), ("torque", "aim_torque"),
                                               ("kpts", "kpts3d")])
    def test_a_missing_file_is_named(self, tmp_path, missing, name):
        report = vr.check_run(make_run(tmp_path, skip=(missing,)))
        assert not _check(report, f"ファイル: {name}")["ok"]

    def test_the_stop_is_seen_in_the_log(self, tmp_path):
        out = make_run(tmp_path)
        (out / "run.log").write_text("✅ aim_torque（ベクトル形式）を保存しました: x\n", encoding="utf-8")
        report = vr.check_run(out, log=out / "run.log")
        assert not _check(report, "ログ: 停止要求を受けた")["ok"]
        assert _check(report, "ログ: 終了時の CSV を書いた")["ok"]

    def test_a_max_frames_exit_is_not_a_stop_request(self, tmp_path):
        """MAX_FRAMES で抜けたときも本体は "[STOP] Reached MAX_FRAMES" と出す。停止要求（§3-2）と取り違えない。"""
        out = make_run(tmp_path)
        (out / "run.log").write_text("[STOP] Reached MAX_FRAMES=300 -> exiting loop\n✅ aim_torque x\n", encoding="utf-8")
        assert not _check(vr.check_run(out, log=out / "run.log"), "ログ: 停止要求を受けた")["ok"]

    def test_expecting_a_stop_without_a_log_is_not_a_pass(self, tmp_path):
        """ログが無いと停止要求を確かめられない。黙って項目を省くと、GUI の停止を確かめたつもりになる。"""
        assert not _check(vr.check_run(make_run(tmp_path, log=False), expect_stop=True), "ログ: 停止要求を受けた")["ok"]

    def test_a_fallback_to_another_recording_fails(self, tmp_path):
        """入力が開けないと本体は別の録画（リポジトリ直下の cam*_output_*）へ黙って切り替える。"""
        out = make_run(tmp_path)
        (out / "run.log").write_text("❌ 入力の読み込みに失敗しました。解決理由: config: file paths\n"
                                     "→ Try pair[1]: cam0_output_0907.mp4 , cam1_output_0907.mp4\n"
                                     "[STOP] 停止要求を受けました。\n✅ aim_torque x\n", encoding="utf-8")
        assert not _check(vr.check_run(out, log=out / "run.log"), "ログ: 指定した入力を読んだ")["ok"]

    def test_raw_and_filtered_rows_match(self, tmp_path):
        out = make_run(tmp_path)
        kpts = next(out.glob("kpts3d_0923*_gZ-.csv"))
        pd.read_csv(kpts).iloc[:50].to_csv(kpts, index=False)
        assert not _check(vr.check_run(out), "行: 生 CSV と kpts3d が 1 行ずつ対応")["ok"]


class TestTorqueAndGauge:
    def test_torque_magnitudes(self, tmp_path):
        torque = vr.check_run(make_run(tmp_path))["torque"]
        assert torque["wrist_R"]["median_abs"] == pytest.approx(12.0)
        assert torque["elbow_R"]["max_abs"] == pytest.approx(30.0)

    def test_gauge_per_cycle_against_the_band(self, tmp_path):
        gauge = vr.check_run(make_run(tmp_path))["gauge"]
        assert gauge["wrist_R"]["cycle_peaks"] == pytest.approx([5.0, 50.0, 150.0])
        assert (gauge["wrist_R"]["reached_low"], gauge["wrist_R"]["reached_high"]) == (2, 1)
        assert gauge["wrist_R"]["band"] == [40.0, 100.0]

    def test_gravity_comes_from_the_sidecar(self, tmp_path):
        assert vr.check_run(make_run(tmp_path))["gravity"]["label"] == "Z-"


class TestRate:
    def test_file_input_only_reports_the_processing_rate(self, tmp_path):
        report = vr.check_run(make_run(tmp_path, file_mode=True, t_step=0.05), log=tmp_path / "run" / "run.log")
        assert report["fps"]["processed_fps"] == pytest.approx(20.0)
        assert report["fps"]["loop_dt_mean"] == pytest.approx(0.045)
        assert _check(report, "処理間隔: dt と実際の間隔が 20% 以内")["ok"], "ファイル入力は遅くても dt が正しい"

    def test_a_live_run_that_cannot_keep_up_fails(self, tmp_path):
        report = vr.check_run(make_run(tmp_path, file_mode=False, t_step=0.05))
        assert not _check(report, "処理間隔: dt と実際の間隔が 20% 以内")["ok"]


class TestEkf:
    def test_rms_between_raw_and_filtered(self, tmp_path):
        ekf = vr.check_run(make_run(tmp_path, raw_offset=(13, "y", 0.004)))["ekf"]
        rms = {(row["landmark"], row["axis"]): row["rms_mm"] for row in ekf["series"]}
        assert rms[(13, "y")] == pytest.approx(4.0, abs=0.06)
        assert rms[(11, "x")] == pytest.approx(0.0, abs=0.06)

    def test_rejection_rate_uses_the_profile(self, tmp_path):
        profile = tmp_path / "ekf_profile.json"
        entry = {"q_acc": 0.1, "r": 1e-4, "gate_std": 1e9}
        series = {str(lid): {axis: dict(entry) for axis in "xyz"} for lid in IDS}
        series["13"]["y"]["gate_std"] = 1e-12
        profile.write_text(json.dumps({"schema_version": 1, "frame": "runtime", "unit": "m", "series": series}),
                           encoding="utf-8")
        out = make_run(tmp_path, provenance={"ekf_noise": {"origin": "profile", "path": str(profile)}})
        ekf = vr.check_run(out)["ekf"]
        assert ekf["noise_origin"] == "profile"
        rate = {(row["landmark"], row["axis"]): row["rejection_rate"] for row in ekf["series"]}
        assert rate[(13, "y")] == pytest.approx(1.0)
        assert rate[(11, "x")] == 0.0

    def test_the_body_scale_ratio_is_applied_like_the_runtime(self, tmp_path):
        """実行時はプロファイルの q・r に体格比の 2 乗を掛ける（ekf_profile.SeriesEntry.scaled）。"""
        profile = tmp_path / "ekf_profile.json"
        series = {str(lid): {axis: {"q_acc": 0.1, "r": 1e-4, "gate_std": 3.0} for axis in "xyz"} for lid in IDS}
        profile.write_text(json.dumps({"schema_version": 1, "frame": "runtime", "unit": "m", "series": series}),
                           encoding="utf-8")
        origin, lookup = vr._noise_params({"ekf_noise": {"origin": "profile", "path": str(profile)},
                                           "ekf_scale_ratio": 2.0}, DT)
        assert origin == "profile"
        assert lookup(11, "x") == pytest.approx((0.4, 4e-4, 3.0))

    def test_a_relative_profile_path_is_found_next_to_the_run(self, tmp_path):
        """サイドカーの path は本体の作業フォルダ（再生なら出力フォルダ）からの相対のことがある。"""
        out = make_run(tmp_path, provenance={"ekf_noise": {"origin": "profile", "path": "ekf_profile.json"}})
        series = {str(lid): {axis: {"q_acc": 0.1, "r": 1e-4, "gate_std": 1e9} for axis in "xyz"} for lid in IDS}
        (out / "ekf_profile.json").write_text(json.dumps({"schema_version": 1, "frame": "runtime", "unit": "m",
                                                          "series": series}), encoding="utf-8")
        assert vr.check_run(out)["ekf"]["noise_origin"] == "profile"

    def test_a_missing_profile_does_not_stop_the_check(self, tmp_path):
        out = make_run(tmp_path, provenance={"ekf_noise": {"origin": "profile", "path": "/nowhere/ekf_profile.json"}})
        ekf = vr.check_run(out)["ekf"]
        assert all(row["rejection_rate"] is None for row in ekf["series"])
        assert "見つからない" in ekf["note"]

    def test_no_gate_means_no_rejection_rate(self, tmp_path):
        """EKF_GATE_STD <= 0 なら実行時は門を使わない（extended_kalman_filter）。100% と出さない。"""
        ekf = vr.check_run(make_run(tmp_path, provenance={"EKF_GATE_STD": 0.0}))["ekf"]
        assert all(row["rejection_rate"] is None for row in ekf["series"])

    def test_the_profile_is_required_when_asked(self, tmp_path):
        report = vr.check_run(make_run(tmp_path), expect_profile=True)
        assert not _check(report, "EKF: 較正プロファイルを使った")["ok"]


class TestReplayEnvironment:
    def _env(self, tmp_path, **kwargs):
        base = {"DEMO_MONO_GAUGE_ON": "1", "DEMO_MONO_CAM0_ONLY": "1", "DT_SEC": "0.3", "HEADLESS": "0",
                OUTPUT_DIR_ENV: "/gui/place"}
        args = dict(cam0=tmp_path / "c0.avi", cam1=tmp_path / "c1.avi", calib=tmp_path, out_dir=tmp_path / "out",
                    fixed_hz=False, subject="7", timestamp=TS, stop_file=tmp_path / "stop")
        args.update(kwargs)
        return vr.replay_environment(base, **args)

    def test_the_demo_switches_are_off_and_the_output_goes_under_the_run(self, tmp_path):
        env = self._env(tmp_path)
        assert env["DEMO_MONO_GAUGE_ON"] == env["DEMO_MONO_CAM0_ONLY"] == "0"
        assert env["HEADLESS"] == "1"
        assert env[OUTPUT_DIR_ENV] == str(tmp_path / "out")
        assert env["CAM0"] == str(tmp_path / "c0.avi") and env["CALIB_BASE_DIR"] == str(tmp_path)
        assert env["AUTO_FALLBACK_TO_FILES"] == env["USE_SAMPLE_VIDEOS"] == "0"
        assert env[STOP_FILE_ENV] == str(tmp_path / "stop")
        assert env["RT_POSE_FIXED_HZ_ON"] == "0" and env["TIMESTAMP_OVERRIDE"] == TS

    def test_a_stale_dt_override_is_removed(self, tmp_path):
        assert "DT_SEC" not in self._env(tmp_path)
        assert self._env(tmp_path, dt_sec=0.25)["DT_SEC"] == "0.25"

    def test_the_gui_settings_are_the_base(self, tmp_path):
        base = vr.gui_environment(Settings({"EKF_Q_ACC": "0.5"}), stop_file=tmp_path / "stop")
        assert base["EKF_Q_ACC"] == "0.5"
        assert base["DEMO_MONO_GAUGE_ON"] == "0", "GUI の既定（デモ表示を切る）が土台になっていない"

    def test_the_child_writes_utf8(self, tmp_path):
        """Windows ではパイプの既定が cp932 で、本体の "✅" の print で落ちる。"""
        env = self._env(tmp_path)
        assert env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONUTF8"] == "1"

    def test_a_relative_profile_is_made_absolute(self, tmp_path, monkeypatch):
        """本体は出力フォルダで動くので、相対パスのままだとプロファイルが見つからず同梱既定値で走る。"""
        monkeypatch.chdir(tmp_path)
        assert self._env(tmp_path, ekf_profile="prof/ekf_profile_0.03333.json")["EKF_PROFILE"] == \
            str(tmp_path / "prof" / "ekf_profile_0.03333.json")
        base = {"EKF_PROFILE": "prof"}
        env = vr.replay_environment(base, cam0="a", cam1="b", calib=".", out_dir=tmp_path / "o", fixed_hz=False,
                                    subject="7", timestamp=TS, stop_file=tmp_path / "s")
        assert env["EKF_PROFILE"] == str(tmp_path / "prof"), "GUI の設定から引き継いだ相対パスも絶対にする"

    def test_an_empty_profile_clears_the_gui_setting(self, tmp_path):
        env = vr.replay_environment({"EKF_PROFILE": "/gui/profile"}, cam0="a", cam1="b", calib=".",
                                    out_dir=tmp_path, fixed_hz=False, subject="7", timestamp=TS,
                                    stop_file=tmp_path / "s", ekf_profile="")
        assert env["EKF_PROFILE"] == ""


class TestReplayRefusesBadInput:
    def test_a_missing_video_is_refused_before_starting(self, tmp_path, monkeypatch):
        """開けない入力のまま起動すると、本体は別の録画へ黙って切り替える。"""
        def must_not_start(*args, **kwargs):
            raise AssertionError("入力が無いのに計測を起動した")

        monkeypatch.setattr(vr.subprocess, "Popen", must_not_start)
        code = vr.main(["replay", "--cam0", str(tmp_path / "missing0.mp4"), "--cam1", str(tmp_path / "missing1.mp4"),
                        "--calib", str(tmp_path), "--out", str(tmp_path / "out"), "--subject", "7"])
        assert code == 2


class TestDtOverride:
    def test_no_override_when_the_camera_kept_its_rate(self):
        assert vr.dt_override({"container_fps": 30.0, "measured_fps": 29.9}, fixed_hz=False) is None

    def test_the_measured_rate_sets_dt(self):
        assert vr.dt_override({"container_fps": 30.0, "measured_fps": 25.0}, fixed_hz=False) == pytest.approx(0.04)

    def test_the_stride_of_the_fixed_rate_is_kept(self):
        # 容器が 30 fps なら 4 Hz 間引きは 8 フレームに 1 回。実際は 25 fps なので 8/25 秒
        assert vr.dt_override({"container_fps": 30.0, "measured_fps": 25.0}, fixed_hz=True) == pytest.approx(0.32)

    def test_the_gui_rate_settings_are_used(self):
        """本体は RT_POSE_FIXED_HZ・SKIP_FRAMES（GUI の設定）で間引く。4 Hz・間引きなしを決め打ちしない。"""
        meta = {"container_fps": 30.0, "measured_fps": 25.0}
        assert vr.dt_override(meta, fixed_hz=True, env={"RT_POSE_FIXED_HZ": "5"}) == pytest.approx(6 / 25)
        assert vr.dt_override(meta, fixed_hz=False, env={"SKIP_FRAMES": "2"}) == pytest.approx(2 / 25)
