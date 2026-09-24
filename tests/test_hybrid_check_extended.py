"""``verify_run check`` が混成の新しい版の記録（EKF・1RM・ゲージの帯・重力の出どころ）を確かめることを固定する（B5）。

**なぜこのテストがあるか。**

混成の計測は EKF・被験者の 1RM・論文の閾値 W_0.70〜W_0.85 のゲージ・重力の選択肢を持つようになった
（``app.runners.network_measure``）。朝の実機の確認は ``check`` の報告で行うので、そこに EKF の前後の差と
棄却率（S9b の材料）、回ごとの W_pos・帯への到達回数・スコア、被験者・1RM・重力の出どころ・盤の傾きが
出ないと、ゲージが論文の閾値で動いたかを記録から確かめられない。

- EKF の手前の生 3D（``kpts3d_raw_<stamp>.csv``）は 1/30 s の格子で、抜けた格子は NaN の行。EKF の後の
  ``kpts3d_<stamp>.csv`` は届いた組だけの行。**行の番号ではなく格子の番号（frames の ``grid_index`` と生 CSV の
  ``frame``）で合わせる**（``grid_index`` の無い古い記録は時刻を格子に丸めて（round(t/dt)））。
  行で合わせると、抜けの後ろがすべて 1 格子以上ずれ、EKF の差が数 cm に化ける
- 新しい版の記録（``output_schema_version`` がある）のときだけ検査を足す。古い記録と既存のテストは合格のまま
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.tuning.raw_capture import RawCaptureWriter
from test_hybrid_verification import IDS, N, _structural_failures, make_hybrid_run
from tools import verify_run as vr

OFFSET_M = 0.002   # EKF の後の 3D を生の 3D からずらす量（RMS の期待値 2 mm）
GAP = range(50, 56)  # 組が届かなかった格子（kpts3d・frames に行が無く、生 CSV は NaN の行）
BANDS = {"elbow_L": [47.5, 56.9], "elbow_R": [4.5, 5.5], "wrist_L": [9.6, 11.6], "wrist_R": [4.5, 6.0]}
W1RM = {"elbow_L": 66.3, "elbow_R": 10.0, "wrist_L": 13.5, "wrist_R": 8.0}


def _stamp(run):
    return next(run.glob("frames_*.csv")).stem[len("frames_"):]


def make_new_run(tmp_path, *, ekf=True, gap=True, raw=True):
    """``make_hybrid_run`` の記録を、A2 の記録器が書く新しい版の形に直す。"""
    run = make_hybrid_run(tmp_path)
    stamp = _stamp(run)
    kpts_path, frames_path = run / f"kpts3d_{stamp}.csv", run / f"frames_{stamp}.csv"
    kpts, frames = pd.read_csv(kpts_path), pd.read_csv(frames_path)
    points = kpts.drop(columns="frame").to_numpy(float).reshape(N, len(IDS), 3)
    # 生 3D（格子の全部の行。抜けは NaN）と、届いた組だけの EKF の後の 3D
    if raw:
        provenance = {"unit": "m", "frame": "runtime", "dt": 1.0 / 30.0, "source": "hybrid", "times": "grid",
                      "file_mode": False, "EKF_ENABLE": ekf, "EKF_GATE_STD": 3.0,
                      "ekf_noise": {"origin": "builtin", "path": None} if ekf else None}
        writer = RawCaptureWriter(run / f"kpts3d_raw_{stamp}.csv", IDS, provenance)
        for k in range(N):
            value = np.full((len(IDS), 3), np.nan) if (gap and k in GAP) else points[k]
            writer.append(k, k / 30.0, value)
        writer.note(ekf_scale_ratio=1.0, gravity=[0.0, 0.0, -9.81], gravity_label="Z-")
        writer.close()
    keep = [k for k in range(N) if not (gap and k in GAP)]
    shifted = points[keep] + (OFFSET_M if ekf else 0.0)
    new_kpts = pd.DataFrame(shifted.reshape(len(keep), -1), columns=kpts.columns[1:])
    new_kpts.insert(0, "frame", range(len(keep)))
    new_kpts.to_csv(kpts_path, index=False)
    new_frames = frames.iloc[keep].copy()
    new_frames["frame"] = range(len(keep))
    new_frames["grid_index"] = keep
    new_frames["dt_s"] = 1.0 / 30.0
    new_frames["dyn_active"] = 0
    new_frames["height_m"] = 0.0
    new_frames["rep"] = 0
    new_frames.to_csv(frames_path, index=False)
    # 回ごとの W+・W−・W_1RM・スコア（肘 R は 1 回目が帯の下、2 回目が過負荷）
    work_path = run / f"cycle_work_{stamp}.csv"
    rows = []
    for rep, (frame, pos) in enumerate(((40, 4.0), (78, 6.0))):
        for joint in vr.JOINTS:
            value = pos if joint == "elbow_R" else 5.0
            rows.append({"frame": frame, "t_ns": 0, "joint": joint, "work_j": value, "work_pos_j": value,
                         "work_neg_j": 0.0, "w1rm_j": W1RM[joint], "score": value / W1RM[joint]})
    pd.DataFrame(rows).to_csv(work_path, index=False)
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["frames"] = len(keep)
    meta["calibration_meta"]["checkerboard_short_axis"] = {"tilt_deg": 4.2, "up_label_runtime": "Z+",
                                                          "vector_runtime": [0.0, 0.07, 0.99], "samples": 10}
    meta.update({
        "output_schema_version": 2, "subject_id": "00", "body_mass_kg": 65.0,
        "one_rm_kg": {"elbow_L": 10.0, "elbow_R": 10.0, "wrist_L": 5.0, "wrist_R": 5.0},
        "forearm_len_m": {"L": 0.25, "R": 0.25}, "w1rm_j": W1RM, "gauge_bands_j": BANDS,
        "gravity": {"source": "checkerboard", "label": "Z-", "vector": [0.0, 0.0, -9.81], "up_label": "Z+",
                    "detail": "盤の短辺を Z+ に吸着"},
        "ekf": {"enabled": ekf, "origin": "builtin" if ekf else None, "path": None, "scale_ratio": 1.0},
        "dyn_gate": True, "mac_camera": {"measured_fps": 29.8},
        "timing": {"frames": len(keep), "median_ms": 4.0, "p95_ms": 9.0, "max_ms": 30.0},
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return run


def test_an_old_record_still_passes_and_says_why_there_is_no_ekf(tmp_path):
    report = vr.check_run(make_hybrid_run(tmp_path))
    assert not _structural_failures(report)
    assert "kpts3d_raw" not in " ".join(c["name"] for c in report["checks"])
    assert "EKF" in report["ekf"]["note"] and "古い" in report["ekf"]["note"]


def test_the_new_record_passes(tmp_path):
    report = vr.check_run(make_new_run(tmp_path))
    assert not _structural_failures(report), _structural_failures(report)
    assert any(c["name"] == "ファイル: kpts3d_raw" and c["ok"] for c in report["checks"])


def test_the_ekf_difference_is_aligned_on_the_grid(tmp_path):
    """組が抜けた後ろも同じ格子どうしで比べる。行で合わせると抜けの後ろが 6 格子ずれる。"""
    ekf = vr.check_run(make_new_run(tmp_path))["ekf"]
    assert ekf["noise_origin"] == "builtin" and ekf["ekf_enabled"]
    assert ekf["matched_rows"] == N - len(GAP) == ekf["kpts_rows"]
    rms = [row["rms_mm"] for row in ekf["series"]]
    assert rms == pytest.approx([OFFSET_M * 1000.0] * len(rms), rel=1e-6)
    assert all(row["rejection_rate"] is not None for row in ekf["series"])


def test_a_disabled_ekf_does_not_break_the_check(tmp_path):
    ekf = vr.check_run(make_new_run(tmp_path, ekf=False))["ekf"]
    assert not ekf["ekf_enabled"]
    assert all(row["rejection_rate"] is None for row in ekf["series"])
    assert [row["rms_mm"] for row in ekf["series"]] == pytest.approx([0.0] * len(ekf["series"]), abs=1e-9)


def test_a_missing_raw_capture_fails_only_the_new_record(tmp_path):
    report = vr.check_run(make_new_run(tmp_path, raw=False))
    failed = [c["name"] for c in _structural_failures(report)]
    assert failed == ["ファイル: kpts3d_raw"]


def test_bands_reps_and_scores(tmp_path):
    gauge = vr.check_run(make_new_run(tmp_path))["gauge"]
    elbow = gauge["elbow_R"]
    assert elbow["work_pos"] == pytest.approx([4.0, 6.0])
    assert elbow["band"] == pytest.approx(BANDS["elbow_R"])
    assert (elbow["reached_low"], elbow["reached_high"]) == (1, 1)
    assert elbow["scores"] == pytest.approx([0.4, 0.6])
    assert elbow["w1rm"] == pytest.approx(10.0)


def test_subject_one_rm_gravity_and_board_tilt(tmp_path):
    report = vr.check_run(make_new_run(tmp_path))
    subject = report["subject"]
    assert subject["subject_id"] == "00" and subject["body_mass_kg"] == 65.0
    assert subject["one_rm_kg"]["elbow_L"] == 10.0
    assert report["gravity"]["source"] == "checkerboard"
    assert report["gravity"]["board_tilt_deg"] == pytest.approx(4.2)
    assert report["timing"]["p95_ms"] == 9.0


def test_the_text_report_shows_the_new_sections(tmp_path):
    text = vr.format_report(vr.check_run(make_new_run(tmp_path)))
    for word in ("被験者 00", "W_pos", "帯", "RMS", "棄却率", "checkerboard", "盤の傾き 4.2", "処理時間"):
        assert word in text, word


def test_an_unfinished_rep_is_not_counted_as_a_rep():
    """止めたときに開いたままの回（``status=unfinished``）は、回の数・仕事・帯への到達に数えない。
    ``status`` 列の無い古い記録は全行を閉じた回として扱う（2026-09-24 の全体レビュー）。"""
    work = pd.DataFrame({"joint": ["elbow_R", "elbow_R"], "work_j": [20.0, 230.0],
                         "status": ["closed", "unfinished"]})
    assert list(vr._closed_reps(work)["work_j"]) == [20.0]
    old = work.drop(columns="status")
    assert len(vr._closed_reps(old)) == 2
