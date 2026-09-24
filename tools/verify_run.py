"""計測の出力を確かめる（§6-2・§6-3・§3-2）。録画を計測に読み込ませる再生もここから行う。

- ``check``: 出力フォルダ（``OUTPUT_DIR``、既定は ``output_data``）を読んで確かめる。合否を出すのは構造
  （ファイルの有無、行の対応、実機での処理間隔、肩・肘・手首の 3D がそろった行の割合）だけ。値（トルク、ゲージ、
  骨の長さ、EKF の RMS 差・棄却率）は並べて出す。
  期待範囲は S6 の実測で決める（``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` :364）
- ``replay``: 録画（``tools/record_stereo.py`` のフォルダか、受け取った ``cameras_raw/<試技>/``）を計測
  （``python -m app --role realtime``、GUI と同じ起動）に読み込ませ、終わったら ``check`` にかける。
  設定は GUI の設定（``entry.worker_environment``）を土台にし、再生に要るものだけを重ねる

- ``hybrid-raw``: 混成ステレオ（Mac＋同じ Wi-Fi の Pixel 7a、``app.runners.hybrid_measure``）の 3D を生 CSV の形に直す。
  S6 の雑音の推定（``ekf_estimate`` / ``tune_ekf``）にかけるため。``check`` は混成の計測フォルダも確かめる

使い方（リポジトリ直下で）::

    python -m tools.verify_run replay --session recordings/S07_0923_213245 --subject 7 --fixed-hz 0
    python -m tools.verify_run replay --cam0 A.mp4 --cam1 B.mp4 --calib DIR --out OUT --subject 7 --fixed-hz 1
    python -m tools.verify_run replay --session ... --subject 7 --stop-after-sec 60   # §3-2 を GUI なしで
    python -m tools.verify_run check ~/Documents/WheelchairTorque/output_data --log run.log
    python -m tools.verify_run check ~/Documents/WheelchairTorque/hybrid/measure --expect-stop   # 混成の最新の回
    python -m tools.verify_run hybrid-raw ~/Documents/WheelchairTorque/hybrid/measure --stride 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

# pylint: disable=no-member
import cv2 as cv
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # python tools/verify_run.py でも動くように
    sys.path.insert(0, str(REPO_ROOT))

from app import entry  # noqa: E402
from app.core.settings import OUTPUT_DIR_ENV, Settings  # noqa: E402
from app.core.stop_request import STOP_FILE_ENV  # noqa: E402
from app.hybrid.retriangulate import retriangulate  # noqa: E402
from app.tuning.ekf_likelihood import innovation_loglik  # noqa: E402
from app.tuning.ekf_profile import builtin_entry, resolve_profile  # noqa: E402
from app.tuning.raw_capture import (  # noqa: E402
    HYBRID_RETRI_SOURCE, RawCaptureWriter, read_raw_capture, sidecar_path)
from tools.parse_fps_stats import parse_file  # noqa: E402

JOINTS = ("wrist_R", "elbow_R", "wrist_L", "elbow_L")
CALIB_FILES = ("c0.dat", "c1.dat", "rot_trans_c0.dat", "rot_trans_c1.dat")
AXES = "xyz"
EXPECTED_TORQUE = "手首・肘 10〜40 N·m 台の見込み"
# 実機（ライブ）で、処理の間隔が dt からこれ以上ずれたら不合格（本体の [DT][警告] と同じ 20%）
DT_TOLERANCE = 0.2
# 録画の実測 fps が容器の fps からこれ以上ずれたら、再生で DT_SEC を渡す（record_stereo と同じ）
FPS_TOLERANCE = 0.05
FIXED_HZ_DEFAULT = 4.0
# 本体のログの文言（master_research_code.py）
INPUT_FAILED = "入力の読み込みに失敗"
STOP_REQUESTED = "[STOP] 停止要求を受けました"
# 再生のログのうち画面にも出す行
ECHO_TAGS = ("Input streams resolved", INPUT_FAILED, "Try pair", "Try media", "[CALIB]", "[DT]", "[RAW]",
             "[INERTIA]", "[GRAVITY]", "[EKF]", "[STOP]", "✅", "[WARN]", "[ERROR]", "Traceback", "Error", "detected")
LOOP_ECHO_EVERY = 60


# --------------------------------------------------------------------------- check


def _latest_timestamp(out_dir: Path) -> str | None:
    raws = sorted(out_dir.glob("kpts3d_raw_*.csv"), key=lambda p: p.stat().st_mtime)
    return raws[-1].stem[len("kpts3d_raw_"):] if raws else None


def _one(out_dir: Path, pattern: str) -> Path | None:
    found = sorted(out_dir.glob(pattern))
    return found[-1] if found else None


def _torque_stats(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    stats = {}
    for joint in JOINTS:
        column = f"{joint}_y"
        values = np.abs(df[column].to_numpy(float)) if column in df else np.array([])
        values = values[np.isfinite(values)]
        stats[joint] = {
            "n": int(values.size),
            "median_abs": float(np.median(values)) if values.size else None,
            "p95_abs": float(np.percentile(values, 95)) if values.size else None,
            "max_abs": float(values.max()) if values.size else None,
        }
    return stats


def _gauge_stats(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    bands = meta.get("thresholds_gauge") or meta.get("thresholds_auto") or {}
    stats = {}
    for joint in JOINTS:
        if joint not in df:
            continue
        peaks = [float(v) for v in df.groupby("cycle_index")[joint].max().to_numpy()]
        band = bands.get(joint)
        stats[joint] = {
            "cycle_peaks": peaks,
            "band": [float(b) for b in band] if band else None,
            "reached_low": sum(p >= band[0] for p in peaks) if band else None,
            "reached_high": sum(p >= band[1] for p in peaks) if band else None,
        }
    return stats


def _profile_path(path: str, base_dir: Path | None) -> Path | None:
    """サイドカーの ``ekf_noise.path``。相対なら、今の作業フォルダ、次に本体の出力フォルダから探す。"""
    candidates = [Path(path)]
    if base_dir is not None and not Path(path).is_absolute():
        candidates.append(base_dir / path)
    return next((c for c in candidates if c.exists()), None)


def _noise_params(provenance: Mapping[str, Any], dt: float, base_dir: Path | None = None):
    """実行時に EKF が使った雑音パラメータの出どころと、系列 (ID, 軸) → (q, r, gate) を返す関数。

    プロファイルが見つからなければ関数の代わりに None を返す。プロファイル由来の値は実行時と同じ解決で求める:
    dt で選び（``ekf_profile.resolve_profile``）、q・r に体格の比（サイドカーの ``ekf_scale_ratio``）の 2 乗を掛け
    （``SeriesEntry.scaled``）、プロファイルに無い系列は全体の中央値で埋める（``ProfileResolution.series_noise``）。
    実行時が選べないプロファイル（dt が合わない・壊れている）なら ValueError。
    """
    noise = provenance.get("ekf_noise") or {}
    origin = noise.get("origin") or "env"
    if origin == "profile" and noise.get("path"):
        found = _profile_path(noise["path"], base_dir)
        if found is None:
            return origin, None
        try:
            resolution = resolve_profile(found, dt=dt, scale_ratio=float(provenance.get("ekf_scale_ratio") or 1.0))
        except (KeyError, TypeError) as error:
            raise ValueError(f"{found.name} を較正プロファイルとして読めない（{error!r}）") from error
        if resolution.reason is not None:
            raise ValueError(f"{found.name} は dt {dt:.5f} s の実行時に選ばれない（{resolution.reason}）")

        def lookup(lid, axis):
            chosen = resolution.series_noise([int(lid)])
            k = AXES.index(axis)
            return float(chosen.q_acc[k]), float(chosen.r[k]), float(chosen.gate_std[k])
        return origin, lookup
    if origin == "builtin":
        builtin = builtin_entry(dt)
        return origin, lambda lid, axis: (builtin.q_acc, builtin.r, builtin.gate_std)
    scalar = (float(provenance.get("EKF_Q_ACC")), float(provenance.get("EKF_R")), float(provenance.get("EKF_GATE_STD")))
    return "env", lambda lid, axis: scalar


def _noise_lookup(provenance: Mapping[str, Any], dt: float, base_dir: Path | None) -> tuple[str, Any, str]:
    """``_noise_params`` に、較正プロファイルが見つからない・使えないときの注記を添える。"""
    try:
        origin, lookup = _noise_params(provenance, dt, base_dir)
    except ValueError as error:
        origin = (provenance.get("ekf_noise") or {}).get("origin") or "env"
        return origin, None, f"較正プロファイルを使えないので棄却率は出さない: {error}"
    note = ""
    if lookup is None:
        note = f"較正プロファイル {(provenance.get('ekf_noise') or {}).get('path')} が見つからないので棄却率は出さない"
    return origin, lookup, note


def _ekf_stats(capture, kpts_path: Path, base_dir: Path | None = None) -> dict[str, Any]:
    """系列ごとの RMS（EKF の前と後の差）[mm] と、実行時の雑音パラメータで数えた棄却率（S9b の材料）。

    棄却率は、生の系列を素の KF（``innovation_loglik``）に通した正規化イノベーションが門（gate_std）の
    外に出た割合。実行時の EKF はロバスト更新で状態が変わるので近似である。門が 0 以下なら実行時は門を
    使わないので、棄却率は出さない。USB の生 CSV と kpts3d は 1 行ずつ対応するので、行の番号で合わせる。
    """
    kpts = pd.read_csv(kpts_path)
    n = min(capture.points.shape[0], len(kpts))
    dt = float(capture.provenance["dt"])
    origin, lookup, note = _noise_lookup(capture.provenance, dt, base_dir)
    return _ekf_series(capture, capture.points[:n], kpts.iloc[:n].reset_index(drop=True), origin, lookup, note)


def _ekf_series(capture, raw_rows: np.ndarray, kpts: pd.DataFrame, origin: str, lookup, note: str) -> dict[str, Any]:
    """合わせ済みの行（``raw_rows`` と ``kpts`` の同じ番号が同じ時刻）で、系列ごとの RMS と棄却率を出す。

    棄却率は生の系列の全体（抜けた格子の NaN を含む、dt 一定）で数える。
    """
    dt = float(capture.provenance["dt"])
    rows = []
    for i, lid in enumerate(capture.landmark_ids):
        for a, axis in enumerate(AXES):
            raw = capture.points[:, i, a]
            column = f"joint_{i}_{axis}"
            diff = raw_rows[:, i, a] - (kpts[column].to_numpy(float) if column in kpts else np.nan)
            finite = np.isfinite(diff)
            rate, gate = None, None
            if lookup is not None:
                q_acc, r, gate = lookup(lid, axis)
                if gate > 0:
                    normalized = innovation_loglik(raw, dt, q_acc, r).normalized
                    usable = normalized[np.isfinite(normalized)]
                    rate = float(np.mean(np.abs(usable) > gate)) if usable.size else None
            rows.append({
                "landmark": int(lid),
                "axis": axis,
                "rms_mm": float(np.sqrt(np.mean(diff[finite] ** 2)) * 1000.0) if finite.any() else None,
                "rejection_rate": rate,
                "n": int(finite.sum()),
                "gate_std": gate,
            })
    return {"noise_origin": origin, "ekf_enabled": bool(capture.provenance.get("EKF_ENABLE", True)),
            "note": note, "series": rows}


def _abs_stats(values) -> dict[str, Any]:
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "median_abs": float(np.median(values)) if values.size else None,
        "p95_abs": float(np.percentile(values, 95)) if values.size else None,
        "max_abs": float(values.max()) if values.size else None,
    }


# --------------------------------------------------------------------------- 混成ステレオ（Mac＋Pixel 7a）

# app.hybrid.recorder.Recorder が書くファイル（<名前>_<stamp>.csv）
HYBRID_FILES = ("kpts3d", "frames", "landmarks2d", "local_torque", "cycle_work")
# カメラごとの速さの下限（30 fps の 8 割）。これを下回ると「30 fps を保てていない」（§6-2）
MIN_CAMERA_FPS = 24.0
ROLE_NAMES = {"cam0": "Mac", "cam1": "Pixel"}
HYBRID_EKF_NOTE = ("この記録は EKF の手前の生 3D（kpts3d_raw_<stamp>.csv）が無い古い版なので、EKF の前後の差と"
                   "棄却率（S9b）は出さない。S6 の雑音の推定は `python -m tools.verify_run hybrid-raw <計測フォルダ>` で"
                   "生 CSV に直して ekf_estimate / tune_ekf にかける")
# 新しい版の記録（app.hybrid.recorder が EKF・1RM・ゲージの帯とともに書く）の目印・ファイル・列。
# meta に output_schema_version があるときだけ、下の検査を足す（古い記録は合格のまま）
HYBRID_SCHEMA_KEY = "output_schema_version"
HYBRID_RAW_PREFIX = "kpts3d_raw"
# 在りかだけ報告する（合否にしない）ファイル: <名前>_<stamp>*.csv
HYBRID_EXTRA_FILES = ("cycle_energy", "gauge_energy", "aim_torque_vec")
HYBRID_WORK_COLUMNS = ("work_pos_j", "work_neg_j", "w1rm_j", "score")
HYBRID_SUBJECT_KEYS = ("subject_id", "body_mass_kg", "one_rm_kg", "forearm_len_m", "w1rm_j")
# EKF の後の 3D の行のうち、生 CSV の同じ格子が見つかる割合の下限
MIN_GRID_MATCH = 0.95


def _is_hybrid(folder: Path) -> bool:
    meta = folder / "meta.json"
    if not meta.is_file():
        return False
    try:
        return "coordinates" in json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False


def hybrid_session(path: str | Path) -> Path | None:
    """混成の計測フォルダ（Recorder の meta.json がある）。計測フォルダの親（measure）なら最新の回。"""
    path = Path(path)
    if _is_hybrid(path):
        return path
    sessions = sorted(d for d in path.iterdir() if d.is_dir() and _is_hybrid(d)) if path.is_dir() else []
    return sessions[-1] if sessions else None


def _hybrid_files(folder: Path) -> tuple[str | None, dict[str, Path | None]]:
    """記録器のファイル（<名前>_<stamp>.csv）。stamp は frames_* から取り、名前を決め打ちする。

    ``kpts3d_*`` の glob だと、``hybrid_raw_capture`` が同じフォルダに書く ``kpts3d_raw_*`` を拾ってしまう。
    """
    frames = _one(folder, "frames_*.csv")
    stamp = frames.stem[len("frames_"):] if frames else None
    files = {name: (folder / f"{name}_{stamp}.csv") if stamp else None for name in HYBRID_FILES}
    return stamp, {name: (path if path is not None and path.is_file() else None) for name, path in files.items()}


def _recorded_raw_path(folder: Path, stamp: str | None) -> Path | None:
    """記録器が書いた EKF の手前の 3D（``kpts3d_raw_<stamp>.csv`` とサイドカー）。古い版の記録には無い（None）。"""
    if not stamp:
        return None
    path = folder / f"{HYBRID_RAW_PREFIX}_{stamp}.csv"
    return path if path.is_file() and sidecar_path(path).is_file() else None


def _intervals(t_s) -> np.ndarray:
    steps = np.diff(np.asarray(t_s, dtype=float))
    return steps[np.isfinite(steps) & (steps > 0)]


# 配置と 3D の質（本番の前の試し計測で、置き方を直すべきかを決める）。目安は mobile/README.md の「実機で残る確認」
# （前腕長の標準偏差 < 1.5 cm）と、三角測量の一般的な目安（視線のなす角 15° 以上）
SEGMENTS = {"肩幅": (11, 12, 0.25, 0.55), "上腕R": (12, 14, 0.12, 0.5), "前腕R": (14, 16, 0.12, 0.5),
            "上腕L": (11, 13, 0.12, 0.5), "前腕L": (13, 15, 0.12, 0.5)}
FRAME_POINTS = {13: "左肘", 14: "右肘", 15: "左手首", 16: "右手首"}
MIN_SEGMENT_SHARE = 0.95
MAX_FOREARM_STD_M = 0.015
MIN_RAY_ANGLE_DEG = 15.0
MIN_INSIDE_SHARE = 0.95
# USB の構造の検査: 肩・肘・手首の 3D がそろった行の割合の下限。人が画面に入るまでの数秒などで抜けるのは普通なので
# 半分とする。灰色の映像のように 3D が 1 点も取れない回（トルクが 0 のまま CSV はそろう）を落とすため
TRACKED_POINTS = (11, 12, 13, 14, 15, 16)
MIN_TRACKED_SHARE = 0.5


def _segment_lengths(points: np.ndarray, ids) -> dict[str, dict[str, Any]]:
    """部位（肩幅・上腕・前腕）ごとの長さの中央値、範囲に入る割合、範囲内のばらつき。

    ``points`` は (行, 点, 3) [m]、並びは ``ids`` の順。長さが 1 つも有限でなければ中央値と割合は None。
    """
    slot = {int(lid): i for i, lid in enumerate(ids)}
    segments = {}
    for name, (a, b, low, high) in SEGMENTS.items():
        if a in slot and b in slot:
            length = np.linalg.norm(points[:, slot[a]] - points[:, slot[b]], axis=1)
            length = length[np.isfinite(length)]
        else:
            length = np.array([])
        inside = length[(length >= low) & (length <= high)]
        segments[name] = {
            "median_m": float(np.median(length)) if length.size else None,
            "share": float(inside.size / length.size) if length.size else None,
            "std_m": float(np.std(inside)) if inside.size > 1 else None,   # 範囲内の値だけのばらつき
            "range_m": [low, high],
        }
    return segments


def _tracked_rows(points: np.ndarray, ids) -> tuple[int, int]:
    """肩・肘・手首（``TRACKED_POINTS``）の 3D がすべて有限の行の数と、全体の行の数。"""
    slot = {int(lid): i for i, lid in enumerate(ids)}
    if not all(lid in slot for lid in TRACKED_POINTS):
        return 0, int(points.shape[0])
    arm = points[:, [slot[lid] for lid in TRACKED_POINTS]]
    return int(np.isfinite(arm).all(axis=(1, 2)).sum()), int(points.shape[0])


def _camera_centres(folder: Path) -> list[np.ndarray]:
    """2 台のカメラの中心 [cm]（cam0＝Mac の座標）。計測フォルダに写した校正から。"""
    from utils import read_rotation_translation

    centres = []
    for index in (0, 1):
        R, T = read_rotation_translation(index, str(folder))
        centres.append(-np.asarray(R, dtype=float).T @ np.asarray(T, dtype=float).ravel())
    return centres


def _hybrid_quality(folder: Path, files: Mapping[str, Path | None], meta: Mapping[str, Any],
                    raw_capture=None) -> dict[str, Any] | None:
    """骨の長さ、肘での 2 本の視線のなす角、肘・手首が各カメラの画面内にある割合。

    三角測量の質を見るので、EKF の手前の 3D（``kpts3d_raw_<stamp>.csv``）があればそれを使う。新しい版の記録の
    ``kpts3d`` は EKF の後の点で、均されて骨の長さのばらつきが小さく出る（置き方の失敗を見逃す）。
    ``raw_capture`` はその生 3D を読んだもの（無ければ None）。
    """
    if files["kpts3d"] is None:
        return None
    ids = [int(i) for i in meta["pose_keypoints"]]
    slot = {lid: i for i, lid in enumerate(ids)}
    if raw_capture is not None:
        points = raw_capture.points
    else:
        table = pd.read_csv(files["kpts3d"])
        points = table.drop(columns="frame").to_numpy(float).reshape(len(table), len(ids), 3)
    quality: dict[str, Any] = {"segments": _segment_lengths(points, ids)}
    try:
        centres = _camera_centres(folder)
    except (OSError, ValueError):
        centres = None
    if centres is not None:
        quality["baseline_cm"] = float(np.linalg.norm(centres[1] - centres[0]))
        # 記録の座標 (-cx, -cz, -cy) [m] を cam0 の座標 [cm] に戻す
        camera = np.stack([-points[..., 0], -points[..., 2], -points[..., 1]], axis=-1) * 100.0
        angles = {}
        for side, lid in (("R", 14), ("L", 13)):
            v0, v1 = camera[:, slot[lid]] - centres[0], camera[:, slot[lid]] - centres[1]
            with np.errstate(invalid="ignore"):
                cos = np.sum(v0 * v1, axis=1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1))
            value = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            value = value[np.isfinite(value)]
            angles[side] = float(np.median(value)) if value.size else None
        quality["angle_deg"] = angles
    if files["landmarks2d"] is not None:
        marks = pd.read_csv(files["landmarks2d"], usecols=["role", "landmark", "x", "y"])
        marks = marks[marks["landmark"].isin(FRAME_POINTS)]
        seen = (marks["x"].between(0, 1)) & (marks["y"].between(0, 1))
        quality["inside"] = {role: {FRAME_POINTS[int(lid)]: float(share) for lid, share in group.groupby("landmark").mean().items()}
                             for role, group in seen.groupby([marks["role"], marks["landmark"]]).mean().groupby(level=0)
                             for group in [group.droplevel(0)]}
    return quality


NO_FINITE_LENGTH = "有限の長さが無い"
NO_FINITE_HINT = "（3D が取れていない。被写体が両方の画面に入っているか、校正を確かめる）"


def _quality_checks(quality: Mapping[str, Any], add) -> None:
    """骨の長さと配置の検査。長さが 1 つも有限でない部位（割合・ばらつきが None）は「有限の長さが無い」で不合格。"""
    segments = quality["segments"]
    short = [f"{name} {NO_FINITE_LENGTH}" if s["share"] is None else f"{name} {s['share']:.0%}"
             for name, s in segments.items() if s["share"] is None or s["share"] < MIN_SEGMENT_SHARE]
    missing = any(s["share"] is None for s in segments.values())
    add("3D: 肩幅・上腕・前腕の長さが妥当な範囲に入る割合 95% 以上", not short,
        "、".join(short) + (NO_FINITE_HINT if missing else ""))
    wobbly = []
    for name in ("前腕R", "前腕L"):
        s = segments[name]
        if s["share"] is None:
            wobbly.append(f"{name} {NO_FINITE_LENGTH}")
        elif s["std_m"] is None:
            wobbly.append(f"{name} 範囲内の長さが 1 個以下")
        elif s["std_m"] >= MAX_FOREARM_STD_M:
            wobbly.append(f"{name} {s['std_m'] * 100:.1f} cm")
    add("3D: 前腕の長さのばらつき（標準偏差）1.5 cm 未満", not wobbly, "、".join(wobbly))
    if "angle_deg" in quality:
        angles = quality["angle_deg"]
        narrow = [f"{'右' if side == 'R' else '左'}肘 {_fmt(v, '.1f')}°" for side, v in angles.items()
                  if v is None or v < MIN_RAY_ANGLE_DEG]
        add("配置: 肘での 2 本の視線のなす角 15° 以上", not narrow,
            ("、".join(narrow) + f"（基線 {quality['baseline_cm']:.1f} cm。被写体を近づけるか 2 台の間を広げ、"
             f"被写体を 2 台の中間の正面に置いて校正し直す。docs/hybrid_field_run.md の表）") if narrow else "")
    if "inside" in quality:
        # 片方のカメラの点が 1 つも無ければ、残った 1 台だけで「両カメラ」を合格にしない
        absent = [f"{ROLE_NAMES[role]} の点が無い" for role in ROLE_NAMES if role not in quality["inside"]]
        cut = [f"{ROLE_NAMES.get(role, role)} の{name} {share:.0%}" for role, shares in sorted(quality["inside"].items())
               for name, share in shares.items() if share < MIN_INSIDE_SHARE]
        add("配置: 肘・手首が両カメラの画面内にある割合 95% 以上", not absent and not cut, "、".join(absent + cut))


def _hybrid_ekf_stats(capture, kpts_path: Path, frames: pd.DataFrame, base_dir: Path | None = None) -> dict[str, Any]:
    """混成の EKF の前後の差と棄却率。行の番号ではなく、同じ格子どうしで合わせる。

    生 3D は同期バッファの格子（既定 1/30 s）で抜けた格子は NaN の行、EKF の後の kpts3d は届いた組だけの行なので、
    行で合わせると抜けの後ろがすべてずれる。生 CSV の ``frame`` は格子の番号で、frames の ``grid_index`` と同じ
    番号（どちらも最初の組を 0 とする）なので、それで直接つなぐ。``grid_index`` の無い古い記録は、時刻を格子に
    丸めて（round(t/dt)）合わせる。時刻はどちらも最初の組からの秒（生 CSV の ``t``、frames の ``t_s``）。
    """
    provenance = capture.provenance
    dt = float(provenance["dt"])
    enabled = bool(provenance.get("EKF_ENABLE", True))
    kpts = pd.read_csv(kpts_path)
    n = min(len(kpts), len(frames))
    usable = np.isfinite(capture.points).any(axis=(1, 2))
    grid = pd.to_numeric(frames["grid_index"], errors="coerce").to_numpy(float)[:n] if "grid_index" in frames else None
    if grid is not None and len(grid) and np.isfinite(grid).all():
        alignment = "grid_index"
        raw_keys = np.asarray(capture.frame, dtype=float)
        keys = grid
    else:
        alignment = "round(t/dt)"
        raw_keys = np.rint(np.asarray(capture.t, dtype=float) / dt)
        keys = np.rint(frames["t_s"].to_numpy(float)[:n] / dt)
    index: dict[int, int] = {}
    for row, key in enumerate(raw_keys):
        if usable[row] and np.isfinite(key):
            index.setdefault(int(key), row)
    pairs = [(index[int(k)], j) for j, k in enumerate(keys) if np.isfinite(k) and int(k) in index]
    try:
        origin, lookup, note = _noise_lookup(provenance, dt, base_dir)
    except (KeyError, TypeError, ValueError):
        origin, lookup = (provenance.get("ekf_noise") or {}).get("origin") or "unknown", None
        note = "雑音のパラメータが記録に無いので棄却率は出さない"
    if not enabled:
        lookup, note = None, "EKF は無効（前後の差は 0 のはず）。棄却率は出さない"
    raw_rows = capture.points[[raw for raw, _ in pairs]] if pairs else np.empty((0, *capture.points.shape[1:]))
    kpts_rows = kpts.iloc[[row for _, row in pairs]].reset_index(drop=True)
    result = _ekf_series(capture, raw_rows, kpts_rows, origin, lookup, note)
    result.update(ekf_enabled=enabled, matched_rows=len(pairs), kpts_rows=int(n), alignment=alignment,
                  scale_ratio=provenance.get("ekf_scale_ratio"))
    return result


def _hybrid_gauge(work: pd.DataFrame, meta: Mapping[str, Any]) -> dict[str, Any]:
    """部位ごとの回の W_pos・スコア・帯（W_0.70〜W_0.85）への到達回数（論文 4.5.2 節）。"""
    bands = meta.get("gauge_bands_j") or {}
    w1rm = meta.get("w1rm_j") or {}
    stats = {}
    for joint in sorted(set(JOINTS) | set(bands)):
        rows = work.loc[work["joint"] == joint]
        positive = pd.to_numeric(rows["work_pos_j"], errors="coerce").to_numpy(float)
        scores = pd.to_numeric(rows["score"], errors="coerce").to_numpy(float)
        band = bands.get(joint)
        finite = positive[np.isfinite(positive)]
        stats[joint] = {
            "work_pos": [float(v) for v in positive],
            "scores": [float(v) for v in scores],
            "band": [float(b) for b in band] if band else None,
            "w1rm": w1rm.get(joint),
            "reached_low": int(np.sum(finite >= band[0])) if band else None,
            "reached_high": int(np.sum(finite >= band[1])) if band else None,
        }
    return stats


def _hybrid_extended(folder: Path, stamp: str | None, files: Mapping[str, Path | None], frames, meta, report, add,
                     raw_path: Path | None, raw_capture) -> None:
    """新しい版の記録（``output_schema_version`` がある）の検査と値。古い記録では何もしない。

    ``raw_path`` は生 3D（``kpts3d_raw_<stamp>.csv``）の場所、``raw_capture`` はそれを読んだもの（kpts3d が無ければ None）。
    """
    has_raw = raw_path is not None
    report["files"][HYBRID_RAW_PREFIX] = raw_path.name if has_raw else None
    for name in HYBRID_EXTRA_FILES:
        found = _one(folder, f"{name}_{stamp}*.csv") if stamp else None
        report["files"][name] = found.name if found else None
    if meta.get(HYBRID_SCHEMA_KEY) is None:
        return
    add(f"ファイル: {HYBRID_RAW_PREFIX}", has_raw,
        raw_path.name if has_raw else "無い（EKF の手前の生 3D。tune_ekf の入力）")
    if not has_raw:
        report["ekf"]["note"] = (f"{HYBRID_RAW_PREFIX}_{stamp}.csv（とサイドカー）が無いので、EKF の前後の差と棄却率は出さない。"
                                 "記録が途中で止まったか、書き出しが漏れている")
    if has_raw and files["kpts3d"] is not None and frames is not None:
        report["ekf"] = _hybrid_ekf_stats(raw_capture, files["kpts3d"], frames, base_dir=folder)
        matched, rows = report["ekf"]["matched_rows"], report["ekf"]["kpts_rows"]
        add(f"行: kpts3d の各行に生 CSV の同じ格子がある（{MIN_GRID_MATCH:.0%} 以上）",
            rows > 0 and matched / rows >= MIN_GRID_MATCH, f"{matched} / {rows} 行")
    if files["cycle_work"] is not None:
        work = pd.read_csv(files["cycle_work"])
        if set(HYBRID_WORK_COLUMNS) <= set(work.columns):
            report["gauge"] = _hybrid_gauge(work, meta)
    report["subject"] = {key: meta.get(key) for key in HYBRID_SUBJECT_KEYS}
    board = (meta.get("calibration_meta") or {}).get("checkerboard_short_axis") or {}
    report["gravity"] = dict(meta.get("gravity") or {}, board_tilt_deg=board.get("tilt_deg"),
                             board_up_label=board.get("up_label_runtime"))
    report["timing"] = meta.get("timing")
    report["hybrid"].update({key: meta.get(key) for key in ("ekf", "dyn_gate", "mac_camera", HYBRID_SCHEMA_KEY)})


def check_hybrid_run(folder: Path, log: str | Path | None = None, expect_stop: bool = True) -> dict[str, Any]:
    """混成の計測フォルダを確かめる（§3-2 は meta.json、§6-2 はトルク・サイクルの仕事・Pixel と Mac の速さ）。"""
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    stamp, files = _hybrid_files(folder)
    report: dict[str, Any] = {
        "kind": "hybrid", "out_dir": str(folder), "timestamp": stamp, "checks": checks,
        "files": {k: (p.name if p else None) for k, p in files.items()},
        "hybrid": {k: meta.get(k) for k in ("status", "exit_code", "stop_reason", "frames", "size_drops", "error",
                                            "body_mass_kg", "gravity_mode", "calibration", "cam0_offset_ms")},
        "ekf": {"note": HYBRID_EKF_NOTE, "noise_origin": None, "ekf_enabled": False, "series": []},
    }
    for name, path in files.items():
        add(f"ファイル: {name}", path is not None, path.name if path else "無い")
    # kill されると meta.json は "recording" のまま（記録器は閉じるときに complete にする）
    add("meta: 記録を正しく閉じた", meta.get("status") == "complete", f"status={meta.get('status')}")
    add("meta: 終了コード 0", meta.get("exit_code") in (0, None), f"exit_code={meta.get('exit_code')}、{meta.get('error') or ''}")
    if expect_stop:
        add("meta: 停止要求で止まった", meta.get("stop_reason") == "stop_request",
            f"stop_reason={meta.get('stop_reason')}（GUI の停止ボタン・停止ファイル・SIGTERM なら stop_request）")

    frames = pd.read_csv(files["frames"]) if files["frames"] else None
    if files["kpts3d"] is not None and frames is not None:
        n_kpts = len(pd.read_csv(files["kpts3d"]))
        ok = n_kpts == len(frames) and meta.get("frames") in (None, n_kpts)
        add("行: kpts3d・frames・meta の frames が一致", ok, f"kpts3d {n_kpts} / frames {len(frames)} / meta {meta.get('frames')}")
    if files["local_torque"] is not None:
        torque = pd.read_csv(files["local_torque"])
        add("行: local_torque が 1 行以上", len(torque) > 0, f"{len(torque)} 行")
        report["torque"] = {joint: _abs_stats(torque.loc[torque["joint"] == joint, "y"]) for joint in JOINTS}
    if frames is not None:
        work = pd.read_csv(files["cycle_work"]) if files["cycle_work"] else pd.DataFrame(columns=["joint", "work_j"])
        report["cycles"] = {
            "detected": int(frames["cycle_detected"].sum()),
            "work": {joint: [float(v) for v in work.loc[work["joint"] == joint, "work_j"]] for joint in JOINTS},
        }
        steps = _intervals(frames["t_s"])
        role_fps, role_frames = {}, {}
        if files["landmarks2d"] is not None:
            # 1 フレームは 33 行。Pixel はつなぎ直すと seq が 0 に戻るので、撮影時刻も合わせてフレームを見分ける
            marks = pd.read_csv(files["landmarks2d"], usecols=["role", "seq", "t_ns"]).drop_duplicates(["role", "seq", "t_ns"])
            for role, group in marks.groupby("role"):
                role_steps = _intervals(np.sort(group["t_ns"].to_numpy(float)) / 1e9)
                role_frames[role] = int(len(group))
                role_fps[role] = 1.0 / float(np.median(role_steps)) if role_steps.size else None
        # 2 台ともそろって初めて「Mac・Pixel とも」。片方の点が 1 つも無い（人を見つけない・別のカメラ）なら不合格
        absent = [role for role in ROLE_NAMES if role not in role_fps]
        slow = {role: fps for role, fps in role_fps.items() if fps is None or fps < MIN_CAMERA_FPS}
        detail = "、".join([f"{ROLE_NAMES[role]}（{role}）の点が無い" for role in absent]
                          + [f"{ROLE_NAMES.get(role, role)}（{role}）{_fmt(fps, '.1f')} fps" for role, fps in sorted(slow.items())])
        add(f"速さ: Mac・Pixel とも {MIN_CAMERA_FPS:g} fps 以上（30 fps の 8 割）", not absent and not slow,
            "landmarks2d が無い" if not role_fps else detail)
        period = float(np.median(steps)) if steps.size else None
        report["fps"] = {
            "processed_fps": 1.0 / period if period else None,
            # 組のうち、遅い方のカメラの実測に基づく割合。残りは同期バッファの線形補間で作った点（点の無いカメラは 0）
            "real_share": (min(1.0, min(role_frames.get(role, 0) for role in ROLE_NAMES) / len(frames))
                           if role_frames and len(frames) else None),
            # 同期バッファが 100 ms を超える穴で組を作らなかった時間（組の間隔が 1.5 倍を超えた分）
            "missing_s": float(steps[steps > 1.5 * period].sum()) if period else None,
            "interval_p05": float(np.percentile(steps, 5)) if steps.size else None,
            "interval_p95": float(np.percentile(steps, 95)) if steps.size else None,
            # 組は同期バッファ（app.net.sync_buffer）が補間して作るので、組の速さはカメラの速さではない。
            # 30 fps を保てたかは各カメラの速さ（role_fps）で見る
            "role_fps": role_fps, "role_frames": role_frames,
            "file_mode": False,
        }
    # EKF の手前の生 3D は _hybrid_extended と _hybrid_quality の両方で使うので 1 回だけ読む（どちらも kpts3d が要る）
    raw_path = _recorded_raw_path(folder, stamp)
    raw_capture = read_raw_capture(raw_path) if raw_path is not None and files["kpts3d"] is not None else None
    _hybrid_extended(folder, stamp, files, frames, meta, report, add, raw_path, raw_capture)
    quality = _hybrid_quality(folder, files, meta, raw_capture)
    if quality is not None:
        report["quality"] = quality
        _quality_checks(quality, add)
    if log is not None and Path(log).is_file():
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        add("ログ: 保存を報告した", "保存:" in text)
    return report


# hybrid-raw の出力名の印。計測は EKF の手前の 3D を kpts3d_raw_<stamp>.csv に書くので、道具の出力は
# kpts3d_raw_<stamp>_retri[_grid][_sN].csv にして分ける
RETRI_SUFFIX = "_retri"


def hybrid_raw_capture(session: str | Path, stride: int | None = None, out_dir: str | Path | None = None, *,
                       grid: bool = False, hz: float | None = None) -> Path:
    """混成の 3D（EKF の手前）を生 CSV（``kpts3d_raw_<stamp>_retri*``、``app.tuning.raw_capture`` の形）に直す。

    S6 の雑音の推定（``app.tuning.ekf_estimate`` / ``app.runners.tune_ekf``）がそのまま使える。サイドカーの source は
    ``hybrid_retri``（比べる用）。混成の計測（実行時）が読むプロファイルは、記録器が書いた ``kpts3d_raw_<stamp>.csv``
    （source が ``hybrid``）から作る。tune_ekf は ``hybrid_retri`` のプロファイルを実行時の置き場へ書かず、収録の隣に書く

    - 既定は、遅い方のカメラ（実機では Pixel、10〜15 Hz）の実際の撮影時刻で記録の 2D から三角測量し直した 3D
      （``app.hybrid.retriangulate``）。計測中の 3D は 30 Hz の格子へ線形補間した点で、補間の区間が直線になり
      雑音の推定が狂う。``grid=True`` なら記録された格子の 3D をそのまま使う（比べる用）。新しい版の記録は
      ``kpts3d`` が EKF の後なので、EKF の手前の ``kpts3d_raw_<stamp>.csv`` を使う（古い版の記録は ``kpts3d``）
    - 混成の経路には間引きの設定が無いので、4 Hz 間引きに当たる 2 設定目は間引いて作る。``hz`` を与えると、
      実際の速さから間引き幅（``stride``）を決める（12 Hz で ``hz=4`` なら 3 組おき）
    - 間隔は揺れ、推定は dt 一定を前提にするので、dt は間隔の中央値とし、揺れの幅（5〜95%）も残す
    """
    folder = hybrid_session(session)
    if folder is None:
        raise ValueError(f"混成の計測フォルダではない: {session}")
    if stride is not None and hz is not None:
        raise ValueError("stride と hz はどちらか一方")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    stamp, files = _hybrid_files(folder)
    ids = [int(i) for i in meta["pose_keypoints"]]
    recorded_raw = _recorded_raw_path(folder, stamp) if grid else None
    if recorded_raw is not None:
        # 新しい版の記録は kpts3d が EKF の後の点なので、EKF の手前の格子（kpts3d_raw_<stamp>.csv、抜けは NaN の行）を使う。
        # EKF で均した点から雑音を推定すると r が小さく出て、その較正を実行時が選んでしまう（dt が 1/30 s で一致する）
        capture = read_raw_capture(recorded_raw)
        points, t, frame_no = capture.points, capture.t, capture.frame
        times, skipped = "grid", 0
    elif grid:
        # 古い版の記録（EKF の手前の CSV が無い）は kpts3d がそのまま三角測量の点
        kpts_path, frames_path = files["kpts3d"], files["frames"]
        if kpts_path is None or frames_path is None:
            raise ValueError(f"kpts3d・frames の CSV が無い: {folder}")
        kpts, frames = pd.read_csv(kpts_path), pd.read_csv(frames_path)
        n = min(len(kpts), len(frames))
        points = kpts.drop(columns="frame").to_numpy(float)[:n].reshape(n, len(ids), 3)
        t = frames["t_s"].to_numpy(float)[:n]
        frame_no = frames["frame"].to_numpy(int)[:n]
        times, skipped = "grid", 0
    else:
        if files["landmarks2d"] is None:
            raise ValueError(f"landmarks2d の CSV が無い: {folder}")
        result = retriangulate(folder)
        points, t = result.points, result.t_ns / 1e9
        frame_no = np.arange(len(t))
        times, skipped = result.reference, result.skipped
    all_steps = _intervals(t)
    if not all_steps.size:
        raise ValueError(f"組の時刻が足りない（{len(t)} 組）")
    real_fps = 1.0 / float(np.median(all_steps))
    if hz is not None:
        stride = max(1, round(real_fps / hz))
    stride = stride or 1
    if stride < 1:
        raise ValueError("stride は 1 以上")
    index = np.arange(0, len(t), stride)
    steps = _intervals(t[index])
    if not steps.size:
        raise ValueError(f"組の時刻が足りない（{len(t)} 組、{stride} 組おき）")
    where = "30 Hz の格子（計測中の線形補間を含む）" if grid else f"{times} の実際の撮影時刻"
    provenance = {
        "unit": "m", "frame": "runtime", "dt": float(np.median(steps)),
        "dt_source": f"混成ステレオの {where} の間隔の中央値（{stride} 組おき）",
        "src_fps": real_fps, "file_mode": False,
        # 記録器の生 CSV（source が hybrid、tune_ekf が実行時の置き場へ書く）と分ける
        "source": HYBRID_RETRI_SOURCE, "hybrid_session": str(folder), "times": times, "stride": stride,
        "skipped_pairs": skipped, "t0_s": float(t[index[0]]),
        "interval_p05": float(np.percentile(steps, 5)), "interval_p95": float(np.percentile(steps, 95)),
        "RT_POSE_FIXED_HZ_ON": stride > 1, "EKF_ENABLE": False, "ekf_noise": None,
        "coordinates": meta.get("coordinates"),
    }
    target = Path(out_dir) if out_dir else folder
    target.mkdir(parents=True, exist_ok=True)
    # 計測中の記録（kpts3d_raw_<stamp>.csv、EKF の手前）を上書きしないよう、道具の出力には _retri を付ける
    name = f"kpts3d_raw_{stamp}{RETRI_SUFFIX}" + ("_grid" if grid else "") + ("" if stride == 1 else f"_s{stride}") + ".csv"
    path = target / name
    writer = RawCaptureWriter(path, ids, provenance)
    try:
        for k in index:
            writer.append(int(frame_no[k]), float(t[k] - t[index[0]]), points[k])
    finally:
        writer.close()
    return path


# --------------------------------------------------------------------------- check（USB）


def check_run(out_dir: str | Path, log: str | Path | None = None, timestamp: str | None = None,
              expect_stop: bool = True, expect_profile: bool = False) -> dict[str, Any]:
    """出力フォルダを確かめる。``checks`` は構造の合否、そのほかは値。

    混成ステレオの計測フォルダ（またはその親の measure）なら ``check_hybrid_run`` に回す。
    ``expect_stop``: 停止ファイル・SIGTERM で止めた回として確かめる（USB はログが要る）。
    ``expect_profile``: EKF が較正プロファイルを使ったことを確かめる（S9b）。
    """
    out_dir = Path(out_dir)
    hybrid = hybrid_session(out_dir)
    if hybrid is not None and timestamp is None:
        return check_hybrid_run(hybrid, log=log, expect_stop=expect_stop)
    ts = timestamp or _latest_timestamp(out_dir)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    report: dict[str, Any] = {"kind": "usb", "out_dir": str(out_dir), "timestamp": ts, "checks": checks}
    raw_path = out_dir / f"kpts3d_raw_{ts}.csv" if ts else None
    has_raw = raw_path is not None and raw_path.is_file() and sidecar_path(raw_path).is_file()
    add("ファイル: kpts3d_raw", has_raw, raw_path.name if raw_path else "kpts3d_raw_*.csv が無い")
    files = {
        "kpts3d": _one(out_dir, f"kpts3d_{ts}*.csv") if ts else None,
        "aim_torque": _one(out_dir, f"aim_torque_vec_{ts}_s*.csv") if ts else None,
        "gauge_energy": _one(out_dir, f"gauge_energy_{ts}_s*.csv") if ts else None,
    }
    for key, path in files.items():
        add(f"ファイル: {key}", path is not None, path.name if path else "無い")
    cycle_debug = _one(out_dir, f"cycle_energy_debug_{ts}_s*.csv") if ts else None
    report["files"] = {k: (p.name if p else None) for k, p in {**files, "cycle_energy_debug": cycle_debug}.items()}

    capture = read_raw_capture(raw_path) if has_raw else None
    if capture is not None and files["kpts3d"] is not None:
        n_raw, n_kpts = capture.points.shape[0], len(pd.read_csv(files["kpts3d"]))
        # 体格の比で計測を止めたとき（終了コード 3）だけ、生 CSV が 1 行多い
        add("行: 生 CSV と kpts3d が 1 行ずつ対応", n_raw - n_kpts in (0, 1), f"生 {n_raw} 行 / kpts3d {n_kpts} 行")
    if files["aim_torque"] is not None:
        n_torque = len(pd.read_csv(files["aim_torque"], encoding="utf-8-sig"))
        add("行: aim_torque が 1 行以上", n_torque > 0, f"{n_torque} 行（慣性の暖機 30 フレームの後から）")
        report["torque"] = _torque_stats(files["aim_torque"])
    if files["gauge_energy"] is not None:
        report["gauge"] = _gauge_stats(files["gauge_energy"])

    if capture is not None:
        prov = capture.provenance
        report["gravity"] = {"label": prov.get("gravity_label"), "vector": prov.get("gravity"),
                             "estimated": prov.get("gravity_set")}
        dt = float(prov["dt"])
        steps = np.diff(capture.t)
        steps = steps[np.isfinite(steps) & (steps > 0)]
        interval = float(np.median(steps)) if steps.size else None
        file_mode = bool(prov.get("file_mode"))
        report["fps"] = {"dt": dt, "dt_source": prov.get("dt_source"), "file_mode": file_mode,
                         "median_interval": interval, "processed_fps": 1.0 / interval if interval else None}
        if file_mode:
            add("処理間隔: dt と実際の間隔が 20% 以内", True, "ファイル入力（処理が遅くても dt は動画の fps で決まる）")
        else:
            ok = interval is not None and abs(interval - dt) / dt <= DT_TOLERANCE
            add("処理間隔: dt と実際の間隔が 20% 以内", ok,
                f"dt {dt:.4f} s / 実際 {interval:.4f} s" if interval else "間隔が取れない")
        # 3D が 1 点も取れない回（人が写っていない・灰色の映像）でも、ファイルと行はそろいトルクは 0 のまま書かれる
        tracked, rows = _tracked_rows(capture.points, capture.landmark_ids)
        add(f"3D: 肩・肘・手首がそろった行 {MIN_TRACKED_SHARE:.0%} 以上", rows > 0 and tracked / rows >= MIN_TRACKED_SHARE,
            f"{tracked} / {rows} 行" + ("" if rows and tracked else "、人を見つけていない。映像と写り方を確かめる"))
        # 骨の長さは混成と同じ節（値として並べる。USB の置き方・校正の目安は混成と違うので合否にしない）
        report["quality"] = {"segments": _segment_lengths(capture.points, capture.landmark_ids)}
        if files["kpts3d"] is not None:
            report["ekf"] = _ekf_stats(capture, files["kpts3d"], base_dir=out_dir)
        if expect_profile:
            origin = (prov.get("ekf_noise") or {}).get("origin")
            add("EKF: 較正プロファイルを使った", origin == "profile", f"雑音の出どころ {origin}")
    elif expect_profile:
        add("EKF: 較正プロファイルを使った", False, "生 CSV が無い")

    if log is not None and Path(log).is_file():
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        # 入力が開けないと、本体は録画の組（作業フォルダの cam*_output_*）やサンプル動画へ黙って切り替える
        add("ログ: 指定した入力を読んだ", INPUT_FAILED not in text,
            "別の録画へ切り替わった" if INPUT_FAILED in text else "")
        if expect_stop:
            # MAX_FRAMES で抜けたときも "[STOP] Reached MAX_FRAMES" と出るので、停止要求の文言で見る
            add("ログ: 停止要求を受けた", STOP_REQUESTED in text, "停止ファイルか SIGTERM で止めたときに出る")
        add("ログ: 終了時の CSV を書いた", "✅ aim_torque" in text)
        parsed = parse_file(Path(log))
        if parsed is not None and parsed[3]:
            report.setdefault("fps", {})["loop_dt_mean"] = parsed[3]["mean"]
            report["fps"]["loop_lines"] = parsed[3]["n"]
    elif expect_stop:
        add("ログ: 停止要求を受けた", False, "ログが無いので確かめていない。GUI の出力欄を保存して --log で渡す")
    return report


def _fmt(value, spec=".2f") -> str:
    return "—" if value is None else format(value, spec)


def _ekf_lines(e: Mapping[str, Any]) -> list[str]:
    """EKF の前後の差と棄却率の行（USB と混成で共用）。"""
    lines = []
    rms = [row["rms_mm"] for row in e["series"] if row["rms_mm"] is not None]
    rates = [row["rejection_rate"] for row in e["series"] if row["rejection_rate"] is not None]
    lines.append(f"[§6-3 EKF（S9b の材料）] 雑音の出どころ {e['noise_origin']}、EKF {'有効' if e['ekf_enabled'] else '無効'}")
    if e.get("note"):
        lines.append(f"  {e['note']}")
    if "matched_rows" in e:
        how = "格子の番号（grid_index）" if e.get("alignment") == "grid_index" else "時刻の格子（round(t/dt)）"
        lines.append(f"  生 CSV と kpts3d を{how}で合わせた行 {e['matched_rows']} / {e['kpts_rows']}、"
                     f"体格の比 {_fmt(e.get('scale_ratio'), '.3f')}")
    if rms:
        lines.append(f"  RMS（前後の差）[mm]: 中央値 {np.median(rms):.2f} / 最大 {max(rms):.2f}")
    if rates:
        lines.append(f"  棄却率: 中央値 {np.median(rates):.3f} / 最大 {max(rates):.3f}")
    worst = sorted((row for row in e["series"] if row["rms_mm"] is not None), key=lambda r: -r["rms_mm"])[:5]
    for row in worst:
        lines.append(f"    {row['landmark']}_{row['axis']}: RMS {row['rms_mm']:.2f} mm、棄却率 {_fmt(row['rejection_rate'], '.3f')}")
    lines.append("  張り付き・n_eff は python -m app.tuning.ekf_estimate <kpts3d_raw の CSV>")
    return lines


def _hybrid_lines(report: Mapping[str, Any]) -> list[str]:
    """混成の新しい版の記録の節（被験者・ゲージ・重力・処理時間）。"""
    lines = []
    if "subject" in report:
        s = report["subject"]
        one_rm = "、".join(f"{k} {_fmt(v, '.1f')}" for k, v in (s.get("one_rm_kg") or {}).items()) or "—"
        forearm = "、".join(f"{k} {_fmt(v, '.3f')}" for k, v in (s.get("forearm_len_m") or {}).items()) or "—"
        lines.append(f"[被験者] 被験者 {s.get('subject_id')}、体重 {_fmt(s.get('body_mass_kg'), '.1f')} kg、"
                     f"1RM [kg]: {one_rm}、前腕長 [m]: {forearm}")
    if "gauge" in report:
        lines.append("[§6-8 ゲージ] 回ごとの W_pos [J]・スコア S = W_pos / W_1RM・帯 W_0.70〜W_0.85（論文 4.5.2 節）")
        for joint, g in report["gauge"].items():
            work = ", ".join(_fmt(v, ".1f") for v in g["work_pos"])
            scores = ", ".join(_fmt(v, ".2f") for v in g["scores"])
            band = (f"帯 {g['band'][0]:.1f}〜{g['band'][1]:.1f} J、W_0.70 到達 {g['reached_low']} 回 / W_0.85 到達 "
                    f"{g['reached_high']} 回、W_1RM {_fmt(g['w1rm'], '.1f')} J") if g["band"] else "帯なし（1RM か前腕長が無い）"
            lines.append(f"  {joint}: W_pos [{work}] / S [{scores}]（{band}）")
    if "gravity" in report and report.get("kind") == "hybrid":
        g = report["gravity"]
        lines.append(f"[重力] 出どころ {g.get('source')}、向き {g.get('label')}（上 {g.get('up_label')}）、"
                     f"盤の傾き {_fmt(g.get('board_tilt_deg'), '.1f')}°（盤 {g.get('board_up_label') or 'なし'}）"
                     + (f"。{g['detail']}" if g.get("detail") else ""))
    if report.get("timing"):
        t = report["timing"]
        lines.append(f"[処理時間] 1 組の処理 中央値 {_fmt(t.get('median_ms'), '.1f')} ms / 95% {_fmt(t.get('p95_ms'), '.1f')} ms / "
                     f"最大 {_fmt(t.get('max_ms'), '.1f')} ms（30 Hz の予算は 33 ms）")
    return lines


def format_report(report: Mapping[str, Any]) -> str:
    lines = [f"== 検証: {report['out_dir']}（{report.get('timestamp')}） =="]
    lines.append("[構造]")
    for check in report["checks"]:
        mark = "✓" if check["ok"] else "✗"
        lines.append(f"  {mark} {check['name']}" + (f"（{check['detail']}）" if check["detail"] else ""))
    if "torque" in report:
        lines.append(f"[§6-2 トルク |τ_y| [N·m]]（{EXPECTED_TORQUE}）")
        for joint, s in report["torque"].items():
            lines.append(f"  {joint}: 中央値 {_fmt(s['median_abs'])} / 95% {_fmt(s['p95_abs'])} / 最大 {_fmt(s['max_abs'])}")
    if "gauge" in report and report.get("kind") != "hybrid":
        lines.append("[§6-2 ゲージ（サイクルごとの最大 [J]。最初と最後は途中のサイクル）]")
        for joint, s in report["gauge"].items():
            peaks = ", ".join(_fmt(p, ".1f") for p in s["cycle_peaks"])
            band = (f"帯 {s['band'][0]:.1f}〜{s['band'][1]:.1f}、E_low 到達 {s['reached_low']} / "
                    f"E_high 到達 {s['reached_high']}") if s["band"] else "帯なし"
            lines.append(f"  {joint}: [{peaks}]（{band}）")
    if "gravity" in report and report.get("kind") != "hybrid":
        g = report["gravity"]
        lines.append(f"[§6-2 重力] g の向き {g['label']}（{'体幹から推定' if g['estimated'] else '既定のまま'}）")
    if "cycles" in report:
        c = report["cycles"]
        lines.append(f"[§6-2 サイクル] 検出 {c['detected']} 回。サイクルごとの仕事 [J]:")
        for joint, work in c["work"].items():
            lines.append(f"  {joint}: [{', '.join(_fmt(w, '.1f') for w in work)}]")
    if "fps" in report and report.get("kind") == "hybrid":
        f = report["fps"]
        roles = "、".join(f"{ROLE_NAMES.get(role, role)}（{role}）{_fmt(v, '.1f')} fps"
                         for role, v in sorted(f.get("role_fps", {}).items()))
        lines.append(f"[§6-2 速さ] {roles}（目標は各 30 fps）。組は同期バッファが 30 Hz の格子へ線形補間して作るので "
                     f"{_fmt(f.get('processed_fps'), '.1f')} fps（間隔の 5〜95%: "
                     f"{_fmt(f.get('interval_p05'), '.3f')}〜{_fmt(f.get('interval_p95'), '.3f')} s）")
        lines.append(f"  組のうち遅い方のカメラの実測に基づく割合 {_fmt(f.get('real_share'), '.2f')}（残りは補間）、"
                     f"100 ms を超える穴で組が抜けた時間 {_fmt(f.get('missing_s'), '.1f')} s")
    elif "fps" in report:
        f = report["fps"]
        loop = f", [LOOP] の dt 平均 {_fmt(f.get('loop_dt_mean'), '.4f')} s" if "loop_dt_mean" in f else ""
        lines.append(f"[§6-2 処理の速さ] 処理 {_fmt(f.get('processed_fps'), '.1f')} fps（dt {_fmt(f.get('dt'), '.4f')} s = "
                     f"{_fmt(1.0 / f['dt'] if f.get('dt') else None, '.1f')} fps 相当、"
                     f"{'ファイル入力' if f.get('file_mode') else '実機'}）{loop}")
    if "quality" in report:
        q = report["quality"]
        lines.append("[配置と 3D の質]")
        for name, s in q["segments"].items():
            lines.append(f"  {name}: 中央値 {_fmt(s['median_m'], '.3f')} m、{s['range_m'][0]}〜{s['range_m'][1]} m に入る割合 "
                         f"{_fmt(s['share'], '.0%')}、範囲内のばらつき {_fmt(None if s['std_m'] is None else s['std_m'] * 100, '.1f')} cm")
        if "angle_deg" in q:
            lines.append(f"  基線 {q['baseline_cm']:.1f} cm、肘での視線のなす角 右 {_fmt(q['angle_deg']['R'], '.1f')}° / "
                         f"左 {_fmt(q['angle_deg']['L'], '.1f')}°（15° 以上が目安）")
        for role, shares in sorted(q.get("inside", {}).items()):
            lines.append(f"  {ROLE_NAMES.get(role, role)} の画面内: " + "、".join(f"{n} {v:.0%}" for n, v in shares.items()))
        lines.append("  肩幅は巻尺で測った値と比べる（mobile/README.md の目安は ±2 cm）")
    if report.get("kind") == "hybrid":
        h = report.get("hybrid", {})
        lines.append(f"[§3-2 記録] status={h.get('status')}、止まった理由 {h.get('stop_reason')}、"
                     f"終了コード {h.get('exit_code')}、解像度違いで捨てた点 {h.get('size_drops')}")
        lines.extend(_hybrid_lines(report))
        if report["ekf"].get("series"):
            lines.extend(_ekf_lines(report["ekf"]))
        else:
            lines.append(f"[§6-3 EKF] {report['ekf']['note']}")
    elif "ekf" in report:
        lines.extend(_ekf_lines(report["ekf"]))
    failed = [c for c in report["checks"] if not c["ok"]]
    lines.append("結果: " + ("構造の検査はすべて合格" if not failed else f"不合格 {len(failed)} 件"))
    return "\n".join(lines)


def _write_report(report: Mapping[str, Any], out_dir: Path) -> Path:
    path = out_dir / "verify_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- replay


def gui_environment(settings: Settings, stop_file: str | Path) -> dict[str, str]:
    """GUI が計測を起動するときと同じ環境変数。"""
    return entry.worker_environment(settings, role="realtime", stop_file=str(stop_file))


def dt_override(meta: Mapping[str, Any], fixed_hz: bool, env: Mapping[str, str] | None = None) -> float | None:
    """録画の実測 fps が容器の fps とずれていれば、再生で渡す DT_SEC。ずれていなければ None。

    再生では dt が容器の fps から決まる。間引きの幅は本体と同じく容器の fps と、GUI の設定
    （``env`` の ``RT_POSE_FIXED_HZ``・``SKIP_FRAMES``）から決まる（``config.resolve_dynamics_dt``。固定 Hz なら
    ``round(fps / Hz) - 1`` フレームを飛ばす）ので、その幅は保ったまま実際の時間に直す。
    """
    env = env or {}
    container = float(meta.get("container_fps") or 0.0)
    measured = meta.get("measured_fps")
    if not measured or container <= 0 or abs(measured - container) / container <= FPS_TOLERANCE:
        return None
    if fixed_hz:
        rate = float(env.get("RT_POSE_FIXED_HZ") or FIXED_HZ_DEFAULT)
        stride = int(max(0, round(container / max(rate, 1e-6)) - 1)) + 1
    else:
        skip = int(float(env.get("SKIP_FRAMES") or 0))
        stride = skip if skip > 0 else 1
    return stride / float(measured)


def replay_environment(base: Mapping[str, str], *, cam0, cam1, calib, out_dir, fixed_hz: bool, subject: str,
                       timestamp: str, stop_file, dt_sec: float | None = None, ekf_profile=None,
                       max_frames: int | None = None) -> dict[str, str]:
    """``base``（GUI と同じ環境変数）に、録画を読み込ませるための値を重ねる。"""
    env = dict(base)
    env.update({
        "CAM0": str(cam0), "CAM1": str(cam1), "CALIB_BASE_DIR": str(calib),
        # 開けなかったとき別の録画へ黙って切り替わらないように
        "USE_SAMPLE_VIDEOS": "0", "AUTO_FALLBACK_TO_FILES": "0",
        "HEADLESS": "1", "DISABLE_IMSHOW": "1", "LOOP_FILE_PLAYBACK": "0",
        # 本体の録画（間引き後のフレームだけ）は要らない
        "DISABLE_WRITE": "1",
        # コードの既定は 1（デモ表示）。1 のままだと力学が回らずトルクが全部 0 になる
        "DEMO_MONO_GAUGE_ON": "0", "DEMO_MONO_CAM0_ONLY": "0",
        "RT_POSE_FIXED_HZ_ON": "1" if fixed_hz else "0",
        "SUBJECT_ID": str(subject), "TIMESTAMP_OVERRIDE": timestamp,
        OUTPUT_DIR_ENV: str(out_dir), STOP_FILE_ENV: str(stop_file),
        "LOOP_TRACE": "0", "CAMERA_DIAG": "0", "PYTHONUNBUFFERED": "1",
        # 親は utf-8 で読む。Windows のパイプの既定（cp932）だと本体の "✅" の print で落ちる
        "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
    })
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    env.pop("DT_SEC", None)
    if dt_sec is not None:
        env["DT_SEC"] = f"{dt_sec:.6g}"
    # 本体は出力フォルダで動くので、相対パスのままだとプロファイルが見つからず同梱既定値で走る。
    # GUI の設定から引き継いだ値も、ここ（再生を起動した場所）を基準に絶対パスにする。"" は明示の解除
    if ekf_profile is not None:
        env["EKF_PROFILE"] = os.path.abspath(ekf_profile) if ekf_profile else ""
    elif env.get("EKF_PROFILE"):
        env["EKF_PROFILE"] = os.path.abspath(env["EKF_PROFILE"])
    if max_frames:
        env["MAX_FRAMES"] = str(int(max_frames))
    return env


def input_problems(cam0: Path, cam1: Path, calib: Path) -> list[str]:
    """再生の入力の問題。開けない動画のまま起動すると、本体は別の録画へ黙って切り替える
    （``master_research_code.py`` の入力のフォールバック）ので、起動の前に確かめる。
    """
    problems = []
    for label, path in (("cam0", cam0), ("cam1", cam1)):
        if not path.is_file():
            problems.append(f"{label} の動画が無い: {path}")
            continue
        cap = cv.VideoCapture(str(path))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok:
            problems.append(f"{label} の動画を読めない: {path}")
    missing = [name for name in CALIB_FILES if not (calib / name).is_file()]
    if missing:
        problems.append(f"校正ファイルが無い: {calib} に {', '.join(missing)}")
    return problems


def _replay(args) -> int:
    if args.session:
        session = Path(args.session)
        meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
        cam0, cam1 = (session / name for name in meta["videos"])
        calib = session
    elif args.cam0 and args.cam1 and args.calib:
        session, meta = None, {}
        cam0, cam1, calib = Path(args.cam0), Path(args.cam1), Path(args.calib)
    else:
        print("[ERROR] --session か、--cam0 --cam1 --calib の組を指定する", file=sys.stderr)
        return 2
    problems = input_problems(Path(cam0), Path(cam1), Path(calib))
    if problems:
        for problem in problems:
            print(f"[ERROR] {problem}", file=sys.stderr)
        return 2
    name = args.name or ("hz4" if args.fixed_hz else "full")
    if args.out:
        out_dir = Path(args.out)
    elif session is not None:
        out_dir = session / "runs" / name
    else:
        print("[ERROR] --cam0 で指定するときは --out も指定する", file=sys.stderr)
        return 2
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # 本体は被験者の 1RM の部位表（m_max_part_<ID>.json）を作業フォルダから読む。GUI をリポジトリから
    # 起動したときと同じ値を使うよう写す（無ければ本体が既定値の雛形を作る）
    m_max = REPO_ROOT / f"m_max_part_{args.subject}.json"
    if m_max.is_file() and not (out_dir / m_max.name).exists():
        shutil.copy2(m_max, out_dir / m_max.name)
    stop_file = out_dir / "stop.request"
    stop_file.unlink(missing_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    base = gui_environment(Settings.load(Settings.default_path()), stop_file)
    dt_sec = dt_override(meta, args.fixed_hz, base) if meta else None
    env = replay_environment(
        base,
        cam0=Path(cam0).resolve(), cam1=Path(cam1).resolve(), calib=Path(calib).resolve(), out_dir=out_dir,
        fixed_hz=args.fixed_hz, subject=args.subject, timestamp=timestamp, stop_file=stop_file,
        dt_sec=dt_sec, ekf_profile=args.ekf_profile, max_frames=args.max_frames)
    log_path = out_dir / f"run_{timestamp}.log"
    print(f"[REPLAY] {cam0.name} / {cam1.name} → {out_dir}（{'4 Hz 間引き' if args.fixed_hz else '間引きなし'}"
          + (f"、DT_SEC={dt_sec:.5f}" if dt_sec else "") + f"）ログ {log_path}", flush=True)

    command = [sys.executable, "-u", "-m", "app", "--role", "realtime"]
    process = subprocess.Popen(command, cwd=out_dir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    timer = None
    loops = 0
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("[LOOP]"):
                    loops += 1
                    if loops % LOOP_ECHO_EVERY == 0:
                        print(line.rstrip(), flush=True)
                elif any(tag in line for tag in ECHO_TAGS):
                    print(line.rstrip(), flush=True)
                # 停止ファイルはループに入ってから置く（入る前に置くと最初の周回で抜ける）
                if args.stop_after_sec and timer is None and line.startswith("[RAW]"):
                    timer = threading.Timer(args.stop_after_sec, stop_file.touch)
                    timer.daemon = True
                    timer.start()
        code = process.wait()
    finally:
        if timer is not None:
            timer.cancel()
        if process.poll() is None:
            process.terminate()
    print(f"[REPLAY] 終了コード {code}", flush=True)

    report = check_run(out_dir, log=log_path, timestamp=timestamp, expect_stop=bool(args.stop_after_sec),
                       expect_profile=bool(args.ekf_profile))
    report["exit_code"] = code
    report["replay"] = {"cam0": str(cam0), "cam1": str(cam1), "calib": str(calib), "fixed_hz": args.fixed_hz,
                        "dt_sec": dt_sec, "subject": args.subject, "stop_after_sec": args.stop_after_sec,
                        "recording_meta": meta or None}
    print(format_report(report))
    print(f"[REPLAY] 報告: {_write_report(report, out_dir)}")
    return 0 if code == 0 and all(c["ok"] for c in report["checks"]) else 1


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="計測の出力を確かめる（§6-2・§6-3・§3-2）。録画の再生もここから")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="出力フォルダを確かめる")
    check.add_argument("out_dir", help="計測の出力フォルダ（GUI なら画面の「出力先」）")
    check.add_argument("--log", default=None, help="計測のログ（[LOOP] の dt と [STOP] を見る）")
    check.add_argument("--timestamp", default=None, help="どの回か（MMDD_HHMMSS。既定は最新）")
    check.add_argument("--expect-stop", action="store_true", help="停止ボタン・停止ファイルで止めた回として確かめる")

    raw = sub.add_parser("hybrid-raw", help="混成ステレオの 3D を生 CSV に直す（S6 の推定用）")
    raw.add_argument("session", help="混成の計測フォルダ（その親の measure なら最新の回）")
    raw.add_argument("--stride", type=int, default=None, help="n 組おきに間引く")
    raw.add_argument("--hz", type=float, default=None, help="この速さに間引く（4 で S6 の 2 設定目。実際の速さから幅を決める）")
    raw.add_argument("--grid", action="store_true", help="計測中の 30 Hz の格子の 3D（線形補間を含む）をそのまま使う（比べる用）")
    raw.add_argument("--out", default=None, help="書き出し先（既定は計測フォルダ）")

    replay = sub.add_parser("replay", help="録画を計測に読み込ませて確かめる")
    replay.add_argument("--session", default=None, help="tools/record_stereo.py の録画フォルダ")
    replay.add_argument("--cam0", default=None, help="cam0 の動画（--session を使わないとき）")
    replay.add_argument("--cam1", default=None, help="cam1 の動画（--session を使わないとき）")
    replay.add_argument("--calib", default=None, help="校正ファイル 4 つのフォルダ（--session を使わないとき）")
    replay.add_argument("--out", default=None, help="出力先（既定は <録画>/runs/<name>）")
    replay.add_argument("--name", default=None, help="出力先の名前（既定は full か hz4）")
    replay.add_argument("--subject", required=True, help="被験者番号（SUBJECT_ID。1RM の部位表を選ぶ）")
    replay.add_argument("--fixed-hz", type=int, choices=(0, 1), default=0, help="1 なら 4 Hz 間引き（S6 の 2 設定目）")
    replay.add_argument("--stop-after-sec", type=float, default=0.0,
                        help="ループに入ってからこの秒数で停止ファイルを置く（§3-2 を GUI なしで確かめる）")
    replay.add_argument("--ekf-profile", default=None, help="EKF_PROFILE（S9b。tune_ekf が作ったファイルかフォルダ）")
    replay.add_argument("--max-frames", type=int, default=None, help="MAX_FRAMES（短く試すとき）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        args.fixed_hz = bool(args.fixed_hz)
        return _replay(args)
    if args.command == "hybrid-raw":
        try:
            path = hybrid_raw_capture(args.session, stride=args.stride, out_dir=args.out, grid=args.grid, hz=args.hz)
        except ValueError as error:
            print(f"[ERROR] {error}", file=sys.stderr)
            return 2
        meta = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
        print(f"生 CSV: {path}（{meta['dt_source']}: dt {meta['dt']:.5f} s、間隔の 5〜95%: "
              f"{meta['interval_p05']:.4f}〜{meta['interval_p95']:.4f} s、組を作れなかった点 {meta['skipped_pairs']}）")
        print(f"次: python -m app.tuning.ekf_estimate {path}")
        print(f"    python -m app.runners.tune_ekf {path}   # 比べる用。プロファイルはこの CSV の隣に書く")
        print("    混成の計測に使うプロファイルは、計測フォルダの kpts3d_raw_<stamp>.csv（記録器の生 CSV）から tune_ekf で作る")
        return 0
    report = check_run(args.out_dir, log=args.log, timestamp=args.timestamp, expect_stop=args.expect_stop)
    print(format_report(report))
    if report.get("timestamp"):
        print(f"報告: {_write_report(report, Path(report['out_dir']))}")
    return 0 if all(c["ok"] for c in report["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
