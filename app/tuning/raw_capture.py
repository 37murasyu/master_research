"""EKF の手前の生 3D 座標を、較正に使える形で書き出し・読み込む。

EKF の最尤推定に使う入力は、実行時と同じ単位・軸・dt でなければならない。
別スクリプトが書いた CSV を使うと単位や軸が食い違う（設計メモの版 2 の訂正）ので、
計測スクリプト自身が EKF の手前で書き出す。

- 1 行ごとに flush する。GUI の停止は停止ファイルでループを抜けて終了時処理を走らせるが
  （``app.core.stop_request``）、猶予を過ぎて kill されたときにも残すため
- 由来（dt・間引き設定・EKF 設定など）はサイドカー JSON に**起動時**に書く
- 読み込み側は ``stage: "pre_ekf"`` のサイドカーが無いファイルを拒否する

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の「実装 0」。
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STAGE = "pre_ekf"
SCHEMA_VERSION = 1


def sidecar_path(csv_path: str | Path) -> Path:
    """生 CSV に対応するサイドカー JSON のパス（拡張子だけを替える）。"""
    return Path(csv_path).with_suffix(".json")


def git_commit(repo_dir: str | Path) -> str | None:
    """``repo_dir`` の HEAD のコミット。git が無い・リポジトリの外（配布版など）なら None。

    記録のために計測を止めてはいけないので、取れなければ黙って None を返す。
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _header(landmark_ids: Sequence[int]) -> list[str]:
    return ["frame", "t"] + [f"{lid}_{axis}" for lid in landmark_ids for axis in ("x", "y", "z")]


class RawCaptureWriter:
    """処理したフレームごとに 1 行を追記し、その都度 flush する。

    ``points`` の並びは ``landmark_ids`` の順（実行時の 3D 点は ID の昇順）。
    値は丸めずに書き、検出されなかった点は NaN のまま残す。
    """

    def __init__(self, csv_path: str | Path, landmark_ids: Sequence[int], provenance: Mapping[str, Any]):
        self.path = Path(csv_path)
        self.landmark_ids = [int(lid) for lid in landmark_ids]

        meta = dict(provenance)
        meta.update(
            stage=STAGE,
            schema_version=SCHEMA_VERSION,
            landmark_ids=self.landmark_ids,
            created=datetime.now().isoformat(timespec="seconds"),
        )
        sidecar_path(self.path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self._fh = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh, lineterminator="\n")
        self._writer.writerow(_header(self.landmark_ids))
        self._fh.flush()

    def append(self, frame: int, t: float, points: np.ndarray) -> None:
        arr = np.asarray(points, dtype=float)
        expected = (len(self.landmark_ids), 3)
        if arr.shape != expected:
            raise ValueError(f"points の形は {expected} のはずが {arr.shape}")
        # repr は float を往復で失わない最短表記にする（NaN は "nan"）
        self._writer.writerow([int(frame), repr(float(t)), *map(repr, arr.reshape(-1).tolist())])
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


@dataclass(frozen=True, eq=False)
class RawCapture:
    """生 CSV 1 本ぶん。``points`` は (フレーム数, 点数, 3) で、並びは ``landmark_ids`` の順。"""

    landmark_ids: tuple[int, ...]
    frame: np.ndarray
    t: np.ndarray
    points: np.ndarray
    provenance: dict[str, Any]


def _read_sidecar(csv_path: Path) -> dict[str, Any]:
    meta_path = sidecar_path(csv_path)
    if not meta_path.is_file():
        raise ValueError(
            f"{csv_path.name} にサイドカー {meta_path.name} が無い。"
            f"較正に使えるのは stage が {STAGE!r} の生 CSV だけ"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("stage") != STAGE:
        raise ValueError(f"{meta_path.name} の stage は {meta.get('stage')!r}。較正に使えるのは {STAGE!r} だけ")
    return meta


def read_raw_capture(csv_path: str | Path) -> RawCapture:
    """生 CSV とサイドカーを読む。

    最終行は常に捨てる。正常なら改行の後ろの空文字、kill で書き込みが途切れていれば
    改行の無い不完全な行だから。それ以外の行の列数が合わなければ、kill では生じない
    壊れ方なので例外にする。
    """
    path = Path(csv_path)
    meta = _read_sidecar(path)
    ids = tuple(int(lid) for lid in meta["landmark_ids"])
    expected = _header(ids)

    lines = path.read_text(encoding="utf-8").split("\n")
    lines.pop()
    if not lines or lines[0].split(",") != expected:
        raise ValueError(f"{path.name} の見出しがサイドカーの landmark_ids と合わない")

    frames: list[int] = []
    times: list[float] = []
    values: list[list[float]] = []
    for number, line in enumerate(lines[1:], start=2):
        cells = line.split(",")
        if len(cells) != len(expected):
            raise ValueError(f"{path.name}:{number} の列数が {len(cells)}（{len(expected)} のはず）")
        frames.append(int(cells[0]))
        times.append(float(cells[1]))
        values.append([float(cell) for cell in cells[2:]])

    points = np.asarray(values, dtype=float).reshape(len(values), len(ids), 3)
    return RawCapture(
        landmark_ids=ids,
        frame=np.asarray(frames, dtype=int),
        t=np.asarray(times, dtype=float),
        points=points,
        provenance=meta,
    )
