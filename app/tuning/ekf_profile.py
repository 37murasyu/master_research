"""較正プロファイルの読み書きと、実行時の解決（dt による探索とフォールバック）。

推定した ``(q_acc, r, gate_std)`` は、**どの条件で測ったか**と一緒でなければ使えない。
実行時の dt は間引き設定で 8 倍変わり（設計メモ 欠陥 2）、再構成スケールは校正手法で変わる（欠陥 5）。

- ``dt`` が合うプロファイルを探す。無ければ同梱既定値に落として ``reason`` を残す。
  **計測は止めない**（決定 6）。落ちた試技は記録から後で見分けられる
- BPF が有効なら較正値は使わない（決定 7）。BPF は位置の DC を落とすので、
  較正時と同じ前処理を再現しない限り当てにならない
- ``schema_version`` / ``frame`` / ``unit`` の食い違いは運用では起こらない「壊れたファイル」なので例外
- 基準長の比 ``s`` は ``q_acc`` と ``r`` に ``s²`` で効かせる（どちらも長さの 2 乗の次元）

同梱既定値は**仮置き**で、版 2 の推定中央値をそのまま置いてある。S6（実機の収録）の
実測が出たら、間引き設定ごとに差し替えること。

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の「実装 4」（S7）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from app.tuning.ekf_estimate import SeriesFit
from app.tuning.ekf_likelihood import innovation_loglik
from app.tuning.raw_capture import RawCapture, git_commit
from extended_kalman_filter import SeriesNoise

SCHEMA_VERSION = 1
FRAME = "runtime"  # 三角測量の軸入れ替え後
UNIT = "m"
AXES = ("x", "y", "z")

# 推定を採用する最少の有効サンプル数。S6 の実測でこの閾値自体を見直す
MIN_N_EFF = 300
# ゲートは正規化イノベーションのこのパーセンタイル（正規分布なら約 3σ）
GATE_PERCENTILE = 99.7
# dt がこの相対差以内なら同じ設定とみなす（29.97 fps のような小さなずれを拾う）
DT_TOLERANCE = 0.05
# スケール不変の基準長に使う組（右肩–右肘）
SCALE_REF_PAIR = (12, 14)


@dataclass(frozen=True)
class SeriesEntry:
    """1 系列ぶんの較正値と、その値がどこから来たか。"""

    q_acc: float
    r: float
    gate_std: float
    n_eff: int
    rho1: float
    source: str  # "fit" | "axis_median" | "global_median" | "builtin"
    reason: str | None = None

    def scaled(self, ratio: float) -> "SeriesEntry":
        factor = float(ratio) ** 2
        return replace(self, q_acc=self.q_acc * factor, r=self.r * factor)


_BUILTIN_GATE = 3.0
# TODO(S6): 実機の収録から間引き設定ごとに測り直す。今は版 2 の推定中央値の仮置き
BUILTIN_DEFAULTS: dict[float, SeriesEntry] = {
    1 / 30: SeriesEntry(q_acc=0.122, r=2.59e-5, gate_std=_BUILTIN_GATE, n_eff=0, rho1=float("nan"), source="builtin"),
    8 / 30: SeriesEntry(q_acc=0.122, r=2.59e-5, gate_std=_BUILTIN_GATE, n_eff=0, rho1=float("nan"), source="builtin"),
}


def builtin_entry(dt: float) -> SeriesEntry:
    """``dt`` にいちばん近い同梱既定値。"""
    nearest = min(BUILTIN_DEFAULTS, key=lambda known: abs(np.log(known / float(dt))))
    return BUILTIN_DEFAULTS[nearest]


def _is_usable(fit: SeriesFit | None, min_n_eff: int) -> bool:
    return fit is not None and not fit.at_bound and fit.n_eff >= min_n_eff


def _gate_std(capture: RawCapture, key: tuple[int, str], fit: SeriesFit, dt: float) -> float:
    """正規化イノベーションの分布から、その系列の門の広さを決める。

    ゲート無しで推定した ``r`` は外れ値を吸収して過大になるので、固定 3.0 だと門が広くなる。
    """
    lid, axis = key
    z = capture.points[:, capture.landmark_ids.index(lid), AXES.index(axis)]
    normalized = innovation_loglik(z, dt, fit.q_acc, fit.r).normalized
    finite = normalized[np.isfinite(normalized)]
    if finite.size == 0:
        return _BUILTIN_GATE
    return float(np.percentile(np.abs(finite), GATE_PERCENTILE))


def _median_entry(entries: Sequence[SeriesEntry], source: str) -> SeriesEntry:
    return SeriesEntry(
        q_acc=float(np.median([e.q_acc for e in entries])),
        r=float(np.median([e.r for e in entries])),
        gate_std=float(np.median([e.gate_std for e in entries])),
        n_eff=0,
        rho1=float("nan"),
        source=source,
    )


def _scale_ref(capture: RawCapture) -> dict[str, Any] | None:
    if not all(point in capture.landmark_ids for point in SCALE_REF_PAIR):
        return None
    first, second = (capture.landmark_ids.index(point) for point in SCALE_REF_PAIR)
    lengths = np.linalg.norm(capture.points[:, first] - capture.points[:, second], axis=1)
    finite = lengths[np.isfinite(lengths)]
    if finite.size == 0:
        return None
    return {"pair": list(SCALE_REF_PAIR), "median_len": float(np.median(finite))}


def build_profile(
    capture: RawCapture,
    fits: Mapping[tuple[int, str], SeriesFit | None],
    *,
    min_n_eff: int = MIN_N_EFF,
) -> dict[str, Any]:
    """推定結果を、実行時が読める形に落とす。採用できない系列は段階を踏んでフォールバックする。"""
    dt = float(capture.provenance["dt"])
    usable = {key: fit for key, fit in fits.items() if _is_usable(fit, min_n_eff)}
    fitted = {
        key: SeriesEntry(
            q_acc=fit.q_acc,
            r=fit.r,
            gate_std=_gate_std(capture, key, fit, dt),
            n_eff=fit.n_eff,
            rho1=fit.rho[0],
            source="fit",
        )
        for key, fit in usable.items()
    }

    series: dict[str, dict[str, Any]] = {}
    for lid in capture.landmark_ids:
        per_axis: dict[str, Any] = {}
        same_point = [fitted[(lid, axis)] for axis in AXES if (lid, axis) in fitted]
        for axis in AXES:
            entry = fitted.get((lid, axis))
            if entry is None and same_point:
                entry = _median_entry(same_point, "axis_median")
            elif entry is None and fitted:
                entry = _median_entry(list(fitted.values()), "global_median")
            elif entry is None:
                entry = replace(builtin_entry(dt), reason="no_usable_fit")
            per_axis[axis] = asdict(entry)
        series[str(lid)] = per_axis

    provenance = capture.provenance
    return {
        "schema_version": SCHEMA_VERSION,
        "frame": FRAME,
        "unit": UNIT,
        "dt": dt,
        "fps": float(provenance.get("src_fps") or 1.0 / dt),
        "bpf": {
            "low": float(provenance.get("EKF_BPF_LOW", 0.0)),
            "high": float(provenance.get("EKF_BPF_HIGH", 0.0)),
            "order": int(provenance.get("EKF_BPF_ORDER", 2)),
        },
        "scale_ref": _scale_ref(capture),
        "source": {
            "n_frames": int(capture.points.shape[0]),
            "git": git_commit(Path(__file__).resolve().parents[2]),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        "series": series,
    }


def write_profile(path: str | Path, profile: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_profile(path: str | Path) -> dict[str, Any]:
    """読んだ時点で、その値が何を意味するかを検査する。"""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, expected in (("schema_version", SCHEMA_VERSION), ("frame", FRAME), ("unit", UNIT)):
        if profile.get(key) != expected:
            raise ValueError(f"{Path(path).name} の {key} が {profile.get(key)!r}（{expected!r} のはず）")
    return profile


@dataclass(frozen=True)
class ProfileResolution:
    """実行時に使う較正値と、その出どころ。``reason`` が付いていれば既定値に落ちている。"""

    entries: dict[tuple[int, str], SeriesEntry]
    dt: float
    path: Path | None
    reason: str | None

    def series_noise(self, landmark_ids: Sequence[int]) -> SeriesNoise:
        """実行時の配列（点 × 3 + 軸）に並べ替える。プロファイルに無い系列だけ落とす。"""
        known = list(self.entries.values())
        spare = _median_entry(known, "global_median") if known else builtin_entry(self.dt)
        chosen = [self.entries.get((int(lid), axis), spare) for lid in landmark_ids for axis in AXES]
        return SeriesNoise(
            q_acc=np.array([e.q_acc for e in chosen], dtype=float),
            r=np.array([e.r for e in chosen], dtype=float),
            gate_std=np.array([e.gate_std for e in chosen], dtype=float),
        )


def _candidates(source: str | Path | None) -> list[Path]:
    if source is None:
        return []
    path = Path(source)
    if path.is_dir():
        return sorted(path.glob("*.json"))
    return [path] if path.is_file() else []


def _builtin_resolution(dt: float, reason: str) -> ProfileResolution:
    return ProfileResolution(entries={}, dt=float(dt), path=None, reason=reason)


def resolve_profile(
    source: str | Path | None = None,
    *,
    dt: float,
    bpf_enabled: bool = False,
    scale_ratio: float = 1.0,
) -> ProfileResolution:
    """その場の条件に合う較正値を選ぶ。合わなければ既定値に落として理由を残す。"""
    if bpf_enabled:
        return _builtin_resolution(dt, "bpf_enabled")

    candidates = _candidates(source)
    if not candidates:
        return _builtin_resolution(dt, "no_profile")

    for path in sorted(candidates, key=lambda p: abs(read_profile(p)["dt"] - dt)):
        profile = read_profile(path)
        if abs(profile["dt"] - dt) / dt > DT_TOLERANCE:
            continue
        entries = {
            (int(lid), axis): SeriesEntry(**entry).scaled(scale_ratio)
            for lid, per_axis in profile["series"].items()
            for axis, entry in per_axis.items()
        }
        return ProfileResolution(entries=entries, dt=float(profile["dt"]), path=path, reason=None)

    return _builtin_resolution(dt, "dt_mismatch")
