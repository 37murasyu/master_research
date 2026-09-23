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
from extended_kalman_filter import EKFConfig, SeriesNoise

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
# 実行時の基準長（肩–肘）として人体でありうる範囲 [m]。外れたら座標の単位が違う（欠陥 5）
PLAUSIBLE_REF_LEN = (0.10, 0.60)
# ディレクトリを渡したときに探すファイル名。生 CSV のサイドカー（kpts3d_raw_*.json）を拾わない
PROFILE_GLOB = "ekf_profile_*.json"


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
    # 生 CSV のサイドカーも schema_version / frame / unit が同じなので、上の検査だけでは通ってしまう
    if not isinstance(profile.get("series"), dict):
        raise ValueError(f"{Path(path).name} は較正プロファイルではない（series が無い）")
    return profile


@dataclass(frozen=True)
class ProfileResolution:
    """実行時に使う較正値と、その出どころ。``reason`` が付いていれば既定値に落ちている。"""

    entries: dict[tuple[int, str], SeriesEntry]
    dt: float
    path: Path | None
    reason: str | None
    # 較正時の基準長（build_profile の scale_ref）。実行時の長さとの比で q・r を掛け直す
    scale_ref: dict[str, Any] | None = None

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
        return sorted(path.glob(PROFILE_GLOB))
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
        return ProfileResolution(entries=entries, dt=float(profile["dt"]), path=path, reason=None,
                                 scale_ref=profile.get("scale_ref"))

    return _builtin_resolution(dt, "dt_mismatch")


def body_scale_ratio(scale_ref: Mapping[str, Any] | None, run_length: float) -> float:
    """実行時の基準長と較正時の基準長の比 L_run / L_cal。``q_acc`` と ``r`` に 2 乗で効かせる（欠陥 5）。

    基準長が人体としてありえない範囲なら例外にする。``r`` が小さすぎれば派手に発散するが、
    大きすぎると静かに素通りするので、黙って使わない。``calib.py`` で校正し直すと座標が
    m の 1/100 になる経路がある。プロファイルに基準長が無ければ 1（掛け直さない）。
    """
    low, high = PLAUSIBLE_REF_LEN
    if not np.isfinite(run_length) or not (low <= run_length <= high):
        raise ValueError(
            f"実行時の肩–肘の長さ {run_length!r} m が人体の範囲 {low}〜{high} m に無い。"
            " 座標の単位（校正の経路）を確かめること")
    if not scale_ref:
        return 1.0
    return float(run_length) / float(scale_ref["median_len"])


@dataclass(frozen=True)
class RuntimeNoise:
    """実行時の LandmarkEKF に渡す雑音パラメータと、その出どころ。"""

    cfg: EKFConfig | SeriesNoise
    origin: str   # "env"（環境変数のスカラー）| "profile" | "builtin"（同梱既定値）
    resolution: ProfileResolution | None
    sources: dict[str, int]   # 系列ごとの source の内訳
    landmark_ids: tuple[int, ...]

    def scaled(self, ratio: float) -> EKFConfig | SeriesNoise:
        """基準長の比で掛け直した雑音パラメータ。環境変数のスカラーは掛け直さない。"""
        if self.origin == "env" or self.resolution is None:
            return self.cfg
        return replace(self.resolution, entries={
            key: entry.scaled(ratio) for key, entry in self.resolution.entries.items()
        }).series_noise(self.landmark_ids)

    def describe(self) -> str:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(self.sources.items()))
        where = "" if self.resolution is None or self.resolution.path is None else f" path={self.resolution.path}"
        reason = "" if self.resolution is None or self.resolution.reason is None else f" reason={self.resolution.reason}"
        return f"origin={self.origin}{where}{reason} 系列: {breakdown}"

    def provenance(self) -> dict[str, Any]:
        """サイドカー JSON に残す形。"""
        return {
            "origin": self.origin,
            "path": None if self.resolution is None or self.resolution.path is None else str(self.resolution.path),
            "reason": None if self.resolution is None else self.resolution.reason,
            "profile_dt": None if self.resolution is None else self.resolution.dt,
            "sources": dict(self.sources),
        }


def runtime_noise(
    source: str | Path | None,
    *,
    dt: float,
    bpf_enabled: bool,
    landmark_ids: Sequence[int],
    scalar: EKFConfig,
) -> RuntimeNoise:
    """実行時の EKF に渡す雑音パラメータを決める（設計メモ 実装 5、S9）。

    - ``source``（``EKF_PROFILE``）が無ければ、環境変数のスカラー（今までの挙動）
    - あれば ``resolve_profile`` で解決し、**プロファイルが勝つ**。GUI は ``EKF_Q_ACC`` / ``EKF_R`` を
      必ず子プロセスに渡すので、スカラーを先にすると GUI 経由ではプロファイルが使われない
    - dt が合わない・BPF が有効なら同梱既定値（理由は ``resolution.reason``）

    系列はランドマーク ID の**昇順**に並べる（実行時の 3D 点列の並び）。``pose_keypoints`` の宣言順は
    ID 順ではないので、そのまま並べると系列が黙って取り違えられる。
    """
    ids = tuple(sorted(int(lid) for lid in landmark_ids))
    n_series = len(ids) * len(AXES)
    if not source:
        return RuntimeNoise(cfg=scalar, origin="env", resolution=None, sources={"env": n_series}, landmark_ids=ids)

    resolution = resolve_profile(source, dt=dt, bpf_enabled=bpf_enabled)
    known = list(resolution.entries.values())
    spare = _median_entry(known, "global_median") if known else builtin_entry(resolution.dt)
    sources: dict[str, int] = {}
    for lid in ids:
        for axis in AXES:
            entry = resolution.entries.get((lid, axis), spare)
            sources[entry.source] = sources.get(entry.source, 0) + 1
    return RuntimeNoise(
        cfg=resolution.series_noise(ids),
        origin="profile" if resolution.reason is None else "builtin",
        resolution=resolution,
        sources=sources,
        landmark_ids=ids,
    )
