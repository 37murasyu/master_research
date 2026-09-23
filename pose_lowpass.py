"""姿勢 CSV の低域通過フィルタ。カットオフは試技ごとに押し上げの基本周波数 f0 から決める。

スコアの前処理（KNOWN_ISSUES §6-1）。論文の `*_lpf.csv` は `tmp_filter_pose_torque.py` が 2 Hz 固定の
Butterworth（4 次、filtfilt）で作っていた。一方、論文の表 5 とリアルタイム経路（`master_research_code.py`
の E_FC_*）は「fc = 6 × f0 を 2.1〜6.0 Hz に制限」で、被験者 8 は 5.63 Hz だった。2026-09-23 に
オフラインもこちらに揃えた。

f0 はリアルタイム経路の `OnlineF0Estimator` と同じ計算: 左右の肘角の平均を unwrap し、Welch
（Hann、nperseg 256）の 0.3 Hz 以上のピーク。ピークと中央値の比（SNR）が 3 dB 未満なら
はっきりした周期が無いとみなし、従来の 2 Hz にする。

使い方::

    python pose_lowpass.py --in-dir "Adjusted 3D Pose" --out-dir output_data/filtered_pose_lpf
    python pose_lowpass.py --in-dir "Adjusted 3D Pose" --out-dir ... --fc 2.0   # 固定にする
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

# リアルタイム経路の E_FC_K / E_FC_MIN / E_FC_MAX / E_F0_FMIN / E_F0_SNR_THRESHOLD の既定値と同じ
# （tests/test_pose_lowpass.py が一致を確かめる）
FC_K = 6.0
FC_MIN_HZ = 2.1
FC_MAX_HZ = 6.0
F0_MIN_HZ = 0.3
F0_SNR_DB = 3.0
# 周期がはっきりしないときのカットオフ。従来の固定値
FALLBACK_FC_HZ = 2.0
NPERSEG = 256
ORDER = 4
DEFAULT_FS = 30.0

_ARMS = {"L": (11, 13, 15), "R": (12, 14, 16)}   # 肩・肘・手首のランドマーク ID


def _points(df: pd.DataFrame, jid: int) -> np.ndarray:
    return df[[f"joint_{jid}_{a}" for a in "xyz"]].to_numpy(float)


def elbow_angle(df: pd.DataFrame) -> np.ndarray:
    """左右の肘角（上腕と前腕のなす角）の平均 [rad]。片側が欠けるフレームはもう片側だけ。"""
    angles = []
    for shoulder, elbow, wrist in _ARMS.values():
        upper = _points(df, elbow) - _points(df, shoulder)
        fore = _points(df, wrist) - _points(df, elbow)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.sum(upper * fore, axis=1) / (np.linalg.norm(upper, axis=1) * np.linalg.norm(fore, axis=1))
        angles.append(np.arccos(np.clip(cos, -1.0, 1.0)))
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(angles), axis=0)


def estimate_f0(theta: np.ndarray, fs: float) -> tuple[float, float]:
    """肘角の系列から押し上げの基本周波数 f0 [Hz] と SNR [dB] を返す。取れなければ (0, 0)。"""
    theta = np.asarray(theta, dtype=float)
    theta = pd.Series(theta).interpolate(limit_direction="both").to_numpy()
    if len(theta) < 64 or not np.all(np.isfinite(theta)):
        return 0.0, 0.0
    freqs, psd = welch(np.unwrap(theta), fs=fs, window="hann", nperseg=min(NPERSEG, len(theta)))
    mask = freqs >= F0_MIN_HZ
    if not mask.any():
        return 0.0, 0.0
    peak = int(np.argmax(psd[mask]))
    snr = 10.0 * np.log10(max(1e-9, psd[mask][peak] / max(1e-12, float(np.median(psd[mask])))))
    return float(freqs[mask][peak]), float(snr)


def cutoff_hz(f0: float, snr_db: float) -> float:
    """fc = FC_K × f0 を [FC_MIN_HZ, FC_MAX_HZ] に収める。周期がはっきりしなければ FALLBACK_FC_HZ。"""
    if snr_db < F0_SNR_DB or f0 <= 0:
        return FALLBACK_FC_HZ
    return float(np.clip(FC_K * f0, FC_MIN_HZ, FC_MAX_HZ))


def butter_lowpass(data: np.ndarray, fs: float, fc: float, order: int = ORDER) -> np.ndarray:
    """列ごとの Butterworth 低域通過（filtfilt、位相ずれなし）。"""
    b, a = butter(order, min(fc / (fs / 2.0), 0.999), btype="low")
    return filtfilt(b, a, data, axis=0)


def lowpass_pose_csv(src: str | Path, out_dir: str | Path, fs: float = DEFAULT_FS, fc: float | None = None) -> dict:
    """姿勢 CSV 1 本を低域通過し、`<stem>_lpf.csv` と `<stem>_lpf_meta.json` を書く。

    fc を与えなければ f0 から決める。欠測は線形補間してから掛ける（従来の前処理と同じ）。
    """
    src, out_dir = Path(src), Path(out_dir)
    df = pd.read_csv(src)
    cols = [c for c in df.columns if c.startswith("joint_") and c.endswith(("_x", "_y", "_z"))]
    filled = df.copy()
    filled[cols] = filled[cols].interpolate(limit_direction="both")

    f0, snr = estimate_f0(elbow_angle(filled), fs)
    chosen = float(fc) if fc is not None else cutoff_hz(f0, snr)
    usable = [c for c in cols if filled[c].notna().all()]
    filled[usable] = butter_lowpass(filled[usable].to_numpy(float), fs, chosen)

    out_dir.mkdir(parents=True, exist_ok=True)
    filled.to_csv(out_dir / f"{src.stem}_lpf.csv", index=False)
    meta = {
        "source": src.name, "fs": fs, "order": ORDER, "f0_hz": f0, "f0_snr_db": snr, "fc_hz": chosen,
        "fc_rule": "fixed" if fc is not None else f"clip({FC_K} x f0, {FC_MIN_HZ}, {FC_MAX_HZ}); "
                                                  f"SNR < {F0_SNR_DB} dB -> {FALLBACK_FC_HZ}",
    }
    (out_dir / f"{src.stem}_lpf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="姿勢 CSV を低域通過する（カットオフは試技ごとに f0 から）")
    parser.add_argument("--in-dir", required=True, help="姿勢 CSV のフォルダ（Adjusted 3D Pose など）")
    parser.add_argument("--out-dir", required=True, help="出力先（<stem>_lpf.csv と <stem>_lpf_meta.json）")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS, help="サンプリング周波数 [Hz]")
    parser.add_argument("--fc", type=float, default=None, help="カットオフを固定する [Hz]（省略すると f0 から）")
    args = parser.parse_args(argv)
    for src in sorted(Path(args.in_dir).glob("*.csv")):
        meta = lowpass_pose_csv(src, args.out_dir, fs=args.fs, fc=args.fc)
        print(f"[LPF] {src.name}: f0={meta['f0_hz']:.3f} Hz (SNR {meta['f0_snr_db']:.1f} dB) -> fc={meta['fc_hz']:.2f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
