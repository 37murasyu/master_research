"""実行時の EKF が、どの雑音パラメータを使うかの決め方を固定する（S9）。

**なぜこのテストがあるか。**

S7 で較正プロファイルを作れるようになったが、実行時の EKF（``master_research_code.py``）には
まだ渡っていなかった。配線にあたって決めたことを固定する。

- プロファイルが解決できたら**プロファイルが勝つ**。GUI は ``EKF_Q_ACC`` / ``EKF_R`` を必ず子プロセスに
  渡すので、「環境変数 → プロファイル」の順だと GUI 経由では一生使われない（設計メモ 実装 5）
- ``EKF_PROFILE`` を指定していなければ、環境変数のスカラー（今までの挙動）のまま
- 指定していても dt が合わない・BPF が有効なら同梱既定値に落とし、理由を残す（決定 6・7）
- 系列はランドマーク ID の昇順に並べる。``pose_keypoints`` は宣言順が ID 順ではないので、
  そのまま渡すと系列が黙って取り違えられる
- ディレクトリを渡したとき、生 CSV のサイドカー（``kpts3d_raw_*.json``）を拾わない。
  サイドカーも ``schema_version`` / ``frame`` / ``unit`` が同じなので検査を通ってしまっていた
- 体格の比（L_run / L_cal）は、人体としてありえない長さなら例外にする。r が大きすぎると
  静かに素通りするので、黙って使わない（欠陥 5）
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from app.tuning.ekf_estimate import SeriesFit
from app.tuning.ekf_profile import (
    body_scale_ratio,
    build_profile,
    builtin_entry,
    read_profile,
    runtime_noise,
    write_profile,
)
from app.tuning.raw_capture import RawCapture
from extended_kalman_filter import EKFConfig, SeriesNoise

REPO_ROOT = Path(__file__).resolve().parents[1]
IDS = (11, 12)
DT = 1 / 30
SCALAR = EKFConfig(q_acc=1e-3, r=1e-3, gate_std=3.0)


def _profile_dir(tmp_path, dt=DT, q_by_id=None):
    """点ごとに q を変えた較正プロファイルを 1 つ置く。"""
    q_by_id = q_by_id or {11: 0.2, 12: 0.4}
    rng = np.random.default_rng(0)
    n = 400
    capture = RawCapture(
        landmark_ids=IDS, frame=np.arange(n), t=np.arange(n) * dt,
        points=rng.normal(0, 0.005, (n, len(IDS), 3)),
        provenance={"dt": dt, "src_fps": 1 / dt, "stage": "pre_ekf", "landmark_ids": list(IDS)},
    )
    fits = {(lid, axis): SeriesFit(q_acc=q_by_id[lid], r=3e-5, loglik=0.0, n_eff=n, at_bound=False, rho=(0.0,) * 5)
            for lid in IDS for axis in "xyz"}
    write_profile(tmp_path / f"ekf_profile_{dt:.5f}.json", build_profile(capture, fits))
    return tmp_path


class TestRuntimeNoise:
    def test_without_a_profile_the_scalar_settings_are_used(self):
        noise = runtime_noise(None, dt=DT, bpf_enabled=False, landmark_ids=IDS, scalar=SCALAR)
        assert noise.origin == "env"
        assert noise.cfg is SCALAR, "EKF_PROFILE 未指定なのに今までのスカラーを使っていない"

    def test_a_matching_profile_wins_over_the_scalars(self, tmp_path):
        noise = runtime_noise(_profile_dir(tmp_path), dt=DT, bpf_enabled=False, landmark_ids=IDS, scalar=SCALAR)
        assert noise.origin == "profile"
        assert isinstance(noise.cfg, SeriesNoise)
        np.testing.assert_allclose(noise.cfg.q_acc, [0.2] * 3 + [0.4] * 3)

    def test_series_follow_ascending_landmark_ids(self, tmp_path):
        """pose_keypoints の宣言順（12 が 11 より先）で渡しても、系列は ID の昇順に並ぶ。"""
        noise = runtime_noise(_profile_dir(tmp_path), dt=DT, bpf_enabled=False, landmark_ids=(12, 11), scalar=SCALAR)
        np.testing.assert_allclose(noise.cfg.q_acc, [0.2] * 3 + [0.4] * 3)

    def test_a_dt_mismatch_falls_back_to_builtin_with_a_reason(self, tmp_path):
        noise = runtime_noise(_profile_dir(tmp_path), dt=8 / 30, bpf_enabled=False, landmark_ids=IDS, scalar=SCALAR)
        assert noise.origin == "builtin"
        assert noise.resolution.reason == "dt_mismatch"
        np.testing.assert_allclose(noise.cfg.r, builtin_entry(8 / 30).r)

    def test_bandpass_falls_back_to_builtin(self, tmp_path):
        noise = runtime_noise(_profile_dir(tmp_path), dt=DT, bpf_enabled=True, landmark_ids=IDS, scalar=SCALAR)
        assert noise.origin == "builtin" and noise.resolution.reason == "bpf_enabled"

    def test_the_breakdown_of_sources_is_reported(self, tmp_path):
        noise = runtime_noise(_profile_dir(tmp_path), dt=DT, bpf_enabled=False, landmark_ids=(11, 12, 13),
                              scalar=SCALAR)
        # 11・12 は fit、プロファイルに無い 13 は他の系列の中央値
        assert noise.sources == {"fit": 6, "global_median": 3}
        assert "fit=6" in noise.describe()

    def test_provenance_is_json_serialisable(self, tmp_path):
        noise = runtime_noise(_profile_dir(tmp_path), dt=DT, bpf_enabled=False, landmark_ids=IDS, scalar=SCALAR)
        record = json.loads(json.dumps(noise.provenance()))
        assert record["origin"] == "profile" and record["reason"] is None


class TestCandidates:
    def test_a_raw_capture_sidecar_in_the_same_folder_is_not_picked(self, tmp_path):
        folder = _profile_dir(tmp_path)
        sidecar = {"schema_version": 1, "frame": "runtime", "unit": "m", "dt": DT, "stage": "pre_ekf"}
        (folder / "kpts3d_raw_0923_120000.json").write_text(json.dumps(sidecar), encoding="utf-8")
        noise = runtime_noise(folder, dt=DT, bpf_enabled=False, landmark_ids=IDS, scalar=SCALAR)
        assert noise.origin == "profile"
        assert noise.resolution.path.name.startswith("ekf_profile_")

    def test_a_sidecar_given_directly_is_rejected(self, tmp_path):
        path = tmp_path / "kpts3d_raw_0923_120000.json"
        path.write_text(json.dumps({"schema_version": 1, "frame": "runtime", "unit": "m", "dt": DT}), encoding="utf-8")
        with pytest.raises(ValueError, match="series"):
            read_profile(path)


class TestBodyScale:
    def test_the_ratio_uses_the_profile_reference_length(self):
        assert body_scale_ratio({"pair": [12, 14], "median_len": 0.30}, 0.33) == pytest.approx(1.1)

    def test_no_reference_length_means_no_scaling(self):
        assert body_scale_ratio(None, 0.33) == 1.0

    @pytest.mark.parametrize("length", [0.003, 3.0, float("nan")])
    def test_an_implausible_length_is_an_error(self, length):
        """calib.py で校正し直すと座標が m の 1/100 になる（欠陥 5）。黙って 100 倍の比を使わない。"""
        with pytest.raises(ValueError):
            body_scale_ratio({"pair": [12, 14], "median_len": 0.30}, length)


@pytest.fixture(scope="module")
def module():
    return ast.parse((REPO_ROOT / "master_research_code.py").read_text(encoding="utf-8"))


class TestRuntimeWiring:
    """``master_research_code.py``（import できない）の配線を AST で確かめる。"""

    @staticmethod
    def _first_line(module, predicate):
        return min((node.lineno for node in ast.walk(module) if predicate(node)), default=None)

    def test_the_filter_is_built_after_the_sample_interval_is_known(self, module):
        dt_line = self._first_line(module, lambda n: isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Tuple) and any(getattr(e, "id", None) == "_DYN_DT" for e in t.elts) for t in n.targets))
        ekf_line = self._first_line(module, lambda n: isinstance(n, ast.Call) and getattr(n.func, "id", None) == "LandmarkEKF")
        raw_line = self._first_line(module, lambda n: isinstance(n, ast.Call) and getattr(n.func, "id", None) == "RawCaptureWriter")
        assert dt_line and ekf_line and raw_line
        assert dt_line < ekf_line < raw_line, "EKF を _DYN_DT の算出後・生 CSV の書き出し前に作っていない"

    def test_the_bandpass_sampling_rate_comes_from_the_sample_interval(self, module):
        call = next(n for n in ast.walk(module) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "LandmarkEKF")
        fs = next(k.value for k in call.keywords if k.arg == "fs")
        assert "_DYN_DT" in ast.unparse(fs), "BPF の fs がカメラの fps のまま（間引きで 8 倍ずれる）"

    def test_runtime_noise_is_resolved(self, module):
        assert any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "runtime_noise" for n in ast.walk(module))
