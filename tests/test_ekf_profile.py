"""較正プロファイルの読み書き・dt による探索・フォールバックを固定する。

**なぜこのテストがあるか。**

系列ごとに推定した ``(q_acc, r, gate_std)`` は、実行時に読み込めて初めて意味を持つ。
そこで起きる事故は次の 3 つで、どれも「静かに間違う」形をしている。

- **dt の食い違い。** 実行時の dt は間引き設定で 0.0333 s と 0.267 s の間を 8 倍変わる
  （設計メモ 欠陥 2）。合わない較正値をそのまま使うと、指標上は正常に見えたまま
  平滑化がほぼ効かなくなる。計測は止めず、dt の合うファイルを探し、
  無ければ同梱既定値に落として**理由を記録する**（決定 6）
- **校正のやり直しでスケールが変わる。** 再構成スケールは校正手法で変わる（欠陥 5）ので、
  基準長の比を ``q`` と ``r`` に 2 乗で効かせる
- **系列の取りこぼし。** 点の構成が変われば、プロファイルに無い系列が出る。
  その系列だけフォールバックし、ファイル全体は捨てない

``schema_version`` / ``frame`` / ``unit`` の食い違いだけは、運用では起こらない
「壊れたファイル」なので例外にする。

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の「実装 4」（S7）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.tuning.ekf_estimate import SeriesFit
from app.tuning.ekf_profile import (
    BUILTIN_DEFAULTS,
    SCHEMA_VERSION,
    build_profile,
    builtin_entry,
    read_profile,
    resolve_profile,
    write_profile,
)
from app.tuning.raw_capture import RawCapture

IDS = (11, 12)
DT = 1 / 30
N_FRAMES = 400


def _capture(dt: float = DT, seed: int = 0) -> RawCapture:
    rng = np.random.default_rng(seed)
    t = np.arange(N_FRAMES) * dt
    wave = np.sin(2 * np.pi * 0.5 * t)[:, None, None] * 0.1
    points = wave + rng.normal(0.0, 0.005, (N_FRAMES, len(IDS), 3))
    return RawCapture(
        landmark_ids=IDS,
        frame=np.arange(N_FRAMES),
        t=t,
        points=points,
        provenance={"dt": dt, "src_fps": 1 / dt, "stage": "pre_ekf", "landmark_ids": list(IDS)},
    )


def _fit(q_acc: float = 0.122, r: float = 2.59e-5, n_eff: int = N_FRAMES, at_bound: bool = False) -> SeriesFit:
    return SeriesFit(q_acc=q_acc, r=r, loglik=1234.5, n_eff=n_eff, at_bound=at_bound, rho=(0.05,) * 5)


def _all_good_fits() -> dict[tuple[int, str], SeriesFit | None]:
    return {(lid, axis): _fit() for lid in IDS for axis in "xyz"}


class TestBuildProfile:
    """推定結果を、実行時が読める形に落とす。"""

    def test_records_what_the_values_are_only_valid_for(self):
        profile = build_profile(_capture(), _all_good_fits())

        assert profile["schema_version"] == SCHEMA_VERSION
        assert profile["frame"] == "runtime", "軸の入れ替え後の座標系であることが記録されていない"
        assert profile["unit"] == "m"
        assert profile["dt"] == pytest.approx(DT), "較正に使った dt が記録されていない"
        assert profile["series"]["11"]["x"]["source"] == "fit"

    def test_calibrates_the_gate_per_series_instead_of_using_the_constant(self):
        # ゲート無しで推定した r は外れ値を吸収して過大になるので、固定 3.0 だと門が広くなる
        entry = build_profile(_capture(), _all_good_fits())["series"]["11"]["x"]

        assert entry["gate_std"] > 0.0
        assert entry["gate_std"] != pytest.approx(3.0), "正規化イノベーションから決めず、定数 3.0 のままになっている"

    def test_falls_back_to_the_other_axes_of_the_same_point(self):
        fits = _all_good_fits()
        fits[(11, "y")] = _fit(q_acc=9.9, r=9.9e-3, n_eff=10)  # 有効サンプルが足りない
        series = build_profile(_capture(), fits)["series"]["11"]

        assert series["y"]["source"] == "axis_median"
        assert series["y"]["q_acc"] == pytest.approx(np.median([series["x"]["q_acc"], series["z"]["q_acc"]]))

    def test_falls_back_to_the_global_median_when_the_whole_point_is_unusable(self):
        fits = _all_good_fits()
        for axis in "xyz":
            fits[(12, axis)] = None  # 一度も検出されなかった点
        series = build_profile(_capture(), fits)["series"]["12"]

        assert all(series[axis]["source"] == "global_median" for axis in "xyz")
        assert series["x"]["q_acc"] == pytest.approx(0.122, rel=0.5)

    def test_marks_a_fit_that_stuck_to_the_search_boundary_as_unusable(self):
        fits = _all_good_fits()
        fits[(11, "z")] = _fit(at_bound=True)
        assert build_profile(_capture(), fits)["series"]["11"]["z"]["source"] != "fit"

    def test_the_gate_sees_dropped_frames_as_missing_rows(self):
        """取りこぼした行（frame 番号の抜け）は、門を決めるイノベーションでも推定と同じく NaN の行として扱う。
        行を詰めたままだと抜けの前後が 1 dt に縮み、正規化イノベーションが大きく出て門が広がる。"""
        full = _capture()
        keep = np.flatnonzero(np.arange(N_FRAMES) % 10 != 3)   # 10 行に 1 行を取りこぼした
        packed = RawCapture(landmark_ids=IDS, frame=full.frame[keep], t=full.t[keep], points=full.points[keep],
                            provenance=full.provenance)
        holed = full.points.copy()
        holed[np.arange(N_FRAMES) % 10 == 3] = np.nan
        filled = RawCapture(landmark_ids=IDS, frame=full.frame, t=full.t, points=holed, provenance=full.provenance)
        gate = build_profile(packed, _all_good_fits())["series"]["11"]["x"]["gate_std"]
        assert gate == pytest.approx(build_profile(filled, _all_good_fits())["series"]["11"]["x"]["gate_std"])


class TestFileFormat:
    """壊れたファイルは読んだ時点で止める。"""

    def test_write_then_read_keeps_the_values(self, tmp_path):
        profile = build_profile(_capture(), _all_good_fits())
        path = tmp_path / "ekf_profile.json"
        write_profile(path, profile)

        assert read_profile(path) == profile

    @pytest.mark.parametrize(
        "key, value",
        [("schema_version", SCHEMA_VERSION + 1), ("frame", "mediapipe"), ("unit", "cm")],
        ids=["形式の版", "座標系", "単位"],
    )
    def test_rejects_a_profile_that_means_something_else(self, tmp_path, key, value):
        profile = build_profile(_capture(), _all_good_fits())
        profile[key] = value
        path = tmp_path / "ekf_profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")

        with pytest.raises(ValueError, match=key):
            read_profile(path)


class TestResolve:
    """実行時に、その場の dt に合う較正値を選ぶ。合わなければ計測を止めずに既定値へ落とす。"""

    @staticmethod
    def _profile_dir(tmp_path):
        for dt in (DT, 8 / 30):
            profile = build_profile(_capture(dt=dt), _all_good_fits())
            write_profile(tmp_path / f"ekf_profile_{dt:.5f}.json", profile)
        return tmp_path

    def test_picks_the_file_whose_dt_matches(self, tmp_path):
        resolution = resolve_profile(self._profile_dir(tmp_path), dt=8 / 30)

        assert resolution.reason is None, "dt の合うファイルがあるのに既定値へ落ちている"
        assert resolution.dt == pytest.approx(8 / 30)
        assert resolution.entries[(11, "x")].source == "fit"

    def test_tolerates_a_small_difference_in_the_camera_rate(self, tmp_path):
        # 29.97 fps のカメラでは dt が 0.1% ずれる。較正をやり直す話ではない
        resolution = resolve_profile(self._profile_dir(tmp_path), dt=1 / 29.97)

        assert resolution.reason is None
        assert resolution.dt == pytest.approx(DT)

    def test_falls_back_to_builtin_when_no_file_matches_the_dt(self, tmp_path):
        resolution = resolve_profile(self._profile_dir(tmp_path), dt=1 / 120)
        noise = resolution.series_noise(IDS)

        assert resolution.reason == "dt_mismatch", "dt が合わないのに理由が記録されていない"
        assert noise.r[0] == pytest.approx(builtin_entry(1 / 120).r), "既定値に落ちていない"
        assert noise.q_acc[0] == pytest.approx(builtin_entry(1 / 120).q_acc)

    def test_falls_back_to_builtin_when_the_bandpass_is_enabled(self, tmp_path):
        # BPF は位置の DC を落とす。較正時と同じ前処理を再現しない以上、較正値は当てにならない
        resolution = resolve_profile(self._profile_dir(tmp_path), dt=DT, bpf_enabled=True)
        noise = resolution.series_noise(IDS)

        assert resolution.reason == "bpf_enabled"
        assert noise.r[0] == pytest.approx(builtin_entry(DT).r), "BPF 有効時に較正値を使っている"

    def test_builtin_defaults_exist_for_both_thinning_settings(self):
        # dt が合わないときの落ち先が 1 組しかないと、落ちた先も dt 違いになる（決定 6 の帰結）
        assert set(np.round(list(BUILTIN_DEFAULTS), 4)) == {round(1 / 30, 4), round(8 / 30, 4)}
        assert builtin_entry(8 / 30).q_acc == BUILTIN_DEFAULTS[8 / 30].q_acc


class TestSeriesNoiseForRuntime:
    """実行時の配列（点 × 3 + 軸）へ渡す。"""

    def test_maps_entries_to_point_times_three_plus_axis(self, tmp_path):
        profile = build_profile(_capture(), _all_good_fits())
        write_profile(tmp_path / "p.json", profile)
        resolution = resolve_profile(tmp_path / "p.json", dt=DT)

        noise = resolution.series_noise(IDS)

        assert noise.r.shape == (len(IDS) * 3,)
        assert noise.r[0] == pytest.approx(resolution.entries[(11, "x")].r)
        assert noise.r[5] == pytest.approx(resolution.entries[(12, "z")].r)

    def test_a_landmark_missing_from_the_profile_does_not_discard_the_file(self, tmp_path):
        profile = build_profile(_capture(), _all_good_fits())
        write_profile(tmp_path / "p.json", profile)
        resolution = resolve_profile(tmp_path / "p.json", dt=DT)

        noise = resolution.series_noise((11, 12, 99))  # 99 はプロファイルに無い点

        assert noise.r.shape == (9,)
        assert noise.r[0] == pytest.approx(resolution.entries[(11, "x")].r), "既知の系列まで捨てている"
        assert np.isfinite(noise.r[6:]).all(), "知らない点の値が埋まっていない"

    def test_a_different_body_scale_is_applied_squared(self, tmp_path):
        # 校正のやり直しで再構成スケールが変わる（欠陥 5）。長さ比 s に対し q も r も s² 倍
        profile = build_profile(_capture(), _all_good_fits())
        write_profile(tmp_path / "p.json", profile)

        plain = resolve_profile(tmp_path / "p.json", dt=DT)
        scaled = resolve_profile(tmp_path / "p.json", dt=DT, scale_ratio=2.0)

        assert scaled.entries[(11, "x")].r == pytest.approx(plain.entries[(11, "x")].r * 4.0)
        assert scaled.entries[(11, "x")].q_acc == pytest.approx(plain.entries[(11, "x")].q_acc * 4.0)
