"""収録から較正プロファイルを作るコマンドを固定する。

**なぜこのテストがあるか。**

較正は「各自が自分の収録から作る」方式（設計メモ 決定 2）なので、収録 → プロファイルの
1 手が用意されていなければ誰も使わない。GUI の解析タスク（S11）もこのコマンドを呼ぶ。

出力のファイル名に dt を入れるのは、実行時の探索が **dt でファイルを選ぶ**ため
（決定 6）。間引き設定ごとに録ったプロファイルが、同じフォルダに並んで共存できる。

入力は EKF の**手前**で書いた生 CSV だけを受け付ける。EKF を通った値で較正すると、
平滑化済みの系列に対する推定になり、実行時の EKF に入れる値として意味をなさない。
判定はサイドカーの ``stage``（実装 0）で行う。

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の S7。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.runners.tune_ekf import main
from app.tuning.ekf_profile import read_profile
from app.tuning.raw_capture import RawCaptureWriter, sidecar_path

IDS = [11, 12]
DT = 8 / 30
N_FRAMES = 400


def _recording(tmp_path, *, with_sidecar: bool = True):
    rng = np.random.default_rng(0)
    t = np.arange(N_FRAMES) * DT
    points = np.sin(2 * np.pi * 0.1 * t)[:, None, None] * 0.1 + rng.normal(0, 0.005, (N_FRAMES, len(IDS), 3))

    csv_path = tmp_path / "kpts3d_raw_0101_000001.csv"
    writer = RawCaptureWriter(csv_path, IDS, provenance={"dt": DT, "src_fps": 30.0})
    for k in range(N_FRAMES):
        writer.append(8 * k, float(t[k]), points[k])
    writer.close()
    if not with_sidecar:
        sidecar_path(csv_path).unlink()
    return csv_path


class TestTuneEkfCommand:
    def test_writes_a_profile_named_after_the_dt(self, tmp_path, capsys):
        csv_path = _recording(tmp_path)

        assert main([str(csv_path)]) == 0
        written = sorted(tmp_path.glob("ekf_profile_*.json"))

        assert len(written) == 1, "プロファイルが収録の隣に 1 つだけ作られていない"
        assert f"{DT:.5f}" in written[0].name, "ファイル名に dt が入っておらず、実行時の探索が選べない"
        profile = read_profile(written[0])
        assert profile["dt"] == pytest.approx(DT)
        assert set(profile["series"]) == {"11", "12"}
        assert str(written[0]) in capsys.readouterr().out, "どこに書いたかが表示されない"

    def test_out_option_chooses_the_destination(self, tmp_path):
        destination = tmp_path / "custom" / "profile.json"

        assert main([str(_recording(tmp_path)), "--out", str(destination)]) == 0
        assert read_profile(destination)["dt"] == pytest.approx(DT)

    def test_refuses_a_recording_that_is_not_from_before_the_ekf(self, tmp_path, capsys):
        csv_path = _recording(tmp_path, with_sidecar=False)

        assert main([str(csv_path)]) == 2, "較正に使えない入力を受け付けている"
        assert "pre_ekf" in capsys.readouterr().err, "なぜ使えないかが表示されない"


# ---------------------------------------------------------------------------
# 混成の収録と --out のフォルダ（B7）
# ---------------------------------------------------------------------------
#
# なぜこのテストがあるか: --out にフォルダを渡すと IsADirectoryError で落ちていた（ファイル名を足していなかった）。
# 混成の計測は EKF の手前の生 3D を 1/30 s の格子で計測フォルダに書く。プロファイルを計測フォルダの隣に置くと、
# 次の計測の設定（HYBRID_EKF_PROFILE）に何を入れればよいか分からないので、決まった場所（hybrid/ekf_profiles）に
# 書いて設定の値を案内する。実行時の探索は dt の相対差 5% 以内のプロファイルしか選ばないので、1/30 s から
# 外れた収録（hybrid-raw の Pixel の時刻など）で作ったものは混成の計測では使われない。それを警告する。


def _hybrid_recording(tmp_path, dt=1 / 30):
    rng = np.random.default_rng(1)
    t = np.arange(N_FRAMES) * dt
    points = np.sin(2 * np.pi * 0.5 * t)[:, None, None] * 0.1 + rng.normal(0, 0.003, (N_FRAMES, len(IDS), 3))
    csv_path = tmp_path / "measure" / "20260924_070000_000000" / "kpts3d_raw_20260924_070000_000000.csv"
    csv_path.parent.mkdir(parents=True)
    writer = RawCaptureWriter(csv_path, IDS, provenance={"dt": dt, "src_fps": 30.0, "source": "hybrid", "times": "grid"})
    for k in range(N_FRAMES):
        writer.append(k, float(t[k]), points[k])
    writer.close()
    return csv_path


class TestTuneEkfDestinations:
    def test_an_existing_folder_gets_the_default_name(self, tmp_path):
        folder = tmp_path / "profiles"
        folder.mkdir()
        assert main([str(_recording(tmp_path)), "--out", str(folder)]) == 0
        assert read_profile(folder / f"ekf_profile_{DT:.5f}.json")["dt"] == pytest.approx(DT)

    def test_a_new_path_without_suffix_is_a_folder(self, tmp_path):
        folder = tmp_path / "new_profiles"
        assert main([str(_recording(tmp_path)), "--out", str(folder)]) == 0
        assert (folder / f"ekf_profile_{DT:.5f}.json").is_file()

    def test_a_hybrid_recording_goes_to_the_profile_folder(self, tmp_path, monkeypatch, capsys):
        from app.runners import tune_ekf

        root = tmp_path / "hybrid" / "ekf_profiles"
        monkeypatch.setattr(tune_ekf, "ekf_profile_root", lambda: root)
        assert main([str(_hybrid_recording(tmp_path))]) == 0
        written = root / f"ekf_profile_{1 / 30:.5f}.json"
        assert written.is_file()
        out = capsys.readouterr().out
        assert "HYBRID_EKF_PROFILE" in out and str(written) in out
        assert "警告" not in out

    def test_a_hybrid_recording_off_the_grid_is_warned(self, tmp_path, monkeypatch, capsys):
        from app.runners import tune_ekf

        monkeypatch.setattr(tune_ekf, "ekf_profile_root", lambda: tmp_path / "ekf_profiles")
        assert main([str(_hybrid_recording(tmp_path, dt=1 / 12))]) == 0
        assert "警告" in capsys.readouterr().out

    def test_the_profile_root_is_under_the_hybrid_root(self):
        from app.hybrid.paths import ekf_profile_root, hybrid_root

        assert ekf_profile_root() == hybrid_root() / "ekf_profiles"


# ---------------------------------------------------------------------------
# hybrid-raw の出力は実行時の置き場へ書かない
# ---------------------------------------------------------------------------
#
# なぜこのテストがあるか: tools.verify_run hybrid-raw が記録から作り直した生 CSV（kpts3d_raw_<stamp>_retri*.csv）も
# サイドカーの source が hybrid だったので、tune_ekf はそのプロファイルを実行時の hybrid/ekf_profiles/ に書いた。
# --grid の出力は dt が 1/30 s で、記録器の kpts3d_raw_<stamp>.csv から作ったプロファイルを同じ名前で上書きしうる。
# 実行時のプロファイルは記録器の生 CSV からだけ作り、hybrid-raw の出力（source が hybrid_retri）は比べる用として
# 収録の隣に書く。


def _retri_recording(tmp_path, provenance):
    rng = np.random.default_rng(2)
    t = np.arange(N_FRAMES) / 30
    points = np.sin(2 * np.pi * 0.5 * t)[:, None, None] * 0.1 + rng.normal(0, 0.003, (N_FRAMES, len(IDS), 3))
    csv_path = tmp_path / "session" / "kpts3d_raw_20260924_070000_000000_retri_grid.csv"
    csv_path.parent.mkdir()
    writer = RawCaptureWriter(csv_path, IDS, provenance=dict({"dt": 1 / 30, "src_fps": 30.0, "times": "grid"},
                                                             **provenance))
    for k in range(N_FRAMES):
        writer.append(k, float(t[k]), points[k])
    writer.close()
    return csv_path


class TestRetriangulatedCaptures:
    @pytest.mark.parametrize("provenance", [
        {"source": "hybrid_retri", "hybrid_session": "/x"},
        {"source": "hybrid", "hybrid_session": "/x"},   # 直す前の hybrid-raw が書いたサイドカー
    ])
    def test_a_hybrid_raw_output_stays_next_to_the_capture(self, tmp_path, monkeypatch, capsys, provenance):
        from app.runners import tune_ekf

        root = tmp_path / "hybrid" / "ekf_profiles"
        monkeypatch.setattr(tune_ekf, "ekf_profile_root", lambda: root)
        csv_path = _retri_recording(tmp_path, provenance)
        assert main([str(csv_path)]) == 0
        assert not root.exists(), "hybrid-raw の出力のプロファイルを実行時の置き場に書いた"
        assert (csv_path.parent / f"ekf_profile_{1 / 30:.5f}.json").is_file()
        out = capsys.readouterr().out
        assert "HYBRID_EKF_PROFILE" not in out
        assert "kpts3d_raw_<stamp>.csv" in out, "実行時のプロファイルを何から作るかを案内していない"

    def test_the_hybrid_raw_output_is_marked(self, tmp_path):
        from app.tuning.raw_capture import read_raw_capture
        from test_hybrid_verification import make_hybrid_run
        from tools import verify_run as vr

        path = vr.hybrid_raw_capture(make_hybrid_run(tmp_path), grid=True)
        assert read_raw_capture(path).provenance["source"] == "hybrid_retri"
