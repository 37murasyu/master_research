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
