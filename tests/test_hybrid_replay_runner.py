"""記録を流し直す入口（``app.runners.hybrid_replay``）が、計測の子と同じ ``@@GAUGE`` の行を出すことを確かめる。

**なぜこのテストがあるか。**

GUI のゲージ窓は子の標準出力の ``@@GAUGE`` の行だけを見て動く（設計書 §6.2）。被験者がいない夜や朝の練習で、
記録を流し直してゲージ窓が動くのを確かめるには、再生の入口が本番の子と同じ行（UI 側の ``protocol.decode`` で
読める形、source は「再生」）を出し、押し上げ 1 回ごとに回が進み、論文の閾値の帯が載っている必要がある。
"""

from __future__ import annotations

import pytest

import app.runners.hybrid_replay as runner
import tools.synth_session as ss
from app.gauge.protocol import decode


@pytest.fixture
def pushups(tmp_path):
    return ss.write_session(tmp_path / "measure", reps=3, pixel_hz=30.0, calibration_root=tmp_path / "calibration")


def test_replay_emits_gauge_lines_with_reps_and_bands(pushups, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "replay_root", lambda: tmp_path / "replay")
    monkeypatch.setenv("SUBJECT_ID", "00")
    monkeypatch.delenv("ONE_RM_CSV", raising=False)
    assert runner.main([str(pushups), "--speed", "0"]) == 0
    frames = [decode(line) for line in capsys.readouterr().out.splitlines() if line.startswith("@@GAUGE ")]
    frames = [f for f in frames if f is not None]
    assert frames, "@@GAUGE の行が 1 行も出ていない"
    assert {f.source for f in frames} == {"replay"}
    last = frames[-1]
    assert last.rep == 3, "押し上げ 3 回で回が 3 つ進んでいない"
    for part in ("elbow_L", "elbow_R", "wrist_L", "wrist_R"):
        reading = last.parts[part]
        assert reading.band is not None and reading.band[0] < reading.band[1], f"{part} に論文の帯が無い"
        assert reading.prev is not None and reading.prev > 0, f"{part} の前回の値が無い"


def test_missing_session_is_reported(tmp_path, capsys):
    assert runner.main([str(tmp_path / "nothing")]) == 2
    assert "計測フォルダではありません" in capsys.readouterr().err


@pytest.mark.parametrize("mass", ["nan", "inf", "-60", "0"])
def test_body_mass_must_be_a_positive_finite_number(tmp_path, mass):
    """計測（hybrid_measure）と同じ入口の検査。NaN を通すと記録の開始で meta.json を書けずに落ちる。"""
    with pytest.raises(SystemExit) as raised:
        runner.main([str(tmp_path), "--body-mass", mass])
    assert raised.value.code == 2


def test_what_to_replay_comes_only_from_the_arguments(tmp_path, monkeypatch):
    """何を流すかは引数だけで決まる。親のシェルに残った ``HYBRID_REPLAY*`` は効かない（GUI は引数で渡す）。"""
    monkeypatch.setenv("HYBRID_REPLAY", str(tmp_path))
    monkeypatch.setenv("HYBRID_REPLAY_FROM", "20")
    monkeypatch.setenv("HYBRID_REPLAY_TO", "90")
    monkeypatch.setenv("HYBRID_REPLAY_SPEED", "0")
    with pytest.raises(SystemExit) as raised:
        runner.main([])
    assert raised.value.code == 2, "計測フォルダを環境変数から読んだ"
    args = runner._parser().parse_args([str(tmp_path)])
    assert (args.start_s, args.end_s, args.speed) == (0.0, None, 1.0)


def test_gui_launch_takes_the_arguments_and_stops_on_the_stop_file(pushups, tmp_path, monkeypatch, capsys):
    """GUI は ``--role hybrid_replay`` に、設定から組み立てた引数（``page_measure.replay_arguments``）を付けて起動する。
    ``--to`` は設定が空なら付かない（終わりまで）。停止ボタンは停止ファイル。"""
    import json
    import threading
    import time

    monkeypatch.setattr(runner, "replay_root", lambda: tmp_path / "replay")
    stop_file = tmp_path / "stop"
    monkeypatch.setenv("APP_STOP_FILE", str(stop_file))
    timer = threading.Timer(1.0, stop_file.touch)
    timer.start()
    started = time.monotonic()
    try:
        # 速さ 1 は実時間。3 回の押し上げは 10 秒を超える
        assert runner.main([str(pushups), "--from", "0.0", "--speed", "1.0"]) == 0
    finally:
        timer.cancel()
    assert time.monotonic() - started < 6.0, "停止ファイルで止まっていない"
    out = capsys.readouterr().out
    folder = next(line.split("保存: ", 1)[1] for line in out.splitlines() if line.startswith("保存: "))
    meta = json.loads((tmp_path / "replay" / folder.rsplit("/", 1)[-1] / "meta.json").read_text(encoding="utf-8"))
    assert meta["stop_reason"] == "stop_request"
