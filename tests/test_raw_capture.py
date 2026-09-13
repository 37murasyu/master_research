"""EKF の手前の生 3D 座標を、較正に使える形で残せることを固定する。

**なぜこのテストがあるか。**

EKF の ``(q_acc, r)`` を最尤推定するには、EKF を通す**前**の 3D 座標が要る。
既存の ``kpts3d_{timestamp}.csv`` は EKF 後の値を 4 桁に丸めたもので、
しかも計測ループを抜けた後にまとめて書いている（``master_research_code.py`` の終了時処理）。

GUI の停止は ``QProcess.terminate()``（``app/runners/worker.py``）で、macOS/Linux では
SIGTERM になる。リポジトリに SIGTERM ハンドラは無く、Python の既定ではこのとき
終了時処理も atexit も走らない。**終了時にまとめて書く方式では、GUI から止めた試技の
生データが残らない。** そこで 1 行ごとに flush し、由来を記すサイドカー JSON は起動時に書く。

設計: ``docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`` の「実装 0」（S1）。
"""

from __future__ import annotations

import ast
import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from app.tuning.raw_capture import RawCaptureWriter, git_commit, read_raw_capture, sidecar_path

REPO_ROOT = Path(__file__).resolve().parent.parent
IDS = [11, 12, 13]


def _points(value: float) -> np.ndarray:
    return np.full((len(IDS), 3), value, dtype=float)


class TestSidecar:
    """由来の記録は、フレームが 1 つも来ないうちに書かれていなければならない。"""

    def test_sidecar_is_written_before_any_frame(self, tmp_path):
        csv_path = tmp_path / "kpts3d_raw_0913_120000.csv"
        RawCaptureWriter(csv_path, IDS, provenance={"dt": 1 / 30})

        meta = json.loads(sidecar_path(csv_path).read_text(encoding="utf-8"))
        assert meta["stage"] == "pre_ekf", "較正ツールが入力を判定する stage が記録されていない"
        assert meta["landmark_ids"] == IDS, "列の並び（ランドマーク ID）が記録されていない"
        assert meta["dt"] == pytest.approx(1 / 30), "呼び出し側から渡した由来が記録されていない"

    def test_sidecar_sits_next_to_the_csv_with_the_same_stem(self, tmp_path):
        csv_path = tmp_path / "kpts3d_raw_0913_120000.csv"
        assert sidecar_path(csv_path) == tmp_path / "kpts3d_raw_0913_120000.json"

    def test_git_commit_identifies_the_code_that_recorded(self):
        # 収録後にコードが変わっても、どの版の三角測量・軸変換で録ったかを辿れるようにする
        commit = git_commit(REPO_ROOT)
        assert commit is not None and len(commit) == 40, "リポジトリ内でコミットを取れていない"
        int(commit, 16)

    def test_git_commit_is_none_outside_a_repository(self, tmp_path):
        # 配布版（凍結 exe）には .git が無い。記録のために計測を止めてはいけない
        assert git_commit(tmp_path) is None


class TestStreaming:
    """途中で強制終了されても、それまでの行が読めること。"""

    def test_rows_are_readable_without_close(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        writer = RawCaptureWriter(csv_path, IDS, provenance={})
        writer.append(0, 0.0, _points(1.0))
        writer.append(8, 0.2667, _points(2.0))
        # close() を呼ばない。SIGTERM で落ちたのと同じ状態

        capture = read_raw_capture(csv_path)
        assert capture.points.shape == (2, len(IDS), 3), "flush されておらず、閉じる前の行が読めない"
        np.testing.assert_array_equal(capture.frame, [0, 8])
        np.testing.assert_allclose(capture.t, [0.0, 0.2667])

    def test_values_are_not_rounded_and_nan_survives(self, tmp_path):
        # 推定する r は σ で数 mm。既存 CSV の 4 桁丸め（0.1 mm）とは別に、丸め自体を持ち込まない
        csv_path = tmp_path / "raw.csv"
        points = _points(0.123456789012345)
        points[1] = np.nan  # 検出されなかった点は NaN のまま残す（欠測長の実測に使う）
        writer = RawCaptureWriter(csv_path, IDS, provenance={})
        writer.append(0, 0.0, points)
        writer.close()

        got = read_raw_capture(csv_path).points[0]
        assert got[0, 0] == points[0, 0], "値が丸められている"
        assert np.isnan(got[1]).all(), "欠測（NaN）が保存・復元されていない"

    def test_append_rejects_a_wrong_number_of_points(self, tmp_path):
        writer = RawCaptureWriter(tmp_path / "raw.csv", IDS, provenance={})
        with pytest.raises(ValueError):
            writer.append(0, 0.0, np.zeros((len(IDS) + 1, 3)))


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM の既定動作は POSIX の前提")
class TestTermination:
    """GUI の停止（SIGTERM）で落ちても、そこまでの行が残ること。

    同時に「SIGTERM では終了時処理が走らない」という設計の前提そのものも確かめる。
    この前提が崩れたら（誰かがハンドラを入れたら）、終了時保存でも足りるようになる。
    """

    def test_rows_survive_sigterm_while_end_of_run_code_never_runs(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        marker = tmp_path / "end_of_run.txt"
        child = textwrap.dedent(
            f"""
            import sys, time
            import numpy as np
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from app.tuning.raw_capture import RawCaptureWriter

            writer = RawCaptureWriter({str(csv_path)!r}, {IDS!r}, provenance={{}})
            for i in range(3):
                writer.append(i, i / 30, np.zeros(({len(IDS)}, 3)))
            print("ready", flush=True)
            time.sleep(60)
            # 計測ループを抜けた後の保存処理に相当
            open({str(marker)!r}, "w").write("done")
            """
        )
        proc = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout.readline().strip() == "ready", "子プロセスが書き出しまで進まなかった"
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.stdout.close()

        assert not marker.exists(), "SIGTERM で終了時処理が走った。前提が変わっている"
        assert read_raw_capture(csv_path).points.shape[0] == 3, "SIGTERM で落ちた後に flush 済みの行が残っていない"


class TestReader:
    """較正の入力として使ってよいファイルかを、事実（サイドカー）で判定する。"""

    def test_rejects_a_csv_without_sidecar(self, tmp_path):
        csv_path = tmp_path / "kpts3d_0913_120000.csv"
        csv_path.write_text("frame,t,11_x,11_y,11_z\n0,0.0,1,2,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="pre_ekf"):
            read_raw_capture(csv_path)

    def test_rejects_a_sidecar_from_another_stage(self, tmp_path):
        csv_path = tmp_path / "raw.csv"
        RawCaptureWriter(csv_path, IDS, provenance={}).append(0, 0.0, _points(1.0))
        meta_path = sidecar_path(csv_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["stage"] = "post_ekf"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="pre_ekf"):
            read_raw_capture(csv_path)

    def test_drops_a_line_cut_off_by_a_kill(self, tmp_path):
        # 書き込みの途中で kill されると、最終行が改行なしで途切れる
        csv_path = tmp_path / "raw.csv"
        writer = RawCaptureWriter(csv_path, IDS, provenance={})
        writer.append(0, 0.0, _points(1.0))
        writer.close()
        with csv_path.open("a", encoding="utf-8") as fh:
            fh.write("8,0.2667,1.0,1.0")

        capture = read_raw_capture(csv_path)
        assert capture.points.shape[0] == 1, "途切れた最終行を読み飛ばしていない"

    def test_rejects_a_broken_line_in_the_middle(self, tmp_path):
        # 途中の壊れた行は kill では生じないので、黙って捨てずに知らせる
        csv_path = tmp_path / "raw.csv"
        writer = RawCaptureWriter(csv_path, IDS, provenance={})
        writer.append(0, 0.0, _points(1.0))
        writer.close()
        with csv_path.open("a", encoding="utf-8") as fh:
            fh.write("8,0.2667,1.0\n")
            fh.write("16,0.5333," + ",".join(["1.0"] * (len(IDS) * 3)) + "\n")
        with pytest.raises(ValueError):
            read_raw_capture(csv_path)


class TestRuntimeWiring:
    """計測スクリプトが、三角測量の直後・EKF の手前で生データを書くこと。

    ``master_research_code.py`` は import するとカメラを開くので、ast で順序だけを見る。
    EKF の後ろで書くと、較正に EKF 済みの値が混ざり、推定がすべて無意味になる。
    """

    def test_raw_rows_are_written_between_triangulation_and_the_ekf(self):
        tree = ast.parse((REPO_ROOT / "master_research_code.py").read_text(encoding="utf-8"))
        loop = next(
            node for node in tree.body
            if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True
        )

        def first_call_line(predicate) -> int | None:
            lines = [n.lineno for n in ast.walk(loop) if isinstance(n, ast.Call) and predicate(n.func)]
            return min(lines) if lines else None

        triangulate = first_call_line(lambda f: isinstance(f, ast.Name) and f.id == "_triangulate_transform_batch")
        raw_append = first_call_line(
            lambda f: isinstance(f, ast.Attribute) and f.attr == "append"
            and isinstance(f.value, ast.Name) and f.value.id == "_raw_capture"
        )
        ekf_step = first_call_line(
            lambda f: isinstance(f, ast.Attribute) and f.attr == "step"
            and isinstance(f.value, ast.Name) and f.value.id == "landmark_ekf"
        )

        assert raw_append is not None, "計測ループに _raw_capture.append の呼び出しが無い"
        assert triangulate < raw_append < ekf_step, (
            f"生データの書き出し（:{raw_append}）が三角測量（:{triangulate}）と "
            f"EKF（:{ekf_step}）の間に無い"
        )
