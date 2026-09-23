"""``verify_run hybrid-raw`` の出力名が、計測中に書く生 CSV と衝突しないことを固定する（B1）。

**なぜこのテストがあるか。**

混成の計測は、EKF の手前の 3D を ``kpts3d_raw_<stamp>.csv``（1/30 s 格子、``RawCaptureWriter``）として計測フォルダに
書くようになった（USB と同じ名前）。``hybrid-raw`` も同じフォルダに ``kpts3d_raw_<stamp>.csv`` を書いていたので、
2D から三角測量し直した点で計測中の記録を黙って上書きしてしまう。上書きされると ``tune_ekf`` にかける元の記録と
``check`` の EKF の統計の材料が消える。道具の出力には ``_retri`` を付けて分ける。
"""

from __future__ import annotations

from test_hybrid_verification import make_body_run, make_hybrid_run
from tools import verify_run as vr


def _stamp(run):
    return next(run.glob("frames_*.csv")).stem[len("frames_"):]


def test_the_grid_output_does_not_overwrite_the_live_raw_capture(tmp_path):
    """計測中に書いた生 CSV（ここでは目印の文字列）が、変換の後も残っている。"""
    run = make_hybrid_run(tmp_path)
    stamp = _stamp(run)
    live = run / f"kpts3d_raw_{stamp}.csv"
    live.write_text("計測中の記録", encoding="utf-8")
    path = vr.hybrid_raw_capture(run, grid=True)
    assert path.name == f"kpts3d_raw_{stamp}_retri_grid.csv"
    assert live.read_text(encoding="utf-8") == "計測中の記録"


def test_the_stride_is_kept_after_the_retri_mark(tmp_path):
    path = vr.hybrid_raw_capture(make_hybrid_run(tmp_path), stride=8, grid=True)
    assert path.name.endswith("_retri_grid_s8.csv")


def test_the_retriangulated_output_is_marked(tmp_path):
    """既定（2D から三角測量し直す）の出力も ``_retri`` で終わる。"""
    run = make_body_run(tmp_path)
    path = vr.hybrid_raw_capture(run)
    assert path.name == f"kpts3d_raw_{_stamp(run)}_retri.csv"
