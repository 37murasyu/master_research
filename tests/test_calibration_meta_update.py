"""校正フォルダの ``meta.json`` に後から鍵を足す ``calibration_io.update_meta`` を固定する（B3）。

**なぜこのテストがあるか。**

盤を立てる段階（B4）は、校正を**保存した後**に行う（盤を立てる途中で止めても校正そのものを失わないため）。
そのため盤の向き（``checkerboard_short_axis``）は保存済みの ``meta.json`` に後から足す。書き直しの途中で
落ちて ``meta.json`` が壊れると、校正全体が読めなくなる（計測が始まらない）。一時ファイルに書いてから
``os.replace`` で置き換え、既存の鍵は残す。
"""

from __future__ import annotations

import json

import pytest

from app.hybrid import calibration_io


def _folder(tmp_path):
    directory = tmp_path / "20260924_060000_000000"
    directory.mkdir()
    calibration_io.write_json(directory / "meta.json", {"units": "cm", "stereo": {"rms": 0.3}})
    return directory


def test_new_keys_are_added_and_old_keys_are_kept(tmp_path):
    directory = _folder(tmp_path)
    result = calibration_io.update_meta(directory, checkerboard_short_axis={"vector_cam0": [0.0, -1.0, 0.0]})
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert meta == result
    assert meta["units"] == "cm" and meta["stereo"] == {"rms": 0.3}
    assert meta["checkerboard_short_axis"]["vector_cam0"] == [0.0, -1.0, 0.0]


def test_no_temporary_file_is_left(tmp_path):
    directory = _folder(tmp_path)
    calibration_io.update_meta(directory, note="x")
    assert sorted(p.name for p in directory.iterdir()) == ["meta.json"]


def test_a_failed_write_keeps_the_old_meta(tmp_path):
    """NaN は JSON に書けない（``allow_nan=False``）。書けなくても元の meta.json は壊れない。"""
    directory = _folder(tmp_path)
    before = (directory / "meta.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        calibration_io.update_meta(directory, bad=float("nan"))
    assert (directory / "meta.json").read_text(encoding="utf-8") == before


def test_a_folder_without_meta_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        calibration_io.update_meta(tmp_path, note="x")
