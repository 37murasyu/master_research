"""実カメラ（USB）の設定を固定する。計測（``master_research_code.py``）と録画ツールで共有する。

オート露出・オート WB・オートフォーカスは既定でオフにし、環境変数で固定値を与えられる。
優先度は ``CAM{n}_*`` > ``CAM_*`` > 既定。例: ``CAM0_EXPOSURE``、``CAM_WIDTH``、``CAM1_FOURCC=MJPG``。

録画した映像を計測に読み込ませて確かめる（§6-2・§6-3）とき、録画と計測で設定が違うと姿勢推定の
雑音が変わり、EKF の較正（S6）が計測の条件に合わなくなる。そこで計測の ``_apply_camera_controls`` の
中身をここへ移した（挙動は変えていない）。UVC 前提なので、ファイル・ネットワーク入力には当てないこと
（``SourceSpec.supports_camera_controls``）。
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# pylint: disable=no-member
import cv2 as cv

__all__ = ["apply_camera_controls"]

_OFF = ("0", "off", "false")
_ON = ("1", "on", "true")


def _set_prop(cap, prop: int, value: float) -> bool:
    try:
        ok = cap.set(prop, value)
        # 反映確認（一部backendでは取得できない）
        _ = cap.get(prop)
        return bool(ok)
    except Exception:
        return False


def apply_camera_controls(cap, index: int | None, env: Mapping[str, str] | None = None) -> None:
    """``cap``（``cv.VideoCapture``）にカメラの設定を当てる。``index`` はカメラの番号（``CAM{n}_*`` の n）。

    設定を受け付けないカメラ・バックエンドもあるので、失敗しても例外は投げない。
    """
    env = os.environ if env is None else env

    def _env(k: str, default: str | None = None) -> str | None:
        if index is not None and (f"CAM{index}_{k}" in env):
            return env.get(f"CAM{index}_{k}")
        return env.get(f"CAM_{k}", default)

    # 1) 自動系OFF（既定でOFFを試みる）
    try:
        # Auto Exposure（backend差異に配慮して複数パターンを試す）
        ae_env = _env('AUTO_EXPOSURE', 'off')
        if ae_env and ae_env.lower() in _OFF:
            for v in (0.0, 0.0, 0.25, 0.75):  # MSMF/DSHOW の差へ便宜上複数トライ
                if _set_prop(cap, cv.CAP_PROP_AUTO_EXPOSURE, v):
                    break
        elif ae_env and ae_env.lower() in _ON:
            _set_prop(cap, cv.CAP_PROP_AUTO_EXPOSURE, 1.0)
    except Exception:
        pass
    try:
        awb_env = _env('AUTO_WB', 'off')
        if hasattr(cv, 'CAP_PROP_AUTO_WB'):
            if awb_env and awb_env.lower() in _OFF:
                _set_prop(cap, cv.CAP_PROP_AUTO_WB, 0.0)
            elif awb_env and awb_env.lower() in _ON:
                _set_prop(cap, cv.CAP_PROP_AUTO_WB, 1.0)
    except Exception:
        pass
    try:
        af_env = _env('AUTOFOCUS', 'off')
        if hasattr(cv, 'CAP_PROP_AUTOFOCUS'):
            if af_env and af_env.lower() in _OFF:
                _set_prop(cap, cv.CAP_PROP_AUTOFOCUS, 0.0)
            elif af_env and af_env.lower() in _ON:
                _set_prop(cap, cv.CAP_PROP_AUTOFOCUS, 1.0)
    except Exception:
        pass

    # 2) 固定値の適用（指定がある場合）
    def _env_float(k: str) -> float | None:
        v = _env(k)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    # 解像度・FPS・FOURCC（設定 → 実値の確認の順で行う）
    w_set = _env_float('WIDTH')
    h_set = _env_float('HEIGHT')
    fps_set = _env_float('FPS')
    if w_set:
        _set_prop(cap, cv.CAP_PROP_FRAME_WIDTH, w_set)
    if h_set:
        _set_prop(cap, cv.CAP_PROP_FRAME_HEIGHT, h_set)
    if fps_set:
        _set_prop(cap, cv.CAP_PROP_FPS, fps_set)
    fourcc_env = _env('FOURCC')
    if fourcc_env and len(fourcc_env) >= 4:
        try:
            cc = cv.VideoWriter_fourcc(*fourcc_env[:4])
            _set_prop(cap, cv.CAP_PROP_FOURCC, float(cc))
        except Exception:
            pass

    exp_set = _env_float('EXPOSURE')
    if exp_set is not None:
        _set_prop(cap, cv.CAP_PROP_EXPOSURE, exp_set)
    gain_set = _env_float('GAIN')
    if gain_set is not None:
        _set_prop(cap, cv.CAP_PROP_GAIN, gain_set)
    wb_set = _env_float('WB_TEMPERATURE')
    if wb_set is not None and hasattr(cv, 'CAP_PROP_WB_TEMPERATURE'):
        _set_prop(cap, cv.CAP_PROP_WB_TEMPERATURE, wb_set)
    focus_set = _env_float('FOCUS')
    if focus_set is not None and hasattr(cv, 'CAP_PROP_FOCUS'):
        _set_prop(cap, cv.CAP_PROP_FOCUS, focus_set)
