"""混成ステレオの出力先。

GUI（``app.shell``）からも使うので、cv2 などの重いモジュールを読まない。出力先を決めるのは
ここだけにする。以前は校正・計測・GUI の 2 画面で同じパスを別々に書いていて、片方だけ
変えると画面の表示と実際の書き込み先が食い違う。
"""

from __future__ import annotations

from pathlib import Path

from app.core.platform_compat import user_config_dir, user_output_dir
from app.core.settings import APP_NAME


def hybrid_root() -> Path:
    """USB 経路の出力（``user_output_dir(APP_NAME)``）と同じ場所の下に置く。"""
    return user_output_dir(APP_NAME) / "hybrid"


def calibration_root() -> Path:
    return hybrid_root() / "calibration"


def measurement_root() -> Path:
    return hybrid_root() / "measure"


def replay_root() -> Path:
    """記録した計測を流し直した結果（``app.hybrid.replay``）。本番の計測の記録と混ざらないよう分ける。"""
    return hybrid_root() / "replay"


def session_file() -> Path:
    """Pixel が覚えておく接続先の session（``app.hybrid.session``）。利用者は見ない設定側に置く。"""
    return user_config_dir(APP_NAME) / "hybrid_session.txt"


def ekf_profile_root() -> Path:
    """混成の EKF の較正プロファイル（``app.runners.tune_ekf`` が混成の収録から作る）。設定 ``HYBRID_EKF_PROFILE`` に入れる。"""
    return hybrid_root() / "ekf_profiles"
