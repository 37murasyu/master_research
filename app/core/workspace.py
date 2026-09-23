"""凍結アプリのワークスペース。開発時のリポジトリルートの役を代わりに担う。

開発時はリポジトリルートが「コードの置き場」と「データの置き場」を兼ね、既存スクリプトは
それを前提に書かれている。CWD 相対で書き（``config.py`` の ``output_data``、``calib.py`` の
``camera_parameters/``）、``config.folder_path`` を基点に読む（係数表、カメラパラメータ）。

凍結すると ``folder_path`` はバンドル内（読み取り専用。書けば署名も壊れる）になり、
Finder から起動したアプリの CWD は ``/`` になる。どちらにも書けず、ワーカーは
``config.py`` の import で落ちる。

そこで凍結時だけ、ワーカーは :func:`prepare_workspace` で用意した場所へ ``chdir`` し、
環境変数 ``APP_WORKSPACE`` で ``config.folder_path`` もそこへ向ける。CWD と
``folder_path`` が同じ場所を指すという開発時の前提が、そのまま成り立つ。

**読み取り専用の同梱資産**（フォント・姿勢推定モデル）は ``resources`` の担当で、
ここでは扱わない。ここにあるのは、ユーザが書き換えうるデータの**初期値**だけ。
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = ["WORKSPACE_ENV", "SEED_DIRNAME", "SEED_FILES", "prepare_workspace"]

# ワーカーに「ここがワークスペースだ」と伝える環境変数。config.py が読む。
WORKSPACE_ENV = "APP_WORKSPACE"

# 初期値を同梱するディレクトリ名（バンドル内）。packaging/app.spec もこの名前で追加する。
SEED_DIRNAME = "seed"

# ワークスペースに初期値として置くファイル（リポジトリルートからの相対パス）。
# spec はこの一覧から datas を組み立てるので、ここが唯一の正本。
SEED_FILES: tuple[str, ...] = (
    # config.folder_path から読む係数表
    "rm_method.csv",
    # 被験者ごとの 1RM。混成のゲージの閾値（app.gauge.thresholds、ONE_RM_CSV が空のとき）が読む
    "m_max_all_merged.csv",
    "Moment of inertia estimation coefficient boys.csv",
    # master_research_code.py が folder_path から読む（空でも存在が要る）
    "max_value.txt",
    # 無ければ雛形を作るが、開発時と同じ値から始める
    "supervision_stats.csv",
    # calib.py が CWD 相対で読む
    "calibration_settings.yaml",
    # 直近のキャリブレーション結果。校正し直すまでの計測に使う
    "camera_parameters/c0.dat",
    "camera_parameters/c1.dat",
    "camera_parameters/rot_trans_c0.dat",
    "camera_parameters/rot_trans_c1.dat",
    # 録画を入力にしたとき（file_mode）に utils.get_projection_matrix が読む
    "camera_parameters/Param_for_MYvideo/c0.dat",
    "camera_parameters/Param_for_MYvideo/c1.dat",
    "camera_parameters/Param_for_MYvideo/rot_trans_c0.dat",
    "camera_parameters/Param_for_MYvideo/rot_trans_c1.dat",
)


def prepare_workspace(root: Path, seed_dir: Path) -> Path:
    """ワークスペースを作り、無いファイルだけ初期値を置く。

    **既にあるファイルは上書きしない**。キャリブレーションをやり直した結果や、
    書き換えた係数表が、起動のたびに同梱の既定値へ戻ってはいけない。

    ``seed_dir`` が無くても例外にしない。同梱が漏れても、書き込み先さえあれば
    計測は始められる（足りないファイルは、それを読むスクリプトが理由付きで失敗する）。
    """
    root.mkdir(parents=True, exist_ok=True)
    if not seed_dir.is_dir():
        return root
    for src in seed_dir.rglob("*"):
        if not src.is_file():
            continue
        dst = root / src.relative_to(seed_dir)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return root
