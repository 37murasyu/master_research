"""エントリポイントの引数処理と、計測ワーカーの起動方法。

PyInstaller で凍結すると ``python master_research_code.py`` は実行できない。
配布物に Python インタプリタが単体で存在しないためだ。そこで
**多重エントリ**の形を取る。アプリは ``sys.executable`` に ``--role=realtime``
を付けて自分自身を再起動し、その役割で既存の計測スクリプトを走らせる。

    開発時: python -m app        →  python -m app --role realtime
    凍結時: MyApp.exe            →  MyApp.exe --role realtime

**開発と配布で同じコードパス**を通るのが要点。ここがずれると「手元では動くのに
凍結すると起動しない」という、最も遅く気づく壊れ方をする。

GUI に依存しないので、Qt を import せずにテストできる。
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.core import resources, workspace
from app.core.platform_compat import user_output_dir
from app.core.settings import APP_NAME, OUTPUT_DIR_ENV, SCHEMA, Settings, measurement_output_dir
from app.core.stop_request import STOP_FILE_ENV

__all__ = [
    "ParsedArgs",
    "parse_args",
    "resolve_module",
    "worker_command",
    "worker_environment",
    "CHILD_OUTPUT_ENV",
    "run_worker",
    "workspace_dir",
    "ROLES",
    "WORKER_MODULES",
    "SCRIPT_ROLE",
    "REPLAY_ROLE",
]

# 記録した混成の計測を流し直す役割（カメラも Pixel も使わない）。実機の計測（hybrid_measure）とは別の role にして、
# 何を流すか（計測フォルダ・範囲・速さ）は引数で渡す（app.shell.page_measure.replay_arguments）。環境変数では
# 切り替えないので、親のシェルに何が残っていても実機の計測が再生になることはない。計測画面もこの名前を引く。
REPLAY_ROLE = "hybrid_replay"

# 役割ごとに __main__ として実行する既存モジュール。
#
# calibrate を別プロセスにするのは macOS の制約による。calib.py の撮影フェーズは
# 対話式の cv.imshow + waitKey で、macOS では GUI 操作がメインスレッド必須のため、
# QThread から呼ぶと "Unknown C++ exception from OpenCV code" で失敗する（実測済み）。
# カメラ検出や行列計算のような GUI を伴わない部分はプロセス内で呼んでよい。
WORKER_MODULES = {
    "realtime": "master_research_code",
    "calibrate": "calib",
    "hybrid_calibrate": "app.runners.hybrid_calibrate",
    "hybrid_measure": "app.runners.hybrid_measure",
    REPLAY_ROLE: "app.runners.hybrid_replay",
}

# オフライン解析用の汎用役割。実行するモジュール名は --module で指定する。
# 解析スクリプトは matplotlib で描画するものが多く、これも GUI 操作なので
# スレッドではなく子プロセスで動かす。対象が多いため個別の役割にはしない。
SCRIPT_ROLE = "script"

ROLES = ("gui", *WORKER_MODULES, SCRIPT_ROLE)


@dataclass
class ParsedArgs:
    role: str = "gui"
    # --role script のとき実行するモジュール名
    module: str | None = None
    # 既存スクリプトへそのまま渡す引数（被験者番号など）
    passthrough: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> ParsedArgs:
    parser = argparse.ArgumentParser(
        prog="app",
        description="車椅子駆動の関節トルク計測アプリ",
        add_help=True,
    )
    parser.add_argument(
        "--role",
        choices=ROLES,
        default="gui",
        help="gui: 画面を開く / realtime, calibrate, script: ワーカーとして動く（内部用）",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="--role script のとき実行するモジュール名（内部用）",
    )
    known, rest = parser.parse_known_args(argv)
    return ParsedArgs(role=known.role, module=known.module, passthrough=list(rest))


def resolve_module(role: str, module: str | None = None) -> str:
    """役割から、``__main__`` として実行するモジュール名を決める。

    役割の妥当性検査もここで行う。以前は ``worker_command`` と ``run_worker`` が
    同じ二分岐を別々に書いており、片方だけ直すと食い違う形だった。
    """
    if role == SCRIPT_ROLE:
        if not module:
            raise ValueError(f"--role {SCRIPT_ROLE} にはモジュール名の指定が必要です")
        return module

    resolved = WORKER_MODULES.get(role)
    if resolved is None:
        valid = ", ".join((*WORKER_MODULES, SCRIPT_ROLE))
        raise ValueError(f"ワーカーの役割が不正: {role}（有効: {valid}）")
    return resolved


# 子の標準出力の書き方（tools/verify_run.py の再生と同じ 3 つ）。親の環境の値より優先する。
# - 文字コード: Windows のパイプの既定（cp932）だと、本体の "✅" の print で子が UnicodeEncodeError で落ちる。
#   PYTHONUTF8 は open() の既定も UTF-8 にする（PYTHONIOENCODING は標準入出力だけ）
# - ためない: Python はパイプへの出力をためるので、flush しない print は子が終わるまでログに出なかった
CHILD_OUTPUT_ENV = {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def uses_stop_file(role: str) -> bool:
    return role in {"realtime", "hybrid_calibrate", "hybrid_measure", REPLAY_ROLE}


def worker_command(
    role: str, passthrough: list[str] | None = None, module: str | None = None
) -> list[str]:
    """指定した役割でワーカーを起動するコマンドを組み立てる。

    **既存スクリプトのパスを直接参照しない**。凍結後はファイルとして
    存在しないため。自分自身を役割つきで呼び直す。
    """
    resolve_module(role, module)  # 妥当性の検査

    role_args = ["--role", role]
    if role == SCRIPT_ROLE:
        role_args += ["--module", module or ""]

    extra = list(passthrough or [])
    if resources.is_frozen():
        return [sys.executable, *role_args, *extra]
    return [sys.executable, "-m", "app", *role_args, *extra]


def worker_environment(
    settings: Settings, role: str = "realtime", stop_file: str | None = None
) -> dict[str, str]:
    """ワーカーに渡す環境変数。

    親の環境を引き継いだうえで、アプリの設定で**上書きする**。
    引き継ぐのは PATH などが必要だから。上書きするのは、シェルに残った
    古い値が勝つと子プロセスの挙動が読めなくなるから。

    設定は全件を明示的に渡す（``Settings.as_env`` を参照）。差分だけ渡すと、
    渡さなかった項目は既存スクリプト側の既定値が効いてしまう。

    設定の名前（``SCHEMA``）は、親の環境から先にまとめて落とす。既定値の無い設定（SUBJECT_ID・CAM0・
    ONE_RM_CSV・MP_THREADS など）は ``as_env`` が渡さないので、上書きだけでは親のシェルの値が子へ漏れる。

    ``stop_file`` は停止要求のファイルのパス（``app.core.stop_request``）。親の環境に
    残った古い値を引き継がないよう、渡さないときは消す。

    計測の CSV は、GUI が「出力先」と表示している場所に書かせる（``OUTPUT_DIR``、config.save_dir が読む）。

    子の標準出力は UTF-8 で、ためずに書かせる（``CHILD_OUTPUT_ENV``）。親（``WorkerRunner``）は UTF-8 で読み、
    ログを逐次出す。
    """
    env = {name: value for name, value in os.environ.items() if name not in SCHEMA}
    env.update(settings.as_env())
    env.update(CHILD_OUTPUT_ENV)
    env["APP_ROLE"] = role
    env[OUTPUT_DIR_ENV] = str(measurement_output_dir())
    env.pop(STOP_FILE_ENV, None)
    if stop_file:
        env[STOP_FILE_ENV] = str(stop_file)
    return env


def workspace_dir() -> Path:
    """凍結時のワークスペース。GUI が「出力先」として表示する場所と同じにする。"""
    return user_output_dir(APP_NAME)


def _enter_workspace() -> None:
    """凍結時だけ、ワークスペースへ移って ``config.folder_path`` もそこへ向ける。

    開発時は何もしない。研究者はリポジトリルートから起動しており、そこが既に
    コードとデータの両方の置き場になっている（``app.core.workspace`` を参照）。
    """
    if not resources.is_frozen():
        return
    seed = resources.resource_root() / workspace.SEED_DIRNAME
    root = workspace.prepare_workspace(workspace_dir(), seed)
    os.chdir(root)
    os.environ[workspace.WORKSPACE_ENV] = str(root)


def run_worker(
    role: str, passthrough: list[str] | None = None, module: str | None = None
) -> int:
    """既存スクリプトを ``__main__`` として実行する。

    ``runpy`` を使うのは、対象がトップレベル直書きのスクリプトだから。
    ``master_research_code`` は import しても意味がなく（``__main__`` ガードが無く
    2,900 行が import 時に走る）、ファイルパス指定も凍結後は使えない。
    モジュール名で解決する必要がある。
    """
    module = resolve_module(role, module)
    _enter_workspace()

    argv_backup = sys.argv[:]
    sys.argv = [module, *(passthrough or [])]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0
    finally:
        sys.argv = argv_backup
