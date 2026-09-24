# -*- mode: python ; coding: utf-8 -*-
"""macOS 用 .app の PyInstaller 設定。``packaging/build_macos.sh`` から呼ぶ。

一覧を手で持たないのが方針。手書きの一覧は、ソース側に何かを足したときに黙ってずれ、
「凍結したときだけ動かない」という最も遅く気づく壊れ方をする。

- hiddenimports: ワーカーとして ``runpy`` で動かすモジュール。静的解析では見つからない。
  ``entry.WORKER_MODULES``（計測・キャリブレーション）と解析画面の ``TASKS`` から集める。
- seed: ワークスペースの初期値。``app.core.workspace.SEED_FILES`` から集める。

onedir で作る。onefile は起動のたびに一時ディレクトリへ全体を展開するので、
計測ワーカーを子プロセスとして起動し直すこのアプリでは、その展開が毎回走って遅い。
"""

import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821  (SPECPATH は PyInstaller が与える)
sys.path.insert(0, str(ROOT))

from app.core import workspace  # noqa: E402
from app.core.settings import APP_NAME  # noqa: E402
from app.entry import WORKER_MODULES  # noqa: E402
from app.shell.page_analyze import TASKS  # noqa: E402

VERSION = "0.1.0"
# 逆ドメイン名の識別子。カメラ等の許可（TCC）はこの識別子に紐づくので、変えると許可を取り直しになる。
BUNDLE_ID = os.environ.get("APP_BUNDLE_ID", "local.masterresearch.WheelchairTorque")
ICON = Path(os.environ.get("APP_ICON", ROOT / "build" / "icon" / "AppIcon.icns"))

hiddenimports = sorted(
    {
        *WORKER_MODULES.values(),
        *(task.module for task in TASKS),
        # app.gauge.demo は ``python -m app.gauge.demo`` でしか呼ばれず、WORKER_MODULES
        # にも TASKS にも載らない（runpy のワーカーでも解析画面のタスクでもない）ので
        # 静的解析では見つからない。手で足す（本体はまだ別の作業ツリーで作成中だが、
        # この一覧はビルド時にしか読まれないので先に足してよい）。
        "app.gauge.demo",
        # ``python -m app.gauge.fonts``（書体の診断）も同じく手で足す
        "app.gauge.fonts",
    }
)

datas = [
    # 読み取り専用の同梱資産（app.core.resources が解決する）
    (str(ROOT / "assets"), "assets"),
    # 設定画面の項目定義。settings.py が自分の隣（Path(__file__).with_name）から読む
    (str(ROOT / "app" / "core" / "settings_schema.json"), "app/core"),
    # 計測スクリプトが os.path.dirname(__file__) から読むもの。凍結後は sys._MEIPASS 直下を指す
    (str(ROOT / "pose_landmarker_lite.task"), "."),
    (str(ROOT / "gauge_layout.json"), "."),
    # master_research_code.py の stats_file。根に無いと、読み取り専用の _MEIPASS に雛形を書こうとする
    (str(ROOT / "supervision_stats.csv"), "."),
    # Mac＋Pixel の校正（app/runners/hybrid_calibrate.py の board_defaults）が resources.resource_root() から読む。
    # seed/ にも入れる（calib.py はワークスペースの CWD 相対で読む）が、seed/ だけでは根に無く起動直後に落ちる
    (str(ROOT / "calibration_settings.yaml"), "."),
]
# 根に置くファイルは tests/test_packaging_spec.py が確かめる
datas += [
    (str(ROOT / rel), str(Path(workspace.SEED_DIRNAME) / Path(rel).parent))
    for rel in workspace.SEED_FILES
]

# PySide6 と同居させない。pyqtgraph は PyQt5 を先に探すので、紛れ込むと GPL の構成で配布しかねない
# （app/__init__.py を参照）。tkinter は使っていない（試作の test2.py だけ）。
excludes = ["PyQt5", "PyQt6", "PySide2", "tkinter", "IPython", "pytest"]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,
    upx=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    upx=False,
    name=APP_NAME,
)

app = BUNDLE(  # noqa: F821
    coll,
    name=f"{APP_NAME}.app",
    icon=str(ICON),
    bundle_identifier=BUNDLE_ID,
    version=VERSION,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "CFBundleDevelopmentRegion": "ja",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        # これが無いと、カメラを開いた瞬間に macOS がプロセスを終了させる（許可ダイアログも出ない）
        "NSCameraUsageDescription": "プッシュアップ中の姿勢を推定するため、カメラの映像を使います。",
        "NSBluetoothAlwaysUsageDescription": "ロードセル（HX711）の計測値を Bluetooth で受け取るために使います。",
        "NSLocalNetworkUsageDescription": "スマートフォンから姿勢データを Wi-Fi で受け取るために使います。",
    },
)
