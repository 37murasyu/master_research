#!/usr/bin/env bash
# macOS 用の .app を作る。
#
#   ./packaging/build_macos.sh            → dist/WheelchairTorque.app
#
# 1. アイコンを生成する（packaging/icon.py → build/icon/AppIcon.icns）
# 2. PyInstaller で固める（packaging/app.spec）
#
# Python は既定で .venv のものを使う（bootstrap.sh が作る）。別のものを使うなら PYTHON で指定する。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
cd "$ROOT"

"$PY" packaging/icon.py build/icon
"$PY" -m PyInstaller --noconfirm --clean \
    --distpath dist --workpath build/pyinstaller \
    packaging/app.spec

echo "作成: $ROOT/dist/WheelchairTorque.app"
