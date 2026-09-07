#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Python のバージョンを固定する。
#
# mediapipe は 3.9〜3.12 にしか対応していない。素の `python3` を使うと、
# 3.13 以降が入っている環境では mediapipe の wheel が見つからず失敗する。
# さらに 3.12 で distutils が標準ライブラリから消えたため、japanize-matplotlib が
# import できるよう setuptools も入れる（詳細は requirements_min.txt を参照）。
# ---------------------------------------------------------------------------
REQUIRED_PY="${REQUIRED_PY:-3.12}"

die() {
  echo "エラー: $*" >&2
  exit 1
}

# --clean を付けると既存の .venv を作り直す。付けなければ再利用して
# 依存関係の更新だけ行う（何度実行しても安全）。
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    # 変数は必ず ${} で囲む。直後に全角文字が続くと、非 UTF-8 ロケールの bash が
    # そのバイト列を変数名の一部と解釈して "unbound variable" になる。
    *) die "不明な引数: ${arg} （使えるのは --clean のみ）" ;;
  esac
done
if [ "$CLEAN" = "1" ] && [ -d .venv ]; then
  echo "==> 既存の .venv を削除"
  rm -rf .venv
fi

if command -v uv >/dev/null 2>&1; then
  # uv は指定バージョンの Python を必要に応じて自動取得する。最も確実で速い。
  if [ -d .venv ]; then
    echo "==> 既存の .venv を再利用（作り直すには --clean）"
  else
    echo "==> uv で Python ${REQUIRED_PY} の仮想環境を作成"
    uv venv --python "${REQUIRED_PY}" .venv
  fi
  PY=".venv/bin/python"

  if [ -d "wheelhouse" ]; then
    uv pip install --python "$PY" --no-index --find-links=wheelhouse -r requirements_min.txt
  else
    uv pip install --python "$PY" -r requirements_min.txt
  fi
else
  # uv が無い場合は python3.12 を直接探す。
  PY_BIN="$(command -v "python${REQUIRED_PY}" || true)"

  if [ -z "$PY_BIN" ] && command -v pyenv >/dev/null 2>&1; then
    PY_BIN="$(find "$(pyenv root)/versions" -maxdepth 3 -type f \
      -path "*/${REQUIRED_PY}.*/bin/python${REQUIRED_PY}" 2>/dev/null | sort -V | tail -1 || true)"
  fi

  [ -n "$PY_BIN" ] || die "python${REQUIRED_PY} が見つかりません。
  mediapipe は Python 3.9〜3.12 のみ対応のため、このバージョンが必要です。
  いずれかで導入してください:
    uv:       curl -LsSf https://astral.sh/uv/install.sh | sh
    pyenv:    pyenv install ${REQUIRED_PY}
    homebrew: brew install python@${REQUIRED_PY}"

  if [ -d .venv ]; then
    echo "==> 既存の .venv を再利用（作り直すには --clean）"
  else
    echo "==> ${PY_BIN} で仮想環境を作成"
  fi
  "$PY_BIN" -m venv .venv
  PY=".venv/bin/python"
  "$PY" -m pip install --upgrade pip

  if [ -d "wheelhouse" ]; then
    "$PY" -m pip install --no-index --find-links=wheelhouse -r requirements_min.txt
  else
    "$PY" -m pip install -r requirements_min.txt
  fi
fi

echo "==> 完了: $("$PY" --version)"
echo "    有効化: source .venv/bin/activate"

# 開発用（テスト・ビルド）が必要なら:
#   pip install -r requirements_dev.txt
# 解析・可視化の追加パッケージが必要なら:
#   pip install -r requirements_extra.txt
