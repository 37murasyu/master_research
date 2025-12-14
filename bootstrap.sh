#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

if [ -d "wheelhouse" ]; then
  pip install --no-index --find-links=wheelhouse -r requirements_min.txt
else
  pip install -r requirements_min.txt
fi

# Extras (必要なら有効化)
# pip install -r requirements_extra.txt
