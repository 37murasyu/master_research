"""``os.getenv`` の呼び出しを走査して設定スキーマの雛形を生成する。

``master_research_code.py`` には ``os.getenv`` が 150 箇所（ユニーク 136 個）ある。
これを手で書き写すのは現実的でないので、ソースから機械的に抽出する。

型は「どう使われているか」から推定する:

    os.getenv('X', '1') in ('1', 'true', 'True')   -> bool
    int(os.getenv('X', '5'))                        -> int
    float(os.getenv('X', '1.2'))                    -> float
    os.getenv('X', 'foo')                           -> str

使い方::

    python tools/extract_env_schema.py > app/core/settings_schema.json

生成物はあくまで雛形。UI に出す項目や説明文は
``app/core/settings.py`` の ``CURATED`` 側で上書きする。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 既定の走査対象。config.py も含めるのは、入力ソースや HEADLESS など
# 実行を左右する設定がそちらで定義されているため。
DEFAULT_SOURCES = (
    REPO_ROOT / "master_research_code.py",
    REPO_ROOT / "config.py",
)

# bool として扱う比較先。既存コードの慣用句。
BOOL_LITERALS = {"1", "true", "True", "yes", "on"}

# 接頭辞からグループ名（UI の見出しに使う）
GROUP_LABELS = {
    "E": "エネルギー・フィルタ",
    "RT": "リアルタイム処理",
    "POSE": "姿勢推定",
    "DEMO": "デモ用",
    "GRAVITY": "重力方向推定",
    "EKF": "カルマンフィルタ",
    "GAUGE": "ゲージ表示",
    "PERF": "性能計測",
    "HX711": "ロードセル",
    "TRACE": "トレース出力",
    "DRAW": "描画",
    "HEALTH": "ヘルスモニタ",
    "LOOP": "メインループ",
    "CAM": "カメラ",
    "CAM0": "カメラ",
    "CAM1": "カメラ",
    "USE": "入力ソース",
    "DISABLE": "機能無効化",
    "MP": "MediaPipe",
    "CALIB": "キャリブレーション",
}


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def _is_getenv(node: ast.AST) -> bool:
    """``os.getenv(...)`` と ``os.environ.get(...)`` の両方を拾う。

    config.py は後者を使っており、片方だけ見ていると HEADLESS や
    USE_SAMPLE_VIDEOS のような重要な設定を取りこぼす。
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False

    # os.getenv(...)
    if (
        node.func.attr == "getenv"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ):
        return True

    # os.environ.get(...)
    return (
        node.func.attr == "get"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "environ"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "os"
    )


def _infer_type(call: ast.Call) -> str:
    """呼び出しの包まれ方から型を推定する。"""
    parent = getattr(call, "parent", None)

    # int(os.getenv(...)) / float(os.getenv(...))
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
        if parent.func.id in ("int", "float"):
            return parent.func.id

    # os.getenv(...) in ('1', 'true', ...)
    if isinstance(parent, ast.Compare) and parent.left is call:
        for op, comparator in zip(parent.ops, parent.comparators):
            if isinstance(op, ast.In) and isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                values = {
                    elt.value
                    for elt in comparator.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
                if values & BOOL_LITERALS:
                    return "bool"

    # .strip() や .lower() を挟んでいる場合は 1 段たどる
    if isinstance(parent, ast.Attribute) and parent.attr in ("strip", "lower", "upper"):
        grandparent = getattr(parent, "parent", None)
        if isinstance(grandparent, ast.Call):
            setattr(grandparent, "parent", getattr(grandparent, "parent", None))
            return _infer_type(grandparent)

    return "str"


def _group_of(name: str) -> str:
    prefix = name.split("_")[0]
    return GROUP_LABELS.get(prefix, "その他")


def extract(source_path: Path) -> dict[str, dict]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    _attach_parents(tree)

    found: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not _is_getenv(node):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue

        name = node.args[0].value
        default = None
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            default = node.args[1].value

        entry = {
            "type": _infer_type(node),
            "default": default,
            "group": _group_of(name),
            "lines": [node.lineno],
        }

        if name in found:
            # 同じ変数が複数箇所で読まれている。既定値が食い違う場合は記録しておく
            # （どちらが効くかが実行経路依存になっている＝潜在的な不具合）。
            prev = found[name]
            prev["lines"].append(node.lineno)
            if prev["default"] != entry["default"]:
                prev.setdefault("conflicting_defaults", []).append(entry["default"])
            if prev["type"] == "str" and entry["type"] != "str":
                prev["type"] = entry["type"]
        else:
            found[name] = entry

    for entry in found.values():
        entry["lines"].sort()
    return dict(sorted(found.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="走査対象の Python ファイル（複数指定可）",
    )
    parser.add_argument("--out", default="-", help="出力先（既定は標準出力）")
    args = parser.parse_args()

    sources = [Path(s) for s in (args.source or DEFAULT_SOURCES)]
    schema: dict[str, dict] = {}
    for source in sources:
        for name, entry in extract(source).items():
            if name in schema:
                schema[name]["lines"].extend(entry["lines"])
                if schema[name]["type"] == "str" and entry["type"] != "str":
                    schema[name]["type"] = entry["type"]
            else:
                schema[name] = entry
    schema = dict(sorted(schema.items()))

    payload = {
        "_generated_from": [s.name for s in sources],
        "_note": "tools/extract_env_schema.py で生成。手で編集せず、上書きは settings.py の CURATED で行う。",
        "settings": schema,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{len(schema)} 件を {args.out} に書き出しました", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
