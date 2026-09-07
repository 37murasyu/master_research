"""アプリのエントリポイント。

役割によって振る舞いが変わる。

    python -m app                        GUI を開く
    python -m app --role realtime        計測ワーカーとして動く（GUI が内部で起動）
    python -m app --role calibrate       キャリブレーションワーカー
    python -m app --role script --module <名前>   解析スクリプトを走らせる

凍結後は ``python -m app`` の代わりに実行ファイル自身が同じ引数で呼ばれる。
詳しくは ``app/entry.py`` を参照。
"""

from __future__ import annotations

import sys

from app import entry


def main(argv: list[str] | None = None) -> int:
    args = entry.parse_args(argv)

    if args.role == "gui":
        # Qt はここで初めて import する。ワーカーとして起動されたときに
        # 不要な GUI ツールキットを読み込まないため（起動時間とメモリの節約）。
        from app.shell.main_window import run_gui

        return run_gui(sys.argv[:1])

    return entry.run_worker(args.role, args.passthrough, module=args.module)


if __name__ == "__main__":
    raise SystemExit(main())
