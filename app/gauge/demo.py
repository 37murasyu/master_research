"""被験者ゲージの合成データのデモ。CLI: ``python -m app.gauge.demo``。

計測の子プロセス（別セッションが実装）が無くても、被験者ゲージ（GUI）と
行の形式 v2（``app.gauge.protocol``）を単体で確かめられるようにするための
道具。4 つの使い方がある:

1. 引数なし: ``GaugeWindow`` を開き、接続待ち 2 秒 → 8 回（うち 1 回は過負荷、
   1 回は不足）→ 終了、を ``QTimer`` で流す。目で見て確かめる用。
2. ``--snapshot DIR``: 窓を開かず、``SCENARIOS`` の状態ごとに 1 枚ずつ PNG を
   書き出す。見た目の回帰確認・スクリーンショットの並べ比べ用。
3. ``--emit``: 標準出力へ ``@@GAUGE `` の行を 1 行ずつ書く。GUI 側（``QProcess``
   で子プロセスの標準出力を読む経路）を、実機・実測が無くても確かめられる。

4. ``--via-worker``: ``--emit`` を子プロセスとして ``WorkerRunner`` で起動し、
   親に届いたフレームの数・ログに漏れたゲージの行・終了コード・頻度を 1 行で出す。
   凍結版（.app）で、実測と同じ経路が通ることを確かめる用。

``SCENARIOS`` は ``app.gauge.model``・``app.gauge.protocol`` だけで組み立てて
あり、Qt には依存しない（``--emit`` や ``test_scenario_covers_every_state`` が
Qt の無い環境でも読み込めるように、Qt を要る道具（``app.gauge.widget``・
``app.gauge.window``）は使うときだけ遅延 import する）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from app.gauge.model import GaugeState, apply_frame, finish, reset, with_joules
from app.gauge.protocol import PART_NAMES, GaugeFrame, PartReading, encode

__all__ = ["SCENARIOS", "build_parser", "main"]

# ---------------------------------------------------------------------------
# 帯の元（constraints.md・task-9-brief.md の例）。
# 肘は W1RM 250 → [175, 212.5]、手首は W1RM 160 → [112, 136]。
# ---------------------------------------------------------------------------

_ELBOW_W1RM = 250.0
_ELBOW_BAND = (175.0, 212.5)
_WRIST_W1RM = 160.0
_WRIST_BAND = (112.0, 136.0)

_BAND: dict[str, tuple[float, float]] = {
    "elbow_L": _ELBOW_BAND,
    "elbow_R": _ELBOW_BAND,
    "wrist_L": _WRIST_BAND,
    "wrist_R": _WRIST_BAND,
}
_W1RM: dict[str, float] = {
    "elbow_L": _ELBOW_W1RM,
    "elbow_R": _ELBOW_W1RM,
    "wrist_L": _WRIST_W1RM,
    "wrist_R": _WRIST_W1RM,
}


def _make_parts(
    now: dict[str, float | None],
    prev: dict[str, float | None],
    band: dict[str, tuple[float, float] | None],
    w1rm: dict[str, float | None],
) -> dict[str, PartReading]:
    """4 つの「部位名→値」の辞書から ``parts`` を組み立てる。無い部位名は None 扱い。"""
    return {
        name: PartReading(now=now.get(name), prev=prev.get(name), band=band.get(name), w1rm=w1rm.get(name))
        for name in PART_NAMES
    }


def _state(frame: GaugeFrame, *, show_joules: bool = True) -> GaugeState:
    return apply_frame(reset(show_joules=show_joules), frame)


# ---------------------------------------------------------------------------
# シナリオ（task-9-brief.md の名前のまま）
# ---------------------------------------------------------------------------


def _scenario_waiting() -> GaugeState:
    """接続待ち。帯（＝どの運動かの割り当て）は接続前から分かっている想定で残し、
    今回値・前回値だけ無い。
    """
    frame = GaugeFrame(link="waiting", rep=0, source="demo", parts=_make_parts({}, {}, _BAND, _W1RM))
    return _state(frame)


def _scenario_first() -> GaugeState:
    """1 回目の途中（まだ 1 回も完了していないので ``prev`` は全部 None）。"""
    now = {"elbow_L": 74.0, "wrist_L": 22.0, "elbow_R": 41.0, "wrist_R": 12.0}
    frame = GaugeFrame(link="connected", rep=0, source="demo", parts=_make_parts(now, {}, _BAND, _W1RM))
    return _state(frame)


def _scenario_nth() -> GaugeState:
    """画面案①の値（mk_subject.py の ``main``）。7 回目の途中、右上腕が過負荷。"""
    now = {"elbow_L": 132.0, "wrist_L": 18.0, "elbow_R": 231.0, "wrist_R": 96.0}
    prev = {"elbow_L": 118.0, "wrist_L": 64.0, "elbow_R": 190.0, "wrist_R": 88.0}
    frame = GaugeFrame(link="connected", rep=6, source="demo", parts=_make_parts(now, prev, _BAND, _W1RM))
    return _state(frame)


def _scenario_joules_off() -> GaugeState:
    """``03_nth`` と同じ値で、J 表示だけ消す（見え方の違いを並べて比べる用）。"""
    return with_joules(_scenario_nth(), False)


def _scenario_band_null() -> GaugeState:
    """1 部位（右前腕）だけ帯が無い。溝と部位名だけになり、J オンなので数字だけ足す。"""
    now = {"elbow_L": 132.0, "wrist_L": 18.0, "elbow_R": 231.0, "wrist_R": 55.0}
    prev = {"elbow_L": 118.0, "wrist_L": 64.0, "elbow_R": 190.0}
    band: dict[str, tuple[float, float] | None] = dict(_BAND)
    band["wrist_R"] = None
    frame = GaugeFrame(link="connected", rep=4, source="demo", parts=_make_parts(now, prev, band, _W1RM))
    return _state(frame)


def _scenario_over_clamp() -> GaugeState:
    """左上腕が 1.25·hi（265.625）を超えて弧が右端で止まる。ほかの 3 部位は
    IN_BAND・IN_BAND・SHORT を 1 つずつ見せる。
    """
    now = {"elbow_L": 300.0, "elbow_R": 190.0, "wrist_L": 120.0, "wrist_R": 90.0}
    prev = {"elbow_L": 260.0, "elbow_R": 170.0, "wrist_L": 110.0, "wrist_R": 95.0}
    frame = GaugeFrame(link="connected", rep=3, source="demo", parts=_make_parts(now, prev, _BAND, _W1RM))
    return _state(frame)


def _scenario_done() -> GaugeState:
    """正常終了。今回値は無く、前回の目盛りだけ残る（mk_subject.py の ``done``）。"""
    prev = {"elbow_L": 141.0, "wrist_L": 97.0, "elbow_R": 176.0, "wrist_R": 102.0}
    frame = GaugeFrame(link="connected", rep=12, source="demo", parts=_make_parts({}, prev, _BAND, _W1RM))
    return finish(_state(frame), 0)


def _scenario_failed() -> GaugeState:
    """異常終了。落ちる直前は RUNNING（link="connected"）だったので、
    ``_effective_phase`` の読み替えにより弧・値はそのまま残って見える。
    """
    now = {"elbow_L": 150.0, "elbow_R": 180.0, "wrist_L": 100.0, "wrist_R": 90.0}
    prev = {"elbow_L": 140.0, "elbow_R": 170.0, "wrist_L": 95.0, "wrist_R": 85.0}
    frame = GaugeFrame(link="connected", rep=4, source="demo", parts=_make_parts(now, prev, _BAND, _W1RM))
    return finish(_state(frame), 1)


def _scenario_wide_band() -> GaugeState:
    """帯が 50〜200 J という、実際の W1RM 比では出ない幅の広い帯（brief 指定）。"""
    wide_band = (50.0, 200.0)
    band = {name: wide_band for name in PART_NAMES}
    now = {"elbow_L": 90.0, "elbow_R": 210.0, "wrist_L": 150.0, "wrist_R": 40.0}
    prev = {"elbow_L": 80.0, "elbow_R": 190.0, "wrist_L": 140.0, "wrist_R": 45.0}
    frame = GaugeFrame(link="connected", rep=2, source="demo", parts=_make_parts(now, prev, band, {}))
    return _state(frame)


def _scenario_replay() -> GaugeState:
    """``source="replay"`` を除けば ``03_nth`` に近い値。見出しに「▶ 再生」が出る。"""
    now = {"elbow_L": 132.0, "wrist_L": 18.0, "elbow_R": 190.0, "wrist_R": 96.0}
    prev = {"elbow_L": 118.0, "wrist_L": 64.0, "elbow_R": 170.0, "wrist_R": 88.0}
    frame = GaugeFrame(link="connected", rep=5, source="replay", parts=_make_parts(now, prev, _BAND, _W1RM))
    return _state(frame)


def _scenario_now_null() -> GaugeState:
    """1 部位（右前腕）だけ今回値が無い。帯はあるので溝・帯までは描くが、
    値の弧・状態・前回の目盛りは（``prev`` があっても）出ない。
    """
    now = {"elbow_L": 150.0, "elbow_R": 190.0, "wrist_L": 120.0}
    prev = {"elbow_L": 140.0, "elbow_R": 180.0, "wrist_L": 115.0, "wrist_R": 90.0}
    frame = GaugeFrame(link="connected", rep=5, source="demo", parts=_make_parts(now, prev, _BAND, _W1RM))
    return _state(frame)


SCENARIOS: dict[str, GaugeState] = {
    "01_waiting": _scenario_waiting(),
    "02_first": _scenario_first(),
    "03_nth": _scenario_nth(),
    "04_joules_off": _scenario_joules_off(),
    "05_band_null": _scenario_band_null(),
    "06_over_clamp": _scenario_over_clamp(),
    "07_done": _scenario_done(),
    "08_failed": _scenario_failed(),
    "09_wide_band": _scenario_wide_band(),
    "10_replay": _scenario_replay(),
    "11_now_null": _scenario_now_null(),
}


# ---------------------------------------------------------------------------
# --snapshot: 状態ごとに PNG を書く
# ---------------------------------------------------------------------------


def _parse_size(text: str) -> tuple[int, int]:
    """``"1600x900"`` を ``(1600, 900)`` に直す。壊れていれば ``ValueError``。"""
    try:
        w_text, h_text = text.lower().split("x", 1)
        w, h = int(w_text), int(h_text)
    except ValueError as exc:
        raise ValueError(f"--size は '幅x高さ' の形式にする（例: 1600x900）: {text!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"--size は正の整数にする: {text!r}")
    return w, h


def _cmd_snapshot(directory: str, size: str, fonts: str | None = None) -> int:
    try:
        w, h = _parse_size(size)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # QApplication を作る前に決める（判断済みのこと）。既に QT_QPA_PLATFORM が
    # 決まっていれば（例: pytest の conftest.py）それを尊重する。
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from app.core.qt import QtWidgets
    from app.gauge.widget import render_image

    # 生成した QApplication は Qt 側がプロセス内で保持し続けるので、変数として
    # 持ち回る必要はない（インスタンスを作る、という呼び出しだけが要る）。
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from app.gauge import fonts as gauge_fonts

    # --fonts all は、書体の組ごとに DIR/<組の名前>/ へ書く（見比べるため）
    presets = list(gauge_fonts.PRESETS) if fonts == "all" else [fonts]
    for preset in presets:
        out_dir = Path(directory) / preset if fonts == "all" else Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        font_set = gauge_fonts.font_set(preset)
        for name, state in SCENARIOS.items():
            image = render_image(state, w, h, font_set)
            image.save(str(out_dir / f"{name}.png"))

    return 0


# ---------------------------------------------------------------------------
# --emit: 行を標準出力へ書く
# ---------------------------------------------------------------------------


def _emit_frame(i: int) -> GaugeFrame:
    """``i`` 回目ぶんの合成フレーム（線形に増える値。decode の往復を確かめる用）。"""
    now = {"elbow_L": 120.0 + i, "elbow_R": 150.0 + i, "wrist_L": 90.0 + i, "wrist_R": 100.0 + i}
    prev = {} if i == 0 else {name: value - 10.0 for name, value in now.items()}
    return GaugeFrame(link="connected", rep=i, source="demo", parts=_make_parts(now, prev, _BAND, _W1RM))


def _cmd_emit(*, count: int, interval: float, exit_code: int) -> int:
    # 「毎回 interval 寝る」ではなく「i 回目は開始から i·interval の時刻まで寝る」。
    # macOS の time.sleep(0.033) は 1 回 8 ms ほど寝過ごし、固定で寝ると 30 Hz が
    # 24 Hz に落ちる。締め切りで寝れば、寝過ごしを次の回で取り返せる。
    start = time.perf_counter()
    for i in range(count):
        sys.stdout.write(encode(_emit_frame(i)))
        sys.stdout.flush()
        if interval > 0 and i < count - 1:
            time.sleep(max(0.0, start + (i + 1) * interval - time.perf_counter()))
    return exit_code


# ---------------------------------------------------------------------------
# --via-worker: 実測と同じ経路（WorkerRunner → 子の --emit → gauge_frame）で流す
# ---------------------------------------------------------------------------

_VIA_WORKER_TIMEOUT_MS = 30000


def _cmd_via_worker(*, count: int, interval: float, exit_code: int) -> int:
    """子に ``--emit`` を走らせ、親の ``WorkerRunner`` にフレームが届くかを数える。

    凍結版では子も凍結版の実行ファイルを ``--role script`` で呼び直すので、
    「子の標準出力 → QProcess → LineDemux → gauge_frame」の経路を、実機が無くても
    .app のまま確かめられる。最後の 1 行に要約を書き、届いた数・ログに漏れた
    ゲージの行・子の終了コードが期待どおりなら 0 を返す。
    """
    from app.core.qt import QtCore
    from app.core.settings import Settings
    from app.runners.worker import WorkerRunner

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    runner = WorkerRunner(role="script")
    stamps: list[float] = []
    leaked = {"n": 0}
    result = {"code": None}

    def on_output(text: str) -> None:
        leaked["n"] += text.count("@@GAUGE")

    def on_finished(code: int) -> None:
        result["code"] = code
        app.quit()

    runner.gauge_frame.connect(lambda _frame: stamps.append(time.perf_counter()))
    runner.output.connect(on_output)
    runner.finished.connect(on_finished)
    QtCore.QTimer.singleShot(_VIA_WORKER_TIMEOUT_MS, app.quit)

    passthrough = ["--emit", "--count", str(count), "--interval", str(interval), "--exit-code", str(exit_code)]
    if not runner.start(Settings(), passthrough, module="app.gauge.demo"):
        print("via-worker: 子を起動できなかった")
        return 1
    app.exec()

    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) >= 2 and stamps[-1] > stamps[0] else 0.0
    print(
        f"via-worker frames={len(stamps)}/{count} gauge_in_log={leaked['n']} "
        f"exit={result['code']} rate_hz={rate:.1f}"
    )
    ok = len(stamps) == count and leaked["n"] == 0 and result["code"] == exit_code
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 引数なし: GaugeWindow を開いて QTimer で流す
# ---------------------------------------------------------------------------

_WAIT_MS = 2000  # 接続待ちの長さ（brief: 2 秒）
_STEP_MS = 700  # 1 回ごとの間隔
_OVERLOAD_REP = 2  # 3 回目（0 始まり）を過負荷にする
_SHORTFALL_REP = 5  # 6 回目（0 始まり）を不足にする
_LIVE_REPS = 8


def _live_frames() -> list[GaugeFrame]:
    """接続待ちの後に流す合成の 8 回ぶん（brief: 過負荷 1 回・不足 1 回を含む）。"""
    frames: list[GaugeFrame] = []
    prev_now: dict[str, float] = {}
    for i in range(_LIVE_REPS):
        if i == _OVERLOAD_REP:
            now = {"elbow_L": 300.0, "elbow_R": 290.0, "wrist_L": 200.0, "wrist_R": 190.0}
        elif i == _SHORTFALL_REP:
            now = {"elbow_L": 60.0, "elbow_R": 55.0, "wrist_L": 40.0, "wrist_R": 35.0}
        else:
            now = {"elbow_L": 190.0, "elbow_R": 185.0, "wrist_L": 122.0, "wrist_R": 118.0}
        frames.append(GaugeFrame(link="connected", rep=i, source="demo", parts=_make_parts(now, prev_now, _BAND, _W1RM)))
        prev_now = now
    return frames


def _cmd_live(font_preset: str | None = None) -> int:
    from app.core.qt import QtCore, QtWidgets
    from app.gauge.window import GaugeWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GaugeWindow(show_joules=True, font_preset=font_preset)
    window.begin(show_joules=True)

    frames = _live_frames()
    progress = {"i": 0}
    timer = QtCore.QTimer()

    def step() -> None:
        if progress["i"] >= len(frames):
            timer.stop()
            window.finish(0)
            return
        window.set_frame(frames[progress["i"]])
        progress["i"] += 1

    timer.timeout.connect(step)
    QtCore.QTimer.singleShot(_WAIT_MS, lambda: timer.start(_STEP_MS))

    return app.exec()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="被験者ゲージの合成データのデモ")
    parser.add_argument("--snapshot", metavar="DIR", help="窓を開かず、シナリオごとに PNG を DIR へ書く")
    parser.add_argument("--size", default="1600x900", help="--snapshot の画像の大きさ（既定 1600x900）")
    parser.add_argument("--emit", action="store_true", help="ゲージの行を標準出力へ書いて終わる（source=demo）")
    parser.add_argument("--count", type=int, default=8, help="--emit で出す行数（既定 8）")
    parser.add_argument("--interval", type=float, default=1.0, help="--emit の行の間隔 [秒]（既定 1.0）")
    parser.add_argument("--exit-code", type=int, default=0, help="--emit の終了コード（既定 0）")
    parser.add_argument(
        "--via-worker",
        action="store_true",
        help="--emit を子プロセスで走らせ、WorkerRunner に届いたフレームを数える（--count などはそのまま子へ）",
    )
    parser.add_argument(
        "--fonts",
        metavar="PRESET",
        help="書体の組（rodin・tsukushi・kaimin・system）。--snapshot では all で組ごとに書く。"
        "省略すると設定 GAUGE_FONT_PRESET の既定",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.via_worker:
        return _cmd_via_worker(count=args.count, interval=args.interval, exit_code=args.exit_code)
    if args.emit:
        return _cmd_emit(count=args.count, interval=args.interval, exit_code=args.exit_code)
    if args.snapshot:
        return _cmd_snapshot(args.snapshot, args.size, args.fonts)
    return _cmd_live(args.fonts)


if __name__ == "__main__":
    raise SystemExit(main())
