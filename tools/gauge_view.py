"""記録を流し直す子プロセスの ``@@GAUGE`` の行でゲージ窓を動かし、数秒ごとに窓を画像に撮る（画面での確かめ）。

    python -m tools.gauge_view <計測フォルダ> --out <画像の置き場> [--every 2] [--speed 1] [--from 0] [--offscreen]

GUI 本体の計測ページを通さずに、本番と同じ道筋（子プロセスの標準出力の行 → ``protocol.decode`` → ``GaugeWindow.set_frame``）
でゲージの見た目と動きを確かめる。子は ``python -m app.runners.hybrid_replay``（カメラと Pixel は使わない）。
``--offscreen`` なら画面に出さずに描く（画面が消えていても撮れる）。撮った画像は ``gauge_<連番>.png``。
"""

from __future__ import annotations

import argparse
import codecs
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="記録の再生でゲージ窓を動かして撮る")
    parser.add_argument("session", help="流し直す計測フォルダ")
    parser.add_argument("--out", required=True, help="画像を書くフォルダ")
    parser.add_argument("--every", type=float, default=2.0, help="撮る間隔 [s]")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--from", dest="start_s", type=float, default=0.0)
    parser.add_argument("--to", dest="end_s", type=float, default=None)
    parser.add_argument("--hide-joules", action="store_true", help="J の数値を隠す")
    parser.add_argument("--offscreen", action="store_true", help="画面に出さずに描く")
    args = parser.parse_args(argv)
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from app.core.qt import QtCore, QtWidgets
    from app.gauge.protocol import PREFIX, decode
    from app.gauge.window import GaugeWindow

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = GaugeWindow(show_joules=not args.hide_joules)
    window.begin(show_joules=not args.hide_joules)
    window.resize(1280, 720)

    process = QtCore.QProcess()
    process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
    process.setWorkingDirectory(str(Path(__file__).resolve().parent.parent))
    child_args = ["-m", "app.runners.hybrid_replay", str(Path(args.session).expanduser()),
                  "--speed", str(args.speed), "--from", str(args.start_s)]
    if args.end_s is not None:
        child_args += ["--to", str(args.end_s)]
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    state = {"buffer": "", "shots": 0, "frames": 0, "last": None}

    def read_stdout() -> None:
        state["buffer"] += decoder.decode(bytes(process.readAllStandardOutput()))
        *lines, state["buffer"] = state["buffer"].split("\n")
        for line in lines:
            frame = decode(line) if line.startswith(PREFIX) else None
            if frame is None:
                if line.strip():
                    print(f"[子] {line}")
                continue
            window.set_frame(frame)
            state["frames"] += 1
            state["last"] = frame

    def read_stderr() -> None:
        text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            print(f"[子 stderr] {text}", file=sys.stderr)

    def shoot() -> None:
        path = out / f"gauge_{state['shots']:03d}.png"
        window.grab().save(str(path))
        state["shots"] += 1
        last = state["last"]
        summary = "" if last is None else (
            f" rep={last.rep} link={last.link} source={last.source} " + ", ".join(
                f"{name}={reading.now}/{reading.prev}" for name, reading in last.parts.items()))
        print(f"[撮影] {path.name}{summary}")

    def finished(code: int, _status=None) -> None:
        read_stdout()
        window.finish(int(code))
        QtCore.QTimer.singleShot(300, lambda: (shoot(), app.quit()))
        print(f"[終了] 子の終了コード {code}、受け取った行 {state['frames']}")

    process.readyReadStandardOutput.connect(read_stdout)
    process.readyReadStandardError.connect(read_stderr)
    process.finished.connect(finished)
    timer = QtCore.QTimer()
    timer.timeout.connect(shoot)
    timer.start(int(args.every * 1000))
    process.start(sys.executable, child_args)
    app.exec()
    return 0 if state["frames"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
