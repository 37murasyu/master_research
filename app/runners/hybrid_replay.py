"""記録した混成の計測を流し直し、計測と同じ ``@@GAUGE`` の行を出す（GUI のゲージ窓の確かめと練習用）。

    python -m app.runners.hybrid_replay <計測フォルダ> [--from 20] [--to 90] [--speed 1]

GUI からは独立の role ``hybrid_replay``（``app.entry.WORKER_MODULES``）で引数なしに起動する。計測フォルダは設定
``HYBRID_REPLAY``、範囲と速さは ``HYBRID_REPLAY_FROM``・``HYBRID_REPLAY_TO``・``HYBRID_REPLAY_SPEED``（空なら既定）を
環境変数で受け取る。これらは再生の子にしか渡らない（``app.entry.worker_environment``）ので、親のシェルに残った
``export`` で実機の計測（``hybrid_measure``）が再生になることはない。停止は計測と同じ停止ファイル。
カメラと Pixel は使わない。記録は ``hybrid/replay/`` に書く（本番の記録と混ざらない）。

設定（被験者番号・1RM・EKF・関所など）は計測の子と同じ ``hybrid_measure.measurement_config`` で環境変数から作る。
記録を流すのは別のスレッド（本番の受信スレッドの代わり）、行を書くのはメインスレッド（本番と同じ）。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from app.core.stop_request import StopRequest
from app.gauge.tracker import GaugeTicker, GaugeTracker
from app.hybrid.paths import replay_root
from app.hybrid.replay import REPLAY_ENV, replay
from config import env_float

__all__ = ["main"]

# メインループの周期（本番の Mac のカメラの 1 周回＝約 30 Hz に合わせる）
_TICK_S = 1.0 / 30.0


def _env_float(name: str, default: float | None) -> float | None:
    def invalid(raw):
        print(f"{name}={raw.strip()!r} は数ではないため {default} を使います", file=sys.stderr)

    return env_float(name, default, on_invalid=invalid)


def main(argv=None) -> int:
    from app.runners.hybrid_measure import _default_body_mass, measurement_config

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="記録した Mac＋Pixel の計測を流し直す（ゲージの確かめ）")
    parser.add_argument("session", nargs="?", default=os.environ.get(REPLAY_ENV), help="計測フォルダ")
    parser.add_argument("--from", dest="start_s", type=float, default=_env_float("HYBRID_REPLAY_FROM", 0.0))
    parser.add_argument("--to", dest="end_s", type=float, default=_env_float("HYBRID_REPLAY_TO", None))
    parser.add_argument("--speed", type=float, default=_env_float("HYBRID_REPLAY_SPEED", 1.0),
                        help="1 で実時間、0 で待たない")
    parser.add_argument("--body-mass", type=float, default=None, help="体重 kg（既定は BODY_MASS_KG）")
    parser.add_argument("--gravity-mode", choices=("axis", "trunk"), default="axis")
    args = parser.parse_args(argv)
    if not args.session:
        parser.error(f"計測フォルダを指定してください（引数か環境変数 {REPLAY_ENV}）")
    session = Path(args.session).expanduser()
    if not (session / "meta.json").exists():
        print(f"計測フォルダではありません: {session}", file=sys.stderr)
        return 2
    body_mass = args.body_mass if args.body_mass is not None else _default_body_mass()
    config = measurement_config(body_mass, args.gravity_mode)
    tracker = GaugeTracker(source="replay")
    ticker = GaugeTicker(tracker)
    stop = StopRequest.from_environment()
    stop.install_signal_handlers()
    print(f"[再生] {session}（{args.start_s}〜{args.end_s or '終わり'} s、速さ {args.speed}）。カメラと Pixel は使いません")
    box: dict = {}

    def run() -> None:
        try:
            box["out"] = replay(session, root=replay_root(), start_s=args.start_s, end_s=args.end_s,
                                speed=args.speed, config=config, session_kwargs={"tracker": tracker},
                                should_stop=stop.requested, on_session=lambda s: box.setdefault("session", s))
        except Exception as exc:  # 理由を GUI のログへ出して終了コードで知らせる
            box["error"] = exc

    worker = threading.Thread(target=run, name="hybrid-replay", daemon=True)
    worker.start()
    # i 回目は開始から i·_TICK_S の時刻まで寝る（固定の sleep だと macOS で 1 回あたり約 8 ms 寝過ごし、30 Hz が 24 Hz に落ちる）
    started, tick = time.monotonic(), 0
    while worker.is_alive():
        ticker.tick()
        tick += 1
        time.sleep(max(0.0, started + tick * _TICK_S - time.monotonic()))
    ticker.tick(force=True)
    measurement = box.get("session")
    if "error" in box:
        print(f"再生できません: {box['error']}", file=sys.stderr)
        return 1
    if box.get("out") is None:
        print("Pixel の点が無く、記録はありません")
    else:
        print(f"保存: {box['out']}")
    if measurement is not None and measurement.error:
        print(measurement.error, file=sys.stderr)
    return 0 if measurement is None else measurement.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
