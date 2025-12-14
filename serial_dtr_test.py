r"""Minimal serial test for HX711-logger DTR toggle and line I/O.

Usage (PowerShell):
    $env:PORT="COM3"
    python .\serial_dtr_test.py --port $env:PORT --baud 115200 --dtr on --send "STATUS"
    python .\serial_dtr_test.py --port $env:PORT --dtr off

Notes:
- No camera, no physics; safe to run standalone.
- Requires: pyserial
"""
from __future__ import annotations
import argparse
import time
try:
    import serial  # type: ignore
except ImportError:
    raise SystemExit("pyserial が未インストールです。`pip install pyserial` を実行してください。")


def open_ser(port: str, baud: int, timeout: float = 0.3) -> serial.Serial:  # type: ignore[name-defined]
    ser = serial.Serial(port, baud, timeout=timeout)
    time.sleep(0.3)
    ser.read_all()  # drain boot text
    return ser


def main():
    ap = argparse.ArgumentParser(description="Serial DTR toggle and simple send/recv")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dtr", choices=["on", "off", "none"], default="none")
    ap.add_argument("--send", help="Optional line to send (newline auto-appended)")
    ap.add_argument("--read", type=float, default=1.0, help="Seconds to read after send")
    args = ap.parse_args()

    with open_ser(args.port, args.baud) as ser:
        if args.dtr != "none":
            ser.dtr = (args.dtr == "on")
            print(f"[Serial] DTR -> {ser.dtr}")
        if args.send:
            line = (args.send.strip() + "\n").encode("utf-8")
            ser.write(line); ser.flush()
            print(f"[Serial] sent: {args.send}")
        t_end = time.time() + args.read
        buf = b""
        while time.time() < t_end:
            chunk = ser.read(256)
            if chunk:
                buf += chunk
        if buf:
            print("[Serial] recv:\n" + buf.decode("utf-8", errors="ignore"))
        else:
            print("[Serial] no data received")


if __name__ == "__main__":
    main()
