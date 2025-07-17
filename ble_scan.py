"""Simple BLE scanner (bleak) to find device name/address.

Usage (PowerShell):
  python .\ble_scan.py                # list all devices for ~5 seconds
  python .\ble_scan.py --name HX711-Logger
  python .\ble_scan.py --uuids 6E400001-B5A3-F393-E0A9-E50E24DCCA9E
"""
from __future__ import annotations
import argparse
import asyncio
try:
    from bleak import BleakScanner  # type: ignore
except ImportError:
    raise SystemExit("bleak が未インストールです。`pip install bleak` を実行してください。")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="Filter by device name")
    ap.add_argument("--uuids", nargs="*", help="Filter by service UUID(s)")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    def _filter(d):
        # Name filter (exact match)
        if args.name and (d.name or "") != args.name:
            return False
        # UUID filter (best-effort; many Bleak versions do not expose service UUIDs here)
        if args.uuids:
            meta = getattr(d, "metadata", None)
            service_uuids = None
            if isinstance(meta, dict):
                # Newer bleak may expose 'uuids' or 'service_uuids' in metadata
                service_uuids = meta.get("service_uuids") or meta.get("uuids")
            if service_uuids is None:
                # Can't evaluate UUID filter reliably -> skip filtering but warn once
                pass
            else:
                try:
                    if not set(args.uuids).issubset(set(service_uuids)):
                        return False
                except Exception:
                    pass
        return True

    print(f"Scanning for {args.timeout}s ...")
    devices = await BleakScanner.discover(timeout=args.timeout)
    rows = []
    for d in devices:
        if not _filter(d):
            continue
        rows.append((d.name, d.address, getattr(d, "rssi", None)))
    if not rows:
        print("No matching BLE devices found.")
        return
    print("Name, Address, RSSI")
    for name, addr, rssi in sorted(rows, key=lambda x: (x[2] is not None, x[2]), reverse=True):
        print(f"{name or '(no-name)'}, {addr}, {rssi if rssi is not None else ''}")


if __name__ == "__main__":
    asyncio.run(main())
