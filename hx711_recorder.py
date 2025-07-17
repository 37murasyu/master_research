"""PC側操作用 Python モジュール

依存:
    pip install pyserial requests bleak

機能:
    from hx711_recorder import RecorderClient

    rc = RecorderClient(serial_port='COM5', baud=115200, base_url='http://192.168.1.50')
    rc.start_with_iso('2025-09-22T13:45:12.345Z')
    # or rc.start_with_epoch(int(time.time()*1000))
    ... 計測待ち ...
    rc.stop()
    csv_data = rc.download_csv()
    with open('log.csv','wb') as f: f.write(csv_data)
    rc.erase()

シリアル側コマンド:
    START <ISO8601>
    START_EPOCH <ms>
HTTP側エンドポイント:
    /status (GET)
    /stop   (GET)
    /download (GET, CSV)
    /erase (POST/GET)
"""
from __future__ import annotations
import time
import typing as _t
import asyncio as _asyncio  # BLE dump 等で使用 (aliased to avoid linter shadow warnings)
import requests
try:  # noqa: E402
    import serial  # type: ignore  # noqa: E401
except ImportError:  # pragma: no cover
    serial = None  # type: ignore

# BLE (optional)
try:
    from bleak import BleakClient, BleakScanner  # type: ignore
except ImportError:  # pragma: no cover
    BleakClient = None  # type: ignore
    BleakScanner = None  # type: ignore

BLE_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_CHAR_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_CHAR_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_DEVICE_NAME_DEFAULT = "HX711-Logger"

__all__ = [
    "RecorderClient",
    "BLERecorderClient",
    "BLERecorderClientSync",
    "create_ble_client",
    "record_serial_wifi_session",
    "record_ble_session",
]

class RecorderClient:
    def __init__(self, serial_port: str, baud: int = 115200, base_url: str = 'http://192.168.4.1', timeout: float = 2.0):
        self.serial_port = serial_port
        self.baud = baud
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._ser: _t.Any | None = None

    # ------------- Serial low-level -------------
    def _ensure_serial(self):
        if serial is None:
            raise RuntimeError('pyserial がインストールされていません: pip install pyserial')
        if self._ser and self._ser.is_open:
            return
        self._ser = serial.Serial(self.serial_port, self.baud, timeout=0.3)
        # 初期ブートメッセージ読み飛ばし
        time.sleep(0.5)
        self._drain()

    def _drain(self):
        if not self._ser:
            return
        while True:
            b = self._ser.read_all()
            if not b:
                break

    def _send_line(self, line: str):
        self._ensure_serial()
        assert self._ser
        self._ser.write((line.strip() + '\n').encode('utf-8'))
        self._ser.flush()

    def _read_until(self, substr: str, timeout: float = 3.0) -> str:
        self._ensure_serial()
        assert self._ser
        end = time.time() + timeout
        buf = ''
        while time.time() < end:
            chunk = self._ser.read(256).decode('utf-8', errors='ignore')
            if chunk:
                buf += chunk
                if substr in buf:
                    return buf
        return buf

    # ------------- High-level operations (Serial start) -------------
    def start_with_iso(self, iso_str: str) -> bool:
        self._send_line(f'START {iso_str}')
        resp = self._read_until('\n', 2.0)
        # 旧実装互換と新OK形式を両対応
        return ('FLASH: recording started' in resp) or ('OK START' in resp)

    def start_with_epoch(self, epoch_ms: int) -> bool:
        self._send_line(f'START_EPOCH {epoch_ms}')
        resp = self._read_until('\n', 2.0)
        return ('FLASH: recording started' in resp) or ('OK START_EPOCH' in resp)

    def status_serial(self) -> str:
        self._send_line('STATUS')
        return self._read_until('\n', 1.0)

    # ------------- Serial: stop/dump/erase -------------
    def stop_serial(self) -> bool:
        self._send_line('STOP')
        resp = self._read_until('\n', 1.0)
        return 'OK STOP' in resp

    def erase_serial(self) -> bool:
        self._send_line('ERASE')
        resp = self._read_until('\n', 1.0)
        return 'OK ERASE' in resp

    def _estimate_dump_timeout(self, min_timeout: float = 12.0, max_timeout: float = 120.0, margin: float = 2.0) -> float:
        """Estimate reasonable dump timeout from STATUS and baud rate.
        bytes/sec ≈ baud/10（スタート/ストップビット含む）。概算fileSizeへマージンを掛ける。
        """
        # 事前にSTATUSを取得して概算バイト数を推定
        try:
            self._send_line('STATUS')
            s = self._read_until('\n', 1.0)
            # 例: STATUS REC 123 5904 1737600000000
            for line in s.splitlines():
                if line.startswith('STATUS '):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        file_size = int(parts[3]) if parts[3].isdigit() else 0
                        # 最低でもヘッダ数十バイト
                        est_bytes = max(file_size, 64)
                        bps = max(self.baud, 9600) / 10.0
                        seconds = est_bytes / bps * margin + 3.0
                        return max(min_timeout, min(seconds, max_timeout))
        except Exception:
            pass
        return min_timeout

    def dump_csv_serial(self, timeout: float | None = None) -> bytes:
        """Request CSV over Serial framed by CSV_BEGIN/CSV_END and return raw bytes."""
        # 大きなログ向けに推定タイムアウト
        if timeout is None:
            timeout = self._estimate_dump_timeout()
        # 直前のゴミをドレイン
        self._ensure_serial(); self._drain()
        self._send_line('DUMP')
        self._ensure_serial()
        assert self._ser
        end = time.time() + float(timeout)
        started = False
        buf: bytearray = bytearray()
        header = b'CSV_BEGIN'
        footer = b'CSV_END'
        tail_keep = 16
        window: bytearray = bytearray()
        resent = False
        while time.time() < end:
            chunk: bytes = self._ser.read(512)
            if chunk:
                if not started:
                    window.extend(chunk)
                    if header in window:
                        # 切り出し開始位置を header の直後に
                        idx = window.find(header)
                        window = window[idx + len(header):]
                        # 改行が直後にある想定で、あれば捨てる
                        while window.startswith(b'\r') or window.startswith(b'\n'):
                            window = window[1:]
                        buf.extend(window)
                        window.clear()
                        started = True
                    else:
                        # window サイズを抑制
                        if len(window) > tail_keep:
                            window = window[-tail_keep:]
                else:
                    buf.extend(chunk)
                    # フッタ検出
                    if footer in buf:
                        end_idx = buf.find(footer)
                        data = bytes(buf[:end_idx])
                        # 直前の改行は落とす
                        data = data.rstrip(b'\r\n')
                        return data
            else:
                # ヘッダ未検出かつ無音が続く場合は1回だけDUMPを再送
                if not started and not resent and (end - time.time()) > 2.0:
                    self._send_line('DUMP'); resent = True
        raise TimeoutError('Serial CSV dump timeout')

    def stop_and_dump_serial(self, timeout: float | None = None) -> bytes:
        """Convenience: STOP then DUMP and return CSV bytes."""
        self.stop_serial()
        return self.dump_csv_serial(timeout=timeout)

    # ------------- Wi-Fi HTTP operations -------------
    def status(self) -> dict:
        r = requests.get(self.base_url + '/status', timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def stop(self) -> bool:
        r = requests.get(self.base_url + '/stop', timeout=self.timeout)
        return r.status_code == 200

    def download_csv(self) -> bytes:
        r = requests.get(self.base_url + '/download', timeout=self.timeout)
        r.raise_for_status()
        return r.content

    def erase(self) -> bool:
        r = requests.get(self.base_url + '/erase', timeout=self.timeout)
        return r.status_code == 200

    # ------------- Context manager -------------
    def close(self):
        if self._ser:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def __enter__(self):
        self._ensure_serial()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

class BLERecorderClient:
    """BLE 経由でESP32のロガーを制御し CSV を取得するクライアント

    典型利用:
        import asyncio
        from hx711_recorder import BLERecorderClient

        async def main():
            cli = BLERecorderClient()
            await cli.connect()  # デフォルトはデバイス名スキャン
            await cli.start_with_epoch(int(time.time()*1000))
            await _asyncio.sleep(3)
            await cli.stop()
            csv = await cli.dump_csv()
            print(csv[:200])
            await cli.erase()
            await cli.disconnect()
        asyncio.run(main())
    """
    def __init__(self, device_name: str = BLE_DEVICE_NAME_DEFAULT, address: str | None = None, timeout: float = 10.0):
        if BleakClient is None:
            raise RuntimeError("bleak がインストールされていません: pip install bleak")
        self.device_name = device_name
        self.address = address
        self.timeout = timeout
        self._client: _t.Optional[object] = None  # 実行時には BleakClient インスタンス
        self._notify_buf: list[str] = []

    async def connect(self):
        if self._client is not None:
            # 動的属性 is_connected を参照可能な場合だけ確認
            if getattr(self._client, "is_connected", False):
                return
        if self.address is None:
            dev = await BleakScanner.find_device_by_filter(lambda d, ad: d.name == self.device_name)
            if dev is None:
                raise RuntimeError(f"BLE device '{self.device_name}' not found")
            self.address = dev.address
        cli = BleakClient(self.address)
        await cli.connect(timeout=self.timeout)
        await cli.start_notify(BLE_CHAR_TX_UUID, self._on_notify)
        self._client = cli

    async def disconnect(self):
        if self._client is not None:
            try:
                await getattr(self._client, 'stop_notify')(BLE_CHAR_TX_UUID)
            finally:
                await getattr(self._client, 'disconnect')()
            self._client = None

    def _on_notify(self, _handle: int, data: bytearray):
        text = data.decode(errors='ignore')
        self._notify_buf.append(text)

    async def _write_cmd(self, cmd: str, wait: float = 0.15):  # noqa: D401
        """Write a command line over BLE and wait a short interval for response buffering."""
        if self._client is None or not getattr(self._client, 'is_connected', False):
            raise RuntimeError("BLE not connected")
        await getattr(self._client, 'write_gatt_char')(BLE_CHAR_RX_UUID, (cmd + "\n").encode())
        await _asyncio.sleep(wait)

    def _collect(self) -> str:
        joined = ''.join(self._notify_buf)
        self._notify_buf.clear()
        return joined

    async def start_with_iso(self, iso: str) -> bool:
        await self._write_cmd(f"START {iso}")
        return "OK START" in self._collect()

    async def start_with_epoch(self, epoch_ms: int) -> bool:
        await self._write_cmd(f"START_EPOCH {epoch_ms}")
        return "OK START_EPOCH" in self._collect()

    async def stop(self) -> bool:
        await self._write_cmd("STOP")
        return "OK STOP" in self._collect()

    async def erase(self) -> bool:
        await self._write_cmd("ERASE")
        return "OK ERASE" in self._collect()

    async def status(self) -> dict:
        await self._write_cmd("STATUS")
        txt = self._collect()
        # 例: STATUS REC 123 4567 1737600000000
        for line in txt.splitlines():
            if line.startswith("STATUS "):
                parts = line.strip().split()
                if len(parts) == 6:
                    return {
                        "state": parts[1],
                        "count": int(parts[2]),
                        "fileSize": int(parts[3]),
                        "baseEpochMs": int(parts[4]),
                    }
        return {"raw": txt}

    async def dump_csv(self) -> str:
        # CSV_BEGIN ... CSV_END を受信
        await self._write_cmd("DUMP", wait=0.05)
        csv_lines: list[str] = []
        t_end = time.time() + self.timeout
        started = False
        while time.time() < t_end:
            await _asyncio.sleep(0.05)
            chunk = self._collect()
            if chunk:
                for line in chunk.splitlines(True):  # keepends
                    if line.startswith("CSV_BEGIN"):
                        started = True
                        continue
                    if line.startswith("CSV_END"):
                        return ''.join(csv_lines)
                    if started:
                        csv_lines.append(line)
        raise TimeoutError("BLE dump timeout")

# --- 同期（簡便）ラッパ ---
class BLERecorderClientSync:
    def __init__(self, *a, **kw):
        self._cli = BLERecorderClient(*a, **kw)

    def __enter__(self):
        _asyncio.run(self._cli.connect())
        return self

    def __exit__(self, exc_type, exc, tb):
        _asyncio.run(self._cli.disconnect())

    def start_with_epoch(self, epoch_ms: int) -> bool:
        return _asyncio.run(self._cli.start_with_epoch(epoch_ms))

    def dump_csv(self) -> str:
        return _asyncio.run(self._cli.dump_csv())

    def stop(self) -> bool:
        return _asyncio.run(self._cli.stop())

    def erase(self) -> bool:
        return _asyncio.run(self._cli.erase())


# ---------------- Convenience helpers (import use) ----------------
def create_ble_client(device_name: str = BLE_DEVICE_NAME_DEFAULT, address: str | None = None, timeout: float = 10.0) -> BLERecorderClient:
    """Factory to create a `BLERecorderClient` only if bleak is available.

    Raises:
        RuntimeError: if bleak is not installed.
    """
    return BLERecorderClient(device_name=device_name, address=address, timeout=timeout)


def record_serial_wifi_session(port: str, base_url: str, *, start_epoch_ms: int | None = None, start_iso: str | None = None, duration_sec: float = 5.0) -> bytes:
    """High-level helper:
    1. (Optionally) start flash logging over Serial command.
    2. Sleep for duration.
    3. Stop over Wi-Fi and download CSV bytes.
    4. Returns CSV bytes (caller persists to file).
    """
    start_epoch_ms = start_epoch_ms if start_epoch_ms is not None else int(time.time() * 1000)
    with RecorderClient(port, 115200, base_url) as client_ctx:
        if start_iso:
            client_ctx.start_with_iso(start_iso)
        else:
            client_ctx.start_with_epoch(start_epoch_ms)
        time.sleep(duration_sec)
        client_ctx.stop()
        data = client_ctx.download_csv()
    return data


async def record_ble_session(*, start_epoch_ms: int | None = None, start_iso: str | None = None, duration_sec: float = 5.0, device_name: str = BLE_DEVICE_NAME_DEFAULT) -> str:
    """High-level BLE helper returning CSV text.
    Starts, waits, stops, dumps, erases (erase optional left to caller if needed).
    """
    start_epoch_ms = start_epoch_ms if start_epoch_ms is not None else int(time.time() * 1000)
    cli = BLERecorderClient(device_name=device_name)
    await cli.connect()
    try:
        if start_iso:
            await cli.start_with_iso(start_iso)
        else:
            await cli.start_with_epoch(start_epoch_ms)
        await _asyncio.sleep(duration_sec)
        await cli.stop()
        csv_text = await cli.dump_csv()
        return csv_text
    finally:
        await cli.disconnect()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='HX711 Recorder Client (Serial/Wi-Fi/BLE)')
    ap.add_argument('--port', help='Serial COM port (omit if only BLE)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--url', help='Base URL of device (Wi-Fi)')
    ap.add_argument('--start-iso')
    ap.add_argument('--start-epoch', type=int)
    ap.add_argument('--stop', action='store_true')
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--erase', action='store_true')
    ap.add_argument('--serial-dump', action='store_true', help='Dump CSV over Serial and save to serial_log.csv')
    ap.add_argument('--serial-erase', action='store_true', help='Erase log over Serial')
    ap.add_argument('--serial-stop-dump', action='store_true', help='STOP and DUMP over Serial, save to serial_log.csv')
    ap.add_argument('--status-serial', action='store_true', help='Print STATUS over Serial')
    ap.add_argument('--ble', action='store_true', help='Use BLE mode')
    args = ap.parse_args()

    if args.ble:
        async def run_ble():
            cli = BLERecorderClient()
            await cli.connect()
            try:
                if args.start_iso:
                    ok_iso_ble = await cli.start_with_iso(args.start_iso)
                    print('ble start_iso:', ok_iso_ble)
                if args.start_epoch is not None:
                    ok_epoch_ble = await cli.start_with_epoch(args.start_epoch)
                    print('ble start_epoch:', ok_epoch_ble)
                if args.stop:
                    print('ble stop:', await cli.stop())
                if args.erase:
                    print('ble erase:', await cli.erase())
                if args.download:
                    csv_text = await cli.dump_csv()
                    print('ble csv bytes:', len(csv_text))
                    with open('ble_log.csv','w', encoding='utf-8') as f_ble: f_ble.write(csv_text)
            finally:
                await cli.disconnect()
        _asyncio.run(run_ble())
    else:
        if not (args.port and args.url):
            # 純シリアルモード（STOP/DUMP/ERASE）を許容
            if not args.port:
                print('純シリアル操作には --port が必要です')
            else:
                rc = RecorderClient(args.port, args.baud, args.url or 'http://0.0.0.0')
                try:
                    if args.start_iso:
                        ok_iso = rc.start_with_iso(args.start_iso); print('start_with_iso(serial):', ok_iso)
                    if args.start_epoch is not None:
                        ok_epoch = rc.start_with_epoch(args.start_epoch); print('start_with_epoch(serial):', ok_epoch)
                    if args.stop:
                        print('stop(serial):', rc.stop_serial())
                    if args.status_serial:
                        print('status(serial):', rc.status_serial().strip())
                    if args.serial_dump:
                        csv_bytes = rc.dump_csv_serial()
                        print('serial dump bytes:', len(csv_bytes))
                        with open('serial_log.csv','wb') as f_sd: f_sd.write(csv_bytes)
                    if args.serial_stop_dump:
                        csv_bytes = rc.stop_and_dump_serial()
                        print('serial stop&dump bytes:', len(csv_bytes))
                        with open('serial_log.csv','wb') as f_sd: f_sd.write(csv_bytes)
                    if args.serial_erase:
                        print('erase(serial):', rc.erase_serial())
                finally:
                    rc.close()
        else:
            rc = RecorderClient(args.port, args.baud, args.url)
            try:
                if args.start_iso:
                    ok_iso = rc.start_with_iso(args.start_iso); print('start_with_iso:', ok_iso)
                if args.start_epoch is not None:
                    ok_epoch = rc.start_with_epoch(args.start_epoch); print('start_with_epoch:', ok_epoch)
                if args.stop:
                    print('stop:', rc.stop())
                if args.download:
                    csv_bytes = rc.download_csv(); print('download bytes:', len(csv_bytes))
                    with open('download_log.csv','wb') as f_dl: f_dl.write(csv_bytes)
                if args.erase:
                    print('erase:', rc.erase())
            finally:
                rc.close()
