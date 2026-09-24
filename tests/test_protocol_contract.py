"""Android が実際に送る電文を、PC 側が解釈できることを検証する。

片側だけのテストでは「両者が同じものを想定している」ことは保証できない。
Kotlin 側のテスト（ContractSampleTest）が組み立てた**生の JSON**を
ファイルに落とし、ここで decode する。どちらかが電文の形を変えれば、
実機を繋ぐ前にここで気づける。

サンプルの更新:
    cd mobile && ./gradlew :app:testDebugUnitTest
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.net import protocol as p

GOLDEN = Path(__file__).resolve().parent.parent / "mobile" / "contract" / "golden_messages.json"

pytestmark = pytest.mark.skipif(
    not GOLDEN.is_file(),
    reason=f"契約サンプルがありません（cd mobile && ./gradlew :app:testDebugUnitTest で生成）: {GOLDEN}",
)


def _messages() -> list[str]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_every_kotlin_message_decodes():
    """Android 側が送るすべての電文が PC 側で読めること。"""
    messages = _messages()
    assert messages, "契約サンプルが空"
    for raw in messages:
        p.decode(raw)  # ProtocolError が飛べば失敗


def test_hello_carries_role_and_session():
    hello = next(m for m in _messages() if json.loads(m).get("type") == "hello")
    decoded = p.decode(hello)
    assert isinstance(decoded, p.Hello)
    assert decoded.role in p.ROLES
    assert decoded.session


def test_hello_can_carry_the_device_id():
    """校正時の端末と照合するための ID。Kotlin 側が省略可能な項目として送れること。"""
    hellos = [p.decode(m) for m in _messages() if json.loads(m).get("type") == "hello"]
    assert any(h.device_id for h in hellos), "device_id 付きの hello が見本に無い"
    assert any(h.device_id is None for h in hellos), "古い形（device_id 無し）も読めること"


def test_calibration_frame_from_the_phone_decodes():
    """撮影要求への応答。base64 の字母と JPEG の先頭（FF D8）の検査を通ること。"""
    raw = next(m for m in _messages() if json.loads(m).get("type") == "calib_frame")
    frame = p.decode(raw)
    assert isinstance(frame, p.CalibrationFrame)
    assert frame.jpeg.startswith(b"\xff\xd8")
    assert (frame.width, frame.height) == (640, 360)


def test_sync_request_timestamp_is_an_integer():
    """浮動小数になっていると 2^53 を超えた時点でナノ秒の精度が落ちる。"""
    raw = next(m for m in _messages() if json.loads(m).get("type") == "sync_req")
    decoded = p.decode(raw)
    assert isinstance(decoded, p.SyncRequest)
    assert isinstance(json.loads(raw)["t1"], int)


def test_landmark_frames_have_the_expected_shape():
    frames = [m for m in _messages() if json.loads(m).get("type") == "landmarks"]
    assert frames, "landmarks の電文が含まれていない"

    for raw in frames:
        frame = p.decode(raw)
        assert isinstance(frame, p.LandmarkFrame)
        assert len(frame.landmarks) == p.LANDMARK_COUNT
        assert frame.width > 0 and frame.height > 0
        assert all(len(point) == 4 for point in frame.landmarks)


def test_capture_timestamp_survives_the_round_trip():
    """ナノ秒の値がそのまま復元できること。

    JSON を経由して float になると 1_725_699_123_456_789_000 は表現できず、
    数百ナノ秒ずれる。同期バッファはこの値でペアを組むので、ここは崩せない。
    """
    raw = next(m for m in _messages() if json.loads(m).get("type") == "landmarks")
    original = json.loads(raw)["t_capture_ns"]
    assert isinstance(original, int)
    assert p.decode(raw).t_capture_ns == original


def test_normalized_coordinates_convert_to_pixels():
    """x, y が [0,1] の正規化座標で、w/h を掛けてピクセルになる規約であること。

    既存の utils.extract_keypoints と同じ扱いにしてある。
    """
    raw = next(m for m in _messages() if json.loads(m).get("type") == "landmarks")
    frame = p.decode(raw)

    for index in range(p.LANDMARK_COUNT):
        x, y, _z, _v = frame.landmarks[index]
        assert 0.0 <= x <= 1.0, f"lm[{index}].x が正規化されていない: {x}"
        assert 0.0 <= y <= 1.0, f"lm[{index}].y が正規化されていない: {y}"

    px, py = frame.pixel_xy(0)
    assert 0 <= px <= frame.width
    assert 0 <= py <= frame.height


def test_frames_flow_through_the_sync_buffer():
    """実際の電文を同期バッファに入れて、ペアが組めること。

    decode できるだけでは足りない。下流まで通ることを確かめる。
    """
    from app.net.sync_buffer import GridSpec, SyncBuffer

    frames = [p.decode(m) for m in _messages() if json.loads(m).get("type") == "landmarks"]
    by_role = {frame.role: frame for frame in frames}
    assert set(by_role) == {"cam0", "cam1"}, "両方のロールのサンプルが要る"

    buffer = SyncBuffer(window_sec=2.0, grid=GridSpec(target_hz=30.0, max_gap_ms=100.0))
    base = min(frame.t_capture_ns for frame in frames)

    # 実サンプルを 100ms 刻みで並べ直して流す
    for step in range(6):
        for role, template in by_role.items():
            buffer.push(
                p.LandmarkFrame(
                    role=role,
                    seq=step,
                    t_capture_ns=base + step * 100_000_000,
                    width=template.width,
                    height=template.height,
                    landmarks=template.landmarks,
                )
            )

    pairs = buffer.drain()
    assert pairs, "実電文からペアが組めていない"
    assert all(set(pair.frames) == {"cam0", "cam1"} for pair in pairs)


PC_MESSAGES = Path(__file__).resolve().parent.parent / "mobile" / "contract" / "pc_messages.json"


def _pc_samples() -> list[str]:
    """PC が端末へ送る電文の見本。Kotlin 側のテストが同じファイルを読む。"""
    return [
        p.encode(p.SyncResponse(t1=1_000_000_000, t2=1_000_500_000, t3=1_000_600_000)),
        p.encode(p.CaptureRequest(id=7, at_ns=1_725_699_123_456_789_000)),
        p.encode(p.CaptureRequest(id=8)),
        # ライブ表示用。縮小と画質の指定が付く。
        p.encode(p.CaptureRequest(id=9, max_width=640, quality=70)),
    ]


class TestPcToDevice:
    """PC → 端末の向きも契約で守る。

    これまで契約テストは端末 → PC の一方向だけだった。撮影指示は逆向きなので、
    こちらが変わっても実機を繋ぐまで気づけない。

    このテストはファイルを**比較するだけ**にしてある。毎回書き出すと、
    テストがリポジトリを書き換えてしまう（Windows では改行も変わる）。
    更新するときは UPDATE_CONTRACT=1 を付けて実行する。
    """

    def test_samples_match_the_file_the_android_tests_read(self):
        samples = _pc_samples()

        if os.getenv("UPDATE_CONTRACT") == "1":
            PC_MESSAGES.parent.mkdir(parents=True, exist_ok=True)
            PC_MESSAGES.write_text(
                json.dumps(samples, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )

        assert PC_MESSAGES.is_file(), (
            f"PC 側の契約サンプルがありません。UPDATE_CONTRACT=1 で再生成してください: {PC_MESSAGES}"
        )
        assert json.loads(PC_MESSAGES.read_text(encoding="utf-8")) == samples

    def test_capture_request_carries_the_target_time(self):
        """端末が「いつのフレームを返すか」を決められること。"""
        decoded = p.decode(json.loads(PC_MESSAGES.read_text(encoding="utf-8"))[1])
        assert isinstance(decoded, p.CaptureRequest)
        assert decoded.at_ns is not None
