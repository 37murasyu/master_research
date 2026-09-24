"""スマホ ⇄ PC のプロトコルを検証する。

これは Android 側と PC 側を並行開発するための**契約**なので、
形が変わったら気づけるように厳しく固定する。

受け取るメッセージはネットワーク越しの入力であり信用できない。
壊れた入力で計測が落ちないことも、ここで担保する。
"""

from __future__ import annotations

import json

import pytest

from app.net import protocol as p


def _valid_landmarks_payload(**overrides):
    payload = {
        "type": "landmarks",
        "role": "cam0",
        "seq": 42,
        "t_capture_ns": 1_725_699_123_456_789_000,
        "w": 1280,
        "h": 720,
        "lm": [[0.5, 0.5, 0.0, 0.9] for _ in range(p.LANDMARK_COUNT)],
    }
    payload.update(overrides)
    return payload


class TestLandmarkMessage:
    def test_round_trip(self):
        original = p.LandmarkFrame(
            role="cam0",
            seq=42,
            t_capture_ns=1_725_699_123_456_789_000,
            width=1280,
            height=720,
            landmarks=[(0.1, 0.2, 0.3, 0.9)] * p.LANDMARK_COUNT,
        )
        decoded = p.decode(p.encode(original))
        assert decoded == original

    def test_decodes_expected_payload(self):
        frame = p.decode(json.dumps(_valid_landmarks_payload()))
        assert isinstance(frame, p.LandmarkFrame)
        assert frame.role == "cam0"
        assert len(frame.landmarks) == p.LANDMARK_COUNT

    def test_pixel_coordinates_use_reported_frame_size(self):
        """正規化座標 × 実サイズ。既存 utils.extract_keypoints と同じ規約にする。"""
        payload = _valid_landmarks_payload(
            w=1280, h=720, lm=[[0.5, 0.25, 0.0, 1.0]] * p.LANDMARK_COUNT
        )
        frame = p.decode(json.dumps(payload))
        x, y = frame.pixel_xy(0)
        assert (x, y) == (640.0, 180.0)

    def test_unknown_fields_are_ignored(self):
        """Android 側が先に項目を増やしても PC 側が落ちないこと（前方互換）。"""
        payload = _valid_landmarks_payload(battery_pct=87, future_field={"a": 1})
        frame = p.decode(json.dumps(payload))
        assert frame.seq == 42


class TestMalformedInput:
    @pytest.mark.parametrize(
        "broken",
        [
            "これはJSONではない",
            "{}",
            json.dumps({"type": "landmarks"}),
            json.dumps(_valid_landmarks_payload(role="cam9")),
            json.dumps(_valid_landmarks_payload(lm=[[0.1, 0.2, 0.3, 0.4]] * 5)),
            json.dumps(_valid_landmarks_payload(t_capture_ns="いつか")),
            json.dumps(_valid_landmarks_payload(w=0)),
            json.dumps(_valid_landmarks_payload(type="不明な種類")),
        ],
    )
    def test_rejects_with_protocol_error(self, broken):
        """壊れた入力は ProtocolError に集約する。呼び出し側は 1 種類だけ捕まえればよい。"""
        with pytest.raises(p.ProtocolError):
            p.decode(broken)

    def test_error_message_names_the_problem(self):
        with pytest.raises(p.ProtocolError) as exc:
            p.decode(json.dumps(_valid_landmarks_payload(role="cam9")))
        assert "cam9" in str(exc.value) or "role" in str(exc.value)


def _deep(depth: int) -> str:
    return "[" * depth + "]" * depth


def _with_point(point: str) -> str:
    """最初のランドマークだけを JSON の生の文字列 ``point`` に差し替えた電文。"""
    raw = json.dumps(_valid_landmarks_payload())
    return raw.replace("[0.5, 0.5, 0.0, 0.9]", point, 1)


class TestHostileInput:
    """壊れた電文・悪意のある電文でも、``decode`` は ProtocolError だけを投げる。

    受信サーバは ProtocolError だけを捕まえて数える。以前は巨大な整数の座標で OverflowError、深い入れ子で
    RecursionError が外へ出て、接続ごと 1011 で切れ、protocol_errors にも数えなかった。NaN・Inf の座標や、
    負・64 bit を超える時刻は素通りしていた（三角測量と同期バッファが黙って壊れる）。
    """

    @pytest.mark.parametrize(
        "broken",
        [
            pytest.param(_with_point("[" + "9" * 400 + ", 0.5, 0, 1]"), id="huge-int-coordinate"),
            pytest.param('{"type":"landmarks","lm":' + _deep(100_000) + "}", id="deep-nesting"),
            pytest.param('{"type":' + _deep(990) + "}", id="deep-type"),
            pytest.param(_with_point("[NaN, 0.5, 0, 1]"), id="nan-x"),
            pytest.param(_with_point("[0.5, 0.5, 0, Infinity]"), id="inf-visibility"),
            pytest.param(_with_point("[0.5, -Infinity, 0, 1]"), id="minus-inf-y"),
            pytest.param(_with_point("[1e400, 0.5, 0, 1]"), id="float-overflow"),
            pytest.param(_with_point('["nan", 0.5, 0, 1]'), id="nan-string"),
            pytest.param(json.dumps(_valid_landmarks_payload(t_capture_ns=-5)), id="negative-time"),
            pytest.param(json.dumps(_valid_landmarks_payload(t_capture_ns=2**63)), id="time-over-int64"),
            pytest.param(json.dumps(_valid_landmarks_payload(t_capture_ns=2**80)), id="time-2**80"),
            pytest.param(json.dumps(_valid_landmarks_payload(seq=10**4000)), id="huge-seq"),
            pytest.param(json.dumps(_valid_landmarks_payload(w=10**9, h=10**9)), id="huge-frame"),
            pytest.param(json.dumps({"type": "sync_req", "t1": -1}), id="negative-sync-t1"),
            pytest.param(json.dumps({"type": "sync_req", "t1": 2**80}), id="huge-sync-t1"),
            pytest.param(json.dumps({"type": "sync_res", "t1": 1, "t2": -2, "t3": 3}), id="negative-sync-t2"),
            pytest.param(json.dumps({"type": "capture_req", "id": 1, "at_ns": -1}), id="negative-at-ns"),
            pytest.param(
                json.dumps({"type": "calib_frame", "role": "cam0", "id": 3, "t_capture_ns": -1,
                            "w": 1280, "h": 720, "jpeg": "/9j/2Q=="}),
                id="negative-calib-time",
            ),
        ],
    )
    def test_rejects_with_protocol_error_only(self, broken):
        with pytest.raises(p.ProtocolError):
            p.decode(broken)

    def test_extreme_but_valid_values_are_accepted(self):
        """64 bit に収まる最大の時刻と、画面の外の座標（MediaPipe は画面外の点も返す）は通す。"""
        frame = p.decode(json.dumps(_valid_landmarks_payload(
            t_capture_ns=2**63 - 1, lm=[[-0.3, 1.4, -2.0, 0.0]] * p.LANDMARK_COUNT,
        )))
        assert frame.t_capture_ns == 2**63 - 1
        assert frame.landmarks[0] == (-0.3, 1.4, -2.0, 0.0)


class TestTimeSync:
    def test_offset_is_zero_for_symmetric_delay(self):
        """往復の遅延が対称なら、時計ずれ 0 と算出されること。"""
        # 端末時刻 t1=1000 で送信 -> PC が 1050 で受信・応答 -> 端末が 1100 で受信
        result = p.compute_clock_offset(t1=1000, t2=1050, t3=1050, t4=1100)
        assert result.offset_ns == 0
        assert result.rtt_ns == 100

    def test_offset_detects_phone_clock_behind(self):
        """端末の時計が PC より 500 遅れている場合を検出できること。"""
        # PC 時刻 = 端末時刻 + 500。往復 100（片道 50）
        result = p.compute_clock_offset(t1=1000, t2=1550, t3=1550, t4=1100)
        assert result.offset_ns == 500

    def test_rtt_excludes_server_processing_time(self):
        """サーバ内の処理時間 (t3-t2) は往復遅延から除く。"""
        result = p.compute_clock_offset(t1=1000, t2=1040, t3=1060, t4=1100)
        assert result.rtt_ns == 80

    def test_best_sample_is_the_one_with_lowest_rtt(self):
        """RTT が最小のサンプルが最も信頼できる（ジッタの影響が小さい）。"""
        samples = [
            p.compute_clock_offset(t1=0, t2=100, t3=100, t4=1000),   # rtt 1000
            p.compute_clock_offset(t1=0, t2=60, t3=60, t4=100),      # rtt 100 ← これ
            p.compute_clock_offset(t1=0, t2=300, t3=300, t4=500),    # rtt 500
        ]
        assert p.best_offset(samples) is samples[1]

    def test_best_offset_of_empty_is_none(self):
        assert p.best_offset([]) is None

    def test_sync_messages_round_trip(self):
        request = p.SyncRequest(t1=1234)
        assert p.decode(p.encode(request)) == request

        response = p.SyncResponse(t1=1234, t2=5678, t3=5680)
        assert p.decode(p.encode(response)) == response


class TestHello:
    def test_hello_round_trip(self):
        hello = p.Hello(role="cam1", device="Pixel 8", session="abc123")
        assert p.decode(p.encode(hello)) == hello

    def test_hello_rejects_unknown_role(self):
        with pytest.raises(p.ProtocolError):
            p.decode(json.dumps({"type": "hello", "role": "left", "device": "x", "session": "s"}))


# JPEG の最小形（SOI + EOI）。中身は見ないので、これで検査の条件を満たす。
TINY_JPEG = b"\xff\xd8\xff\xd9"


class TestCaptureRequest:
    """PC → 端末の撮影指示。校正用の画像を撮ってもらう。"""

    def test_round_trip(self):
        original = p.CaptureRequest(id=7, at_ns=1_725_699_123_456_789_000)
        assert p.decode(p.encode(original)) == original

    def test_target_time_is_optional(self):
        """時刻を指定しない（次のフレームでよい）場合も読めること。"""
        decoded = p.decode('{"type": "capture_req", "id": 7}')
        assert decoded == p.CaptureRequest(id=7, at_ns=None)

    def test_size_and_quality_round_trip(self):
        """ライブ表示用に小さく軽い JPEG を頼めること。校正は全解像度のまま。"""
        original = p.CaptureRequest(id=9, max_width=640, quality=70)
        assert p.decode(p.encode(original)) == original

    def test_size_and_quality_are_omitted_when_unset(self):
        """古い端末が知らない項目を送らない。"""
        encoded = p.encode(p.CaptureRequest(id=1))
        assert "max_width" not in encoded
        assert "quality" not in encoded

    @pytest.mark.parametrize(
        "field, value",
        [
            ("max_width", 0),
            ("max_width", -640),
            ("max_width", True),
            ("max_width", 640.5),
            ("quality", 0),
            ("quality", 101),
            ("quality", "90"),
        ],
    )
    def test_rejects_invalid_size_or_quality(self, field, value):
        payload = {"type": "capture_req", "id": 1, field: value}
        with pytest.raises(p.ProtocolError):
            p.decode(json.dumps(payload))


class TestCalibrationFrame:
    """端末 → PC の校正用画像。姿勢推定と同じフレームを JPEG で送る。"""

    def _frame(self, **overrides):
        values = dict(
            role="cam0",
            id=3,
            t_capture_ns=1_725_699_123_456_789_000,
            width=1280,
            height=720,
            jpeg=TINY_JPEG,
        )
        values.update(overrides)
        return p.CalibrationFrame(**values)

    def test_round_trip(self):
        original = self._frame()
        assert p.decode(p.encode(original)) == original

    def test_rejects_data_that_is_not_jpeg(self):
        """画像でないものを掴むと、検出側が黙って 0 枚になる。"""
        import base64

        payload = json.dumps({
            "type": "calib_frame", "role": "cam0", "id": 3,
            "t_capture_ns": 1, "w": 1280, "h": 720,
            "jpeg": base64.b64encode("これは画像ではない".encode("utf-8")).decode("ascii"),
        })
        with pytest.raises(p.ProtocolError):
            p.decode(payload)

    def test_rejects_broken_base64(self):
        payload = json.dumps({
            "type": "calib_frame", "role": "cam0", "id": 3,
            "t_capture_ns": 1, "w": 1280, "h": 720, "jpeg": "!!!壊れている!!!",
        })
        with pytest.raises(p.ProtocolError):
            p.decode(payload)

    def test_rejects_an_oversized_image(self):
        """無線の相手からの入力なので、受け取る大きさに上限を置く。"""
        import base64

        oversized = TINY_JPEG + b"\x00" * (p.MAX_CALIBRATION_BYTES + 1)
        payload = json.dumps({
            "type": "calib_frame", "role": "cam0", "id": 3,
            "t_capture_ns": 1, "w": 1280, "h": 720,
            "jpeg": base64.b64encode(oversized).decode("ascii"),
        })
        with pytest.raises(p.ProtocolError):
            p.decode(payload)


class TestHelloDeviceId:
    """同じ機種 2 台を区別するための端末 ID。

    ``Build.MODEL`` は 2 台とも "Pixel 7a" になる。機種名だけでは、
    内部パラメータを取り違えても、役割を入れ替えても気づけない。
    """

    def test_device_id_round_trip(self):
        original = p.Hello(role="cam1", device="Google Pixel 7a", session="ab12", device_id="9f3c1d")
        assert p.decode(p.encode(original)) == original

    def test_older_apps_without_device_id_still_connect(self):
        decoded = p.decode(json.dumps({
            "type": "hello", "role": "cam0", "device": "Google Pixel 7a", "session": "ab12",
        }))
        assert decoded.device_id is None
