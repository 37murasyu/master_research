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
