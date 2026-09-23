"""被験者ゲージの行の書式 v2（``@@GAUGE ``）を検証する。

これは計測の子プロセス（別セッションが実装）と GUI（Qt）の**契約**。子は
``sys.stdout.write(encode(frame))`` で 1 行を出し、GUI 側が ``decode`` する。

子プロセスからの入力は信用できない（壊れた JSON・未知の版・変な値）ので、
``decode`` は例外を投げず ``None`` を返す設計にしてある。GUI 側は 1 行読むたびに
None チェックだけすればよく、壊れた 1 行のために描画ループを止めずに済む。
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from app.gauge import protocol as g


def _reading(**overrides):
    values = dict(now=12.3, prev=10.1, band=(10.0, 20.0), w1rm=66.0)
    values.update(overrides)
    return g.PartReading(**values)


def _full_frame(**overrides):
    values = dict(
        link="connected",
        rep=3,
        source="measure",
        parts={name: _reading() for name in g.PART_NAMES},
    )
    values.update(overrides)
    return g.GaugeFrame(**values)


class TestEncode:
    def test_encode_writes_one_line_with_prefix_and_newline(self):
        line = g.encode(_full_frame())
        assert line.startswith(g.PREFIX)
        assert line.endswith("\n")
        # 途中に改行が紛れ込むと、行単位で読む側が壊れる
        assert line.count("\n") == 1

    def test_encode_writes_null_for_non_finite(self):
        frame = _full_frame(
            parts={
                "elbow_L": g.PartReading(
                    now=float("nan"),
                    prev=float("inf"),
                    band=(float("-inf"), 20.0),
                    w1rm=float("nan"),
                )
            }
        )
        line = g.encode(frame)
        body = json.loads(line[len(g.PREFIX) :])
        reading = body["parts"]["elbow_L"]
        assert reading["now"] is None
        assert reading["prev"] is None
        assert reading["w1rm"] is None
        # 片方だけでも非有限なら band 全体を null にする（lo/hi 片方欠けは無効な組）
        assert reading["band"] is None

    def test_encode_rounds_to_one_decimal(self):
        frame = _full_frame(
            parts={"elbow_L": g.PartReading(now=12.34, prev=10.16, band=(9.951, 20.049), w1rm=66.0)}
        )
        line = g.encode(frame)
        body = json.loads(line[len(g.PREFIX) :])
        reading = body["parts"]["elbow_L"]
        assert reading["now"] == 12.3
        assert reading["prev"] == 10.2
        assert reading["band"] == [10.0, 20.0]

    def test_encode_fits_in_pipe_buf(self):
        """macOS の PIPE_BUF（512 バイト）未満に収まること。最悪の場合で確かめる。"""
        worst = g.GaugeFrame(
            link="connected",
            rep=99999,
            source="replay",
            parts={
                name: g.PartReading(now=12345.6, prev=98765.4, band=(11111.1, 99999.9), w1rm=54321.0)
                for name in g.PART_NAMES
            },
        )
        line = g.encode(worst)
        assert len(line.encode("ascii")) < 512

    def test_encoded_line_is_ascii(self):
        line = g.encode(_full_frame())
        assert line.isascii()
        # デコードできない非 ASCII を混入させないことを、エンコード結果そのもので確かめる
        line.encode("ascii")


class TestRoundTrip:
    @pytest.mark.parametrize(
        "frame",
        [
            pytest.param(
                g.GaugeFrame(link="waiting", rep=0, source="measure", parts={}),
                id="接続待ちで値なし",
            ),
            pytest.param(
                _full_frame(
                    rep=1,
                    parts={
                        "elbow_L": g.PartReading(now=12.3, prev=None, band=(10.0, 20.0), w1rm=66.0)
                    },
                ),
                id="1回目でprev=None",
            ),
            pytest.param(
                _full_frame(
                    parts={
                        "elbow_L": g.PartReading(now=12.3, prev=10.1, band=None, w1rm=66.0)
                    }
                ),
                id="band=None",
            ),
            pytest.param(
                _full_frame(
                    parts={
                        "elbow_L": g.PartReading(now=12.3, prev=10.1, band=(10.0, 20.0), w1rm=None)
                    }
                ),
                id="w1rm=None",
            ),
            pytest.param(
                _full_frame(
                    parts={
                        "elbow_L": g.PartReading(now=None, prev=10.1, band=(10.0, 20.0), w1rm=66.0)
                    }
                ),
                id="now=None",
            ),
            pytest.param(_full_frame(source="replay"), id="source=replay"),
        ],
    )
    def test_round_trip_restores_the_frame(self, frame):
        assert g.decode(g.encode(frame)) == frame


class TestDecodeRejects:
    def test_decode_rejects_line_without_prefix(self):
        body = json.dumps({"v": 2, "link": "waiting", "rep": 0, "source": "measure", "parts": {}})
        assert g.decode(body + "\n") is None
        assert g.decode("@@OTHER " + body + "\n") is None

    def test_decode_rejects_broken_json(self):
        assert g.decode(g.PREFIX + "{これは JSON ではない\n") is None
        assert g.decode(g.PREFIX + "\n") is None

    @pytest.mark.parametrize("version", [1, 3, "2"])
    def test_decode_rejects_other_versions(self, version):
        body = json.dumps(
            {"v": version, "link": "waiting", "rep": 0, "source": "measure", "parts": {}}
        )
        assert g.decode(g.PREFIX + body + "\n") is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"link": "bogus"},
            {"link": None},
            {"rep": -1},
            {"rep": 1.5},
            {"rep": True},
            {"rep": "1"},
            {"rep": None},
            {"source": "bogus"},
            {"source": None},
            {"parts": "not-a-dict"},
            {"parts": [1, 2]},
            {"parts": None},
        ],
    )
    def test_decode_rejects_bad_required_fields(self, overrides):
        body = dict(v=2, link="waiting", rep=0, source="measure", parts={})
        body.update(overrides)
        assert g.decode(g.PREFIX + json.dumps(body) + "\n") is None

    def test_decode_accepts_crlf(self):
        body = json.dumps({"v": 2, "link": "connected", "rep": 1, "source": "measure", "parts": {}})
        with_lf = g.decode(g.PREFIX + body + "\n")
        with_crlf = g.decode(g.PREFIX + body + "\r\n")
        assert with_crlf == with_lf
        assert with_crlf is not None

    def test_decode_ignores_unknown_keys_and_parts(self):
        body = {
            "v": 2,
            "link": "connected",
            "rep": 4,
            "source": "measure",
            "future_field": "未知の鍵",
            "parts": {
                "elbow_L": {"now": 12.3, "prev": 10.1, "band": [10.0, 20.0], "w1rm": 66.0},
                "ankle_L": {"now": 1.0},  # 契約に無い部位
            },
        }
        frame = g.decode(g.PREFIX + json.dumps(body) + "\n")
        assert frame is not None
        assert "ankle_L" not in frame.parts
        assert frame.parts["elbow_L"].now == 12.3

    def test_decode_turns_invalid_band_into_none(self):
        for bad_band in ([10.0], [10.0, 20.0, 30.0], [20.0, 10.0], [10.0, 10.0], ["a", "b"], "10,20"):
            body = {
                "v": 2,
                "link": "connected",
                "rep": 1,
                "source": "measure",
                "parts": {"elbow_L": {"now": 12.3, "band": bad_band}},
            }
            frame = g.decode(g.PREFIX + json.dumps(body) + "\n")
            assert frame is not None
            assert frame.parts["elbow_L"].band is None, bad_band

    def test_decode_turns_invalid_prev_and_w1rm_into_none(self):
        """prev・w1rm が数でなければ None にする（now と違い、部位ごと捨てはしない）。"""
        body = {
            "v": 2,
            "link": "connected",
            "rep": 1,
            "source": "measure",
            "parts": {"elbow_L": {"now": 12.3, "prev": "abc", "w1rm": [1, 2]}},
        }
        frame = g.decode(g.PREFIX + json.dumps(body) + "\n")
        assert frame is not None
        reading = frame.parts["elbow_L"]
        assert reading.now == 12.3
        assert reading.prev is None
        assert reading.w1rm is None

    def test_decode_drops_part_with_non_numeric_now(self):
        body = {
            "v": 2,
            "link": "connected",
            "rep": 1,
            "source": "measure",
            "parts": {
                "elbow_L": {"now": "abc"},
                "elbow_R": {"now": 12.3},
            },
        }
        frame = g.decode(g.PREFIX + json.dumps(body) + "\n")
        assert frame is not None
        assert "elbow_L" not in frame.parts
        assert "elbow_R" in frame.parts


def test_protocol_does_not_import_qt():
    """子プロセスは Qt を読まない。protocol.py が誤って import しないことを別プロセスで確かめる。"""
    script = (
        "import sys; import app.gauge.protocol; "
        "print('PySide6' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
