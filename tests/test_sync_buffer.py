"""時刻同期バッファを検証する。

三角測量は「2 台の同時刻の 2D 点」を前提にしている（``utils.DLT``）。
既存コードはこれを「同じループ回で読んだフレーム＝同時刻」という
暗黙の仮定で満たしていた（``master_research_code.py:2945-2946`` の逐次 grab）。
リポジトリ全体で ``CAP_PROP_POS_MSEC`` の使用箇所はゼロで、
タイムスタンプによる対応付けは存在しない。

無線では到着順が当てにならないので、**時刻でペアを組む**。
そのうえで共通の等間隔グリッドへ線形補間して再標本化する。最近傍で組むより
整合が良く、ランドマークは単なる点列なので補間が自明かつ安価。
"""

from __future__ import annotations

import pytest

from app.net.protocol import LANDMARK_COUNT, LandmarkFrame
from app.net.sync_buffer import GridSpec, SyncBuffer

MS = 1_000_000  # 1 ミリ秒 = 10^6 ナノ秒


def _frame(role: str, seq: int, t_ms: float, value: float) -> LandmarkFrame:
    """全ランドマークが同じ値を持つフレーム。補間結果を検算しやすくする。"""
    return LandmarkFrame(
        role=role,
        seq=seq,
        t_capture_ns=int(t_ms * MS),
        width=1280,
        height=720,
        landmarks=[(value, value, value, 1.0)] * LANDMARK_COUNT,
    )


def _buffer(target_hz: float = 10.0, window_sec: float = 2.0, max_gap_ms: float = 250.0) -> SyncBuffer:
    # 10 Hz（周期 100ms）にしておくと、テストの時刻が読みやすい。
    return SyncBuffer(window_sec=window_sec, grid=GridSpec(target_hz=target_hz, max_gap_ms=max_gap_ms))


class TestPairing:
    def test_nothing_emitted_with_only_one_role(self):
        """片方しか来ていない間は出さない。片眼では三角測量できない。"""
        buf = _buffer()
        for i in range(5):
            buf.push(_frame("cam0", i, i * 100, 0.5))
        assert buf.drain() == []

    def test_emits_pairs_when_both_roles_present(self):
        buf = _buffer()
        for i in range(4):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100, 0.2))
        pairs = buf.drain()
        assert pairs, "両方揃っているのにペアが出ない"
        assert all(set(pair.frames) == {"cam0", "cam1"} for pair in pairs)

    def test_pairs_are_in_increasing_time_order(self):
        buf = _buffer()
        for i in range(6):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100, 0.2))
        times = [pair.t_ns for pair in buf.drain()]
        assert times == sorted(times)

    def test_same_pair_is_not_emitted_twice(self):
        buf = _buffer()
        for i in range(6):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100, 0.2))
        first = {pair.t_ns for pair in buf.drain()}
        second = {pair.t_ns for pair in buf.drain()}
        assert not (first & second), "同じ時刻を二度出している"


class TestInterpolation:
    def test_midpoint_is_linearly_interpolated(self):
        """グリッド時刻が 2 サンプルの中間なら、値も中間になること。"""
        buf = _buffer(target_hz=10.0, window_sec=2.0, max_gap_ms=250.0)
        # cam1 を 50ms ずらす。グリッドは cam1 の最初の時刻 50ms から始まる。
        buf.push(_frame("cam0", 0, 0, 0.0))
        buf.push(_frame("cam0", 1, 100, 1.0))
        buf.push(_frame("cam0", 2, 200, 2.0))
        buf.push(_frame("cam1", 0, 50, 10.0))
        buf.push(_frame("cam1", 1, 150, 11.0))
        buf.push(_frame("cam1", 2, 250, 12.0))

        pairs = buf.drain()
        assert pairs

        first = pairs[0]
        assert first.t_ns == 50 * MS
        # cam0 は 0ms(0.0) と 100ms(1.0) の中間 -> 0.5
        assert first.frames["cam0"].landmarks[0][0] == pytest.approx(0.5)
        # cam1 は 50ms にサンプルがあるのでそのまま
        assert first.frames["cam1"].landmarks[0][0] == pytest.approx(10.0)

    def test_exact_sample_time_is_not_distorted(self):
        buf = _buffer()
        for i in range(4):
            buf.push(_frame("cam0", i, i * 100, float(i)))
            buf.push(_frame("cam1", i, i * 100, float(i) * 10)
                     )
        pairs = buf.drain()
        for pair in pairs:
            k = pair.t_ns // (100 * MS)
            assert pair.frames["cam0"].landmarks[0][0] == pytest.approx(float(k))
            assert pair.frames["cam1"].landmarks[0][0] == pytest.approx(float(k) * 10)

    def test_frame_size_is_preserved(self):
        buf = _buffer()
        for i in range(4):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100, 0.2))
        pair = buf.drain()[0]
        assert pair.frames["cam0"].width == 1280
        assert pair.frames["cam0"].height == 720


class TestGapHandling:
    def test_grid_point_inside_a_long_gap_is_dropped(self):
        """欠測が長すぎる区間は補間せず捨てる。

        無線ではパケットロスが日常的に起きる。長い穴を線形補間で埋めると、
        実際には動いていた手を「まっすぐ動いた」ことにしてしまう。
        """
        buf = _buffer(target_hz=10.0, window_sec=5.0, max_gap_ms=150.0)
        for i in range(3):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100, 0.2))
        # cam1 が 200ms から 800ms まで 600ms 欠測（許容 150ms を超える）
        for i, t in enumerate((300, 400, 500, 600, 700, 800)):
            buf.push(_frame("cam0", 10 + i, t, 0.1))
        buf.push(_frame("cam1", 10, 800, 0.2))

        emitted = {pair.t_ns for pair in buf.drain()}
        for t_ms in (300, 400, 500, 600, 700):
            assert t_ms * MS not in emitted, f"{t_ms}ms は欠測区間なので出してはいけない"

    def test_recovers_after_the_gap(self):
        buf = _buffer(target_hz=10.0, window_sec=5.0, max_gap_ms=150.0)
        for i in range(3):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100, 0.2))
        buf.drain()
        # 長い欠測のあと、両方が復帰する
        for i, t in enumerate((800, 900, 1000, 1100)):
            buf.push(_frame("cam0", 20 + i, t, 0.1))
            buf.push(_frame("cam1", 20 + i, t, 0.2))
        assert buf.drain(), "欠測から復帰したのにペアが出ない"


class TestOutOfOrderAndBounds:
    def test_out_of_order_arrival_is_handled(self):
        """到着順が前後しても、時刻で並べ直して組めること。"""
        buf = _buffer()
        for t in (0, 200, 100, 300):
            buf.push(_frame("cam0", t, t, 0.1))
        for t in (300, 0, 100, 200):
            buf.push(_frame("cam1", t, t, 0.2))
        times = [pair.t_ns for pair in buf.drain()]
        assert times == sorted(times)
        assert len(times) >= 2

    def test_buffer_does_not_grow_without_bound(self):
        """長時間の計測でメモリを食い潰さないこと。"""
        buf = _buffer(target_hz=30.0, window_sec=1.0, max_gap_ms=100.0)
        for i in range(3000):  # 100 秒相当
            t = i * 33.3
            buf.push(_frame("cam0", i, t, 0.1))
            buf.push(_frame("cam1", i, t, 0.2))
            buf.drain()
        assert buf.buffered_count("cam0") < 100
        assert buf.buffered_count("cam1") < 100

    def test_one_role_alone_does_not_grow_without_bound(self):
        """片方がまだ来ていない間も、時間窓より古いものは捨てること。

        混成構成では PC のカメラが先に流れ始め、スマホが QR を読むまで何分も
        片側だけになる。組めないまま溜め続けると、30fps で 10 分に 18,000 フレーム残る。
        """
        buf = _buffer(target_hz=30.0, window_sec=2.0, max_gap_ms=100.0)
        for i in range(300):  # 10 秒相当
            buf.push(_frame("cam0", i, i * 33.3, 0.1))
            buf.drain()
        assert buf.buffered_count("cam0") <= 2 * 30 + 2

    def test_pairs_form_once_the_late_role_arrives(self):
        """捨てたあとでも、遅れて来たロールと直近のフレームで組めること。"""
        buf = _buffer(target_hz=10.0, window_sec=1.0, max_gap_ms=250.0)
        for i in range(50):  # cam0 だけ 5 秒
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.drain()
        for i in range(45, 50):  # cam1 は最後の 0.5 秒ぶんから
            buf.push(_frame("cam1", i, i * 100, 0.2))
        pairs = buf.drain()
        assert pairs, "遅れて来たロールと組めていない"
        assert pairs[0].t_ns == 4_500 * MS

    def test_frames_older_than_the_window_are_discarded(self):
        buf = _buffer(target_hz=10.0, window_sec=0.5, max_gap_ms=250.0)
        for i in range(20):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100, 0.2))
            buf.drain()
        assert buf.buffered_count("cam0") <= 8


def _stalled_arrivals(stall_s: float, total_s: float = 25.0, stall_at_s: float = 5.0) -> list[LandmarkFrame]:
    """PC のカメラ（cam0）は撮影直後に、Pixel（cam1）は 150 ms 遅れて届く。ただし Wi-Fi が ``stall_s`` 秒
    詰まっている間の Pixel の点は、回復した瞬間にまとめて（順序は保って）届く。到着順に並べて返す。"""
    events: list[tuple[float, int, LandmarkFrame]] = []
    for i in range(int(total_s * 30)):
        t_ms = i * 1000 / 30
        events.append((t_ms + 5, 0, _frame("cam0", i, t_ms, 0.1)))
        arrive = t_ms + 150
        if stall_at_s * 1000 <= arrive < (stall_at_s + stall_s) * 1000:
            arrive = (stall_at_s + stall_s) * 1000 + i * 0.001
        events.append((arrive, 1, _frame("cam1", i, t_ms + 7, 0.2)))
    events.sort(key=lambda e: (e[0], e[1]))
    return [frame for _, _, frame in events]


class TestBurstAfterStall:
    """Wi-Fi が詰まった後にまとめて届いた Pixel の点を、組にできること。

    Pixel は詰まっている間の点を送信キューに溜め、回復した瞬間にまとめて送る。その間も PC のカメラの点は
    届き続けるので、保持時間が短いと相手（PC のカメラの点）が先に捨てられ、まとめて届いた点が組にならない。
    保持 2 s のころは、3 s の詰まりで 33 組、8 s で 183 組を失っていた。組は両方そろった時点で出るので、
    保持時間を延ばしても表示は遅れない。
    """

    @pytest.mark.parametrize("stall_s", [3.0, 8.0, 14.0])
    def test_default_window_keeps_every_pair_through_a_stall(self, stall_s):
        """Pixel の ping の切断（15 s）より短い詰まりなら、既定の保持時間で 1 組も失わない。"""
        buf = SyncBuffer()
        frames = _stalled_arrivals(stall_s)
        emitted = 0
        for frame in frames:
            buf.push(frame)
            emitted += len(buf.drain())
        assert buf.stats["dropped_gap"] == 0
        assert buf.stats["dropped_late"] == 0
        assert emitted >= len(frames) // 2 - 2

    def test_default_window_outlasts_the_pixel_ping_timeout(self):
        """詰まりが ping の間隔（15 s、mobile の SensorClient）を超えると端末が切断するので、それより古い点は来ない。"""
        assert SyncBuffer().window_ns >= 15_000_000_000

    def test_default_window_still_bounds_memory(self):
        """保持時間を延ばしても、長時間の計測で溜め続けないこと（30 Hz で保持時間ぶん + 補間用の 2 個）。"""
        buf = SyncBuffer()
        limit = round(buf.window_ns / 1e9 * 30) + 2
        for i in range(30 * 120):  # 2 分相当
            t_ms = i * 1000 / 30
            buf.push(_frame("cam0", i, t_ms, 0.1))
            buf.push(_frame("cam1", i, t_ms + 7, 0.2))
            buf.drain()
        assert buf.buffered_count("cam0") <= limit
        assert buf.buffered_count("cam1") <= limit


class TestStats:
    def test_counts_emitted_and_dropped(self):
        buf = _buffer()
        for i in range(5):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100, 0.2))
        emitted = len(buf.drain())
        assert buf.stats["emitted"] == emitted

    def test_reports_measured_offset_between_roles(self):
        """2 台の時刻ずれを可視化できること。UI で同期品質を見せるのに使う。"""
        buf = _buffer()
        for i in range(4):
            buf.push(_frame("cam0", i, i * 100, 0.1))
            buf.push(_frame("cam1", i, i * 100 + 30, 0.2))  # 30ms 遅れ
        buf.drain()
        assert buf.stats["last_role_skew_ms"] == pytest.approx(30.0, abs=5.0)
