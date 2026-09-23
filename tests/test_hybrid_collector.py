import numpy as np
from app.hybrid.checkerboard import Board
from app.hybrid.collector import BoardCollector


def corners(value):
    return np.full((28, 1, 2), value, np.float32)


def make_collector():
    return BoardCollector(
        Board(),
        detect=lambda image, board: None if image[0, 0] < 0 else corners(image[0, 0]),
        mono_required=1,
        pairs_required=1,
    )


def image(value):
    return np.full((20, 40), value, np.float32)


def test_waits_for_future_and_accepts_still_pair_once():
    c = make_collector()
    for t in range(0, 301, 30):
        c.add_mac(t * 1_000_000, image(10))
    c.add_remote(150_000_000, image(10))
    assert len(c.pairs) == 1
    assert c.ready
    c.add_remote(150_000_000, image(10))
    assert len(c.pairs) == 1


def test_moving_board_and_missing_time_are_not_pairs():
    c = make_collector()
    c.add_remote(150_000_000, image(10))
    for t in range(0, 301, 30):
        c.add_mac(t * 1_000_000, image(t / 10))
    assert not c.pairs
    assert not c.ready
    c = make_collector()
    c.add_mac(0, image(10))
    c.add_mac(500_000_000, image(10))
    c.add_remote(200_000_000, image(10))
    assert not c.pairs


def test_detection_results_are_reused():
    """表示のために検出した結果を渡せば、同じ画像を 2 度検出しない。

    盤が写っていないフレームでも 1 回ずつ全解像度の検出が走っていた。メインスレッドの
    fps が落ち、静止判定の ±150 ms の標本の取り方も崩れる。
    """
    calls = []

    def counting_detect(image, board):
        calls.append(1)
        return corners(image[0, 0])

    c = BoardCollector(Board(), detect=counting_detect, mono_required=1, pairs_required=1)
    for t in range(0, 301, 30):
        found = None if t == 60 else corners(10)  # 60 ms は「検出したが見つからない」
        c.add_mac(t * 1_000_000, image(10), corners=found)
    c.add_remote(150_000_000, image(10), corners=corners(10))
    assert calls == [], "渡した検出結果（見つからない場合を含む）を使っていない"


def test_accepts_a_pair_with_the_real_detector():
    """本物の detect_board を通す。

    OpenCV 5.0 の角点は (N, 2) で返る。偽の検出器が (N, 1, 2) を返していたため、
    静止判定の axis=2 が実機で初めて AxisError になった（2026-09-23）。
    """
    from app.hybrid.checkerboard import detect_board
    from test_hybrid_calibration import render_board

    board = Board()
    frame = render_board(board)
    c = BoardCollector(board, detect=detect_board, mono_required=1, pairs_required=1)
    for t in range(0, 301, 30):
        c.add_mac(t * 1_000_000, frame, corners=detect_board(frame, board))
    c.add_remote(150_000_000, frame)
    assert len(c.pairs) == 1


def test_one_sided_views_and_bounded_ring():
    c = make_collector()
    for t in range(0, 3001, 30):
        c.add_mac(t * 1_000_000, image(10))
    c.add_remote(2800_000_000, image(-1))
    assert len(c.mono[0]) == 1
    assert not c.mono[1] and not c.pairs
    assert c.ring[-1][0] - c.ring[0][0] <= 1_500_000_000
