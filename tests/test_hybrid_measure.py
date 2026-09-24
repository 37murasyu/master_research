import json
import time
import cv2 as cv
import numpy as np
from config import pose_keypoints
from app.hybrid.checkerboard import Intrinsics, Stereo, Board
from app.hybrid.calibration_io import save_calibration, load_calibration
from app.hybrid.measurement import MeasurementSession
from app.hybrid.link import PhoneLink, OFF
from app.hybrid.live import LiveSession
from app.net.mock_sender import MockPhone
from app.net.protocol import LandmarkFrame, Hello
from app.runners.network_measure import NetworkMeasurement
from test_network_measure import _body_points, _pair_from_pixels
from test_hybrid_link import _run_phone
from hybrid_fakes import FakeCamera


def geometry():
    k = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    d = np.array([0.15, -0.08, 0.002, 0.001, 0.0])
    intr = Intrinsics(k, d, (1280, 720), 0)
    stereo = Stereo(np.eye(3), np.array([[-35.0], [0], [0]]), 0, [], list(range(12)))
    return intr, stereo


def projected():
    intr, stereo = geometry()
    truth = _body_points(0)
    truth[:, 1] -= 25  # Keep the entire synthetic subject inside both images.
    a = cv.projectPoints(truth, np.zeros(3), np.zeros(3), intr.K, intr.distortion)[
        0
    ].reshape(-1, 2)
    b = cv.projectPoints(truth, np.zeros(3), stereo.T, intr.K, intr.distortion)[
        0
    ].reshape(-1, 2)
    return truth, a, b


def calibration(tmp_path):
    intr, stereo = geometry()
    return load_calibration(
        save_calibration(
            intr,
            intr,
            stereo,
            Board(),
            cameras=[
                {"kind": "mac", "device_id": "mac"},
                {"kind": "pixel", "device_id": "pixel-1"},
            ],
            root=tmp_path / "calibration",
        )
    )


def test_distortion_corrected_3d_and_outside_nan():
    intr, stereo = geometry()
    truth, a, b = projected()
    m = NetworkMeasurement(
        intr.K @ np.c_[np.eye(3), np.zeros(3)],
        intr.K @ np.c_[stereo.R, stereo.T],
        pose_keypoints,
        lens={"cam0": intr, "cam1": intr},
    )
    result = m.process(_pair_from_pixels(1, a, b))
    expected = truth[:, [0, 2, 1]] * -0.01
    assert np.max(np.linalg.norm(result.points_3d - expected, axis=1)) < 0.005
    # 余白（幅・高さの 10%）より遠く外の点は、歪みの多項式の外挿が暴れるので使わない。
    # EKF はその点を予測で埋めるので、三角測量の結果は EKF の手前（points_raw）で見る（T7 で EKF を入れた）
    a[0] = [-400, 30]
    result = m.process(_pair_from_pixels(2, a, b))
    assert np.isnan(result.points_raw[0]).all()
    assert np.isfinite(result.points_3d[0]).all()


def test_edge_points_survive_undistortion():
    """樽型歪みの補正で、画像の内側の点が端より外（負の座標）へ押し出されても消さない。

    OpenCV 側の三角測量（macOS では常にこちら）は負の座標を「未検出」として捨てる。
    手首が画面の上端近くにあるだけで、その関節が 3D から消えていた。
    """
    k, stereo = geometry()[0].K, geometry()[1]
    intr = Intrinsics(k, np.array([-0.25, 0.0, 0.0, 0.0, 0.0]), (1280, 720), 0)
    truth = _body_points(0)
    truth[:, 1] -= 25
    truth[0] = [0.0, -0.41111 * 200, 200.0]  # 補正後の理想座標で上端の 10 px 外
    a = cv.projectPoints(truth, np.zeros(3), np.zeros(3), intr.K, intr.distortion)[0].reshape(-1, 2)
    b = cv.projectPoints(truth, np.zeros(3), stereo.T, intr.K, intr.distortion)[0].reshape(-1, 2)

    # 前提: 写っているのは画像の内側だが、補正すると負になる
    assert 0 <= a[0, 1] < 20 and 0 <= b[0, 1] < 20
    ideal = cv.undistortPoints(a[:1].reshape(-1, 1, 2), intr.K, intr.distortion, P=intr.K)
    assert ideal.reshape(2)[1] < 0

    m = NetworkMeasurement(
        intr.K @ np.c_[np.eye(3), np.zeros(3)],
        intr.K @ np.c_[stereo.R, stereo.T],
        pose_keypoints,
        lens={"cam0": intr, "cam1": intr},
    )
    result = m.process(_pair_from_pixels(1, a, b))
    expected = truth[:, [0, 2, 1]] * -0.01
    assert np.linalg.norm(result.points_3d[0] - expected[0]) < 0.005


def test_no_measurement_folder_until_the_phone_streams(tmp_path):
    """Pixel が一度も点を送らなければ、計測フォルダを作らない。

    以前は起動直後の定期処理（flush）と、Mac の前に人がいるだけで記録を始めていた。
    端末 ID の違いや古い QR で断られて終えると、中身の無い「完了」の計測が残った。
    """
    session = MeasurementSession(calibration(tmp_path), root=tmp_path / "measure")
    mac = LandmarkFrame("cam0", 0, 1, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33)
    session.flush()
    session.on_landmarks(mac)
    session.flush()
    session.close()
    assert session.directory is None
    assert not (tmp_path / "measure").exists()

    session = MeasurementSession(calibration(tmp_path), root=tmp_path / "measure")
    session.on_landmarks(LandmarkFrame("cam1", 0, 2, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))
    session.close()
    assert session.directory is not None and session.directory.exists()


def test_the_gauge_link_follows_the_pixel(tmp_path):
    """ゲージ（と GUI）の接続表示は、Pixel の点が 1 s 途絶えるか接続が切れたら waiting に戻り、再開で connected に戻る。

    以前は記録を始めたときに 1 回 connected にするだけで、Pixel が切れても connected のままだった。
    更新は受信スレッドの定期処理（on_tick＝flush）で行う。
    """
    from app.gauge.tracker import GaugeTracker

    now = [100.0]
    connected = [True]
    tracker = GaugeTracker()
    session = MeasurementSession(calibration(tmp_path), root=tmp_path / "measure", tracker=tracker,
                                 clock=lambda: now[0])
    session.remote_connected = lambda: connected[0]

    def pixel(seq):
        session.on_landmarks(LandmarkFrame("cam1", seq, seq + 1, 1280, 720, [(0.5, 0.5, 0.0, 1.0)] * 33))

    def link():
        session.flush()
        return tracker.snapshot().link

    assert link() == "waiting", "点が来る前"
    pixel(0)
    assert link() == "connected"
    now[0] += 0.9
    assert link() == "connected", "1 s 以内の途切れ"
    now[0] += 0.2
    assert link() == "waiting", "最後の点から 1 s を超えた"
    pixel(1)
    assert link() == "connected", "点が戻った"
    connected[0] = False
    assert link() == "waiting", "接続が切れた"
    session.close()


def test_identity_and_dimension_rejection(tmp_path):
    session = MeasurementSession(calibration(tmp_path), root=tmp_path / "measure")
    assert session.check_hello(Hello("cam1", "Pixel", "s", "wrong")) is not None
    assert session.check_hello(Hello("cam1", "Pixel", "s", "pixel-1")) is None
    frame = LandmarkFrame("cam1", 0, 1, 640, 360, [(0.5, 0.5, 0, 1)] * 33)
    for _ in range(29):
        assert not session.accept_frame(frame)
    assert session.exit_code == 0
    assert not session.accept_frame(frame)
    assert session.exit_code == 3
    session.close()


def _kpts3d(folder):
    """EKF の後の 3D（``kpts3d_<stamp>.csv``）。2026-09-24 から同じフォルダに EKF の手前の ``kpts3d_raw_<stamp>.csv``
    も書くので、``kpts3d_*`` の glob では取り違える。"""
    return next(p for p in folder.glob("kpts3d_*.csv") if not p.name.startswith("kpts3d_raw_"))


def test_mock_measure_records_during_run_and_closes_after_drain(tmp_path):
    cal = calibration(tmp_path)
    session = MeasurementSession(cal, root=tmp_path / "measure")
    link = PhoneLink(
        host="127.0.0.1",
        port=0,
        capture_mode=OFF,
        on_pairs=session.on_pairs,
        on_landmarks=session.on_landmarks,
        on_hello=session.check_hello,
        accept_frame=session.accept_frame,
        on_tick=session.flush,
        on_stop=session.close,
    )
    truth, a, b = projected()
    pair = _pair_from_pixels(1, a, b)

    class Detector:
        def detect(self, image, stamp):
            return pair.frames["cam0"].landmarks

    link.start()
    phone = MockPhone(
        link.url,
        "cam1",
        session=link.session,
        device_id="pixel-1",
        pose_fn=lambda t, role: pair.frames["cam1"].landmarks,
    )
    thread = _run_phone(phone, 3.0)
    live = LiveSession(FakeCamera(), Detector(), link, output=lambda image: None)
    start = time.monotonic()
    checked = False
    try:
        while time.monotonic() - start < 3.1:
            live.step()
            time.sleep(0.025)
            if not checked and time.monotonic() - start > 1.5:
                csv = _kpts3d(session.directory)
                assert len(csv.read_text().splitlines()) > 2
                checked = True
    finally:
        thread.join(5)
        stop_started = time.monotonic()
        link.stop()
        assert time.monotonic() - stop_started < 2.0
    assert not thread.errors
    meta = json.loads((session.directory / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["frames"] > 40
    data = np.loadtxt(_kpts3d(session.directory), delimiter=",", skiprows=1)
    expected = truth[:, [0, 2, 1]] * -0.01
    assert np.max(np.abs(data[:, 1:].reshape(-1, len(truth), 3) - expected)) < 0.01
    raw = np.loadtxt(next(session.directory.glob("kpts3d_raw_*.csv")), delimiter=",", skiprows=1)
    assert np.nanmax(np.abs(raw[:, 2:].reshape(-1, len(truth), 3) - expected)) < 0.01
    assert meta["writer_thread"] != "MainThread"


def test_rejected_frame_does_not_enter_sync_buffer(tmp_path):
    from app.net.server import SessionHandler
    from app.net.sync_buffer import SyncBuffer
    from app.net import protocol as p

    session = MeasurementSession(calibration(tmp_path), root=tmp_path / "measure")
    buffer = SyncBuffer()
    handler = SessionHandler(
        buffer, on_hello=session.check_hello, accept_frame=session.accept_frame
    )
    handler.handle(p.encode(Hello("cam1", "Pixel", "s", "pixel-1")))
    bad = LandmarkFrame("cam1", 0, 1, 640, 360, [(0.5, 0.5, 0, 1)] * 33)
    handler.handle(p.encode(bad))
    assert not buffer.drain()
    assert session.size_drops == 1


def test_wrong_identity_is_closed_with_1008(tmp_path):
    import asyncio
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
    from app.net import protocol as p

    session = MeasurementSession(calibration(tmp_path), root=tmp_path / "measure")
    link = PhoneLink(host="127.0.0.1", port=0, on_hello=session.check_hello)
    link.start()

    async def probe():
        async with connect(link.url) as ws:
            await ws.send(p.encode(Hello("cam1", "Pixel", link.session, "wrong")))
            try:
                await ws.recv()
            except ConnectionClosed as exc:
                assert exc.rcvd.code == 1008
            else:
                raise AssertionError("wrong device was accepted")

    try:
        asyncio.run(probe())
    finally:
        link.stop()


def test_camera_size_mismatch_returns_two(tmp_path, monkeypatch):
    from app.runners import hybrid_measure

    cal = calibration(tmp_path)

    def wrong(*args, **kwargs):
        raise ValueError("size mismatch")

    # 校正したカメラ（識別子 "mac"）を開いた。止まるのは識別子の違いではなく寸法の違い
    monkeypatch.setattr(hybrid_measure, "mac_identity", lambda index: "mac")
    monkeypatch.setattr(hybrid_measure, "MacCamera", wrong)
    assert hybrid_measure.main(["--calibration", str(cal.directory)]) == 2
