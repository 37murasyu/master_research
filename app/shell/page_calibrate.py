"""キャリブレーション画面。

処理を 2 つに分けている。

- **カメラの検出**（デバイス名の列挙、インデックスの走査）: GUI を伴わないので
  プロセス内のワーカースレッドで実行する
- **チェッカーボードの撮影と校正**: 対話式で ``cv.imshow`` + ``waitKey`` を使う。
  macOS では GUI 操作がメインスレッド必須で、スレッドから呼ぶと
  "Unknown C++ exception from OpenCV code" になるため、**子プロセス**で動かす
"""

from __future__ import annotations

from app.core.platform_compat import (
    camera_backends,
    enumerate_camera_device_names,
    is_macos,
)
from app.core.qt import QtCore, QtWidgets
from app.core.settings import Settings
from app.runners.worker import WorkerRunner
from app.shell.widgets import LogView, StatusBadge

__all__ = ["CalibratePage"]


class CameraProbe(QtCore.QThread):
    """接続カメラを調べる。開くのに時間がかかるので画面を止めない。"""

    found = QtCore.Signal(list)  # [(index, width, height), ...]
    names_found = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    MAX_INDEX = 6

    def run(self) -> None:  # pragma: no cover - 実カメラ依存
        try:
            self.names_found.emit(enumerate_camera_device_names())
        except Exception as exc:
            self.failed.emit(f"デバイス名の取得に失敗: {exc}")

        try:
            import cv2 as cv  # pylint: disable=no-member

            available = []
            backends = camera_backends()
            for index in range(self.MAX_INDEX):
                capture = None
                for backend in backends:
                    capture = cv.VideoCapture(index, backend)
                    if capture is not None and capture.isOpened():
                        break
                    if capture is not None:
                        capture.release()
                        capture = None
                if capture is None:
                    continue
                ok, frame = capture.read()
                if ok and frame is not None:
                    available.append((index, frame.shape[1], frame.shape[0]))
                capture.release()
            self.found.emit(available)
        except Exception as exc:
            self.failed.emit(f"カメラの走査に失敗: {exc}")


class CalibratePage(QtWidgets.QWidget):
    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._runner = WorkerRunner("calibrate", self)
        self._probe: CameraProbe | None = None

        self._runner.output.connect(lambda text: self._log.append_text(text))
        self._runner.state_changed.connect(self._on_state)

        self._build_ui()
        self._on_state("stopped")

    # -- 画面 --------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("カメラキャリブレーション")
        font = title.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch(1)
        self._badge = StatusBadge()
        header.addWidget(self._badge)
        outer.addLayout(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_camera_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setSizes([360, 640])
        outer.addWidget(splitter, 1)

    def _build_camera_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        box = QtWidgets.QGroupBox("接続されているカメラ")
        box_layout = QtWidgets.QVBoxLayout(box)

        self._camera_list = QtWidgets.QListWidget()
        box_layout.addWidget(self._camera_list)

        self._detect_button = QtWidgets.QPushButton("カメラを検出")
        self._detect_button.clicked.connect(self._detect_cameras)
        box_layout.addWidget(self._detect_button)

        layout.addWidget(box)

        steps = QtWidgets.QGroupBox("キャリブレーションの実行")
        steps_layout = QtWidgets.QVBoxLayout(steps)

        note = QtWidgets.QLabel(
            "チェッカーボードの撮影は別ウィンドウで行います。"
            "画面の指示に従って操作してください。\n"
            "※ カメラの位置や向きを変えたら、必ずやり直してください。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b7280;")
        steps_layout.addWidget(note)

        self._start_button = QtWidgets.QPushButton("キャリブレーションを開始")
        self._start_button.clicked.connect(self._start)
        steps_layout.addWidget(self._start_button)

        self._stop_button = QtWidgets.QPushButton("中止")
        self._stop_button.clicked.connect(self._runner.stop)
        steps_layout.addWidget(self._stop_button)

        layout.addWidget(steps)
        layout.addStretch(1)
        return panel

    def _build_log_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("ログ"))
        self._log = LogView()
        layout.addWidget(self._log, 1)
        return panel

    # -- 動作 --------------------------------------------------------------
    def _detect_cameras(self) -> None:
        if self._probe is not None and self._probe.isRunning():
            return

        self._camera_list.clear()
        self._camera_list.addItem("検出中…")
        self._detect_button.setEnabled(False)

        self._probe = CameraProbe(self)
        self._probe.names_found.connect(self._on_names)
        self._probe.found.connect(self._on_cameras)
        self._probe.failed.connect(lambda msg: self._log.append_text(f"[エラー] {msg}\n"))
        self._probe.finished.connect(lambda: self._detect_button.setEnabled(True))
        self._probe.start()

    def _on_names(self, names: list) -> None:
        if names:
            self._log.append_text("[検出] デバイス名: " + ", ".join(names) + "\n")
        elif is_macos():
            self._log.append_text("[検出] デバイス名を取得できませんでした。\n")

    def _on_cameras(self, cameras: list) -> None:
        self._camera_list.clear()
        if not cameras:
            self._camera_list.addItem("見つかりませんでした")
            self._log.append_text(
                "[検出] カメラが見つかりません。接続と、他アプリが占有していないかを確認してください。\n"
            )
            return
        for index, width, height in cameras:
            self._camera_list.addItem(f"インデックス {index}  ({width}x{height})")
        self._log.append_text(f"[検出] {len(cameras)} 台見つかりました。\n")

    def _start(self) -> None:
        self._runner.start(self._settings)

    def _on_state(self, state: str) -> None:
        self._badge.set_state(state)
        running = state in ("starting", "running")
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._detect_button.setEnabled(not running)

    def shutdown(self) -> None:
        self._runner.stop()
        if self._probe is not None and self._probe.isRunning():
            self._probe.wait(3000)

    @property
    def is_running(self) -> bool:
        return self._runner.is_running
