"""キャリブレーション画面。

処理を 2 つに分けている。

- **カメラの検出**（デバイス名の列挙、インデックスの走査）: GUI を伴わないので
  プロセス内のワーカースレッドで実行する
- **チェッカーボードの撮影と校正**: 対話式で ``cv.imshow`` + ``waitKey`` を使う。
  macOS では GUI 操作がメインスレッド必須で、スレッドから呼ぶと
  "Unknown C++ exception from OpenCV code" になるため、**子プロセス**で動かす
"""

from __future__ import annotations

from app.core.platform_compat import enumerate_camera_device_names, is_macos
from app.core.qt import QtCore, QtWidgets
from app.core.settings import Settings
from app.hybrid import paths as hybrid_paths
from app.shell.widgets import RunnerPage

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

        # video_source は cv2 を読み込む。GUI の起動時に払わないよう、
        # ワーカースレッドに入ってから import する。
        # pylint: disable=import-outside-toplevel
        from app.core.video_source import open_source

        try:
            available = []
            for index in range(self.MAX_INDEX):
                source = open_source(index)
                if source is None:
                    continue
                ok, frame = source.read()
                if ok and frame is not None:
                    available.append((index, frame.shape[1], frame.shape[0]))
                source.release()
            self.found.emit(available)
        except Exception as exc:
            self.failed.emit(f"カメラの走査に失敗: {exc}")


class CalibratePage(RunnerPage):
    TITLE = "カメラキャリブレーション"

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        self._probe: CameraProbe | None = None
        super().__init__(settings, "calibrate", parent)

    # -- 骨格への差し込み --------------------------------------------------
    def widgets_disabled_while_running(self) -> list[QtWidgets.QWidget]:
        return [self._start_button, self._detect_button, self._input_mode]

    def widgets_enabled_while_running(self) -> list[QtWidgets.QWidget]:
        return [self._stop_button]

    def start_widgets(self) -> list[QtWidgets.QWidget]:
        return [self._start_button]

    def build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self._input_mode = QtWidgets.QComboBox()
        self._input_mode.addItems(["入力: USB カメラ 2 台", "入力: Mac＋Pixel（混成）"])
        self._input_mode.currentIndexChanged.connect(self._change_input)
        layout.addWidget(self._input_mode)
        self._output_label = QtWidgets.QLabel("出力先: camera_parameters")
        self._output_label.setWordWrap(True)
        layout.addWidget(self._output_label)

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
        self._start_button.clicked.connect(lambda: self._runner.start(self._settings))
        steps_layout.addWidget(self._start_button)

        self._stop_button = QtWidgets.QPushButton("中止")
        self._stop_button.clicked.connect(self._runner.stop)
        steps_layout.addWidget(self._stop_button)

        layout.addWidget(steps)
        layout.addStretch(1)
        return panel

    def _change_input(self, index: int) -> None:
        self._runner.role = "hybrid_calibrate" if index else "calibrate"
        directory = hybrid_paths.calibration_root() if index else "camera_parameters"
        self._output_label.setText(f"出力先: {directory}")

    # -- カメラ検出 --------------------------------------------------------
    def _detect_cameras(self) -> None:
        if self._probe is not None and self._probe.isRunning():
            return

        self._camera_list.clear()
        self._camera_list.addItem("検出中…")
        self._detect_button.setEnabled(False)

        self._probe = CameraProbe(self)
        self._probe.names_found.connect(self._on_names)
        self._probe.found.connect(self._on_cameras)
        self._probe.failed.connect(lambda msg: self.append_log(f"[エラー] {msg}\n"))
        self._probe.finished.connect(lambda: self._detect_button.setEnabled(True))
        self._probe.start()

    def _on_names(self, names: list) -> None:
        if names:
            self.append_log("[検出] デバイス名: " + ", ".join(names) + "\n")
        elif is_macos():
            self.append_log("[検出] デバイス名を取得できませんでした。\n")

    def _on_cameras(self, cameras: list) -> None:
        self._camera_list.clear()
        if not cameras:
            self._camera_list.addItem("見つかりませんでした")
            self.append_log(
                "[検出] カメラが見つかりません。接続と、他アプリが占有していないかを確認してください。\n"
            )
            return
        for index, width, height in cameras:
            self._camera_list.addItem(f"インデックス {index}  ({width}x{height})")
        self.append_log(f"[検出] {len(cameras)} 台見つかりました。\n")

    def shutdown(self) -> None:
        super().shutdown()
        if self._probe is not None and self._probe.isRunning():
            self._probe.wait(3000)
