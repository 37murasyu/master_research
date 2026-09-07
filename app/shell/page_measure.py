"""計測画面。設定を編集し、計測ワーカーを起動・停止し、その出力を見る。"""

from __future__ import annotations

from app.core.platform_compat import user_output_dir
from app.core.qt import QtCore, QtWidgets
from app.core.settings import APP_NAME, SCHEMA, Settings
from app.runners.worker import WorkerRunner
from app.shell.widgets import LogView, SettingsForm, StatusBadge

__all__ = ["MeasurePage"]


class MeasurePage(QtWidgets.QWidget):
    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._runner = WorkerRunner("realtime", self)

        self._runner.output.connect(self._on_output)
        self._runner.state_changed.connect(self._on_state)

        self._build_ui()
        self._on_state("stopped")

    # -- 画面 --------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        outer.addLayout(self._build_header())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 640])
        outer.addWidget(splitter, 1)

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()

        title = QtWidgets.QLabel("リアルタイム計測")
        font = title.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        title.setFont(font)
        row.addWidget(title)

        row.addStretch(1)

        self._badge = StatusBadge()
        row.addWidget(self._badge)

        self._start_button = QtWidgets.QPushButton("計測を開始")
        self._start_button.clicked.connect(self._start)
        row.addWidget(self._start_button)

        self._stop_button = QtWidgets.QPushButton("停止")
        self._stop_button.clicked.connect(self._runner.stop)
        row.addWidget(self._stop_button)

        return row

    def _build_settings_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        note = QtWidgets.QLabel(
            "「デモ用」の項目はアプリ側で無効にしてあります。"
            "有効にすると計算結果が意味を失うことがあります（項目にカーソルを合わせると説明が出ます）。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b7280;")
        layout.addWidget(note)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._form = SettingsForm(self._settings)
        self._form.changed.connect(self._on_settings_changed)
        scroll.setWidget(self._form)
        layout.addWidget(scroll, 1)

        self._output_label = QtWidgets.QLabel()
        self._output_label.setWordWrap(True)
        self._output_label.setStyleSheet("color: #6b7280;")
        self._output_label.setText(f"出力先: {user_output_dir(APP_NAME)}")
        layout.addWidget(self._output_label)

        return panel

    def _build_log_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel("計測ログ"))
        self._log = LogView()
        layout.addWidget(self._log, 1)

        return panel

    # -- 動作 --------------------------------------------------------------
    def _start(self) -> None:
        self._runner.start(self._settings)

    def _on_output(self, text: str) -> None:
        self._log.append_text(text)

    def _on_state(self, state: str) -> None:
        self._badge.set_state(state)
        running = state in ("starting", "running")
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        # 計測中に設定を変えても子プロセスには届かない。誤解を招くので触れなくする。
        self._form.setEnabled(not running)

    def _on_settings_changed(self) -> None:
        changed = self._settings.overrides
        if changed:
            names = ", ".join(sorted(changed))
            self._output_label.setText(
                f"出力先: {user_output_dir(APP_NAME)}\n既定から変更: {names}"
            )
        else:
            self._output_label.setText(f"出力先: {user_output_dir(APP_NAME)}")

    # -- 後片付け ----------------------------------------------------------
    def shutdown(self) -> None:
        """ウィンドウを閉じるとき、子プロセスを残さない。"""
        self._runner.stop()

    @property
    def is_running(self) -> bool:
        return self._runner.is_running
