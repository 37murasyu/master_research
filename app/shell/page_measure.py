"""計測画面。設定を編集し、計測ワーカーを起動・停止し、その出力を見る。"""

from __future__ import annotations

from app.core.qt import QtWidgets
from app.core.settings import Settings, measurement_output_dir
from app.hybrid import paths as hybrid_paths
from app.shell.widgets import RunnerPage, SettingsForm

__all__ = ["MeasurePage"]


class MeasurePage(RunnerPage):
    TITLE = "リアルタイム計測"
    LOG_LABEL = "計測ログ"

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        super().__init__(settings, "realtime", parent)

    # -- 骨格への差し込み --------------------------------------------------
    def header_widgets(self) -> list[QtWidgets.QWidget]:
        self._start_button = QtWidgets.QPushButton("計測を開始")
        self._start_button.clicked.connect(
            lambda: self._runner.start(self._settings)
        )
        self._stop_button = QtWidgets.QPushButton("停止")
        self._stop_button.clicked.connect(self._runner.stop)
        return [self._start_button, self._stop_button]

    def widgets_disabled_while_running(self) -> list[QtWidgets.QWidget]:
        # 計測中に設定を変えても子プロセスには届かない。誤解を招くので触れなくする。
        return [self._start_button, self._form, self._input_mode]

    def widgets_enabled_while_running(self) -> list[QtWidgets.QWidget]:
        return [self._stop_button]

    def build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self._input_mode = QtWidgets.QComboBox()
        self._input_mode.addItems(["入力: USB カメラ 2 台", "入力: Mac＋Pixel（混成）"])
        self._input_mode.currentIndexChanged.connect(self._change_input)
        layout.addWidget(self._input_mode)

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
        self._form.changed.connect(self._refresh_output_label)
        scroll.setWidget(self._form)
        layout.addWidget(scroll, 1)

        self._output_label = QtWidgets.QLabel()
        self._output_label.setWordWrap(True)
        self._output_label.setStyleSheet("color: #6b7280;")
        layout.addWidget(self._output_label)
        self._refresh_output_label()

        return panel

    # -- 表示更新 ----------------------------------------------------------
    def _change_input(self, index: int) -> None:
        self._runner.role = "hybrid_measure" if index else "realtime"
        self._refresh_output_label()

    def _refresh_output_label(self) -> None:
        destination = (hybrid_paths.measurement_root()
                       if self._runner.role == "hybrid_measure" else measurement_output_dir())
        text = f"出力先: {destination}"
        changed = self._settings.overrides
        if changed:
            text += "\n既定から変更: " + ", ".join(sorted(changed))
        self._output_label.setText(text)
