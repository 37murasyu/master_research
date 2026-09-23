"""計測画面。設定を編集し、計測ワーカーを起動・停止し、その出力を見る。

主ボタンは 1 つで、停止中は「計測を開始」、実行中は「停止」に入れ替わる（設計書 §5.2、R16-01）。
入力が Mac＋Pixel（混成）のときは、開始と同時に被験者ゲージの窓（``app.gauge.window``）を開き、
子の出力のゲージの行（``WorkerRunner.gauge_frame``）をそこへ流す。「J の数値」スイッチは
混成のときだけ出し、実行中も切り替えられる（ゲージ窓は同じプロセスにあるので、その場で効く）。

見出しの状態は、骨格の色付きバッジ（緑を含む）を隠し、点と文字の組（``StatusText``）で出す。
1 つめは実行の状態（混成なら回数も）、2 つめは混成の実行中だけの Pixel 接続。回数と接続は
ゲージの行から取る。
"""

from __future__ import annotations

from app.core.qt import QtWidgets
from app.core.settings import Settings, measurement_output_dir
from app.gauge.protocol import GaugeFrame
from app.gauge.window import GaugeWindow
from app.hybrid import paths as hybrid_paths
from app.shell.controls import StatusText, ToggleSwitch
from app.shell.widgets import RunnerPage, SettingsForm

__all__ = ["MeasurePage"]

_HYBRID_ROLE = "hybrid_measure"


class MeasurePage(RunnerPage):
    TITLE = "リアルタイム計測"
    LOG_LABEL = "計測ログ"

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        # RunnerPage.__init__ が _on_state を呼ぶので、そこで触るものは先に作っておく。
        # ゲージ窓は親を持たない独立の窓（第 2 モニタに全画面で出すため）。閉じるのは shutdown。
        self._gauge_window = GaugeWindow(show_joules=bool(settings.get("GAUGE_SHOW_JOULES")))
        # 今（または最後）の実行の role。入力の選択は実行の後で変わりうるので、終了時の振る舞いは
        # 選択ではなくこちらで決める。
        self._run_role: str | None = None
        # 見出しの局面（"stopped" / "running" / "success" / "error"）と、今の実行で最後に届いたフレーム。
        self._header_phase = "stopped"
        self._last_frame: GaugeFrame | None = None
        super().__init__(settings, "realtime", parent)
        # 見せるかどうかは、ページに入ってから決める（親の無いうちに見せると独立の窓になる）。
        self._joules_switch.setVisible(self._runner.role == _HYBRID_ROLE)
        self._badge.hide()  # 色だけで状態を示すバッジ。代わりに _run_status を出す

        self._runner.gauge_frame.connect(self._gauge_window.set_frame)
        self._runner.gauge_frame.connect(self._on_gauge_frame)
        self._runner.finished.connect(self._on_finished)

    # -- 骨格への差し込み --------------------------------------------------
    def header_widgets(self) -> list[QtWidgets.QWidget]:
        self._run_status = StatusText()
        self._link_status = StatusText()  # 見せるかどうかは _refresh_header が決める

        self._joules_switch = ToggleSwitch("J の数値")
        self._joules_switch.setChecked(bool(self._settings.get("GAUGE_SHOW_JOULES")))
        self._joules_switch.toggled.connect(self._on_joules_toggled)

        self._main_button = QtWidgets.QPushButton("計測を開始")
        self._main_button.clicked.connect(self._on_main_button)
        return [self._run_status, self._link_status, self._joules_switch, self._main_button]

    def widgets_disabled_while_running(self) -> list[QtWidgets.QWidget]:
        # 計測中に設定を変えても子プロセスには届かない。誤解を招くので触れなくする。
        return [self._form, self._input_mode]

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

    def shutdown(self) -> None:
        super().shutdown()
        self._gauge_window.close()

    # -- 開始・停止 --------------------------------------------------------
    def _on_main_button(self) -> None:
        if self.is_running:
            self._runner.stop()
        else:
            self._start()

    def _start(self) -> None:
        role = self._runner.role
        self._last_frame = None
        if not self._runner.start(self._settings):
            return  # 起動に失敗したときは窓を開かない
        self._run_role = role
        # start の中で出た "running" の時点では _run_role が前回のままなので、ここで出し直す
        self._refresh_header()
        if role == _HYBRID_ROLE:
            self._gauge_window.begin(
                show_joules=bool(self._settings.get("GAUGE_SHOW_JOULES")),
                avoid_screen=self.window().screen(),
            )

    def _on_finished(self, exit_code: int) -> None:
        if self._run_role == _HYBRID_ROLE:
            self._gauge_window.finish(exit_code)
        self._header_phase = "success" if exit_code == 0 else "error"
        self._refresh_header()

    def _on_gauge_frame(self, frame: GaugeFrame) -> None:
        self._last_frame = frame
        self._refresh_header()

    def _on_joules_toggled(self, checked: bool) -> None:
        self._settings.set("GAUGE_SHOW_JOULES", checked)
        self._gauge_window.set_show_joules(checked)

    # -- 表示更新 ----------------------------------------------------------
    def _on_state(self, state: str) -> None:
        super()._on_state(state)
        running = state in ("starting", "running")
        self._main_button.setText("停止" if running else "計測を開始")
        # 終了の結果（✓／✕）は、この後に届く finished で上書きする
        self._header_phase = "running" if running else "stopped"
        self._refresh_header()

    def _refresh_header(self) -> None:
        hybrid_running = self._header_phase == "running" and self._run_role == _HYBRID_ROLE
        connected = (hybrid_running and self._last_frame is not None
                     and self._last_frame.link == "connected")
        # 回数は、ゲージの見出しと同じく「完了した回数＋1」。つながる前は出さない
        rep = self._last_frame.rep + 1 if connected else None
        self._run_status.set_status(self._header_phase, rep=rep)
        self._link_status.setVisible(hybrid_running)
        if hybrid_running:
            self._link_status.set_link(connected)

    def _change_input(self, index: int) -> None:
        self._runner.role = _HYBRID_ROLE if index else "realtime"
        self._joules_switch.setVisible(self._runner.role == _HYBRID_ROLE)
        self._refresh_output_label()

    def _refresh_output_label(self) -> None:
        destination = (hybrid_paths.measurement_root()
                       if self._runner.role == _HYBRID_ROLE else measurement_output_dir())
        text = f"出力先: {destination}"
        changed = self._settings.overrides
        if changed:
            text += "\n既定から変更: " + ", ".join(sorted(changed))
        self._output_label.setText(text)
