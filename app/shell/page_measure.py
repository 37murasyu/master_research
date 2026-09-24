"""計測画面。設定を編集し、計測ワーカーを起動・停止し、その出力を見る。

主ボタンは 1 つで、停止中は「計測を開始」、実行中は「停止」に入れ替わる（設計書 §5.2、R16-01）。
入力が Mac＋Pixel（混成）のときは、開始と同時に被験者ゲージの窓（``app.gauge.window``）を開き、
子の出力のゲージの行（``WorkerRunner.gauge_frame``）をそこへ流す。「J の数値」スイッチは
混成のときだけ出し、実行中も切り替えられる（ゲージ窓は同じプロセスにあるので、その場で効く）。

見出しの状態は、骨格の色付きバッジ（緑を含む）を隠し、点と文字の組（``StatusText``）で出す。
1 つめは実行の状態（混成なら回数も）、2 つめは混成の実行中だけの Pixel 接続。回数と接続は
ゲージの行から取る。

設定は「実験者用の詳細設定」の開示（既定で閉じる。R11-03）にしまう。中身は入力のラジオ、被験者番号、
体重、使う校正の日時と「変更」リンク（混成のときだけ）、「開発・診断用」の入れ子の開示（残りの
設定フォーム）。画面に説明文は置かない。出力先は、終わった後の「出力フォルダ」リンクで開く。
計算を壊す設定（``app_default`` で既定を無効にした 4 つ）が有効なら、その件数を開示の見出しに出す
（R20-03）。ログは、まだ何も流れていないうちは「未実行」と出す（R19-01）。
"""

from __future__ import annotations

import json
from datetime import datetime

from app.core.qt import QtCore, QtGui, QtWidgets
from app.core.settings import SCHEMA, Settings, measurement_output_dir
from app.gauge import fonts as gauge_fonts
from app.gauge.protocol import GaugeFrame
from app.gauge.window import GaugeWindow
from app.hybrid import paths as hybrid_paths
from app.shell.controls import Disclosure, StatusText, ToggleSwitch
from app.shell.widgets import RunnerPage, SettingsForm

__all__ = ["MeasurePage"]

_HYBRID_ROLE = "hybrid_measure"
# 入力のラジオの id と role の対応（id は並びの順）
_INPUT_ROLES = ("realtime", _HYBRID_ROLE)
# 専用の欄を置くので、入れ子の設定フォームには並べない項目
_DEDICATED_SETTINGS = frozenset({"SUBJECT_ID", "BODY_MASS_KG"})
# 混成の校正のフォルダ名の書式（校正を保存する側が datetime.now() から付ける名前）
_CALIBRATION_DIR_FORMAT = "%Y%m%d_%H%M%S_%f"


def calibration_time_text() -> str:
    """混成で使う校正（``latest.json`` が指すもの）の日時。読めなければ「未校正」。

    ``latest.json`` を直接読む。校正の読み書きのモジュールは numpy などを読み込むので、
    画面の表示のためだけには import しない。
    """
    try:
        latest = json.loads((hybrid_paths.calibration_root() / "latest.json").read_text(encoding="utf-8"))
        taken = datetime.strptime(latest["directory"], _CALIBRATION_DIR_FORMAT)
    except (OSError, ValueError, KeyError, TypeError):
        return "未校正"
    return taken.strftime("%Y-%m-%d %H:%M")


def broken_flag_count(settings: Settings) -> int:
    """アプリが既定で無効にした設定（``app_default`` を持つもの）のうち、有効になっている数。

    どれも有効だと計算が壊れる（``settings.CURATED`` の (a)）。
    """
    return sum(
        1 for setting in SCHEMA.values()
        if setting.app_default is not None
        and settings.get(setting.name)
        and setting.name in settings.overrides
    )


class MeasurePage(RunnerPage):
    TITLE = "リアルタイム計測"
    LOG_LABEL = "計測ログ"

    # 「変更」リンク（校正）。MainWindow がキャリブレーション画面へ移る
    calibration_requested = QtCore.Signal()
    # その場で保存したい設定の変更（「J の数値」スイッチ）。MainWindow が設定を保存する
    settings_edited = QtCore.Signal()

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        # RunnerPage.__init__ が _on_state を呼ぶので、そこで触るものは先に作っておく。
        # ゲージ窓は親を持たない独立の窓（第 2 モニタに全画面で出すため）。閉じるのは shutdown。
        gauge_fonts.set_preset(settings.get("GAUGE_FONT_PRESET"))
        self._gauge_window = GaugeWindow(show_joules=bool(settings.get("GAUGE_SHOW_JOULES")))
        # 今（または最後）の実行の role。入力の選択は実行の後で変わりうるので、終了時の振る舞いは
        # 選択ではなくこちらで決める。
        self._run_role: str | None = None
        # 見出しの局面（"stopped" / "running" / "success" / "error"）と、今の実行で最後に届いたフレーム。
        self._header_phase = "stopped"
        self._last_frame: GaugeFrame | None = None
        super().__init__(settings, "realtime", parent)
        # 見せるかどうかは、ページに入ってから決める（親の無いうちに見せると独立の窓になる）。
        self._sync_input_widgets()
        self._badge.hide()  # 色だけで状態を示すバッジ。代わりに _run_status を出す
        self._log.setPlaceholderText("未実行")
        self._refresh_broken_flags()

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
        # 計測中に設定を変えても子プロセスには届かない。誤解を招くので触れなくする
        # （開示そのものは開閉でき、値は見られる）。理由は _locked_reason に出す。
        return [self._editors, self._form]

    def build_side_panel(self) -> QtWidgets.QWidget:
        self._advanced = Disclosure("実験者用の詳細設定", self._build_advanced())

        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(self._advanced)
        inner_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(inner)

        self._output_link = QtWidgets.QLabel('<a href="#">出力フォルダ</a>')
        self._output_link.setTextFormat(QtCore.Qt.RichText)
        self._output_link.setTextInteractionFlags(
            QtCore.Qt.LinksAccessibleByMouse | QtCore.Qt.LinksAccessibleByKeyboard
        )
        self._output_link.linkActivated.connect(self._open_output_folder)
        self._output_link.setVisible(False)  # 終わった後に出す

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll, 1)
        layout.addWidget(self._output_link)
        return panel

    def _build_advanced(self) -> QtWidgets.QWidget:
        """詳細設定の開示の中身。入力・被験者番号・体重・校正の欄と、入れ子の「開発・診断用」。"""
        self._locked_reason = QtWidgets.QLabel("計測中は変更できません")
        self._locked_reason.setVisible(False)

        # 入力（R2-01）。n=2 なのでコンボボックスではなくラジオ
        self._input_usb = QtWidgets.QRadioButton("USB カメラ 2 台")
        self._input_hybrid = QtWidgets.QRadioButton("Mac＋Pixel")
        self._input_group = QtWidgets.QButtonGroup(self)
        for button_id, button in enumerate((self._input_usb, self._input_hybrid)):
            self._input_group.addButton(button, button_id)
        self._input_group.button(_INPUT_ROLES.index(self._runner.role)).setChecked(True)
        self._input_group.idToggled.connect(self._on_input_toggled)
        inputs = QtWidgets.QWidget()
        inputs_layout = QtWidgets.QHBoxLayout(inputs)
        inputs_layout.setContentsMargins(0, 0, 0, 0)
        inputs_layout.addWidget(self._input_usb)
        inputs_layout.addWidget(self._input_hybrid)
        inputs_layout.addStretch(1)

        self._subject_edit = QtWidgets.QLineEdit(str(self._settings.get("SUBJECT_ID") or ""))
        self._subject_edit.textChanged.connect(lambda text: self._settings.set("SUBJECT_ID", text))

        self._body_mass = QtWidgets.QDoubleSpinBox()
        self._body_mass.setDecimals(1)
        self._body_mass.setRange(20.0, 200.0)
        self._body_mass.valueChanged.connect(lambda value: self._settings.set("BODY_MASS_KG", value))
        # 結線してから値を入れる。保存値が範囲外なら欄は丸めた値を見せるので、設定もそれにそろえる
        # （先に入れると、欄は 200 と見せたまま子には 300 が渡る）
        self._body_mass.setValue(float(self._settings.get("BODY_MASS_KG")))
        self._settings.set("BODY_MASS_KG", self._body_mass.value())
        mass = QtWidgets.QWidget()
        mass_layout = QtWidgets.QHBoxLayout(mass)
        mass_layout.setContentsMargins(0, 0, 0, 0)
        mass_layout.addWidget(self._body_mass)
        mass_layout.addWidget(QtWidgets.QLabel("kg"))  # 単位は欄の直後（R9-08）
        mass_layout.addStretch(1)

        # 使う校正の日時（R13-05）と「変更」リンク（R16-10）。混成の校正なので混成のときだけ出す
        self._calibration_time = QtWidgets.QLabel(calibration_time_text())
        self._calibration_link = QtWidgets.QLabel('<a href="#">変更</a>')
        self._calibration_link.setTextFormat(QtCore.Qt.RichText)
        self._calibration_link.setTextInteractionFlags(
            QtCore.Qt.LinksAccessibleByMouse | QtCore.Qt.LinksAccessibleByKeyboard
        )
        self._calibration_link.linkActivated.connect(lambda _href: self.calibration_requested.emit())
        self._calibration_row = QtWidgets.QWidget()
        calibration_layout = QtWidgets.QHBoxLayout(self._calibration_row)
        calibration_layout.setContentsMargins(0, 0, 0, 0)
        calibration_layout.addWidget(self._calibration_time)
        calibration_layout.addWidget(self._calibration_link)
        calibration_layout.addStretch(1)

        self._editors = QtWidgets.QWidget()
        self._rows = QtWidgets.QFormLayout(self._editors)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.addRow("入力", inputs)
        self._rows.addRow("被験者番号", self._subject_edit)
        self._rows.addRow("体重", mass)
        self._rows.addRow("校正", self._calibration_row)

        self._form = SettingsForm(self._settings, exclude=_DEDICATED_SETTINGS)
        self._form.changed.connect(self._refresh_broken_flags)
        self._dev = Disclosure("開発・診断用", self._form)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 4, 0, 0)
        layout.addWidget(self._locked_reason)
        layout.addWidget(self._editors)
        layout.addWidget(self._dev)
        return content

    def shutdown(self) -> None:
        super().shutdown()
        self._gauge_window.close()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802  (Qt の命名規則)
        # キャリブレーション画面で校正し直して戻ってきたときに、新しい日時を出す
        super().showEvent(event)
        self._calibration_time.setText(calibration_time_text())

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
            # 開発・診断用の欄で書体の組を変えていれば、ここで効かせる
            gauge_fonts.set_preset(self._settings.get("GAUGE_FONT_PRESET"))
            self._gauge_window.begin(
                show_joules=bool(self._settings.get("GAUGE_SHOW_JOULES")),
                avoid_screen=self.window().screen(),
            )

    def _on_finished(self, exit_code: int) -> None:
        if self._run_role == _HYBRID_ROLE:
            self._gauge_window.finish(exit_code)
        self._header_phase = "success" if exit_code == 0 else "error"
        self._refresh_header()
        self._output_link.setVisible(True)  # 異常終了でも、途中までの出力はある

    def _on_gauge_frame(self, frame: GaugeFrame) -> None:
        self._last_frame = frame
        self._refresh_header()

    def _on_joules_toggled(self, checked: bool) -> None:
        self._settings.set("GAUGE_SHOW_JOULES", checked)
        self._gauge_window.set_show_joules(checked)
        self.settings_edited.emit()

    def _on_input_toggled(self, button_id: int, checked: bool) -> None:
        if not checked:
            return  # 外れた側の通知。入った側の通知で切り替える
        self._runner.role = _INPUT_ROLES[button_id]
        self._sync_input_widgets()

    def _sync_input_widgets(self) -> None:
        """入力が Mac＋Pixel のときだけ出すもの（J のスイッチ・校正の行。R15-01）。"""
        hybrid = self._runner.role == _HYBRID_ROLE
        self._joules_switch.setVisible(hybrid)
        self._rows.setRowVisible(self._calibration_row, hybrid)

    def _open_output_folder(self, _href: str) -> None:
        folder = (hybrid_paths.measurement_root()
                  if self._run_role == _HYBRID_ROLE else measurement_output_dir())
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    # -- 表示更新 ----------------------------------------------------------
    def _refresh_broken_flags(self) -> None:
        # 計算を壊す設定は入れ子の「開発・診断用」にあるので、閉じていても見えるよう両方の見出しに出す
        n = broken_flag_count(self._settings)
        self._advanced.set_badge(n)
        self._dev.set_badge(n)

    def _on_state(self, state: str) -> None:
        super()._on_state(state)
        running = state in ("starting", "running")
        self._main_button.setText("停止" if running else "計測を開始")
        self._locked_reason.setVisible(running)
        if running:
            self._output_link.setVisible(False)
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
