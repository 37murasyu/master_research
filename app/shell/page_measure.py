"""計測画面。設定を編集し、計測ワーカーを起動・停止し、その出力を見る。

主ボタンは 1 つで、停止中は「計測を開始」、実行中は「停止」に入れ替わる（設計書 §5.2、R16-01）。
入力は 3 つ（USB カメラ 2 台・Mac＋Pixel・記録の再生）。Mac＋Pixel（混成）と記録の再生のときは、開始と同時に
被験者ゲージの窓（``app.gauge.window``）を開き、子の出力のゲージの行（``WorkerRunner.gauge_frame``）をそこへ流す。
「J の数値」スイッチはゲージ窓を開く入力のときだけ出し、実行中も切り替えられる（ゲージ窓は同じプロセスにあるので、
その場で効く）。記録の再生は実機の計測とは別の role（``hybrid_replay``）で、再生する計測フォルダを選ぶまで
主ボタンを押せない（押せない理由を主ボタンの左に出す）。

見出しの状態は、骨格の色付きバッジ（緑を含む）を隠し、点と文字の組（``StatusText``）で出す。
1 つめは実行の状態（ゲージの行を出す入力なら回数も）、2 つめは混成の実行中だけの Pixel 接続（再生は Pixel を
使わない。ゲージ窓の見出しが「▶ 再生」を出す）。回数と接続はゲージの行から取る。

設定は「実験者用の詳細設定」の開示（既定で閉じる。R11-03）にしまう。中身は入力のラジオ、被験者番号、
体重、使う校正の日時と「変更」リンク（混成のときだけ）、再生する計測フォルダと「選ぶ…」（再生のときだけ）、
「開発・診断用」の入れ子の開示（残りの設定フォーム）。画面に説明文は置かない。出力先は、終わった後の「出力フォルダ」リンクで開く。
計算を壊す設定（``app_default`` で既定を無効にした 4 つ）が有効なら、その件数を開示の見出しに出す
（R20-03）。ログは、まだ何も流れていないうちは「未実行」と出す（R19-01）。

入力ごとに何を出し入れするか（ゲージ窓・J のスイッチ・接続・校正の行・出力フォルダ）は ``MEASURE_INPUTS`` の
1 か所に書き、画面のコードは role の文字列を比べない。入力を足すときは表に 1 行足す。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.core.qt import QtCore, QtGui, QtWidgets
from app.core.settings import SCHEMA, Settings, measurement_output_dir
from app.gauge.protocol import GaugeFrame
from app.gauge.window import GaugeWindow
from app.hybrid import paths as hybrid_paths
from app.shell.controls import Disclosure, StatusText, ToggleSwitch
from app.shell.widgets import RunnerPage, SettingsForm

__all__ = ["MeasurePage", "MeasureInput", "MEASURE_INPUTS", "measure_input"]


@dataclass(frozen=True)
class MeasureInput:
    """計測画面の入力 1 つぶんの事実。画面は role の文字列を比べず、すべてこれを引く。

    ``gauge``: 子がゲージの行を出すので、開始と同時にゲージ窓を開き、J のスイッチを出し、見出しに回数を出す。
    ``pixel_link``: 見出しに Pixel の接続を出す。``calibration_row``: 使う校正の日時と「変更」リンクの行を出す。
    ``output_root``: 終わった後の「出力フォルダ」リンクが開く場所（呼ぶたびに求める。試験で差し替えられるように）。
    ``replay_folder``: 再生する計測フォルダ（設定 ``HYBRID_REPLAY``）の行を出し、選ぶまで開始させない。
    """

    role: str
    label: str
    output_root: Callable[[], Path]
    gauge: bool = False
    pixel_link: bool = False
    calibration_row: bool = False
    replay_folder: bool = False


# 入力のラジオはこの並び（ラジオの id は添字）
MEASURE_INPUTS = (
    MeasureInput("realtime", "USB カメラ 2 台", output_root=lambda: measurement_output_dir()),
    MeasureInput(
        "hybrid_measure", "Mac＋Pixel", output_root=lambda: hybrid_paths.measurement_root(),
        gauge=True, pixel_link=True, calibration_row=True,
    ),
    # 記録した計測を流し直す（カメラも Pixel も使わない）。記録は本番の計測と混ざらないよう replay_root に書く
    MeasureInput(
        "hybrid_replay", "記録の再生", output_root=lambda: hybrid_paths.replay_root(),
        gauge=True, replay_folder=True,
    ),
)


def measure_input(role: str) -> MeasureInput:
    """role の入力の記述。表に無い role は ValueError（画面が知らない子を起動しない）。"""
    for spec in MEASURE_INPUTS:
        if spec.role == role:
            return spec
    raise ValueError(f"計測画面の入力に無い role: {role}")

# 専用の欄を置くので、入れ子の設定フォームには並べない項目
_DEDICATED_SETTINGS = frozenset({"SUBJECT_ID", "BODY_MASS_KG", "HYBRID_REPLAY"})
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


def replay_folder_problem(text: str) -> str | None:
    """再生する計測フォルダ（設定 ``HYBRID_REPLAY``）で開始できない理由。開始できるなら None。

    子（``app.runners.hybrid_replay``）と同じく ``meta.json`` のあるフォルダを計測フォルダとみなす。
    """
    if not text.strip():
        return "再生する計測フォルダを選んでください"
    folder = Path(text.strip()).expanduser()
    if not folder.is_dir():
        return "計測フォルダがありません"
    if not (folder / "meta.json").is_file():
        return "計測フォルダではありません"
    return None


def _link_label(text: str, slot) -> QtWidgets.QLabel:
    """押せるリンクの文字（``<a>`` 1 つ）。押されたら ``slot(href)`` を呼ぶ。"""
    label = QtWidgets.QLabel(f'<a href="#">{text}</a>')
    label.setTextFormat(QtCore.Qt.RichText)
    label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse | QtCore.Qt.LinksAccessibleByKeyboard)
    label.linkActivated.connect(slot)
    return label


def _hrow(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """``widgets`` を左詰めで横に並べた行（余白なし、右は伸び縮みで埋める）。"""
    row = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    return row


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
        self._gauge_window = GaugeWindow(show_joules=bool(settings.get("GAUGE_SHOW_JOULES")),
                                         font_preset=settings.get("GAUGE_FONT_PRESET"))
        # 今（または最後）の実行の入力。入力の選択は実行の後で変わりうるので、終了時の振る舞いは
        # 選択ではなくこちらで決める。
        self._run_input: MeasureInput | None = None
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

        # 主ボタンを押せない理由（R15-04）。押せるときは隠す
        self._start_blocked = QtWidgets.QLabel()
        self._start_blocked.setVisible(False)

        self._main_button = QtWidgets.QPushButton("計測を開始")
        self._main_button.clicked.connect(self._on_main_button)
        return [self._run_status, self._link_status, self._joules_switch, self._start_blocked, self._main_button]

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

        self._output_link = _link_label("出力フォルダ", self._open_output_folder)
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

        # 入力（R2-01）。数が少ないのでコンボボックスではなくラジオ。並びと名前は MEASURE_INPUTS
        self._input_group = QtWidgets.QButtonGroup(self)
        self._input_radios = [QtWidgets.QRadioButton(spec.label) for spec in MEASURE_INPUTS]
        for button_id, button in enumerate(self._input_radios):
            self._input_group.addButton(button, button_id)
        self._input_usb, self._input_hybrid, self._input_replay = self._input_radios
        self._input_group.button(MEASURE_INPUTS.index(self._input)).setChecked(True)
        self._input_group.idToggled.connect(self._on_input_toggled)
        # 3 つを横に並べると左の欄の既定の幅（360）を超えて切れるので、縦に並べる
        inputs = QtWidgets.QWidget()
        inputs_layout = QtWidgets.QVBoxLayout(inputs)
        inputs_layout.setContentsMargins(0, 0, 0, 0)
        for button in self._input_radios:
            inputs_layout.addWidget(button)

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
        mass = _hrow(self._body_mass, QtWidgets.QLabel("kg"))  # 単位は欄の直後（R9-08）

        # 使う校正の日時（R13-05）と「変更」リンク（R16-10）。混成の校正なので混成のときだけ出す
        self._calibration_time = QtWidgets.QLabel(calibration_time_text())
        self._calibration_link = _link_label("変更", lambda _href: self.calibration_requested.emit())
        self._calibration_row = _hrow(self._calibration_time, self._calibration_link)

        # 再生する計測フォルダ。再生のときだけ出す。空・無いフォルダのうちは主ボタンを押せない
        self._replay_edit = QtWidgets.QLineEdit(str(self._settings.get("HYBRID_REPLAY") or ""))
        self._replay_edit.textChanged.connect(self._on_replay_folder_edited)
        self._replay_choose = QtWidgets.QPushButton("選ぶ…")
        self._replay_choose.clicked.connect(self._choose_replay_folder)
        self._replay_row = QtWidgets.QWidget()
        replay_layout = QtWidgets.QHBoxLayout(self._replay_row)
        replay_layout.setContentsMargins(0, 0, 0, 0)
        replay_layout.addWidget(self._replay_edit, 1)
        replay_layout.addWidget(self._replay_choose)

        self._editors = QtWidgets.QWidget()
        self._rows = QtWidgets.QFormLayout(self._editors)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.addRow("入力", inputs)
        self._rows.addRow("被験者番号", self._subject_edit)
        self._rows.addRow("体重", mass)
        self._rows.addRow("校正", self._calibration_row)
        self._rows.addRow("記録", self._replay_row)

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

    @property
    def _input(self) -> MeasureInput:
        """今選んでいる入力。"""
        return measure_input(self._runner.role)

    # -- 開始・停止 --------------------------------------------------------
    def _on_main_button(self) -> None:
        if self.is_running:
            self._runner.stop()
        else:
            self._start()

    def _start(self) -> None:
        spec = self._input
        self._last_frame = None
        if not self._runner.start(self._settings):
            return  # 起動に失敗したときは窓を開かない
        self._run_input = spec
        # start の中で出た "running" の時点では _run_input が前回のままなので、ここで出し直す
        self._refresh_header()
        if spec.gauge:
            self._gauge_window.begin(
                show_joules=bool(self._settings.get("GAUGE_SHOW_JOULES")),
                # 開発・診断用の欄で書体の組を変えていれば、ここで効かせる
                font_preset=self._settings.get("GAUGE_FONT_PRESET"),
                avoid_screen=self.window().screen(),
            )

    def _on_finished(self, exit_code: int) -> None:
        if self._run_input is not None and self._run_input.gauge:
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
        self._runner.role = MEASURE_INPUTS[button_id].role
        self._sync_input_widgets()

    def _sync_input_widgets(self) -> None:
        """入力によって出し入れするもの（J のスイッチ・校正の行・再生の計測フォルダの行。R15-01）。"""
        spec = self._input
        self._joules_switch.setVisible(spec.gauge)
        self._rows.setRowVisible(self._calibration_row, spec.calibration_row)
        self._rows.setRowVisible(self._replay_row, spec.replay_folder)
        self._refresh_main_button()

    def _on_replay_folder_edited(self, text: str) -> None:
        self._settings.set("HYBRID_REPLAY", text.strip())
        self._refresh_main_button()

    def _choose_replay_folder(self) -> None:
        current = Path(self._replay_edit.text().strip()).expanduser()
        start = current if self._replay_edit.text().strip() and current.is_dir() else hybrid_paths.measurement_root()
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "再生する計測フォルダ", str(start))
        if chosen:  # 取り消しは空の文字
            self._replay_edit.setText(chosen)

    def _start_problem(self) -> str | None:
        """今の入力で開始できない理由。開始できるなら None。"""
        if self._input.replay_folder:
            return replay_folder_problem(self._replay_edit.text())
        return None

    def _open_output_folder(self, _href: str) -> None:
        spec = self._run_input or self._input
        folder = spec.output_root()
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    # -- 表示更新 ----------------------------------------------------------
    def _refresh_broken_flags(self) -> None:
        # 計算を壊す設定は入れ子の「開発・診断用」にあるので、閉じていても見えるよう両方の見出しに出す
        n = broken_flag_count(self._settings)
        self._advanced.set_badge(n)
        self._dev.set_badge(n)

    def _refresh_main_button(self) -> None:
        """主ボタンを押せるか。実行中は常に押せる（停止）。止まっているときは開始できない理由があれば押せない。"""
        problem = None if self._header_phase == "running" else self._start_problem()
        self._main_button.setEnabled(problem is None)
        self._start_blocked.setText(problem or "")
        self._start_blocked.setVisible(problem is not None)

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
        self._refresh_main_button()

    def _refresh_header(self) -> None:
        spec = self._run_input if self._header_phase == "running" else None
        connected = (spec is not None and spec.gauge and self._last_frame is not None
                     and self._last_frame.link == "connected")
        # 回数は、ゲージの見出しと同じく「完了した回数＋1」。つながる前は出さない
        rep = self._last_frame.rep + 1 if connected else None
        self._run_status.set_status(self._header_phase, rep=rep)
        show_link = spec is not None and spec.pixel_link
        self._link_status.setVisible(show_link)
        if show_link:
            self._link_status.set_link(connected)
