"""画面をまたいで使う小さな部品。"""

from __future__ import annotations

from app.core.qt import QtCore, QtGui, QtWidgets
from app.core.settings import SCHEMA, Setting, Settings
from app.runners.worker import WorkerRunner

__all__ = ["LogView", "SettingsForm", "StatusBadge", "RunnerPage"]


class LogView(QtWidgets.QPlainTextEdit):
    """子プロセスの出力を流す領域。

    行数に上限を設ける。計測は毎フレーム print するので、放っておくと
    長時間の計測でメモリを食い潰す。
    """

    MAX_BLOCKS = 5000

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_BLOCKS)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

    def append_text(self, text: str) -> None:
        """末尾に追記し、利用者が上を見ていなければ自動で追従する。"""
        scrollbar = self.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4

        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)

        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())


class StatusBadge(QtWidgets.QLabel):
    """状態を色付きで示す小さな表示。"""

    _COLORS = {
        "stopped": ("#6b7280", "停止中"),
        "starting": ("#d97706", "起動中"),
        "running": ("#059669", "計測中"),
        "stopping": ("#d97706", "停止処理中"),
        "error": ("#dc2626", "エラー"),
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumWidth(88)
        self.set_state("stopped")

    def set_state(self, state: str) -> None:
        color, label = self._COLORS.get(state, self._COLORS["stopped"])
        self.setText(label)
        self.setStyleSheet(
            f"background-color: {color}; color: white; "
            f"border-radius: 4px; padding: 4px 10px; font-weight: 600;"
        )


class SettingsForm(QtWidgets.QWidget):
    """UI に出す設定項目をグループごとに並べる。

    ``Setting.ui_visible`` が真のものだけを扱う。135 個すべてを並べても
    使えないので、意味のあるものに絞ってある（``settings.CURATED`` を参照）。
    ``exclude`` に挙げた項目は並べない（画面が専用の欄を別に置いているもの）。
    """

    changed = QtCore.Signal()

    def __init__(
        self,
        settings: Settings,
        exclude: frozenset[str] = frozenset(),
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._settings = settings

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        visible = [s for s in SCHEMA.values() if s.ui_visible and s.name not in exclude]
        groups: dict[str, list[Setting]] = {}
        for setting in visible:
            groups.setdefault(setting.group, []).append(setting)

        for group_name, items in sorted(groups.items()):
            box = QtWidgets.QGroupBox(group_name)
            form = QtWidgets.QFormLayout(box)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            for setting in sorted(items, key=lambda s: s.name):
                label, editor = self._build_row(setting)
                form.addRow(label, editor)
            layout.addWidget(box)

        layout.addStretch(1)

    def _build_row(self, setting: Setting) -> tuple[QtWidgets.QWidget, QtWidgets.QWidget]:
        label = QtWidgets.QLabel(setting.name)
        label.setToolTip(setting.description)

        value = self._settings.get(setting.name)

        if setting.type == "bool":
            editor = QtWidgets.QCheckBox()
            editor.setChecked(bool(value))
            editor.toggled.connect(
                lambda checked, name=setting.name: self._on_change(name, checked)
            )
        elif setting.type == "int":
            editor = QtWidgets.QSpinBox()
            editor.setRange(-1_000_000, 1_000_000)
            editor.setValue(int(value or 0))
            editor.valueChanged.connect(
                lambda v, name=setting.name: self._on_change(name, v)
            )
        elif setting.type == "float":
            editor = QtWidgets.QDoubleSpinBox()
            editor.setRange(-1_000_000.0, 1_000_000.0)
            editor.setDecimals(4)
            editor.setValue(float(value or 0.0))
            editor.valueChanged.connect(
                lambda v, name=setting.name: self._on_change(name, v)
            )
        else:
            editor = QtWidgets.QLineEdit(str(value or ""))
            editor.textChanged.connect(
                lambda text, name=setting.name: self._on_change(name, text)
            )

        # 説明はツールチップだけでなく、その場に出す。「なぜこの既定値か」は
        # 隠すべきではない（特に既定を意図的に変えた 4 つのフラグ）。
        if setting.description:
            editor.setToolTip(setting.description)

        return label, editor

    def _on_change(self, name: str, value: object) -> None:
        try:
            self._settings.set(name, value)
        except (TypeError, ValueError):
            return  # 入力途中の不正値は無視する
        self.changed.emit()


class RunnerPage(QtWidgets.QWidget):
    """「左に操作パネル、右にログ」という 3 画面共通の骨格。

    計測・キャリブレーション・解析はどれも「子プロセスを起動して、
    その出力を眺める」画面で、違うのは左パネルの中身だけだった。
    以前は 3 ファイルがヘッダ・ログパネル・状態遷移・終了処理を
    それぞれ書いており、合わせて約 90 行が同じ仕事をしていた。

    サブクラスが決めるのは:
        TITLE / LOG_LABEL / SPLIT_SIZES
        build_side_panel()                左パネル（唯一の必須実装）
        header_widgets()                  ヘッダに置く追加ウィジェット
        widgets_disabled_while_running()  実行中（停止を待つ間も）に触れなくするもの
        widgets_enabled_while_running()   実行中だけ押せるもの（中止ボタンなど。停止を待つ間は押せない）
    """

    TITLE: str = ""
    LOG_LABEL: str = "ログ"
    SPLIT_SIZES: tuple[int, int] = (360, 640)

    # 子がまだ動いている状態（停止を求めて終わるのを待つ "stopping" を含む）
    BUSY_STATES = ("starting", "running", "stopping")

    def __init__(
        self,
        settings: Settings,
        role: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._settings = settings
        # 最後に届いた実行の状態（WorkerRunner.state_changed）
        self._state = "stopped"

        self._runner = WorkerRunner(role, self)
        self._runner.output.connect(self.append_log)
        self._runner.state_changed.connect(self._on_state)

        self._build_ui()
        self._on_state("stopped")

    # -- 骨格 --------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)
        outer.addLayout(self._build_header())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.build_side_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(self.SPLIT_SIZES))
        outer.addWidget(splitter, 1)

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()

        title = QtWidgets.QLabel(self.TITLE)
        font = title.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        title.setFont(font)
        row.addWidget(title)
        row.addStretch(1)

        self._badge = StatusBadge()
        row.addWidget(self._badge)
        for widget in self.header_widgets():
            row.addWidget(widget)
        return row

    def _build_log_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(self.LOG_LABEL))
        self._log = LogView()
        layout.addWidget(self._log, 1)
        return panel

    # -- サブクラスが実装する部分 ------------------------------------------
    def build_side_panel(self) -> QtWidgets.QWidget:
        raise NotImplementedError

    def header_widgets(self) -> list[QtWidgets.QWidget]:
        return []

    def widgets_disabled_while_running(self) -> list[QtWidgets.QWidget]:
        return []

    def widgets_enabled_while_running(self) -> list[QtWidgets.QWidget]:
        return []

    # -- 共通の振る舞い ----------------------------------------------------
    def append_log(self, text: str) -> None:
        self._log.append_text(text)

    def _on_state(self, state: str) -> None:
        self._state = state
        self._badge.set_state(state)
        busy = state in self.BUSY_STATES
        for widget in self.widgets_disabled_while_running():
            widget.setEnabled(not busy)
        # 停止を待つ間は中止も押せない（もう求めてある）
        for widget in self.widgets_enabled_while_running():
            widget.setEnabled(busy and state != "stopping")

    def shutdown(self) -> None:
        """ウィンドウを閉じるとき、子プロセスを残さない（終わるまで待つ。猶予を過ぎたら強制終了）。"""
        self._runner.stop_and_wait()

    @property
    def is_running(self) -> bool:
        return self._runner.is_running
