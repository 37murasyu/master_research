"""画面をまたいで使う小さな部品。"""

from __future__ import annotations

from typing import Callable

from app.core.qt import QtCore, QtGui, QtWidgets
from app.core.settings import SCHEMA, Setting, Settings

__all__ = ["LogView", "SettingsForm", "StatusBadge"]


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
    """

    changed = QtCore.Signal()

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._editors: dict[str, Callable[[], object]] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        visible = [s for s in SCHEMA.values() if s.ui_visible]
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
            self._editors[setting.name] = editor.isChecked
        elif setting.type == "int":
            editor = QtWidgets.QSpinBox()
            editor.setRange(-1_000_000, 1_000_000)
            editor.setValue(int(value or 0))
            editor.valueChanged.connect(
                lambda v, name=setting.name: self._on_change(name, v)
            )
            self._editors[setting.name] = editor.value
        elif setting.type == "float":
            editor = QtWidgets.QDoubleSpinBox()
            editor.setRange(-1_000_000.0, 1_000_000.0)
            editor.setDecimals(4)
            editor.setValue(float(value or 0.0))
            editor.valueChanged.connect(
                lambda v, name=setting.name: self._on_change(name, v)
            )
            self._editors[setting.name] = editor.value
        else:
            editor = QtWidgets.QLineEdit(str(value or ""))
            editor.textChanged.connect(
                lambda text, name=setting.name: self._on_change(name, text)
            )
            self._editors[setting.name] = editor.text

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
