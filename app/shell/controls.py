"""計測画面（実験者用）で使う小さな部品。

``ToggleSwitch``（切り替えスイッチ）・``Disclosure``（開閉できる見出し）・
``CountBadge``（件数バッジ）・``StatusText``（状態の文字）の 4 つ。

計測画面の見出しは白地（Qt の標準パレット）の上に置く前提なので、文字の色は
基本 Qt の標準パレットに任せ、状態を示す色（琥珀・赤）だけを theme.py の
定数から明示的に取る。constraints.md の「色だけで状態を示さない」を守り、
色を使うところには必ず記号か位置（つまみの左右・矢印の向き）を添える。
"""

from __future__ import annotations

from app.core.qt import QtCore, QtGui, QtWidgets
from app.shell import theme

__all__ = ["ToggleSwitch", "Disclosure", "CountBadge", "StatusText"]


class ToggleSwitch(QtWidgets.QAbstractButton):
    """溝とつまみを自前で描く on/off スイッチ。

    QCheckBox ではなく QAbstractButton を継承して自前で描くのは、計測画面の
    ほかの部品（記号＋文字で状態を示す）と見た目をそろえるため。状態は
    つまみの位置（左/右）でも示すので、色（琥珀/溝の色）だけに頼らない。
    """

    _TRACK_W = 44
    _TRACK_H = 22
    _MARGIN = 2

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setText(text)
        self.setAccessibleName(text)
        self.toggled.connect(lambda _checked: self.update())

    def sizeHint(self) -> QtCore.QSize:
        metrics = self.fontMetrics()
        text_w = metrics.horizontalAdvance(self.text()) if self.text() else 0
        gap = 8 if self.text() else 0
        width = self._TRACK_W + gap + text_w
        height = max(self._TRACK_H + 6, metrics.height() + 6)
        return QtCore.QSize(width, height)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        checked = self.isChecked()
        track_top = (self.height() - self._TRACK_H) / 2.0
        track_rect = QtCore.QRectF(0.0, track_top, float(self._TRACK_W), float(self._TRACK_H))

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(theme.AMBER if checked else theme.TRACK))
        painter.drawRoundedRect(track_rect, self._TRACK_H / 2.0, self._TRACK_H / 2.0)

        thumb_d = self._TRACK_H - 2 * self._MARGIN
        thumb_x = (
            track_rect.right() - thumb_d - self._MARGIN
            if checked
            else track_rect.left() + self._MARGIN
        )
        thumb_rect = QtCore.QRectF(thumb_x, track_rect.top() + self._MARGIN, float(thumb_d), float(thumb_d))
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.drawEllipse(thumb_rect)

        if self.hasFocus():
            pen = QtGui.QPen(QtGui.QColor(theme.HEADER))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            focus_rect = track_rect.adjusted(-2, -2, 2, 2)
            painter.drawRoundedRect(focus_rect, self._TRACK_H / 2.0 + 2, self._TRACK_H / 2.0 + 2)

        if self.text():
            text_rect = QtCore.QRectF(
                self._TRACK_W + 8, 0.0, self.width() - self._TRACK_W - 8, float(self.height())
            )
            painter.setPen(self.palette().color(QtGui.QPalette.WindowText))
            painter.drawText(text_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, self.text())


class Disclosure(QtWidgets.QWidget):
    """開閉できる見出し（QToolButton）と、その下の中身をひとまとめにする部品。

    矢印の向き（閉:右／開:下）が開閉状態を示す。中身は QWidget の
    setVisible で出し入れするだけで、破棄・再生成はしない（開閉のたびに
    中身を作り直すと、中の入力値が消えてしまう）。
    """

    toggled = QtCore.Signal(bool)

    def __init__(
        self,
        title: str,
        content: QtWidgets.QWidget,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._content = content

        self._button = QtWidgets.QToolButton(self)
        self._button.setCheckable(True)
        self._button.setText(title)
        self._button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._button.setArrowType(QtCore.Qt.RightArrow)
        self._button.setAutoRaise(True)
        self._button.toggled.connect(self._on_toggled)

        self._badge = CountBadge(self)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._button)
        header.addWidget(self._badge)
        header.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self._content)

        self._content.setVisible(False)

    def _on_toggled(self, checked: bool) -> None:
        self._button.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self._content.setVisible(checked)
        self.toggled.emit(checked)

    def set_open(self, open_: bool) -> None:
        self._button.setChecked(open_)

    def is_open(self) -> bool:
        return self._button.isChecked()

    def set_badge(self, n: int) -> None:
        self._badge.set_count(n)


class CountBadge(QtWidgets.QLabel):
    """件数を丸背景の数字で示す。0 件のときは何も知らせることが無いので隠れる。"""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.set_count(0)

    def set_count(self, n: int) -> None:
        if n <= 0:
            self.setVisible(False)
            self.setText("")
            return
        self.setText(f"✕ {n}")
        self.setStyleSheet(
            f"background-color: {theme.OVER}; color: #ffffff;"
            f" border-radius: 8px; padding: 1px 6px; font-weight: 600;"
        )
        self.setVisible(True)


class StatusText(QtWidgets.QLabel):
    """記号と文字の組で状態を示すラベル（色だけで状態を示さない）。

    リッチテキストで琥珀の点（●）だけに色を付ける。点の形は「今動いている
    ものがある」ときは ● （塗り）、無いときは ○ （抜き）で区別する
    （``set_link`` の接続待ち）。
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setTextFormat(QtCore.Qt.RichText)
        self.set_status("stopped")

    def _dot(self, color: str) -> str:
        return f'<span style="color:{color};">●</span>'

    def set_status(self, state: str, rep: int | None = None) -> None:
        if state == "running":
            rep_text = f" {rep} 回目" if rep is not None else ""
            self.setText(f"{self._dot(theme.AMBER)} 計測中{rep_text}")
        elif state == "success":
            self.setText("✓ 正常終了")
        elif state == "error":
            self.setText(f'<span style="color:{theme.OVER};">✕</span> 異常終了')
        else:  # "stopped" を含め、既定は停止中扱い
            self.setText("停止中")

    def set_link(self, connected: bool) -> None:
        if connected:
            self.setText(f"{self._dot(theme.AMBER)} Pixel 接続")
        else:
            self.setText("○ Pixel 接続待ち")
