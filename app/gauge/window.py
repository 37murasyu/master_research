"""被験者ゲージの窓。第 2 モニタがあれば全画面、無ければ 1280×720 の窓。

被験者が見るだけの画面なので、中身は ``app.gauge.widget.GaugeWidget`` 1 つ
だけで、ボタンなどの部品は置かない。別セッション（``WorkerRunner`` と
つなぐ側）はここが公開する ``set_frame``・``set_show_joules``・``begin``・
``finish`` を口として使う想定（controller の指示）。

第 2 モニタの選び方（``choose_screen``）は ``Gauge_display.py``
（675〜686 行、触らないファイル）の ``QApplication.screens()`` と
``availableGeometry`` を参考にしたが、あちらの「screen_idx を環境変数で選ぶ」
仕組みは要らないので、「作業者の窓がある画面を避けて 1 番目に見つかった画面」
という単純な規則にしてある。
"""

from __future__ import annotations

from app.core.qt import QtCore, QtGui, QtWidgets
from app.gauge.protocol import GaugeFrame
from app.gauge.widget import GaugeWidget

__all__ = ["choose_screen", "GaugeWindow"]

WINDOW_TITLE = "上肢の仕事量"

# 第 2 モニタが無いときの窓の大きさ（brief）。
_FALLBACK_SIZE = QtCore.QSize(1280, 720)


def choose_screen(screens: list[QtGui.QScreen], avoid: QtGui.QScreen | None) -> QtGui.QScreen | None:
    """``avoid``（作業者の窓がある画面）と違う最初の画面を返す。無ければ ``None``。

    ``avoid`` が ``None`` のときは ``screens[0]``（主画面）を避ける対象とみなす
    （controller の「判断済みのこと」）。作業者の実験者用の窓が必ず主画面に
    あるとは限らないが、明示的に渡されなければ「主画面ではないほうを被験者に
    見せる」が一番安全な既定になる。
    """
    if not screens:
        return None
    target = avoid if avoid is not None else screens[0]
    for screen in screens:
        if screen is not target:
            return screen
    return None


class GaugeWindow(QtWidgets.QWidget):
    """被験者ゲージの窓そのもの。中身は ``GaugeWidget`` 1 つだけ。

    閉じても隠れるだけにする（``WA_DeleteOnClose`` を立てない。既定で
    立っていないが、ここで消してはいけない性質だと分かるように書いておく）。
    被験者の窓を閉じてもアプリ全体が終了しないよう ``WA_QuitOnClose`` は
    立てない。
    """

    def __init__(self, *, show_joules: bool = True, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(WINDOW_TITLE)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

        self._gauge = GaugeWidget(show_joules=show_joules)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._gauge)

    # -- 公開の口 ------------------------------------------------------------

    @property
    def gauge(self) -> GaugeWidget:
        """中身の ``GaugeWidget``。診断・試験で状態を覗く用（brief の 4 口とは別）。"""
        return self._gauge

    def set_frame(self, frame: GaugeFrame) -> None:
        self._gauge.set_frame(frame)

    def set_show_joules(self, show_joules: bool) -> None:
        self._gauge.set_show_joules(show_joules)

    def finish(self, exit_code: int) -> None:
        self._gauge.finish(exit_code)

    def begin(self, *, show_joules: bool, avoid_screen: QtGui.QScreen | None = None) -> None:
        """次の計測の前に呼ぶ。まっさらな状態に戻し、J の表示を反映して画面に出す。

        ``GaugeWidget.reset()`` は J 表示の有無を変えないので、ここで
        ``reset`` の直後に ``set_show_joules`` を呼び、呼び出し側が指定した
        値で上書きする（widget タスクの docstring に書いた前提どおり）。
        """
        self._gauge.reset()
        self._gauge.set_show_joules(show_joules)
        self.present(avoid_screen)

    def present(self, avoid_screen: QtGui.QScreen | None = None) -> None:
        """画面に出す。第 2 モニタがあれば全画面、無ければ 1280×720 の窓を中央に。"""
        screen = choose_screen(QtWidgets.QApplication.screens(), avoid_screen)
        if screen is not None:
            # ネイティブの窓を先に作ってから setScreen しないと、画面をまたぐ
            # 移動が効かないことがある（Qt の既知の癖。brief の手順どおり）。
            self.winId()
            self.setScreen(screen)
            self.setGeometry(screen.geometry())
            self.showFullScreen()
        else:
            self.resize(_FALLBACK_SIZE)
            self._center_on_primary_screen()
            self.showNormal()

    # -- 内部 ------------------------------------------------------------------

    def _center_on_primary_screen(self) -> None:
        primary = QtWidgets.QApplication.primaryScreen()
        if primary is None:
            return
        geo = primary.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + (geo.height() - self.height()) // 2
        self.move(x, y)
