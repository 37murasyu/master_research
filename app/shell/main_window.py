"""メインウィンドウ。3 つの画面を切り替える。"""

from __future__ import annotations

from app.core.platform_compat import user_config_dir
from app.core.qt import QtCore, QtWidgets
from app.core.settings import APP_NAME, Settings
from app.shell.page_analyze import AnalyzePage
from app.shell.page_calibrate import CalibratePage
from app.shell.page_measure import MeasurePage
from app.shell.widgets import RunnerPage

__all__ = ["MainWindow"]

WINDOW_TITLE = "車椅子駆動 関節トルク計測"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self._settings_path = Settings.default_path()
        self._settings = settings if settings is not None else Settings.load(self._settings_path)

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1100, 720)

        self._pages: list[RunnerPage] = []
        self._build_ui()

    # -- 画面 --------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QtWidgets.QStackedWidget()
        self._nav = self._build_nav()

        layout.addWidget(self._nav)
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        measure = MeasurePage(self._settings)
        calibrate = CalibratePage(self._settings)
        for label, page in (
            ("計測", measure),
            ("キャリブレーション", calibrate),
            ("解析", AnalyzePage(self._settings)),
        ):
            self._nav.addItem(label)
            self._stack.addWidget(page)
            self._pages.append(page)

        # 計測画面の校正の「変更」リンクで、キャリブレーション画面へ移る
        measure.calibration_requested.connect(
            lambda: self._nav.setCurrentRow(self._pages.index(calibrate))
        )
        # スイッチの切り替えは、窓を閉じるのを待たずにその場で保存する
        measure.settings_edited.connect(self._save_settings)
        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)

        self.statusBar().showMessage(f"設定: {self._settings_path}")

    def _build_nav(self) -> QtWidgets.QListWidget:
        nav = QtWidgets.QListWidget()
        nav.setFixedWidth(180)
        nav.setFrameShape(QtWidgets.QFrame.NoFrame)
        nav.setStyleSheet(
            "QListWidget { background: #f3f4f6; border-right: 1px solid #e5e7eb; }"
            "QListWidget::item { padding: 14px 16px; }"
            "QListWidget::item:selected { background: #2563eb; color: white; }"
        )
        return nav

    # -- 終了 --------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802  (Qt の命名規則)
        """閉じる前に、動いている子プロセスを片付け、設定を保存する。

        黙って親だけ終了すると、計測プロセスが残って次回の起動時に
        カメラを掴んだままになる。
        """
        running = [page for page in self._pages if page.is_running]
        if running:
            answer = QtWidgets.QMessageBox.question(
                self,
                "実行中の処理があります",
                "実行中の処理を停止して終了しますか？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return

        for page in self._pages:
            page.shutdown()

        self._save_settings()  # 保存できなくても終了は妨げない
        event.accept()

    def _save_settings(self) -> None:
        try:
            self._settings.save(self._settings_path)
        except OSError:
            pass


def run_gui(argv: list[str] | None = None) -> int:
    """Qt アプリケーションを起動する。"""
    from app.core import qt

    qt.assert_lgpl_backend()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(list(argv or []))
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)

    # 設定ディレクトリは初回起動時に無い。作っておく。
    user_config_dir(APP_NAME).mkdir(parents=True, exist_ok=True)

    window = MainWindow()
    window.show()
    return app.exec()
