"""解析画面。収録済みデータに対してオフライン解析を実行する。

解析スクリプトはどれも argparse の CLI で、多くが matplotlib で描画する。
描画も GUI 操作なので、キャリブレーションと同じ理由で**子プロセス**で動かす。
対象が多いため個別の役割は作らず、``--role script --module <名前>`` の
汎用経路を使う。

**計算の中身には手を入れていない**。KNOWN_ISSUES.md の §1（角速度の30倍、
unwrap 漏れ、慣性テンソルが負）は論文の数値に直結する研究上の判断であり、
今回のスコープ外。ここは既存スクリプトをそのまま呼ぶ薄い層に留める。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.platform_compat import user_output_dir
from app.core.qt import QtCore, QtWidgets
from app.core.settings import APP_NAME, Settings
from app.runners.worker import WorkerRunner
from app.shell.widgets import LogView, StatusBadge

__all__ = ["AnalyzePage"]


@dataclass(frozen=True)
class AnalysisTask:
    """画面に並べる解析の 1 項目。"""

    label: str
    module: str
    description: str
    # 入力として選ぶのがフォルダかファイルか
    input_kind: str = "dir"
    # 入力パスを渡すときのオプション名。None なら位置引数として渡す。
    input_option: str | None = None


# README_pose_workflow.md に記載のワークフローから、主要なものを拾ってある。
TASKS: tuple[AnalysisTask, ...] = (
    AnalysisTask(
        label="ステレオ再構成（動画 → 3D姿勢）",
        module="stereo_triangulate_pose",
        description="キャリブ済み2カメラの動画から3D関節位置を復元する。",
        input_kind="dir",
        input_option="--input-dir",
    ),
    AnalysisTask(
        label="3D姿勢をCSVに書き出す",
        module="stereo_reconstruct_to_csv",
        description="再構成した3D姿勢を CSV 形式で保存する。",
        input_kind="dir",
        input_option="--input-dir",
    ),
    AnalysisTask(
        label="姿勢からトルクを計算",
        module="compute_torque_from_pose",
        description="3D姿勢の時系列から逆動力学で関節トルクを求める。",
        input_kind="file",
        input_option="--pose",
    ),
    AnalysisTask(
        label="局所トルクの再計算",
        module="compute_local_torque_offline",
        description="計測時に保存されたグローバルトルクをリンク座標系に変換し直す。",
        input_kind="dir",
        input_option="--base-dir",
    ),
    AnalysisTask(
        label="動画から姿勢を抽出",
        module="video_pose_extractor",
        description="単一カメラの動画から MediaPipe で姿勢ランドマークを抽出する。",
        input_kind="dir",
        input_option="--input-dir",
    ),
)


class AnalyzePage(QtWidgets.QWidget):
    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._runner = WorkerRunner("script", self)

        self._runner.output.connect(lambda text: self._log.append_text(text))
        self._runner.state_changed.connect(self._on_state)

        self._build_ui()
        self._on_state("stopped")
        self._on_task_changed(0)

    # -- 画面 --------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("収録データの解析")
        font = title.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch(1)
        self._badge = StatusBadge()
        header.addWidget(self._badge)
        outer.addLayout(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setSizes([380, 620])
        outer.addWidget(splitter, 1)

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        box = QtWidgets.QGroupBox("解析の種類")
        box_layout = QtWidgets.QVBoxLayout(box)

        self._task_combo = QtWidgets.QComboBox()
        for task in TASKS:
            self._task_combo.addItem(task.label)
        self._task_combo.currentIndexChanged.connect(self._on_task_changed)
        box_layout.addWidget(self._task_combo)

        self._task_description = QtWidgets.QLabel()
        self._task_description.setWordWrap(True)
        self._task_description.setStyleSheet("color: #6b7280;")
        box_layout.addWidget(self._task_description)
        layout.addWidget(box)

        input_box = QtWidgets.QGroupBox("入力")
        input_layout = QtWidgets.QVBoxLayout(input_box)
        row = QtWidgets.QHBoxLayout()
        self._input_edit = QtWidgets.QLineEdit()
        self._input_edit.setPlaceholderText("解析するフォルダまたはファイル")
        row.addWidget(self._input_edit, 1)
        browse = QtWidgets.QPushButton("選択…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        input_layout.addLayout(row)

        input_layout.addWidget(QtWidgets.QLabel("追加オプション（任意）"))
        self._extra_edit = QtWidgets.QLineEdit()
        self._extra_edit.setPlaceholderText("例: --fps 30 --verbose")
        input_layout.addWidget(self._extra_edit)
        layout.addWidget(input_box)

        run_row = QtWidgets.QHBoxLayout()
        self._run_button = QtWidgets.QPushButton("解析を実行")
        self._run_button.clicked.connect(self._run)
        run_row.addWidget(self._run_button)
        self._stop_button = QtWidgets.QPushButton("中止")
        self._stop_button.clicked.connect(self._runner.stop)
        run_row.addWidget(self._stop_button)
        layout.addLayout(run_row)

        note = QtWidgets.QLabel(
            f"出力先: {user_output_dir(APP_NAME)}\n"
            "※ 解析の計算内容は既存スクリプトのままです。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b7280;")
        layout.addWidget(note)

        layout.addStretch(1)
        return panel

    def _build_log_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("解析ログ"))
        self._log = LogView()
        layout.addWidget(self._log, 1)
        return panel

    # -- 動作 --------------------------------------------------------------
    @property
    def _current_task(self) -> AnalysisTask:
        return TASKS[max(0, self._task_combo.currentIndex())]

    def _on_task_changed(self, _index: int) -> None:
        self._task_description.setText(self._current_task.description)

    def _browse(self) -> None:
        task = self._current_task
        start = self._input_edit.text() or str(user_output_dir(APP_NAME))
        if task.input_kind == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "入力ファイルを選択", start)
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "入力フォルダを選択", start)
        if path:
            self._input_edit.setText(path)

    def _run(self) -> None:
        task = self._current_task
        target = self._input_edit.text().strip()
        if not target:
            self._log.append_text("[エラー] 入力を選択してください。\n")
            return

        args: list[str] = []
        if task.input_option:
            args += [task.input_option, target]
        else:
            args.append(target)
        args += self._extra_edit.text().split()

        self._runner.module = task.module
        self._runner.start(self._settings, args)

    def _on_state(self, state: str) -> None:
        self._badge.set_state(state)
        running = state in ("starting", "running")
        self._run_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._task_combo.setEnabled(not running)

    def shutdown(self) -> None:
        self._runner.stop()

    @property
    def is_running(self) -> bool:
        return self._runner.is_running
