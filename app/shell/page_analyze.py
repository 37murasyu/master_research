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

import os
import shlex
from dataclasses import dataclass

from app.core.qt import QtWidgets
from app.core.settings import Settings, measurement_output_dir
from app.shell.widgets import RunnerPage

__all__ = ["AnalyzePage", "split_options"]


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
    # 必須の位置引数（計測の ID）の欄に出す例。None なら欄を出さない。値は引数の先頭に置く
    id_hint: str | None = None


def split_options(text: str, windows: bool | None = None) -> list[str]:
    """「追加オプション」の文字列を引数に分ける。引用符で囲めば、空白を含むパスも 1 つの引数になる。

    空白で分けるだけ（``str.split``）だと、空白を含むパスが壊れていた。閉じていない引用符は ValueError。
    Windows（``windows`` の既定は ``os.name == "nt"``）では ``\\`` を逃がし文字にしない。パスの ``\\`` が消えるため。
    """
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    if os.name == "nt" if windows is None else windows:
        lexer.escape = ""
    return list(lexer)


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
        # 必須の位置引数。渡さないと argparse が必ず exit 2 で終わっていた
        id_hint="例: 0924_095256（kpts3d_<ID>.csv と aim_torque_vec_<ID>.csv の <ID>）",
    ),
    AnalysisTask(
        label="動画から姿勢を抽出",
        module="video_pose_extractor",
        description="単一カメラの動画から MediaPipe で姿勢ランドマークを抽出する。",
        input_kind="dir",
        input_option="--input-dir",
    ),
    # EKF の自己チューニング（設計メモ 実装 5、S11）。計測が書き出す生 CSV から、系列ごとの
    # (q_acc, r, gate_std) を最尤推定する。USB の収録は隣に書いて設定の EKF_PROFILE へ、混成の記録器の
    # 収録は hybrid/ekf_profiles/ に書いて設定の HYBRID_EKF_PROFILE へ（app.runners.tune_ekf が振り分ける）。
    AnalysisTask(
        label="EKF の較正プロファイルを作る",
        module="app.runners.tune_ekf",
        description=(
            "計測が書き出した kpts3d_raw_*.csv（EKF の手前の 3D 座標）から、ランドマークの"
            "平滑化の雑音パラメータを推定し、ekf_profile_*.json を書く。USB の収録は収録の隣に書き、"
            "設定の EKF_PROFILE に指定する。Mac＋Pixel の収録は hybrid/ekf_profiles/ に書き、"
            "設定の HYBRID_EKF_PROFILE にそのフォルダを入れる。どちらも次の計測から使われる。"
        ),
        input_kind="file",
        input_option=None,
    ),
)


class AnalyzePage(RunnerPage):
    TITLE = "収録データの解析"
    LOG_LABEL = "解析ログ"
    SPLIT_SIZES = (380, 620)

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None):
        super().__init__(settings, "script", parent)
        self._on_task_changed(0)

    # -- 骨格への差し込み --------------------------------------------------
    def widgets_disabled_while_running(self) -> list[QtWidgets.QWidget]:
        return [self._run_button, self._task_combo]

    def widgets_enabled_while_running(self) -> list[QtWidgets.QWidget]:
        return [self._stop_button]

    def start_widgets(self) -> list[QtWidgets.QWidget]:
        return [self._run_button]

    def build_side_panel(self) -> QtWidgets.QWidget:
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

        # 必須の位置引数（計測の ID）。要る解析のときだけ出す（_on_task_changed）
        self._id_label = QtWidgets.QLabel("ID")
        input_layout.addWidget(self._id_label)
        self._id_edit = QtWidgets.QLineEdit()
        input_layout.addWidget(self._id_edit)

        input_layout.addWidget(QtWidgets.QLabel("追加オプション（任意）"))
        self._extra_edit = QtWidgets.QLineEdit()
        self._extra_edit.setPlaceholderText('例: --fps 30 --out "空白 を含む/パス.csv"')
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
            f"計測の出力先（入力の既定の場所）: {measurement_output_dir()}\n"
            "※ 解析の計算内容は既存スクリプトのままです。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #6b7280;")
        layout.addWidget(note)

        layout.addStretch(1)
        return panel

    # -- 動作 --------------------------------------------------------------
    @property
    def _current_task(self) -> AnalysisTask:
        return TASKS[max(0, self._task_combo.currentIndex())]

    def _on_task_changed(self, _index: int) -> None:
        task = self._current_task
        self._task_description.setText(task.description)
        self._id_label.setVisible(task.id_hint is not None)
        self._id_edit.setVisible(task.id_hint is not None)
        self._id_edit.setPlaceholderText(task.id_hint or "")

    def _browse(self) -> None:
        task = self._current_task
        start = self._input_edit.text() or str(measurement_output_dir())
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
            self.append_log("[エラー] 入力を選択してください。\n")
            return

        args: list[str] = []
        if task.id_hint is not None:
            run_id = self._id_edit.text().strip()
            if not run_id:
                self.append_log("[エラー] ID を入力してください。\n")
                return
            args.append(run_id)
        if task.input_option:
            args += [task.input_option, target]
        else:
            args.append(target)
        try:
            args += split_options(self._extra_edit.text())
        except ValueError as exc:
            self.append_log(f"[エラー] 追加オプションの引用符が閉じていません（{exc}）。\n")
            return

        self._runner.start(self._settings, args, module=task.module)
