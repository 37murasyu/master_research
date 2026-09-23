# Codex への引き継ぎ（2026-09-23 夜）

車椅子の座位プッシュアップの関節トルクを、ステレオの姿勢推定から計算する研究コード。ここまで Claude Code が進めた作業を引き継ぐ。
**最初にこのファイルを最後まで読み、次に `KNOWN_ISSUES.md` の冒頭（残タスク・対応状況）と `HANDOFF_TODO.md` を読むこと。**
**直近の状態と次にやることは `HANDOFF_NEXT.md`（2026-09-23 深夜）が新しい。食い違ったらそちらを正とする。**

## 1. いまの状態

- ブランチ `murayama/fix-left-right-dynamics`。HEAD は `a6abf3a`（統合コミット `a29ee51` の後に、別のスレッドが配布版 `1aa6962` とゲージの設計書を足した）。**push していない**
- 実際の計測構成は **Mac の内蔵カメラ＋同じ Wi-Fi の Pixel 7a 1 台**（混成ステレオ、`app/hybrid/`、`python -m app --role hybrid_measure`）。
  USB カメラ 2 台の経路（`master_research_code.py`、Windows の研究室 PC）も残っている
- テスト: `.venv/bin/python -m pytest -q tests` で **740 passed / 1 skipped**。**引数なしの `pytest` は INTERNALERROR になるので必ず `tests` を付ける**
- 統合コミットより後の作業は**すべて未コミット**（下の 3.）

## 2. 守ること

- main / master に直接コミット・push しない。**コミットと push はユーザーが指示したときだけ**。ブランチ名は `murayama/<英小文字ケバブケース>`
- ユーザーに見せる文章・コミットメッセージ・コメント・文書は**日本語**（常体。既存のコミットの書き方 `fix(dynamics): ...` に合わせる）。
  コミットメッセージの末尾に `Co-Authored-By:` などの行は付けない（Claude のセッションの印は Codex では不要）
- バグ修正・機能追加は**テストを先に書いて失敗を確かめてから**直す。テストは `tests/`、docstring に「なぜこのテストがあるか」を書く
- `master_research_code.py` は import できない（import するとカメラを開く）。ロジックは import できるモジュールに置いてテストし、
  配線は AST のテスト（`tests/test_path_consistency.py`・`tests/test_gauge_energy.py` の流儀）で確かめる。変更後は `python -m py_compile master_research_code.py`
- 設定スキーマ `app/core/settings_schema.json` は手で編集しない（`python tools/extract_env_schema.py > app/core/settings_schema.json`）
- 静的検査は `uvx --quiet pyflakes <ファイル>`。`master_research_code.py` には既存の 23 件がある（増やさない）
- **`git add -A` をしない**。同じ作業ツリーで別の作業（配布版）が未コミットのまま置かれている（下の 3.）
- Windows の `MAINCODE` フォルダ（論文当時の証拠）では git pull・checkout・reset をしない。実機用の新しいコードは別フォルダに clone する
- 「判断待ち」（下の 5.）の項目は勝手に決めない

## 3. 未コミットの作業と、コミットのしかた

`git status` に出るものの持ち主。**配布版の作業（別のセッション）には触らない**。

| 持ち主 | ファイル |
|---|---|
| 配布版 | コミット済み（`1aa6962`）。`app/entry.py` と `config.py` に残る変更は Claude の分だけ |
| 不明（触らない） | `.superpowers/` |
| Claude の作業 | 下の表 |

Claude の作業の中身（内容ごとにコミットを分けるとよい）:

| まとまり | ファイル | 中身 |
|---|---|---|
| ゲージ（§6-8） | `gauge_energy.py`、`master_research_code.py`、`tests/test_gauge_energy.py` | 肘のゲージをサイクルごとにリセット。ゲージの値と閾値を HEADLESS でも `gauge_energy_*.csv/.json` に残す |
| カメラ設定の共有 | `app/core/camera_controls.py`、`master_research_code.py`、`tests/test_camera_controls.py` | `_apply_camera_controls` の中身を移しただけ（挙動は同じ） |
| 出力先（§3-3） | `app/core/settings.py`、`app/shell/page_measure.py`、`app/shell/page_analyze.py`、`app/entry.py`（一部）、`config.py`（一部）、`tests/test_output_dir.py` | 計測の CSV を GUI の表示どおり `~/Documents/WheelchairTorque/output_data` に書く（環境変数 `OUTPUT_DIR`） |
| USB の録画と検証 | `tools/record_stereo.py`、`tools/verify_run.py`、`tests/test_record_stereo.py`、`tests/test_verify_run.py`、`docs/usb_stereo_verification.md` | USB 2 台の全フレーム録画、録画の再生（計測に読み込ませる）、出力フォルダの検査 |
| 混成の検証 | `app/hybrid/measurement.py`、`app/runners/hybrid_measure.py`、`app/hybrid/retriangulate.py`、`app/hybrid/calibration_io.py`、`app/runners/network_measure.py`、`tests/test_hybrid_verification.py`、`docs/hybrid_verification.md`（`tools/verify_run.py` も） | 止まった理由を `meta.json` に残す、`check` の混成対応、Pixel の実際の撮影時刻で三角測量し直す `hybrid-raw` |
| 実機の準備（§6-10） | `tools/verify_run.py`（配置と 3D の質の検査）、`tests/test_hybrid_verification.py`、`docs/hybrid_field_run.md`、`mobile/app/src/main/kotlin/.../pose/StageRates.kt`・`PoseAnalyzer.kt`・`camera/CameraSetup.kt`・`MainActivity.kt`、`mobile/app/src/main/res/layout/activity_main.xml`・`values/strings.xml`、`mobile/app/src/test/.../pose/StageCounterTest.kt`、`mobile/README.md` | 実機の計測で 3D が壊れた原因（置き方）を試し計測で見つける検査、Pixel の段階ごとの速さと 30fps 固定、当日の手順 |
| 文書 | `KNOWN_ISSUES.md`、`HANDOFF_TODO.md`、`HANDOFF_CODEX.md`、`docs/superpowers/plans/2026-09-23-remaining-tasks.md`、`docs/windows-transfer-audit-2026-09-23.md`、`tmp_filter_pose_torque.py`（docstring のみ） | 上の反映 |

（2026-09-23 深夜に配布版がコミットされたので、以下の部分だけのステージは不要になった。記録として残す）
**`app/entry.py` と `config.py` は配布版と Claude の変更が同じファイルにあった**。コミットするときは Claude の部分だけをステージする
（`git add -p` は対話式なので、HEAD の内容に Claude の変更だけを足したファイルを作って `git update-index --cacheinfo` で入れるなど）。

- `config.py` の Claude の部分: `save_dir` の 5 行（`_OUTPUT_DIR_ENV = "OUTPUT_DIR"` と `save_dir = os.environ.get(...)`）。
  配布版の部分: `folder_path = os.environ.get("APP_WORKSPACE") or ...` とその上のコメント
- `app/entry.py` の Claude の部分: import 行に足した `OUTPUT_DIR_ENV, measurement_output_dir`、`worker_environment` の docstring の最後の段落と
  `env[OUTPUT_DIR_ENV] = str(measurement_output_dir())`。配布版の部分: `Path`・`workspace`・`user_output_dir`・`APP_NAME` の import、
  `__all__` の `workspace_dir`、`workspace_dir()` と `_enter_workspace()`、`run_worker` の `_enter_workspace()` の呼び出し。
  `WORKER_MODULES` の hybrid の 2 行と `uses_stop_file` は統合コミットに入っている
- ステージの中身を先に確かめる（`git diff --cached`）。`git commit -- <パス>` の形にすると、他人がステージした内容を巻き込まない

## 4. タスク（優先順）

### T1. コミット（ユーザーの指示があれば）
3. のまとまりごとに分けてコミットする。コミットの前に全テスト・`py_compile`・pyflakes。

### T2. Pixel の速さの診断表示と 30 fps の固定 【実装済み・未コミット。実機での確認が残り】

2026-09-23 夜に実装した: `mobile/.../pose/StageRates.kt`（段階ごとの数、`StageCounterTest` 5 件）、`PoseAnalyzer`・`CameraSetup`
（Camera2 の撮影完了を数える、`CONTROL_AE_TARGET_FPS_RANGE` を 30 fps、GPU 推論の選択と推論にかかる時間）、`MainActivity` と画面（「毎秒 カメラ → 解析 → 推論 → 人 → 送信／推論 xx ms」の行と
「30fps固定」「GPU推論」のチェック）。Pixel には 23:45 の版を入れた。`./gradlew :app:testDebugUnitTest`（42 件）と `:app:assembleRelease` は通り、署名は入っているアプリと同じ鍵。
残りは実機での確認（`docs/hybrid_field_run.md` の 4.）。以下は実装前の見立て（参考）。

背景: 実機の Pixel 7a は 10〜15 Hz しか出ない（hybrid ブランチでの実測）。計測中は同期バッファ（`app/net/sync_buffer.py`）が 30 Hz の格子へ
線形補間して組むので、組の Pixel 側の 6〜7 割は補間の点になる。モデルは既に最軽量（`pose_landmarker_lite`）で、公開ベンチマークでは
Pixel 6 級で 1 フレーム 20 ms 前後（要実機確認）なので、失われているのはモデルの外と見ている。有力な順:

1. カメラが 15 fps しか出していない。`mobile/.../camera/CameraSetup.kt` の `applyFixedOptics` は AE をロックするが
   `CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE` を指定していない。暗い室内では露光が 1/15 s に延びて 15 fps になり、ロックでそのまま固まる
2. 人を検出できないフレームは送らない（`pose/PoseAnalyzer.kt` の `handleResult`、`if (poses.isEmpty()) return`）
3. `LIVE_STREAM` モードは推論中に来たフレームを捨てる。解析スレッドで毎フレーム `toBitmap` の複製もしている
4. 推論は CPU（`BaseOptions` に delegate の指定なし）。GPU にすると速くなるが効き目は 1〜3 より小さい見込み
5. 画像を小さくするのは見込みが薄い（MediaPipe は内部で数百 px に縮める）うえ、Pixel の校正のやり直しが要る

やること:
- 段階ごとの 1 秒あたりの数（カメラから届いたフレーム・推論の結果・人が写っていた結果・送信）を数え、計測中の画面
  （`MainActivity.updateStreamingStatus`）に出す。数える部分は Android に依存しない小さなクラスにして JVM のユニットテストを書く
- `applyFixedOptics` で `CONTROL_AE_TARGET_FPS_RANGE` を `Range(30, 30)` にする（映像は暗くなる。照明の注意を `mobile/README.md` に書く）
- 試験: `cd mobile && ./gradlew :app:testDebugUnitTest`。実機での確認はユーザーが行う（Pixel に入れて画面の数を見る）
- 注意: 解析する画像の大きさ（1280×720）は校正と結びついているので変えない。release 鍵・`mobile/local.properties`・`keystore*` は Git 管理外（コピーが要る）

### T3. §6-9 サイクル検出（判断待ち。決まったら）
USB（`utils.PushCycleDetector`、`master_research_code.py`）とスマホ／混成（`app/runners/network_measure.py` の `MeasurementConfig.cycle_*`）は同じ検出器で、
左肩の `y`（重力を体幹から推定すると奥行きになる）を **1 フレームあたり**の速さ（1 cm/フレーム）で見る。間引きなし（30 Hz）では 1 回も検出しない
（受け取った被験者 7 の動画の再生で確認。上下動は `z` に出ていた）。案は「見る軸を推定した重力の上の軸に、速さの閾値を m/s に」。
直すなら `tools/verify_run.py replay` で被験者の動画を再生し、検出の回数と位置を確かめてから。

### T4. 小さな不具合（ユーザーの指示があれば）
- `app/tuning/ekf_estimate.py` の `format_report` は、`fit_series` の `ValueError` をすべて「推定不能（有限値が足りない）」と出す。
  実データでは「どの間隔でも過程の寄与が観測誤差を上回らず、q_acc の初期値を取れない」が多い。理由を持ち回って出す
- `python -m app.runners.tune_ekf <csv> --out <フォルダ>` が `IsADirectoryError` で落ちる（ファイルのパスしか取らない）
- `master_research_code.py` で閾値の計算をモジュール直下に移したので、シェルから `BODY_MASS_KG` に数値でない値を渡すと起動時に落ちる（GUI では起きない。KNOWN_ISSUES に記録済み、未修正）

## 5. 判断待ち（勝手に決めない）

§6-9 サイクル検出、PC カメラの実装の重複（`app.net.local_camera_sender` と `app/hybrid/mac_camera.py`）、§6-1 集計範囲、§1-5 重力の既定（`axis`／`trunk`）、
§1-4・§6-6 論文本文の直し方、§2-6 手首の 1RM の列（`wrist_*`／`wrist_*_inner`）、§6-4 `elbow_*_outer` の元の列、§6-8 ゲージの閾値の式、
§6-1 被験者 8 のカットオフ（今回 2.11 Hz、論文の表 5 は 5.63 Hz）。材料は `KNOWN_ISSUES.md` の各節。

## 6. 実機でしか確かめられないこと（手順書あり）

- Mac＋Pixel（`docs/hybrid_verification.md`）: GUI の停止ボタン → `python -m tools.verify_run check ~/Documents/WheelchairTorque/hybrid/measure --expect-stop`（§3-2）、
  Mac・Pixel それぞれの fps（§6-2。24 fps 未満で不合格にする）、80 秒以上の計測で S6（`hybrid-raw` → `ekf_estimate`／`tune_ekf`。混成は EKF が無いので S9b は対象外）
- USB 2 台（`docs/usb_stereo_verification.md`）: 録画ツール、ライブの 30 fps、GUI の停止、S6 → S9b、`BUILTIN_DEFAULTS` と `EKF_MAX_GAP_S` の既定値
- 受け取った被験者 7 の古い動画での S6 は、設計メモの中止条件（端に張り付く系列が過半数・|ρ1| > 0.3）に当たった。新しい構成で取り直して判断する
- **2026-09-23 の実機の混成計測は置き方で 3D が壊れていた**（基線 36.7 cm、肘での視線の角 8°、Mac で手が画面外。KNOWN_ISSUES §6-10）。
  当日の手順は `docs/hybrid_field_run.md`（先に置き直して校正し直す）。置き直しても飛ぶ組が残るなら、骨の長さで外れ値を除く処理を足すかをユーザーに聞く

## 7. 落とし穴

- `config.py` は import 時に作業フォルダへ `output_data` を作る。`video_io` も config を import する
- `master_research_code.py` のデモ表示（`DEMO_MONO_GAUGE_ON`・`DEMO_MONO_CAM0_ONLY`）は**コードの既定が 1**。1 のままだと力学が回らずトルクが全部 0。GUI は 0 を渡す
- 本体は入力の動画が開けないと、作業フォルダの別の録画（`cam*_output_*`）へ黙って切り替える。`tools/verify_run.py replay` は起動前に入力を確かめている
- 本体は `m_max_part_<被験者>.json`・`app.log`・`cam*_output_*.mp4` を作業フォルダに書く
- 混成の計測フォルダには、`hybrid-raw` が `kpts3d_raw_*` を書き足す。`kpts3d_*` の glob で拾うと取り違える（`tools/verify_run._hybrid_files` は `frames_*` の stamp で名前を決め打ちする）
- 混成の組の速さ（30 Hz）はカメラの速さではない（補間）。30 fps の判定はカメラごとの速さで見る
- 受け取った Windows のデータは `~/Downloads/transfer_20260923_review/extracted/`（`cameras_raw/` に被験者 3〜8 のステレオ動画と校正）。原本は書き換えない
