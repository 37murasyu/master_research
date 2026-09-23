# 別スレッドへの引き継ぎ（2026-09-23 深夜）

Mac の内蔵カメラ＋同じ Wi-Fi の Pixel 7a（混成ステレオ）で、座位プッシュアップの関節トルクを計測する研究コード。
前のスレッドでは、検証の道具を作り、実機の記録から壊れていた原因を突き止め、Pixel のアプリを速さの診断ができる版に更新した。
**次は実機で Pixel の速さを測り、置き方を直して本番の確認をする段階。**

最初に読む順: このファイル → `docs/hybrid_field_run.md`（当日の手順）→ `KNOWN_ISSUES.md` の冒頭と §6-10 → `HANDOFF_CODEX.md`（守ること・落とし穴の詳細）。

## 0. 言語

ユーザーに見える文章は、途中の経過報告や道具の前後のつなぎの一文も含めて**すべて日本語**。前のスレッドでは説明の約半分が英語に流れ、
ユーザーに何度も指摘された。`~/.claude/CLAUDE.md` の先頭に規則があり、道具を呼ぶたびに日本語の規則を差し込む hook
（`~/.claude/hooks/japanese-reminder.sh`、PostToolUse）も入れてある。一文でも英語が出たら、次の文から日本語に戻す。

## 1. いまの状態

- ブランチ `murayama/fix-left-right-dynamics`、HEAD `a6abf3a`。**push していない**。コミット・push はユーザーの指示があるときだけ
  - `a29ee51`: `murayama/hybrid-stereo` を統合したコミット（前のスレッド）
  - `1aa6962`（配布版の .app）と `a6abf3a`（被験者が見るゲージの設計書）: 別のスレッドがコミットした
- テスト: `.venv/bin/python -m pytest -q tests` で 740 passed / 1 skipped（必ず `tests` を付ける）。
  スマホは `cd mobile && JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./gradlew :app:testDebugUnitTest --offline` で 44 件
- **Pixel 7a は USB でつながっている**（シリアル `41211JEHN01099`。Wi-Fi のデバッグ接続も同時にあるので、`adb` には必ず `-s 41211JEHN01099` を付ける）。
  23:45 に新しいアプリ（下の 3.）を入れた。設定は「30fps固定」オン、「GPU推論」オフ。今は前回の PC への再接続を待っている
- 実機の記録: 校正 `~/Documents/WheelchairTorque/hybrid/calibration/20260923_215123_212760`、
  計測 `~/Documents/WheelchairTorque/hybrid/measure/20260923_215653_236295`（4.5 分。置き方のせいで 3D が壊れている。下の 4.）

## 2. 未コミットの作業

`git status` に出る変更は**すべて前のスレッドの作業**（`.superpowers/` だけは持ち主が不明なので触らない）。配布版の作業はコミット済みなので、
`app/entry.py` と `config.py` も含めて部分だけをステージする必要はもう無い。まとまりごとに分けてコミットするとよい:

| まとまり | 主なファイル |
|---|---|
| ゲージ（§6-8） | `gauge_energy.py`、`master_research_code.py`、`tests/test_gauge_energy.py` |
| カメラ設定の共有 | `app/core/camera_controls.py`、`master_research_code.py`、`tests/test_camera_controls.py` |
| 出力先（§3-3） | `app/core/settings.py`、`app/entry.py`、`config.py`、`app/shell/page_measure.py`・`page_analyze.py`、`tests/test_output_dir.py` |
| USB の録画と検証 | `tools/record_stereo.py`、`tools/verify_run.py`、`tests/test_record_stereo.py`・`test_verify_run.py`、`docs/usb_stereo_verification.md` |
| 混成の検証（§6-2・§6-3・§3-2） | `app/hybrid/measurement.py`・`retriangulate.py`・`calibration_io.py`、`app/runners/hybrid_measure.py`・`network_measure.py`、`tests/test_hybrid_verification.py`、`docs/hybrid_verification.md` |
| 実機の準備（§6-10） | `docs/hybrid_field_run.md`、`tools/verify_run.py`（配置と 3D の質の検査） |
| Pixel のアプリ | `mobile/.../pose/StageRates.kt`・`PoseAnalyzer.kt`、`camera/CameraSetup.kt`、`MainActivity.kt`、`res/layout/activity_main.xml`・`values/strings.xml`、`mobile/app/src/test/.../pose/`、`mobile/README.md` |
| 文書 | `KNOWN_ISSUES.md`、`HANDOFF_TODO.md`、`HANDOFF_CODEX.md`、`HANDOFF_NEXT.md`、`docs/superpowers/plans/2026-09-23-remaining-tasks.md`、`docs/windows-transfer-audit-2026-09-23.md`、`tmp_filter_pose_torque.py`（docstring のみ） |

`git add -A` はしない（`.superpowers/` を巻き込む）。コミットメッセージは日本語、末尾に `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`。

## 3. Pixel のアプリに入れたもの（23:45 の版）

- 計測中の画面に 1 行: 「毎秒 カメラ … → 解析 … → 推論 … → 人 … → 送信 …／推論 xx ms（CPU か GPU）」。
  前の段より 2 割以上減った最初の段と、カメラが 24 fps 未満なら「カメラが遅い」を示す（`pose/StageRates.kt`）
  - カメラ: Camera2 の撮影完了の数（解析に渡らなかった分も数える）
  - 推論 xx ms: フレームを受け取ってから結果が返るまで（Bitmap への変換を含む）
- 「30fps固定」（既定オン）: 自動露出の目標フレームレートを 30 fps に固定（`CONTROL_AE_TARGET_FPS_RANGE`）。暗い室内でカメラが 15 fps に落ちるのを防ぐ。映像は暗くなる
- 「GPU推論」（既定オフ）: MediaPipe を GPU で動かす。使えなければ CPU に戻り、画面の「（CPU）」で分かる
- どちらも計測中に切り替えると、その場でカメラ（GPU は推定器も）を開き直す
- release 版を同じ鍵で作って入れた（debug 版は端末 ID が変わり、校正を流用できなくなるので使わない）

## 4. 前のスレッドで分かったこと（要点）

- **3D が壊れていた原因は置き方**（KNOWN_ISSUES §6-10）。Mac と Pixel の間が 36.7 cm、被写体まで約 2.5 m で、肘での 2 本の視線のなす角が
  右 7.7°・左 16.4°。右腕の長さが 2 割の組で 0.12〜0.5 m の外（最大 4,734 m）に出て、手首のトルクが最大 100 万 N·m を超えた。
  左右の取り違えでも Pixel 側の補間でもない（確かめた）。Mac の画面の下で手首が切れていた（左 55%、右 74% しか画面内に無い）
- **ユーザーは 2 台を 1 m 以上離せない**。角は「2 台の間 ÷ 被写体までの距離」でほぼ決まるので、**被写体を近づける**（基線 37 cm なら 1.4 m 以内、
  50〜60 cm なら 1.8 m 以内で 15° 以上）。被写体は 2 台の中間の正面に置き、2 台とも被写体へ向ける（`docs/hybrid_field_run.md` の表）
- **Pixel は 14.3 fps**（ユーザーの実測も 10〜15 Hz）。同期バッファ（`app/net/sync_buffer.py`）が 30 Hz の格子へ線形補間するので、組の Pixel 側の
  4 割ほどが補間の点。Pixel の間隔が 100 ms を超えると組が抜ける（271 秒のうち 83 秒）
- 混成の経路には EKF も外れ値の除き方も無く、飛んだ 3D がそのままトルクになる
- サイクル検出（§6-9）は左肩の奥行き方向を 1 フレームあたりの速さで見ており、当てにならない（USB も混成も同じ検出器）

## 5. 次にやること（順番）

### A. Pixel の速さを実機で測る（まずユーザーに了承を取る）

Pixel は PC とつながって計測の状態にならないと数えない。前のスレッドの最後に「Mac 側のプレビューを起動してよいか」を聞いたところで止まっている。

1. ユーザーに、Pixel の画面のロックを外してアプリを前に出し、Pixel の前に座ってもらう（人が写らないと「人」「送信」が 0 になる）
2. Mac で `.venv/bin/python -m app.runners.hybrid_preview --camera 1`（Camo が入っていて 0 番が Camo、1 番が内蔵 FaceTime HD だった）。Pixel は自動でつながる
3. Pixel の画面の段階の行を USB で読む（数秒おきに数回）:
   ```sh
   ~/Library/Android/sdk/platform-tools/adb -s 41211JEHN01099 exec-out uiautomator dump /dev/tty \
     | python3 -c "import re,sys; x=sys.stdin.read(); [print(t) for t in re.findall(r'text=\"([^\"]*)\" resource-id=\"com\.murayama\.wheelchairsensor:id/(?:statusText|detailText)\"', x)]"
   ```
   ロックされていると系統の画面しか読めない（何も出ない）
4. 「30fps固定」オン／オフ × 「GPU推論」オン／オフの 4 通りを、ユーザーに Pixel の画面で切り替えてもらって読む
5. 減っている段で次の手を決める:

| 減っている段 | 次の手 |
|---|---|
| カメラ < 24 | 30fps固定＋照明（入れ済み）。暗くて人の検出が落ちるなら照明を足す |
| 解析 < カメラ、または推論が 33 ms を超える | GPU 推論（入れ済み）。足りなければ、CameraX の回転（`setOutputImageRotationEnabled(true)`）と毎フレームの `toBitmap` の複製をやめ、回転を MediaPipe に渡す（`ImageProcessingOptions`）。座標の向きが変わるので校正との整合を確かめてから |
| 推論 < 解析 | 推論中に来たフレームを捨てている。GPU で推論が速くなれば減る |
| 人 < 推論 | 写り方と照明（ソフトの問題ではない） |
| 送信 < 人 | Wi-Fi（5 GHz） |

結果は `docs/hybrid_field_run.md` の記録シートと KNOWN_ISSUES §6-10 に書く。

### B. 置き直して本番の確認（`docs/hybrid_field_run.md` の 2〜5）

置く → 校正し直す（`python -m app.runners.hybrid_calibrate --camera 1`）→ 試し計測 20〜30 秒 →
`.venv/bin/python -m tools.verify_run check ~/Documents/WheelchairTorque/hybrid/measure --expect-stop` で「配置」「3D」が合格するまで置き直す →
本計測 90 秒以上 → 停止ボタン → 同じ check（§3-2・§6-2）→ `hybrid-raw` と `--hz 4` → `ekf_estimate` / `tune_ekf`（§6-3 の S6）。
check が何を見るかは `docs/hybrid_verification.md`。

### C. コミット（ユーザーの指示があれば）

2. のまとまりごと。前に全テスト・`python -m py_compile master_research_code.py`・`uvx --quiet pyflakes <変えたファイル>`。

## 6. 判断待ち（勝手に決めない）

- 置き直しても 3D が飛ぶ組が残るなら、骨の長さで外れ値を除く処理を混成の経路に足すか
- §6-9 サイクル検出（見る軸を推定した重力の上に、速さの閾値を秒あたりに）
- PC カメラの実装の重複（`app.net.local_camera_sender` と `app/hybrid/mac_camera.py`）
- ほか（§6-1 集計範囲、§1-5 重力の既定、論文本文、§2-6 手首の 1RM の列、§6-4 outer の出典、§6-8 ゲージの閾値の式）は `HANDOFF_CODEX.md` の 5.

## 7. 落とし穴（このスレッドで踏んだもの）

- `mobile/install.sh` は端末が 2 つ見えると止まる（USB と Wi-Fi のデバッグ接続で同じ Pixel が 2 つ出る）。`adb -s 41211JEHN01099 install -r mobile/app/build/outputs/apk/release/app-release.apk` で入れた
- スマホのビルドには Android Studio 同梱の Java が要る（`JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"`）
- 混成の組の速さ（30 Hz）はカメラの速さではない（補間）。30 fps の判定はカメラごとの速さで見る（`check` がそうしている）
- 混成の計測フォルダには `hybrid-raw` が `kpts3d_raw_*` を書き足す。`kpts3d_*` の glob で拾うと取り違える
- 本体（`master_research_code.py`）の注意（デモ表示の既定が 1、入力が開けないと別の録画へ黙って切り替える、など）は `HANDOFF_CODEX.md` の 7.
