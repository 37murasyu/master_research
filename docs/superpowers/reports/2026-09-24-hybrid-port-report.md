# 混成経路への USB 経路の機能の移植と被験者ゲージ UI の報告書（2026-09-24）

計画: `docs/superpowers/plans/2026-09-24-hybrid-usb-port.md`（末尾の「状態と引き継ぎ」の「残り」1〜4 をこの報告で閉じる）。
UI 側の計画と台帳: `docs/superpowers/plans/2026-09-24-subject-gauge.md`・`2026-09-24-subject-gauge-ledger.md`。
ブランチ `murayama/fix-left-right-dynamics`。

## 1. 結論

- 混成（Mac＋Pixel）の計測の子は、EKF・体格の検査・力学の関所・回の区切り（`RepDetector`）・dt を直した仕事の積分・論文の閾値
  W_0.70／W_0.85・`@@GAUGE` の行（約 10〜30 Hz、1 行 512 B 未満）まで入っている（ローカルのセッションで済み。計画の「状態と引き継ぎ」）
- 被験者ゲージの UI（protocol・model・scene・widget・window・worker の振り分け・計測画面 §5.2）を**すべて**このブランチへ取り込んだ。
  UI 側で途中だった Task 16（計測画面 P2）はこのセッションで仕上げた
- `/code-review --fix`（UI の取り込み部分）と `/simplify`（今夜の差分 `64590fd..HEAD` 全体）を済ませた。出力の値は変えていない
- 試験: **1411 passed / 3 skipped / 4 failed（Linux の offscreen）**。4 件はどれも環境の差（下の §4）で、取り込み前の同じ環境の基準でも同じ 4 件が落ちる
- 合成の押し上げ 10 回の再生（R3）は、整理の後も引き継ぎ時と同じ値（下の §3）

**Mac でしかできないこと**（§5）: `.app` の作り直し、GUI で Mac＋Pixel を選んで開始しゲージ窓が第 2 モニタに出るかの目視、朝の実機の手順。

## 2. このセッションで行ったこと

| # | 中身 | コミット |
|---|---|---|
| 1 | `origin/murayama/gauge-measure-page`（`murayama/subject-gauge` を含む。UI の Task 13〜15 と Task 16 の wip）を merge。衝突は `app/shell/page_measure.py` だけで UI 側の版を取った（こちらの最小の配線 36841ac は置き換わる）。その試験 `tests/test_page_measure_gauge.py` は、同じ振る舞いを `tests/test_gauge_measure_page.py` が確かめるので消した。UI 側の版にもゲージ窓を開いて `gauge_frame` をつなぐ処理があることを確かめた | d71d238 |
| 2 | Task 16 の残り: 計算を壊す設定（`app_default` を持つ 4 つ）の件数バッジを外と入れ子の開示の見出しに（R20-03）、ログの「未実行」（R19-01）、「J の数値」スイッチの即時保存（`settings_edited` → `MainWindow._save_settings`）。校正の日時と「変更」リンクは wip に入っていた UI 側のセッションも同じ 3 つを並行して `subject-gauge` に入れていた（0dd99ef）ので後から取り込み、実装と試験はそちら（`broken_flag_count`、戻す・起動時・他の設定の試験が厚い）に揃えた | 8ff1070・0dd99ef を merge |
| 3 | `/code-review --fix`（`36841ac..HEAD` の `app/shell`・`app/gauge`・`worker.py`）: 不具合なし。指摘の危険 1 件（`QFormLayout.setRowVisible` は Qt 6.4 から）に対して `PySide6>=6.4` に固定 | cd457df |
| 4 | `/simplify`（下の表） | b884cdf・f284908・d05cdcb・f290c6d・8854e02・8d37c2c |
| 5 | 文書: `docs/hybrid_field_run.md` §0 の手順 1 を新しい計測画面に合わせる、`KNOWN_ISSUES.md` の一覧の §6-8〜6-10 の行、この報告書 | このコミット |

### /simplify で直したこと（値は同じ）

| 観点 | 直したこと |
|---|---|
| 二重の積分 | 回の W_pos を `GaugeTracker.add` と `RepAccumulator` の 2 か所で積んでいた → `RepAccumulator` だけにし、ゲージは `set_now` で写す。合成 5 場面の各フレームの結果と tracker が NaN を含めて一致。毎フレーム gauge now == rep_work pos の試験を足した |
| 受信スレッドの I/O | 生 3D（`kpts3d_raw_`）を 30 Hz の受信スレッドで毎行 flush していた → Recorder の 1 秒ごとの flush にまとめる（USB の経路の既定は毎行のまま） |
| GUI の描画 | `LineDemux.feed` の O(k²) の切り出し、描画ごとの QFont と文字幅の作り直し、変わらないフレームでの描き直しをやめた（デモ 11 場面×3 大きさと窓の描画 55 枚で画素一致） |
| 重複 | 環境変数の読み方（`config.env_flag` に `env` を足し `env_float` を並べて ekf・energy_pipeline・再生・校正・計測で使う）、cam0→実行時の座標変換と `_unit`（gravity の 1 つに）、`_finite`（`protocol.finite_or_none`）、骨の長さの中央値（`_median_segment`）、部位名（`PART_NAMES`）、meta への追記（`update_meta`）、`gauge_view` の行の分割（本番と同じ `LineDemux`） |
| 使われないもの | `gravity_board.board_up_runtime`、`RepDetector` の `open_elapsed_s`・`max_lift_m`・`reps`・`discarded`、EKF の `failures`、誰も色を渡さないレンダラのキャッシュの引数 |
| その他 | `verify_run check` で生 CSV を 1 度だけ読む、Recorder の `getattr` の既定値をやめて `FrameResult` を直接読む、毎フレームの定数を事前に計算 |

**据え置いたもの（整理の範囲を超える、または値が変わる）**

- `gravity_board.up_label` → `push_up_model.nearest_axis`: 同点の決め方が違う（(1,1,0) で X+ と Y+）
- `demo_gauge` の肘角 → `angle_between`: 式（acos と atan2）が違い数値が一致しない
- `pose_detector._env_float`: 空白だけの値で警告する挙動が `env_float` と違う
- `worker` が 1 塊のフレームの最後だけを出す: 試験とデモの `--via-worker` が全フレームを数える
- 設計の見直しとして次の機会に回すもの:
  - 同期の格子（30 Hz・100 ms）を `SyncBuffer` から配る（今は `1/30` が 10 か所近くに直書き）
  - `LandmarkEKF` に公開の作り直し（`GridEkf._reinit` が私的な状態を書く。USB と共有のモジュールなので今回は触らない）
  - 再生を環境変数 `HYBRID_REPLAY` ではなく別の role にする（親のシェルに残った `export` で本番が再生になりうる。
    計測画面の「出力フォルダ」は再生でも `hybrid/measure` を開く）
  - 計測画面の role の文字列の比較を役割の記述にまとめる

## 3. 検証（このコンテナ、Linux・Python 3.11・PySide6 offscreen）

- 全試験 `QT_QPA_PLATFORM=offscreen python -m pytest -q tests`: 1411 passed / 3 skipped / 4 failed、`py_compile master_research_code.py` 通過、
  変えたファイルの pyflakes は `main_window.py` の `QtCore`（今夜より前からある）だけ
- R3（合成の押し上げ 10 回を `hybrid_replay --speed 0`、被験者 00・65 kg）: 終了コード 0、`@@GAUGE` 173 行・最長 331 B、rep 0→10。
  回ごとに now が 0 に戻り prev が直前の回の値。肘 L の W_pos 19.1〜20.1 J（最後 19.7 J）、手首 約 6.9 J、
  帯 肘 L 47.7〜57.2 J・手首 L 9.7〜11.6 J（引き継ぎ時の値と同じ）。**普通の押し上げでは不足に留まる**（研究上の所見で不具合ではない）

## 4. このコンテナで落ちる 4 件（環境の差）

- `tests/test_text_drawing.py` の 3 件: 日本語の文字を PIL の描画と画素で突き合わせる。入っている日本語フォントが macOS と違う
- `tests/test_gauge_widget.py::TestGaugeWidgetPaintEvent::test_paint_event_matches_render_image`: 窓の描画と QImage の描画を画素で
  突き合わせる。差を画像にして見ると、動く層の文字の縁のアンチエイリアスだけ（形・色・人物は一致）。Linux の offscreen の窓の面と
  QImage で文字の縁の塗り方が違う。macOS では通っていた（UI 側の台帳）

どちらも取り込みの前（c78e155）に同じ環境で落ちており、今回の変更によるものではない。

## 5. Mac で行うこと（朝）

1. `git pull` の後、`.venv/bin/python -m pytest -q tests`（macOS で 4 件も通ることの確認）
2. `.app` の作り直し: 今の `dist/WheelchairTorque.app` を別名で残してから `./packaging/build_macos.sh`。
   凍結版で `--role script --module app.gauge.demo --snapshot <絶対パス>` と `--via-worker` を確かめる（UI 側の計画 Task 18）
3. GUI（`python -m app`）の計測画面で「実験者用の詳細設定」を開き、Mac＋Pixel・被験者番号 00・体重を入れて開始 →
   ゲージ窓が作業者の窓と別の画面（無ければ 1280×720 の窓）に出て「接続待ち」になること。被験者なしなら
   `HYBRID_REPLAY=<合成の計測フォルダ> python -m app` で同じ画面が「▶ 再生」で動く
4. 実機の手順は `docs/hybrid_field_run.md` §0 から（最初の 2 秒は静止、盤を立てる校正、終了コード 3 の意味、停止後の `verify_run check`）
5. 実機でしか分からないこと: 置き方、実際の押し上げで回が閉じるか、実際の W_pos が W_0.70 に届く大きさか、GPU の発熱後の fps、
   第 2 モニタの全画面、高 DPI での見た目、.app の許可の取り直し
