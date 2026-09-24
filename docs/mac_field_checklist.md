# Mac での実機検証チェックリスト（プロジェクト全体、2026-09-24 時点）

クラウドや再生・合成では確かめられず、**Mac（と Pixel 7a）の実機でしか分からないこと**を、ここ 1 か所にまとめた。
細かい手順は各項目の「手順」の文書を見る。散らばっていた出どころ: `KNOWN_ISSUES.md`（実機で行う・§3-2・§6-2・§6-3・§6-10）、
`docs/hybrid_field_run.md`、`docs/hybrid_verification.md`、`docs/usb_stereo_verification.md`、`mobile/README.md`、
`docs/phone_pc_capture.md`、`docs/hybrid-stereo-todo.md`、`HANDOFF_*.md`、ゲージ UI の台帳（Task 7・8・18）、
`docs/superpowers/reports/2026-09-24-hybrid-port-report.md` §5。

以下の `python` はリポジトリ直下の `.venv/bin/python`。上から順に行うと、前の段の結果を後の段で使える。

---

## 0. 準備（Mac の上で、実機の前に）

- [ ] `git pull`（ブランチ `murayama/fix-left-right-dynamics`）→ `python -m pytest -q tests` が**全部**通る
      （クラウドの Linux では文字の画素比較の 4 件が環境の差で落ちる。macOS で通ることを確かめる）
- [ ] `python -m pip install -r requirements_min.txt`（`PySide6>=6.4` に上げた）
- [ ] Pixel に release 版を入れる: `cd mobile && ./install.sh`（段階ごとの速さの表示・「30fps固定」「GPU推論」）
- [ ] 被写体なしの練習: GUI の計測画面で入力「記録の再生」→「選ぶ…」で合成の記録を選んで開始
      （合成は `python -m tools.synth_session --out ~/Documents/WheelchairTorque/hybrid/synth --reps 10`）
  - [ ] フォルダ選びのダイアログが開き、選ぶまで「計測を開始」が押せず理由が出る
  - [ ] ゲージ窓が「▶ 再生」の印つきで開き、回が 1 ずつ進み、今回値が回ごとに 0 に戻る
  - [ ] 終わったら「出力フォルダ」が `hybrid/replay` を開く

## 1. アプリと画面（被験者ゲージ）

| 確かめること | 合格 | 出どころ |
|---|---|---|
| `.app` の作り直し（今の `dist/WheelchairTorque.app` は別名で残す）→ `./packaging/build_macos.sh` | ビルドが通り起動する | 報告書 §5・UI Task 18 |
| 凍結版で `--role script --module app.gauge.demo --snapshot <絶対パス>` | PNG で人物に切れ目なし・文字が重ならない・✓ ✕ と日本語が出る | UI Task 18 |
| 凍結版で `--via-worker` | frames > 0、ゲージの行がログに 0 件、exit 0 | UI Task 18 |
| 入力「Mac＋Pixel」で開始 → ゲージ窓が**第 2 モニタに全画面**（無ければ 1280×720 の窓）で「接続待ち」 | 作業者の窓と別の画面に出る | UI 台帳 Task 8 ⚠️（offscreen では通らない経路） |
| **高 DPI（Retina）**でのゲージの見た目 | 縁・文字がにじまず、静止層と動く層がずれない | UI 台帳 Task 7 ⚠️ |
| J スイッチを切り替え → アプリを閉じずに `settings.json` を見る | その場で保存されている | 設計書 §5.2 |
| 「実験者用の詳細設定」の校正の日時・「変更」リンク・赤い件数バッジ | 日時が最新の校正、「変更」でキャリブレーション画面へ | 設計書 §5.2 |
| `.app` で初回のカメラ・ローカルネットワークの許可（「受信接続を許可」） | 許可の後に Pixel がつながる | 報告書 §5・`hybrid-stereo-todo.md` 段階 2 |

## 2. 置き方と校正（混成、いちばん大事）

手順: `docs/hybrid_field_run.md` §2・§3。2026-09-23 の計測は置き方（基線 36.7 cm・肘での視線の角 8°・Mac で手が画面外）で 3D が壊れた（§6-10）。

- [ ] Mac のカメラ番号（Camo が入っていると 0 番が Camo、内蔵は 1 番）
- [ ] **天板を鉛直に**（重力の `axis` モードは cam0 が水平の前提）
- [ ] 校正: RMS ≤ 1 px、マス寸法の誤差 ≤ 1 mm、**基線が巻尺と ±5%**（`mobile/README.md`）
- [ ] 校正の最後の「盤を立てて静止」（§3b）: 傾き 10° 以内、`verify_run check` の「[重力]」が `checkerboard`
- [ ] Mac のカメラ設定（13）: 起動時の 1 行で露出・WB・焦点を設定できたか（OpenCV の AVFoundation では多くが効かない見込み）と**実測 fps** を記録

## 3. 試し計測（20〜30 秒）→ その場で判定

手順: `docs/hybrid_field_run.md` §4。判定は `python -m tools.verify_run check ~/Documents/WheelchairTorque/hybrid/measure --expect-stop`。

| 確かめること | 合格 | 出どころ |
|---|---|---|
| 配置: 肘での視線のなす角 | 15° 以上 | §6-10 |
| 配置: 肘・手首が両カメラの画面内 | 95% 以上 | §6-10 |
| 3D: 長さが妥当な範囲／前腕のばらつき | 95% 以上／1.5 cm 未満 | §6-10 |
| 速さ: Mac・Pixel | どちらも 24 fps 以上（目安 Mac ≥ 25、Pixel ≥ 20、組 ≥ 20/秒、平均位相差 < 20 ms） | §6-2・`mobile/README.md` |
| Pixel の段階の行（カメラ → 解析 → 推論 → 人 → 送信） | 減っている段が無い。「30fps固定」「GPU推論」オン（09-24 に GPU で 29〜30 を確認） | `mobile/README.md` |
| 終了コード 3 で止まらない | 先頭の窓の肩–肘が人体の範囲（最初の 2 秒は静止） | 体格の検査 |

**配置が合格するまで本計測に入らない。**

## 4. 本計測（90 秒以上）

手順: `docs/hybrid_field_run.md` §0・§5。被験者番号 00・体重を詳細設定に入れ、最初の 2 秒は静止、押し上げ 5 回以上。

| 確かめること | 合格 | 出どころ |
|---|---|---|
| **§3-2 GUI の停止ボタン** | 停止から 2 秒以内に終わり、`check` の「記録を正しく閉じた」「停止要求で止まった」が合格 | §3-2 |
| **§6-2 ライブの 30 fps** | `check` の処理間隔。組が抜けた時間 | §6-2 |
| §6-2 トルクの大きさ | 手首・肘とも 10〜40 N·m 台（09-23 は 100 万 N·m） | §6-2 |
| 肩幅・前腕長 | 肩幅が巻尺 ±2 cm、押し中の前腕長の標準偏差 < 1.5 cm | `mobile/README.md` |
| ゲージが動作に呼応する | 持ち上げで弧が伸び、座ると回が進み、今回値が 0 に戻り前回の目盛りが残る | 計画の目標 |
| 回が閉じる | 押し上げ 1 回ごとに回が 1 増える（持ち上げ 3 cm 未満は捨てる） | §6-9（混成） |
| W_pos が W_0.70 に届くか | **届かなくても不具合ではない**（合成では肘 約 20 J／帯 47〜57 J、手首 約 7 J／帯 9.7〜11.6 J）。実測の値を記録 | 報告書 §3 |
| 押し上げの開始から今回値が増えるまで | 0.3 s 以内 | 計画 R3 |
| 長く使ったとき | Pixel の発熱の後の fps、Mac の処理時間 | 報告書 §5 |

## 5. EKF の較正（§6-3 の S6 → S9b）

手順: `docs/hybrid_field_run.md` §0 の 7・§5 の 3、`docs/hybrid_verification.md` §2。

- [ ] S6: 4 の本計測（90 秒以上）の記録器の `kpts3d_raw_<stamp>.csv`（EKF の手前の 1/30 s の格子。`_retri` の付かないもの）から、
      解析ページ「EKF の較正プロファイルを作る」（または `python -m app.runners.tune_ekf <計測フォルダ>/kpts3d_raw_<stamp>.csv`）で
      プロファイルを作る。`~/Documents/WheelchairTorque/hybrid/ekf_profiles/ekf_profile_0.03333.json` に書かれる
- [ ] 比べる用（任意）: `verify_run hybrid-raw`（実際の撮影時刻で三角測量し直した `kpts3d_raw_<stamp>_retri.csv`）と `--hz 4`
      （4 Hz 相当の `_retri_s3.csv`。名前の s は画面に出る）を `python -m app.tuning.ekf_estimate` にかけ、格子の推定と比べる。
      これらを `tune_ekf` にかけても、プロファイルは CSV の隣に書かれ、実行時の置き場には入らない
- [ ] 中止条件に当たらないか: 探索範囲の端に張り付く系列が過半数、または |ρ1| > 0.3（被験者 7 の古い映像では当たった。
      新しい構成でも当たるなら設計を見直す）
- [ ] S9b: 設定 `HYBRID_EKF_PROFILE` にプロファイルのフォルダを入れてもう 1 試技 → `verify_run check` の「§6-3 EKF」で雑音の出どころが
      `profile`、RMS 差と棄却率が期待の範囲
- [ ] その後で決める: 同梱の既定値（今は q=0.122、r=2.59e-5）の差し替え、`EKF_MAX_GAP_S` の既定値（今は 0＝無制限。欠測の長さの分布で決める）

## 6. 任意（時間があれば）

| 確かめること | 合格 | 出どころ |
|---|---|---|
| `HYBRID_POSE_ROI=1` で ROI（IMAGE モード） | 起動の 1 行が「IMAGE モード … ROI あり」。fps と 3D の質を ROI なしと比べて記録（既定は変えない） | KNOWN_ISSUES（Mac の姿勢推定 14） |
| Pixel の表示の負荷 | 表示 ≥ 3 Hz、表示 0 Hz と 4 Hz でランドマークの fps の低下 ≤ 10% | `mobile/README.md` |
| 部屋の明るさ | 「30fps固定」オン／オフでカメラと人の段の数を比べて記録 | `hybrid_field_run.md` §4 |
| Pixel の自動再接続 | 切断の後、3 秒ごとのつなぎ直しで戻る | `KNOWN_ISSUES`（混成の申し送り） |
| USB カメラ 2 台の経路（2 台目のカメラがあるとき） | 録画ツール `tools/record_stereo.py`、GUI の停止、ライブ 30 fps（`CAM_WIDTH=1280 CAM_HEIGHT=720`） | `docs/usb_stereo_verification.md` |

## 7. 記録すること

`docs/hybrid_field_run.md` §6 の記録シートに加えて: 各段の `check` の出力、Pixel の段階の行、ゲージの画面（スクリーンショット）、
実際の W_pos と帯、ゲージ窓の出た画面、`.app` の許可で困ったこと。結果は `KNOWN_ISSUES.md` の §3-2・§6-2・§6-3・§6-10 に書き戻す。

## 実機でも決まらないこと（研究上の判断）

§6-9 USB 経路のサイクル検出の直し方、§6-8 USB 経路のゲージの閾値の式、§2-6 手首の 1RM の列、§6-4 `elbow_*_outer` の元の列。
