# 引き継ぎ: 今後の TODO（2026-09-23 時点）

Copilot などの AI アシスタントに続きを頼むためのメモ。詳細の正本は `KNOWN_ISSUES.md`（冒頭の「残タスク」と「対応状況」）。

## いまの状態

- ブランチ `murayama/fix-left-right-dynamics`。2026-09-23 に 15 コミット（`4f2e722`〜`aa9197f`）を積んだ。**push していない**
- 未コミット: 肘のゲージのリセットとゲージの記録（§6-8）、出力先（§3-3）、USB の録画と検証の道具、
  混成ステレオの検証（止まった理由の記録、`check` の混成対応、`hybrid-raw`）。下の §3
- テスト: `.venv/bin/python -m pytest -q tests` で **740 passed / 1 skipped**（統合した hybrid と、別作業の配布版のテストを含む）。
  **リポジトリ直下で引数なしの `pytest` は無関係なスクリプトを拾って INTERNALERROR になるので、必ず `tests` を指定する**
- 力学モデルは `push_up_model.py` に集約済み。オフライン（`compute_torque_from_pose.py`）・USB（`master_research_code.py`）・
  スマホ（`app/runners/network_measure.py`）の 3 経路がこれを呼ぶ
- 計画と経過: `docs/superpowers/plans/2026-09-23-remaining-tasks.md`。EKF の設計: `docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`
- **同じ作業ツリーで別の作業（配布版: `app/core/workspace.py`、`packaging/`、`app/entry.py`・`config.py` の一部）が未コミットで進んでいる**。
  `git add -A` をしない。`app/entry.py` と `config.py` は配布版と §3-3 の変更が同じファイルにあるので、コミットのときは自分の部分だけステージする
- **`murayama/hybrid-stereo` は統合済み**（統合コミット `a29ee51`、hybrid 側は `4ab12f6` まで）。未コミットの変更は
  `git checkout -m` で統合後に持ち越した。`app/entry.py` は配布版・出力先・hybrid の変更が同じファイルに並ぶ

## 作業の決まり

- main / master に直接コミット・push しない。作業は `murayama/<内容>` ブランチで。コミット・push はユーザーの指示があるときだけ
- コミットメッセージ・コメント・文書は日本語（既存のコミットの書き方に合わせる: `fix(dynamics): ...`）
- バグ修正と機能追加はテストを先に書いて失敗を確かめてから直す。テストは `tests/`、docstring に「なぜこのテストがあるか」を書く
- `master_research_code.py` は import できない（カメラを開く）。ロジックは import できるモジュールに置いてテストし、
  配線は AST のテスト（`tests/test_path_consistency.py`、`tests/test_gauge_energy.py` など）で確かめる。変更後は `python -m py_compile master_research_code.py`
- 設定スキーマ `app/core/settings_schema.json` は手で編集せず `python tools/extract_env_schema.py > app/core/settings_schema.json` で再生成
- 静的検査は `uvx --quiet pyflakes <ファイル>`（venv にリンターは無い）。`master_research_code.py` は既存の 23 件がある

## 1. ユーザーの判断待ち・保留（勝手に決めない）

| 項目 | 決めること | 材料 |
|---|---|---|
| **§6-9 USB 経路のサイクル検出（新規）** | 見る軸を重力の推定結果（上）にし、速さの閾値を秒あたりに直すか | 左肩の y（奥行き）をフレームあたりの速さで見ており、被験者 7 の再生では間引きなし（GUI の既定）で 1 回も検出しない。上下動は z に出る |
| §6-1 集計範囲 | 3_1・4_0・5_stereo を入れるか、5_1（データの質が怪しい）の扱い、右側だけを報告するか | 論文は右側・7 名。最新の表は KNOWN_ISSUES §6-1 |
| §1-5 重力（水平設置は不明。現行設定を保留） | 「最も近い座標軸」（既定 `--gravity-mode axis`）か「体幹の向きそのもの」（`trunk`）か | 手首・肘で最大 2 倍違う |
| §1-4・§6-6 本文 | 本文の記述と計算の食い違い（I_xx、慣性項を「無視した」、GCVSPL、図の分母）をどう直すか | 論文 `~/master/村山_260831_final.pdf` |
| §2-6 手首の 1RM の列 | `wrist_*` か `wrist_*_inner` か（今は `wrist_*`） | `m_max_all_merged.csv` |
| §6-4 outer の出典 | `elbow_*_outer` が元のスプレッドシートのどの列か | 値は受領 CSV から回収済み |
| §6-8 ゲージの閾値の式 | 旧来のゲージの式（手首の r_x = 0.30 m 固定、帯は約 95〜128 J）のままでよいか | 1 サイクルの正の仕事は 1〜12 J 程度で帯に届かない |

2026-09-23 に決めたこと（済み）: §2-6 手首の分母は手の中心（手長 × 0.506）、§6-4 肘の 1RM は outer（伸展の筋力）、
§6-1 前処理のカットオフは f0 から（`pose_lowpass.py`）、§6-8 肘のゲージはサイクルごとにリセット、§3-3 出力を GUI の表示に合わせる。

## 2. Windowsデータ受領済み（2026-09-23）

詳細は [転送データの照合](docs/windows-transfer-audit-2026-09-23.md)。
原本 `~/Downloads/transfer_20260923.zip`、展開先 `~/Downloads/transfer_20260923_review/extracted/`。
受領版を解析の基準とする（送信側の最終 SHA は確認できない）。

- §6-6: 論文0.99と初回サイクル除外後0.88をCSVから再現。分母14件は旧ゲージ式と一致。GCVSPL系列の分子・図の最終生成版の追跡は残る。
- §1-7: 旧torque_wristbaseの8試技を当時のコード・倍率0.01で再現。
- §6-4: mergedのouter列を回収し、リポジトリ直下に `m_max_all_merged.csv` として置いた（`*.csv` は追跡外）。
- 手の点を含む処理済み姿勢・NPYは無し。3・4・5・6・7・8の元動画18本は受領。被験者 7 は USB 経路の再生で骨長が妥当に出た。
- §3-1: Windows側は7件を隔離したと報告。移動後一覧とREADMEは未確認。
- Windows当時のHEADは `2fb8ff6`。論文時点の証拠として元データ・コードを保存し、実機用の新コードは別フォルダを使う。

## 3. 実機で行う（まとめて 1 回）

手順は、実際の構成（Mac＋Pixel 7a）なら **`docs/hybrid_verification.md`**、USB カメラ 2 台なら **`docs/usb_stereo_verification.md`**。
実機確認をする PC に新しいコードが要る（Windows なら push して `MAINCODE` とは別のフォルダに clone）。

- **当日の手順は `docs/hybrid_field_run.md`**。2026-09-23 の実機の計測は置き方（基線 36.7 cm、肘での視線の角 8°、Mac で手が画面外）で
  3D が壊れていた（KNOWN_ISSUES §6-10）。先に置き直し（2 台を離せなければ被写体を近づける）、校正し直し、試し計測の `check` で配置を確かめる。
  Pixel のアプリ（段階ごとの速さの表示と 30fps 固定）は release 版を作ってある: `cd mobile && ./install.sh`
- Mac＋Pixel: GUI の計測ページで入力を Mac＋Pixel にして計測し、停止ボタンで止める →
  `python -m tools.verify_run check ~/Documents/WheelchairTorque/hybrid/measure --expect-stop`。
  S6 は `python -m tools.verify_run hybrid-raw ...`（Pixel の実際の撮影時刻で三角測量し直す。`--hz 4` で 4 Hz 相当）→
  `ekf_estimate` / `tune_ekf`。混成の経路は EKF を使っていないので S9b は対象外。模擬 Pixel での試走は合格
- 実機の Pixel 7a は 10〜15 Hz しか出ない（hybrid ブランチで実測）。「30 fps を保つか」は現状では不合格の見込みで、
  組の Pixel 側の大半は補間。Pixel 側の推定を速くするか、格子を Pixel に合わせるかは未決

- 道具: USB 2 台の全フレームを録画する `python -m tools.record_stereo`、録画を計測に読み込ませる `python -m tools.verify_run replay`、
  出力フォルダを確かめる `python -m tools.verify_run check`
- 受け取った被験者 7 の動画で試走済み（間引きなし・4 Hz・途中停止のどれも構造の検査は合格）。結果は KNOWN_ISSUES §6-2・§6-3

1. §6-2 ライブ計測（GUI の既定）で 30 fps を保つか（`check` の「処理間隔」）。解像度を校正と同じ 1280×720 にする
   （`CAM_WIDTH=1280 CAM_HEIGHT=720` を GUI 起動前の環境変数で）。ゲージの値は `gauge_energy_*`
2. §3-2 GUI の停止ボタン → `python -m tools.verify_run check ~/Documents/WheelchairTorque/output_data --expect-stop`
3. §6-3 新しい構成で 90 秒以上録画 → 2 設定で再生 → `tune_ekf` → S9b。被験者 7 の古い映像では設計メモの中止条件
   （端に張り付く系列が過半数・|ρ1| > 0.3）に当たったので、新しい構成でも同じなら設計を見直す。
   `BUILTIN_DEFAULTS` と `EKF_MAX_GAP_S` の既定値はその後で決める

## 4. 残る調査・統合

- 統合で、PC のカメラを片方にする実装が 2 つ並んだ（`app.net.local_camera_sender` と `app/hybrid/mac_camera.py`）。どちらを残すか決める
- 上腕質量比0.0227は初期コミットにも存在。出典は確定できなかった（KNOWN_ISSUES §5-5）。数値は変更していない。
- スマホ経路の実機校正・同期遅延・30 fps、およびGUI／保存との接続（KNOWN_ISSUES §6-7）。
- 小さなもの: `ekf_estimate` の「推定不能（有限値が足りない）」の文言が実際の理由（初期値が取れない）と違う、
  `tune_ekf --out` にフォルダを渡すと落ちる（KNOWN_ISSUES §6-3）。計測自身の録画は時間が詰まる（§3-4、記録のみ）
