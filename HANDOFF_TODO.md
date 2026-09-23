# 引き継ぎ: 今後の TODO（2026-09-23 時点）

Copilot などの AI アシスタントに続きを頼むためのメモ。詳細の正本は `KNOWN_ISSUES.md`（冒頭の「残タスク」と「対応状況」）。

## いまの状態

- ブランチ `murayama/fix-left-right-dynamics`。今回の追加修正は未コミット。2026-09-23 に 8 コミット（`4f2e722`〜`cd94230`）を積んだ。**push していない**
- テスト: `.venv/bin/python -m pytest -q tests` で **506 passed / 1 skipped**（引き継ぎ追加対応後。先行対応時は482 passed）。
  **リポジトリ直下で引数なしの `pytest` は無関係なスクリプトを拾って INTERNALERROR になるので、必ず `tests` を指定する**
- 力学モデルは `push_up_model.py` に集約済み。オフライン（`compute_torque_from_pose.py`）・USB（`master_research_code.py`）・
  スマホ（`app/runners/network_measure.py`）の 3 経路がこれを呼ぶ
- 計画と経過: `docs/superpowers/plans/2026-09-23-remaining-tasks.md`。EKF の設計: `docs/superpowers/specs/2026-09-08-ekf-self-tuning-design.md`

## 作業の決まり

- main / master に直接コミット・push しない。作業は `murayama/<内容>` ブランチで。コミット・push はユーザーの指示があるときだけ
- コミットメッセージ・コメント・文書は日本語（既存のコミットの書き方に合わせる: `fix(dynamics): ...`）
- バグ修正と機能追加はテストを先に書いて失敗を確かめてから直す。テストは `tests/`、docstring に「なぜこのテストがあるか」を書く
- `master_research_code.py` は import できない（カメラを開く）。ロジックは import できるモジュールに置いてテストし、
  配線は AST のテスト（`tests/test_path_consistency.py` など）で確かめる。変更後は `python -m py_compile master_research_code.py`
- 設定スキーマ `app/core/settings_schema.json` は手で編集せず `python tools/extract_env_schema.py > app/core/settings_schema.json` で再生成
- 静的検査は `uvx --quiet pyflakes <ファイル>`（venv にリンターは無い）

## 1. ユーザーの判断待ち（勝手に決めない）

| 項目 | 決めること | 材料 |
|---|---|---|
| §2-6 手首の分母 | ダンベルのてこの腕が 0（論文 53 ページの定義どおり）のままか、手の重心の距離などを与えるか | 今は手首のスコアが 2〜30。手の重心なら 0.09〜2.5（KNOWN_ISSUES §2-6 の表） |
| §6-4 1RM | `elbow_*_outer` をスプレッドシートのどの列から取るか | 受領outer列で仮再計算済み。元シート列との対応は未確定 |
| §6-1 集計範囲 | 3_1・4_0・5_stereo を入れるか、5_1（データの質が怪しい）の扱い、右側だけを報告するか | 論文は右側・7 名 |
| §1-5 重力（ユーザー回答: 水平設置は不明。現行設定を保留） | 「最も近い座標軸」（既定 `--gravity-mode axis`）か「体幹の向きそのもの」（`trunk`）か | 手首・肘で最大 2 倍違う。撮影時にカメラが水平だったか |
| §6-1 前処理 | 論文の2 Hzフィルタを使うか、`filter_pose3d.py` を使うか | スコアが10〜30%変わるため保留 |
| §1-4・§6-6 本文 | 本文の記述と計算の食い違い（I_xx、慣性項を「無視した」、GCVSPL、図の分母）をどう直すか | 論文 `~/master/村山_260831_final.pdf` |

判断が出たら、該当するコードを直してスコアを再計算する（手順は KNOWN_ISSUES §6-1）。

## 2. Windowsデータ受領済み（2026-09-23）

詳細は [転送データの照合](docs/windows-transfer-audit-2026-09-23.md)。
原本 `~/Downloads/transfer_20260923.zip`、展開先 `~/Downloads/transfer_20260923_review/extracted/`。
一覧361件のパス・サイズと全CRCは一致。送信側の最終SHAは確認できないとのユーザー回答を受け、再作成・再送は求めず受領版を解析の基準とする。提示SHAとの不一致は記録に残すが、作業の待ち条件にはしない。

- §6-6: 論文0.99と初回サイクル除外後0.88をCSVから再現。分母14件は旧ゲージ式と一致。
  同梱説明の16.73の式は別のLPF系列との混同。GCVSPL系列の分子・図の最終生成版の追跡は残る。
- §1-7: 旧torque_wristbaseの8試技を当時のコード・倍率0.01で再現。他の中間系列へはまだ一般化しない。
- §6-4: mergedのouter列を回収し、同じ7試技を現行モデルで仮再計算。
  結果は `~/Downloads/transfer_20260923_review/current_provisional/` とKNOWN_ISSUES §6-1。
  手首分母・重力・集計条件は保留、元シートの列対応は未確定。
- 手の点を含む処理済み姿勢・NPYは無し。3・4・5・6・7・8の元動画18本は受領。
  手の点を再推定する場合は校正・単位・追跡・歪み補正の確認が先。今回のmetaも全件 `hand=0`。
- §3-1: Windows側は7件を隔離したと報告。移動前一覧のみ受領しており、移動後一覧とREADMEは未確認。
- Windows当時のHEADは `2fb8ff6`。論文時点の証拠として元データ・コードを保存し、実機用の新コードは別フォルダを使う。

## 3. 実機で行う（まとめて 1 回）

実機確認をする PC に新しいコードが要る（Windows なら push して `MAINCODE` とは別のフォルダに clone）。

1. §6-2 USB・スマホ経路でゲージの値が妥当か（手首・肘 10〜40 N·m 台の見込み）、30 fps を保つか。
   起動ログの `[GRAVITY] 体幹から推定: up=Z+`（カメラが水平なら）を確認
2. §3-2 GUI の停止ボタンで `kpts3d_*.csv`・`aim_torque_vec_*_s2*.csv` などが書かれるか
3. §6-3 EKF の S6: 間引きなしと 4 Hz 間引きの 2 設定で収録 → 解析ページの「EKF の較正プロファイルを作る」→
   S9b（RMS 差と棄却率の受け入れ判定）、`app/tuning/ekf_profile.py` の `BUILTIN_DEFAULTS` と
   `EKF_MAX_GAP_S` の既定値（今は 0 = 無制限）を実測で決める

## 4. 追加対応済み（2026-09-23、未コミット）

- 廃止済み `py_native_overlay.py` と `native_overlay/` の追跡ファイル、および専用の旧手動テストを削除。
- §4-4: 真偽値の残り7か所＋ `E_FC_ADAPTIVE_ON` を `config.env_flag` に統一。スキーマ再生成。
- §6-5: 範囲外だけのサイクル検出で落ちないよう修正。旧トルク推定CLIの出力APIを修正。
  旧CLIのモデルは互換用。論文再計算には `compute_torque_from_pose.py` を使う。
- `EFFECTIVE_MASS_COEFFS` の `upper_limb` を `torso` に改名。数値と分母は維持。
- ノイズ評価で被験者4を固定除外しない。必要なら `--exclude-subjects 4` を付ける。
  被験者4のサイクル範囲は未確定で、8名の再集計はまだ行っていない。
- スマホ1台＋PC内蔵カメラ用に `app.net.local_camera_sender` を追加。
  模擬入力で同期を検証。手順: `docs/phone_pc_capture.md`。

## 5. 残る調査・統合

- 上腕質量比0.0227は初期コミットにも存在。そこに記載された参考URLのコードは0.027で、
  0.0227の出典は確定できなかった（KNOWN_ISSUES §5-5）。数値は変更していない。
- スマホ経路の実機校正・同期遅延・30 fps、およびGUI／保存との接続（KNOWN_ISSUES §6-7）。
  追加した送信器と既存サーバの単体起動は入力診断用で、トルクの保存やゲージ表示はしない。
