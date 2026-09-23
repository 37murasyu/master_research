# 残タスク一括対応 実装計画（2026-09-23）

> 実行者向け: この計画はセッション内で順に実行した（ユーザー指示「残りも全部進めて」）。
> 各タスクは TDD（失敗するテスト → 実装 → 全体テスト）で進める。

**Goal:** KNOWN_ISSUES.md の残タスクのうち、実機なしで進められるものをすべて片付け、スコアを通しで再計算する。

**Architecture:** 3 経路（オフライン・USB・スマホ）で別々に書かれていた座位プッシュアップの力学モデルを
`push_up_model.py` に集約し、各経路はそれを呼ぶ。重力・関節の基準点・局所軸・体幹荷重・慣性テンソルの
約束はそこで 1 回だけ決める。

**Tech Stack:** Python 3.12（.venv）、NumPy、pandas、pytest、PySide6（GUI）

**Spec:** `KNOWN_ISSUES.md`（2026-09-13 版）の「残タスク」と、2026-09-23 のユーザー決定。

## ユーザー決定（2026-09-23）

| 項目 | 決定 |
|---|---|
| §1-5 重力 | 初期段階の体幹の向きから推定する |
| §2-2 等価質量 | 分母に手を含める |
| §5-7 関節ラベル | `wrist_R` のトルクを右手首まわりにする |
| §5-4 局所座標系を作れないフレーム | 警告する（値は今のまま） |
| H-A 体幹の質量比 | 今の比（Winter、60 kg で 34.68 kg）のまま |
| 残り | 全部進める |

## Global Constraints

- main に直接コミットしない。コミットはユーザーの指示があるときだけ（`~/.claude/CLAUDE.md`）
- `master_research_code.py` は import できないので、ロジックは import できるモジュールに置いてテストする
- 既存の出力ファイルは書き換えない（規約の版はタグ無し＝v1）
- 実機が要るもの（§6-2、§3-2 の確認、EKF S6）は実施しない。実機で確かめる手順を残す

## Review Focus

1. カメラ座標（y 下向き）と z 上向きの座標系が混在する入力 → どちらでも重力が鉛直下向きになること
2. 手の点が無い／手首がまっすぐに近いフレーム → 手首の局所軸が肘の屈曲軸に落ちること
3. 局所座標系を作れないフレーム → 値は今のまま・警告が出る・件数が要約に出ること
4. `--torque-scale` の既定 0.01 で m 入力のトルクが 1/100 になる罠 → 既定 1.0
5. 左右の鏡映で τ_y と仕事率が変わらないこと（3 経路とも）

---

## Phase A: 論文の数値に効くもの（オフライン）

### Task A1: 局所座標系の警告と上向きの基準軸（§5-4・§1-5）✅
- Modify: `utils.py`（`LocalFrameFallbackWarning`、`compute_local_torque(..., up_axis=None)`、`compute_joint_power(..., up_axis=None)`）
- Test: `tests/test_local_frame.py`

### Task A2: 共有モデル `push_up_model.py` ✅
- Create: `push_up_model.py`
  - `estimate_gravity(trunk_up, magnitude, mode="axis") -> GravityEstimate`（"axis" は最寄りの座標軸へ吸着、"trunk" は体幹そのもの）
  - `trunk_up_vectors`, `torso_load_mass`, `hand_mass`, `hand_point`
  - `joint_axes(shoulder, elbow, wrist, hand=None, other_shoulder=None) -> {joint: (link, parent)}`
  - `inertia_about_link(inertia, link)`（§2-3、軸対称近似）
  - `SegmentState`, `segment_from_storage`, `push_up_torques(forearm, upper_arm, wrist, elbow, shoulder, gravity, load_mass, hand_mass_kg) -> {"wrist","elbow","shoulder"}`
- Test: `tests/test_push_up_model.py`（静止姿勢のトルクを「重さ × 水平のてこの腕」で手計算して比較）

### Task A3: オフラインのトルク（`compute_torque_from_pose.py`） ✅
- 重力を初期フレームの体幹から推定（`--gravity-mode axis|trunk`、`--gravity-frames 30`）
- `--wrist-base` を既定オンに（`--no-wrist-base` で旧来の自由振りの鎖だけ）。手首・肘・肩は `push_up_torques`
- 局所軸は `joint_axes`（手の点 17〜20 があれば手のひら、無ければ肘の屈曲軸）、フォールバック軸は重力の逆向き
- `--torque-scale` の既定を 1.0 に（m 入力で 1/100 になっていた。新規の発見）
- 外部荷重（ダンベル）の向きも重力に合わせる
- meta.json に重力の推定結果・規約の版・フォールバック件数
- Test: `tests/test_offline_torque.py`（y 下向きと z 上向きの同じ動きで同じ |τ|、静止姿勢で `push_up_torques` と一致、手の点ありなしで手首軸が変わる、meta の中身）

### Task A4: スコア（`compute_cycle_energy_elbow_wrist.py`）と §1-6 ✅
- `_joint_powers` の局所軸を `joint_axes` から取る（手首は手の点または肘の屈曲軸）。射影した τ_y・ω_y を返す関数を分けて §1-6 から使う
- 分母は変えない（手を含める、§2-2 決定）。理由をコメントに残す
- `compute_cycle_noise_contrib.py` は fps 掛けと角度経由の ω をやめ、スコアと同じ τ_y・ω_y を使う
- Test: `tests/test_cycle_energy_angles.py` に手首軸のテスト、`tests/test_cycle_noise_contrib.py`（ω が rad/s、スコアと同じ仕事になる）

### Task A5: `recalc_elbow_local_torque.py` を `joint_axes` に揃える ✅

### Task A6: 規約の版（§5-6） ✅
- `config.OUTPUT_SCHEMA_VERSION = 2`（v1 = タグ無し）。オフラインの meta、USB 経路の npy・CSV 名に入れる

## Phase B: リアルタイム表示（§5-7・§5-8・§5-1）

### Task B1: スマホ経路（`app/runners/network_measure.py`）を `push_up_model` に ✅
- 鎖を前腕 → 上腕＋体幹荷重に。胴体・大腿・地面反力は鎖から外す
- 重力は慣性テンソルを確定する初期フレームの体幹から推定
- 仕事率: 手首 = 前腕（手は固定）、肘 = 上腕 − 前腕、肩 = 上腕 − 上胴体
- Test: 既存 `tests/test_network_measure.py` の肘の仕事のテストをラベル修正後の `elbow_R` に、静止姿勢でオフラインと一致、鏡映テストは緑のまま

### Task B2: USB 経路（`master_research_code.py`）を同じ関数に ✅
- `run_specs` の胴体・大腿を外し、`push_up_torques` を呼ぶ。手首の仕事（ゲージ）は正の仕事 ∫max(P,0)dt に
- Test: AST テスト（`push_up_torques`・`arm_axes`・`segment_from_storage` を呼び、旧来の鎖の関数を呼んでいない）と py_compile

## Phase C: 再計算と照合

### Task C1: スコアの通し再計算（§6-1）— スクラッチに全被験者を流し、表にする ✅
### Task C2: 論文本文との照合（§1-4、R-5 の積分範囲、§2-3〜§2-5）— 本文は編集せず、要修正箇所を KNOWN_ISSUES に ✅（KNOWN_ISSUES §6-6）

## Phase D: 運用とコード品質

### Task D1: §3-2 GUI 停止で CSV が書かれない — 停止ファイルでループを抜ける（OS 非依存） ✅
### Task D2: §4-1 ネイティブ描画 ✅（実施時に変更: 到達不能で Windows 専用のネイティブ分岐を外し、スプライトに一本化）
### Task D3: §4-2 ラベル描画の二重実装を `utils.put_text_jp` に集約 ✅
### Task D4: §4-4 計時変数の統一と `config.env_flag` ✅
### Task D5: §5-5 定義の出典（大腿 COM は Winter、手の慣性行は未使用の理由） ✅

## Phase E: EKF（S6 を待たない部分）

### Task E1: S9a 配線（プロファイルが無ければ環境変数のスカラー＝今の挙動、候補を `ekf_profile_*.json` に限定、生成を `_DYN_DT` の後へ、`fs = 1/_DYN_DT`、`sorted(pose_keypoints)`） ✅
### Task E2: S10 ロバスト更新（S' = y²/c²、`max_gap` 秒） ✅（`max_gap` の既定は 0 = 無制限。S6 で決める）
### Task E3: S11 解析ページに「EKF の較正プロファイルを作る」 ✅

## Phase F: 仕上げ
- KNOWN_ISSUES.md を更新（対応状況・残タスク・新規の発見）
- 全テスト、コードレビュー
