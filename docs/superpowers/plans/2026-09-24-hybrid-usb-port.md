# 混成経路へ USB 経路の機能を移植する（1〜8・11〜14）

## Context

実際の計測構成は Mac 内蔵カメラ＋Pixel 7a の混成ステレオだが、研究の値を支える機能（EKF、体格の検査、エネルギーの前処理、
力学の関所、重力の選択肢、被験者と 1RM、ゲージ、出力の形、EKF 較正の GUI、カメラ設定、姿勢推定の工夫）は USB 2 台の経路
（`master_research_code.py`）にしか無い。2026-09-23 の実機計測では、EKF も外れ値の除去も無いため 3D の飛びがそのままトルク
（最大 100 万 N·m）になり、1RM も使えずスコアを出せなかった。ユーザーの依頼（2026-09-24 未明）: 一覧の 1〜8・11〜14 を混成経路に
実装し、**朝にすぐ本番データで検証できる状態**にする。

## 目標（/goal、2026-09-24 00:3x）

計画後は自動承認で実装。**別スレッド（master-research-65）で作るゲージ UI も含めて、朝には本番（Mac＋Pixel の実機、GUI）で、
腕の負荷が実際の動作にリアルタイムに呼応し、論文に則った閾値でゲージが反応している**こと。完了したら `/code-review --fix` と `/simplify`。

## 決めたこと（判断待ちだった項目を、目標に沿ってこう決める。理由は各項）

| 項目 | 決定 | 理由 |
|---|---|---|
| ゲージの閾値（§6-8） | 論文 4.5.2 節の W_0.70・W_0.85 を部位・左右ごとに計算（`theoretical_1rm_work(joint, 体重, 実測前腕長, c·1RM)`）。v<W_0.70 不足、W_0.70≤v<W_0.85 目標帯、v≥W_0.85 過負荷 | ユーザーの指示「論文に則った閾値」。表 2 の S≈0.72/0.86 と同じ境界 |
| ゲージの値 | 今の回の W_pos = Σmax(P,0)·dt（論文の定義、設計書 §6.4） | 論文と設計書が一致 |
| 1RM の出どころ | `m_max_all_merged.csv` を被験者番号（整数、"00"→0）で引く。肘 `elbow_{side}_outer`、手首 `wrist_{side}` | §6-4 は決定済み（outer）、§2-6 は今の `wrist_*`。オフラインのスコアと同じ |
| サイクル検出（§6-9、混成だけ） | 肩の中点の「重力の上向き」への射影（高さ, m）。基準は慣性を決める先頭の窓の中央値。新しい状態機械 `RepDetector`（速さは m/s、関所と共用。詳細は「A の追加の決定」）。USB 経路は触らない | 今の検出器は上下だけの押し上げで 0 回（合成データで確認）。回が閉じないとゲージが回ごとに戻らない |
| EKF の雑音（混成） | `HYBRID_EKF_PROFILE` が空なら同梱の既定値（USB は環境変数のスカラーのまま） | GUI の 1e-3 だと W_pos が +52%、同梱値なら +10%（合成） |
| Mac のカメラ設定（13） | pyobjc は入れず、`apply_camera_controls` を best effort で試して結果と実測 fps を報告・記録 | OpenCV の AVFoundation 経路は露出・WB・焦点を設定できない。依存と .app の同梱を増やさない |
| 姿勢推定の工夫（14） | 移植するが既定は今と同じ（`HYBRID_POSE_*` で受ける） | Pixel と揃える、VIDEO モードの追跡を壊さない、GUI の USB 向けの既定を漏らさない |
| 仕事の積分（`network_measure.py:455` の不具合） | 1 フレームごとに P·dt。dt が 100 ms を超える抜けはまたがない（その区間は積まない） | 組の抜けで仕事が数倍に化ける |
| 力学の関所 | 高さが基準＋余白を超える／上向きの速さで開き、基準近くに N フレーム戻ると閉じる。閉じている間はトルクを記録するが、仕事とゲージには積まない（列 `dyn_active` を残す） | 座っている間の雑音の仕事（論文 5.4 の過大評価の機序）を積まない。記録は捨てない |
| 重力のチェッカーボード | 校正の最後に任意の「盤を立てて静止」を足し、短辺の向きを校正フォルダに保存。あれば最寄りの軸に吸着して使う（USB と同じ） | 混成の校正は盤を自由に傾けるので、別の撮影が要る |
| 体格の比の検査 | プロファイルの有無によらず、先頭の窓の肩–肘の中央値が人体の範囲外なら止める（終了コード 3）。プロファイルがあれば比で掛け直す | 混成は 4,734 m の骨でも止まらなかった |
| ゲージ UI | master-research-65 が実装し、こちらが取り込む（下の調査メモ） | ユーザーの指示で別セッションが担当 |

## 調査メモ（確認済み。計画の材料）

- EKF: `extended_kalman_filter.LandmarkEKF(n_points, fs, cfg, bpf_*, vectorized, robust_gate, max_gap_s)`、`.step(meas(n,3), dt) -> (pos, vel, acc)`、
  `.set_noise()`。雑音は `app.tuning.ekf_profile.runtime_noise(EKF_PROFILE or None, dt=, bpf_enabled=, landmark_ids=pose_keypoints, scalar=EKFConfig(...))`
  → `.cfg`・`.origin`・`.resolution.scale_ref`・`.scaled(ratio)`・`.provenance()`。本体の使い方は `master_research_code.py:2558-2578, 2941-2975`
- 体格の比: `ekf_profile.body_scale_ratio(scale_ref, run_len)`（範囲外は ValueError、`PLAUSIBLE_REF_LEN`）、`SCALE_REF_PAIR=(12,14)`、
  `EXIT_IMPLAUSIBLE_SCALE=3`（本体 2648 行に直書き）。本体はプロファイル使用時だけ判定する
- 生 3D: `app.tuning.raw_capture.RawCaptureWriter(path, sorted(pose_keypoints), provenance={...})`、`.append(frame, t, p3ds)`、`.note(**kw)`、`.close()`
- エネルギー（本体に直書き）: 肘は `compute_cycle_energy_filtered(theta, tau, dt, fc_override)`（LPF→80 点に再標本化→τ を分位で切る→dθ 制限→∫τdθ の正負）
  `master_research_code.py:514`、f0 推定 `OnlineF0Estimator`（:220）、`_fc_scheduler`（:301）、`_butter_lowpass_filtfilt` ほか。`energy_pipeline.py` は古い版
  （`fc_override` と有限値の選別が無い）。手首は Σmax(P,0)·dt、肩は ΣP·dt（:3398-3425）
- 力学の関所: 本体 :3002-3057（`_dyn_active`、`detector.initial_z`、`RT_*`）。サイクルの値は左肩の `RT_CYCLE_AXIS`（既定 y）
- 重力: チェッカーボードは `calib.py:37 _save_checkerboard_short_axis`（校正の全ビューの短辺の中央値）→ 本体 :1534 で `_pick_axis_from_vector`
  （`push_up_model.nearest_axis` で最寄りの軸に吸着）。水平面の制約 `_candidate_axis_labels`（:349）。混成の校正は盤を自由に傾けるので、
  「盤を立てて静止」の撮影を別に足さないと成り立たない
- 理論 1RM 仕事: `compute_cycle_energy_elbow_wrist.theoretical_1rm_work(joint, body_mass, forearm_len, m_db)`、1RM の列は
  `ONE_RM_COLUMNS = {"elbow": "elbow_{side}_outer", "wrist": "wrist_{side}"}`（`m_max_all_merged.csv`、被験者番号は整数）
- ゲージの設計書 `docs/superpowers/specs/2026-09-23-subject-gauge-design.md`: 混成専用、GUI プロセスの Qt 窓、子は `@@GAUGE {json}` を約 10 Hz、
  モジュール `app/gauge/{model,protocol,tracker,widget,window,demo}.py`・`app/shell/{theme,pictograms}.py`、値は今の回の Σmax(P,0)·dt、
  閾値は `gauge_layout.json` の固定値（§9 で未確定）。別の作業ツリー `../master_research-gauge`（`murayama/subject-gauge`）は手つかず
- 不具合: `network_measure.py:455` がサイクル全体の仕事率の和に確定フレームの dt だけを掛ける（組の抜けで数倍に化ける）

- 論文（`~/master/村山_260831_final.pdf` 4.5.2 節、表 2）: FB 尺度は 1 サイクルの W_pos = Σmax(τω,0)Δt を 1RM 相当の理論仕事で割ったスコア S。
  閾値は負荷率 c = 0.70・0.85 の理論仕事 W_c = (m_x r_g + c·m_max r_x)(√2/2+1)g、スコアで S ≈ 0.72・0.86。「UI のゲージはこの 2 点を境に表示色を切り替える」。
  実装は `theoretical_1rm_work(joint, M, L, c·m_max)`（§2-6 で直したてこの腕）で W_c を出す
- ゲージ UI の分担（2026-09-24 00:40、master-research-65 と合意）: あちらが `murayama/subject-gauge`（作業ツリー `../master_research-gauge`）で
  `app/gauge/{protocol,model,widget,window,demo}.py`・`app/shell/{theme,pictograms}.py`・`app/runners/worker.py` の行の振り分け・
  `page_measure.py` の §5.2・`GAUGE_SHOW_JOULES` を実装（01:30 開始、protocol.py を最初にコミットしてハッシュを知らせる）。
  こちらは `app/gauge/tracker.py`・`app/gauge/thresholds.py`・子プロセス側の配線。行の形式 v2:
  `@@GAUGE {"v":2,"link":..,"rep":..,"source":"measure|demo","parts":{"elbow_L":{"now","prev","band":[lo,hi]|null,"w1rm"}...}}`。
  状態は v<lo 不足、lo≤v<hi 目標帯、v≥hi 過負荷。完了の知らせを受けて、こちらのブランチへ取り込む
- 本番環境: `dist/WheelchairTorque.app` は 2026-09-23 22:56 の版。朝の確認はリポジトリからの `python -m app` を基本にし、最後に .app を作り直す

- 混成の流れ（調査 1）: メインスレッド＝`LiveSession.step`（Mac の取得・推定・表示）、受信スレッド＝`PhoneLink`（`on_landmarks`・`on_pairs`・
  `on_tick` 50 ms・`on_stop`）。`SyncBuffer` は 30 Hz 格子、抜けると次の組の t_ns が n×33 ms 跳ぶが `frame_index` は 1 ずつ。
  `NetworkMeasurement.process`（:232）の順は points_3d → `_timestep` → links → baseline → 慣性（:375、30 フレームの中央値＋重力）→ トルク → サイクル。
  3D は USB と同じ (−x,−z,−y)×0.01 m、cam0 基準、関節は `pose_keypoints` 昇順の 16 点（`slot_of`）。`_cycle_value` は左肩（位置 0）
- 記録（`recorder.py`）: 受信スレッド専用、`kpts3d_`・`frames_`・`landmarks2d_`・`local_torque_`・`cycle_work_`・`meta.json`、1 秒ごとに flush
- 校正（`hybrid_calibrate.py`）: 解が出た後（:220）・自動保存（:239-251）と s 保存（:262-267）の前に段階を足せる。Mac の角点
  `session.local_corners` と `cv.solvePnP(board.object_points, corners, K0, dist0)` で盤の向きが出る。`save_calibration` の meta に足せば
  `Recorder` が `calibration_meta` として写す。テストの偽物（`tests/test_hybrid_calibrate_runner.py`）は Space を返し続けるので終わり方が要る
- Mac のカメラ: 幅・高さしか設定していない。OpenCV 5.0 の AVFoundation 経路に露出・焦点のモード設定が無く、pyobjc の AVFoundation も未導入
  （`platform_compat.py:122` に「依存を増やさない」とある）。`apply_camera_controls` は `set` の失敗を握りつぶすので足しても落ちない
- 姿勢推定: Mac は lite・VIDEO モード・閾値 0.5 固定・全解像度。USB の ROI は本体 :1875-1969 に直書き（`_roi_from_keypoints`・`_expand_roi`・
  `_remap_landmarks_to_fullframe`）、推定器は `pose_runtime.PoseEstimator`（IMAGE モード）。USB の横の切り出しは座標を全体へ戻していない
  （USB 側の不具合の疑い。移植では戻す）。モデルは `pose_landmarker_lite.task` だけ
- テストの部品: `tests/test_network_measure.py`（`_body_points`・`_pair_from_pixels`）、`tests/test_hybrid_measure.py`（`calibration(tmp_path)`、
  本物の PhoneLink＋MockPhone で 3 秒回す通しのテスト :139）、`tests/test_hybrid_verification.py`（`make_body_run`）、`tests/hybrid_fakes.py`。
  `hybrid_measure.main` のループを通すテストは無い。新モジュールは `tests/test_cross_platform.py` の一覧に足す

- GUI（調査 2）: `page_measure.py` は子に引数を渡さず `settings.as_env()` の全件を環境変数で渡す（`entry.worker_environment` :137-161）。
  `BODY_MASS_KG` は画面に出ず常に 65、`SUBJECT_ID`・`EKF_PROFILE` は画面に出ている（`settings.CURATED` :75-171）。スキーマの生成元
  `tools/extract_env_schema.py` は `master_research_code.py`・`config.py` だけを見るので、**混成だけが読む変数は `CURATED` に足す**。
  `worker.py:147-151` は塊のまま emit（行の振り分けは UI 側が担当）
- EKF の較正（調査 2）: `tune_ekf` は生 CSV（見出し `frame,t,{id}_x..`、ID 昇順）とサイドカー（`stage=pre_ekf`・`landmark_ids`・`dt` が必須）で通る。
  `scale_ref` は生の点列の ID 12–14 から作る。実行時の `resolve_profile` は dt の相対差 5% 以内を選ぶ → **混成の計測中に 1/30 s 格子で生 CSV を書き、
  抜けた格子は NaN の行で埋める**（行を詰めると dt 一定の前提が崩れる）。`hybrid-raw` の既定の出力名 `kpts3d_raw_{stamp}.csv` は計測中の生 CSV と
  衝突するので名前を変える。`tune_ekf --out <フォルダ>` の `IsADirectoryError` も直す。解析ページのファイル選択は `output_data` から始まる
- `check`（調査 2）: 混成の見分けは `meta.json` の `coordinates` キーだけ（`_is_hybrid` :220）。ファイル名は `_hybrid_files` が stamp で固定。
  混成には S9b の材料が無く `HYBRID_EKF_NOTE` を出すだけ → `_ekf_stats`（:157-195）・ゲージの帯の統計を混成でも回す

- USB のゲージ（調査 3）: 閾値は旧来の式 `r_x·g·(0.42·m1+{0.3,0.7}·m_max)·K`（r_x 直書き、m1＝体重×有効質量係数、体重が 2 系統）で、§2-6 の
  修正に追随していない → 混成では使わず論文の W_c で作る。`gauge_energy.ElbowGaugeEnergy`（Σmax(τ_y·dθ,0)）・`gauge_values` は import できる。
  描画 `Gauge_display.py` は PyQtGraph（混成では使わない。UI 側が新しく作る）
- USB の単眼デモ（調査 3）: 既定では動いていない（肘角の添字が古い並び順 :472-476、肩は Tasks 経路で world landmarks が無く常に None）。
  段階は肩の上昇 0.02/0.10 m・肘角の変化 8/45° → 比 0.30/0.80、1 フレームごとに +0.025/−0.035。→ 三角測量した 3D の高さと肘角で作り直す
- USB の関所（調査 3）: `_dyn_should_run`（:3182）が偽ならトルクを 0 にして行は書く。開く条件はバーストの分岐の中だけ（:3005-3035）、
  閉じるのは基準＋6 mm を 2 回。`_dyn_start_frame`・`RT_DYN_PREV_FRAMES` はログだけ。状態の列は無い
- USB の出力（調査 3）: `aim_torque_vec_{ts}_s2{_grav_tag}.csv`（局所トルク横長、frame は暖機の後から）、`cycle_energy_debug_*`（肘だけ、
  `frame,part,e_pos,e_neg,fc_current,dt_sec,lpf_dt_sec,n_u`）、`gauge_energy_*.csv/.json`、`OFFLINE_WRIST_CAPTURE` の npy
  （`forearm_*` (N,3)・`tau_wrist_*` (N,)）。`OUTPUT_SCHEMA_VERSION=2`（`config.py:366`）、`_grav_tag=_g{label}`
- オフラインのスコア（調査 3）: `compute_cycle_energy_elbow_wrist.py` は `*_with_cycles.csv`（`joint_<ランドマーク ID>`、1 始まりの `cycle_index`）と
  全体座標のトルク（`compute_torque_from_pose.py` の出力）、dt 一定を前提。混成の出力は形が違う → 実行時にスコアを出して記録する（主）＋
  オフライン用の書き出し（従）

- 計画担当 C の発見: `m_max_all_merged.csv` は `.gitignore` の `*.csv` で**追跡されていない**（worktree と .app に入らない）。
  2026-09-23 の記録は先頭 30 フレームの骨の長さが壊れている（右上腕 6.4 m）→ 再生は 20 s から。上下の動きがある区間は 40〜90 s・170〜230 s だけで、
  回が閉じることの証明には合成の押し上げが要る。worktree には `.venv` が無い（主の venv を絶対パスで使う）。GUI は親の環境変数を子へ引き継ぐ
  （`HYBRID_REPLAY` を GUI の起動時に付ければ子に届く）。ゲージの行は 512 バイト未満（QProcess の MergedChannels）

## 段取り

### 担当とファイルの持ち分

作業ツリーを分ける（段取り 0 のコミットの後に作る）。テストは主の venv を絶対パスで使う:
`/Users/s.murayama/projects/master_research/.venv/bin/python -m pytest -q tests`

| 担当 | 作業ツリー（ブランチ） | 持ち分 |
|---|---|---|
| 私（統合） | `master_research`（`murayama/fix-left-right-dynamics`） | 段取り 0 のコミット、`app/core/settings.py` の CURATED、`tests/test_cross_platform.py`、`KNOWN_ISSUES.md`・`docs/*`・`HANDOFF_*`、`app/hybrid/paths.py`（`replay_root()`）、新規 `app/hybrid/replay.py`・`app/runners/hybrid_replay.py`・`tools/synth_session.py`・`tools/gauge_view.py`・`tests/test_hybrid_replay.py`・`tests/test_gauge_contract.py`、取り込み（merge）と検証 |
| 実装 A（計測の核） | `../master_research-core`（`murayama/hybrid-core`） | `app/runners/network_measure.py`、`app/hybrid/recorder.py`・`measurement.py`、`app/runners/hybrid_measure.py`（再生の分岐は私が最後に足す）、新規 `app/gauge/tracker.py`・`app/gauge/thresholds.py`・計測のモジュール（例 `app/hybrid/dynamics_ext.py`・`app/core/cycle_energy.py`）、そのテスト |
| 実装 B（撮影・校正・道具） | `../master_research-tools`（`murayama/hybrid-tools`） | `app/hybrid/{mac_camera,pose_detector,live,calibration_io}.py`、`app/runners/{hybrid_calibrate,tune_ekf}.py`、`app/shell/page_analyze.py`、`tools/verify_run.py`、`app/core/workspace.py`、`packaging/app.spec` の hiddenimports、そのテスト |
| master-research-65（UI） | `../master_research-gauge`（`murayama/subject-gauge`） | `app/gauge/{__init__,protocol,model,widget,window,demo}.py`、`app/shell/{theme,pictograms}.py`、`app/runners/worker.py`、`app/shell/page_measure.py`、CURATED の末尾の自分の項目 |

A と B は CURATED・`test_cross_platform.py`・KNOWN_ISSUES を触らず、新しいモジュール名と記録すべき事実を完了の報告で私に渡す。
既存の呼び出し形は壊さない（新しい引数はキーワード専用・既定値つき）。

**契約**（A と B の依頼文に同じものを書く）
1. 盤を立てた向き: B が校正の `meta.json` に `"gravity_board": {"up_cam0": [x,y,z], "frames": n}`（cam0 のカメラ座標の単位ベクトル、上向き＝−y_cam 側を正）を保存。
   A はこれを (−x,−z,−y) に直して `push_up_model.nearest_axis`（水平面の制約・優先軸・曖昧さの幅つき）で吸着させ、無ければ体幹から決める
2. 記録のファイルは足すだけ。新しい接頭辞は `kpts3d` で始めない: `raw3d_{stamp}.csv`（EKF の手前、`RawCaptureWriter`、1/30 s 格子、抜けは NaN の行、
   サイドカーつき）、`aim_torque_vec_{stamp}_s2_g{label}.csv`、`cycle_energy_{stamp}.csv`、`cycle_score_{stamp}.csv`、`gauge_{stamp}.csv/.json`、
   `forearm_*/tau_wrist_*.npy`（`OFFLINE_WRIST_CAPTURE=1` のとき）。`frames_` の列 `dyn_active` は末尾に足す。B の `hybrid-raw` の出力は `retri3d_{stamp}.csv` に改名
3. `NetworkMeasurement.points_3d` は状態を持たないまま残す（`retriangulate` が使う）
4. `GaugeTracker(parts)`: `add(part, P, dt)`（Σmax(P,0)·dt）・`close_rep()`・`set_bands(bands)`・`set_link(bool)`・`snapshot() -> dict`（ロック付き）。
   `GaugeTicker(tracker, source, interval_s=0.1, write=print).tick()` が `protocol.encode` で 1 行を出す（J は小数 1 桁、NaN は null、512 バイト未満）
5. 閾値: `thresholds.gauge_bands(subject_id, body_mass_kg, forearm_len_by_side, csv=None) -> {part: Band(lo, hi, w1rm) | None}`。
   表は `ONE_RM_CSV` → `config.folder_path/m_max_all_merged.csv`。`SUBJECT_ID` は "0"・"00"・"S000" を 0 と読む。無ければ None と理由のログ
6. 終了コード 3 は解像度の不一致と体格の検査で共用し、`meta.error` で見分ける
7. `app/gauge/__init__.py` は UI 側が protocol のコミットで作る

### 段取り（時刻は目安）

| # | 目安 | 担当 | 内容 | 完了の条件 |
|---|---|---|---|---|
| 0a | 承認直後 | 私 | 全体で 1 回: 全テスト・`py_compile master_research_code.py`・`uvx --quiet pyflakes <変えた .py>`（本体は HEAD の件数より増えない） | 通る |
| 0b | 〜+30 分 | 私 | 未コミットの作業を 8 つのコミットに（下の表）。`git add <パス>` → `git diff --cached --stat` → commit。`git add -A` はしない | `git status --short` が `?? .superpowers/` と今夜の新規だけ |
| 0c | 0b の直後 | 私 | UI 側へ HEAD のハッシュを送る。worktree を 2 つ作る（`git worktree add ../master_research-core -b murayama/hybrid-core HEAD` など） | A・B を起動できる |
| 1 | +15 分 | 私 | UI 側の protocol.py のコミットを `git merge --no-ff <hash>`（a6abf3a の上なら cherry-pick）で取り込み、core に渡す | 全テスト |
| 2 | 〜3 時間 | A | 下の「A のタスク」をテスト先行で順に。節目 M1（dt・高さ・サイクル・関所・tracker・閾値・@@GAUGE）→ M2（EKF・体格・重力の盤・エネルギー）→ M3（出力・デモ） | 節目ごとに全テスト、1 機能 1 コミット |
| 3 | 〜2.5 時間 | B | 下の「B のタスク」をテスト先行で | 全テスト |
| 4 | A・B と並行 | 私 | 再生の入口（`app/hybrid/replay.py`: `landmarks2d_*` を 2 ロール時刻順に SyncBuffer へ push → `MeasurementSession`、実時間の `--speed`、`--from/--to`、出力は `hybrid/replay/<stamp>`、meta に `replay_of`・`replay_timing`、GUI からは `HYBRID_REPLAY` 環境変数）、合成の押し上げ（`tools/synth_session.py`: 2 s 静止→3 s 周期で 10 回、肩 6 cm、肘 95→165°、良い配置と 09-23 の校正、雑音 1 px、Pixel 30/15 Hz、150 ms の抜け、本物の Recorder で書く）、画面の道具（`tools/gauge_view.py`: WorkerRunner＋GaugeWindow＋5 s ごとの `grab().save()`） | speed 0 の通しのテスト |
| 5 | M1 の後 | 私 | core の M1 を取り込み、再生から @@GAUGE の行が出ることを確かめる | 合成で rep が増え now が 0 に戻る |
| 6 | A・B の完了後 | 私 | A・B を `git merge --no-ff` で取り込み、`hybrid_measure` の先頭に再生の分岐、CURATED、`test_cross_platform.py` | 全テスト・py_compile・pyflakes |
| 7 | 6 の後 | 私 | 端末の検証 R1〜R4（下）と `verify_run check` | 数値の条件 |
| 8 | UI の完了の知らせ | 私 | `murayama/subject-gauge` を merge（UI の持ち分は UI の版、共有の一覧は両方の行）、`tests/test_gauge_contract.py`（子の tracker/ticker の本物の行を UI の decode と model に通す: 部位 4 つ・lo<hi・境界の状態・512 B 未満） | 全テスト |
| 9 | 8 の後 | 私 | 画面の確認: `tools/gauge_view.py` を再生（実データ 20〜90 s と合成）で回して grab を Read で見る。GUI 本体も `HYBRID_REPLAY=… python -m app` で起動し `screencapture -x`（撮れなければ grab だけ）。実機の煙の試験（`--camera 1` で 20 s、Pixel が来れば connected まで） | 目視で回数・針・帯・状態・「再生」の印・人物に切れ目が無い |
| 10 | 9 の後 | 私 | `/code-review --fix`（範囲は 0b の最後のコミットから HEAD）→ 全テスト → コミット → `/simplify` → 全テスト・R3 → コミット | 通る |
| 11 | 10 と並行 | 私 | 文書: KNOWN_ISSUES §6-8（混成は論文の W_c）・§6-9（混成は高さ・m/s、USB は判断待ちのまま）・§6-10（EKF・体格・関所、R1〜R3 の数値）、`docs/hybrid_field_run.md`（被験者番号・体重の入れ方、開始 2 秒は静止、盤を立てる校正、ゲージの見方、GPU 推論、終了コード 3、再生での練習） | 更新した |
| 12 | 最後（任意） | 私（UI 側と） | .app の作り直し（今の `dist/WheelchairTorque.app` は別名で残す） | 起動する |
| 13 | 朝 | 私 | ユーザーへの報告と朝の確認の手順 | 渡した |

判断の関所: M1 が 3 時間目までに入らなければ私が A に加わり、A の M3 を削る。UI が 05:15 までに来なければ、届いた分（protocol・worker・window）だけ取り込み、
それも無ければ子の OpenCV の窓の文字行（`live.step(lines=…)`）に部位ごとの now・帯・状態を出す代替にする。06:30 以降はコードを変えない。

削る順（先頭から）: .app → `/simplify` → B の周辺（解析ページの既定フォルダ・SEED_FILES・check の帯の統計）→ 生 3D の記録 → Mac の ROI・推定の工夫 →
肘のエネルギーの前処理 → 盤を立てる校正 → 合成の画面確認 → 力学の関所。**削らない**: サイクル・dt・閾値 W_c・tracker・@@GAUGE・UI の取り込み・EKF・体格の検査・再生。

### 段取り 0b のコミット（依存の順）

| # | 中身 | メッセージ |
|---|---|---|
| 1 | `app/core/settings.py`、`app/entry.py`、`config.py`、`app/shell/page_measure.py`、`app/shell/page_analyze.py`、`tests/test_output_dir.py` | `fix(app): 計測の CSV を GUI が表示する出力先に書く（§3-3）` |
| 2 | `app/core/camera_controls.py`、`tests/test_camera_controls.py`、`master_research_code.py` の塊 1（`@@ -21`）と塊 6（`@@ -2476`）だけ（`git diff` を塊に分けて `git apply --cached --recount`。5 分で無理なら 2 と 3 を 1 つに） | `refactor(usb): カメラ設定の固定を app/core/camera_controls に移す` |
| 3 | `gauge_energy.py`、`tests/test_gauge_energy.py`、`master_research_code.py` の残り | `fix(usb): ゲージの値をどの部位も今のサイクルの正の仕事にし、HEADLESS でも記録する（§6-8）` |
| 4 | `app/hybrid/measurement.py`、`app/runners/hybrid_measure.py`、`app/runners/network_measure.py`、`app/hybrid/calibration_io.py`、`app/hybrid/retriangulate.py` | `feat(hybrid): 止まった理由を記録し、計測フォルダの 2D から三角測量し直せるようにする` |
| 5 | `tools/record_stereo.py`、`tools/verify_run.py`、`tests/test_record_stereo.py`・`test_verify_run.py`・`test_hybrid_verification.py`、`docs/usb_stereo_verification.md`・`hybrid_verification.md` | `feat(tools): 録画・再生と、USB・混成の出力の検査（配置と 3D の質）を足す` |
| 6 | mobile の 8 ファイル | `feat(mobile): 段階ごとの速さの表示、30fps 固定、GPU 推論の切り替え` |
| 7 | `docs/hybrid_field_run.md`、`KNOWN_ISSUES.md`、`HANDOFF_TODO.md`・`HANDOFF_CODEX.md`・`HANDOFF_NEXT.md`、`docs/superpowers/plans/2026-09-23-remaining-tasks.md`、`docs/windows-transfer-audit-2026-09-23.md`、`tmp_filter_pose_torque.py` | `docs: 実機の手順（置き方・Pixel の速さ）と既知の問題・引き継ぎを更新する（§6-10）` |
| 8 | `git add -f m_max_all_merged.csv` | `chore(data): 1RM の表を追跡する（混成のゲージの閾値が読む）` |

各メッセージの末尾に空行と `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`・`Claude-Session: https://claude.ai/code/session_018KBZTFeyteYSHuhB36zGkf`。push しない。

### 契約の確定（計画担当 B の案を採る点）

- 計測中の生 3D は USB と同じ名前 `kpts3d_raw_<stamp>.csv`（`_hybrid_files` は stamp で名前を固定するので `kpts3d_<stamp>.csv` と取り違えない）。
  provenance に `source="hybrid"`・`times="grid"`・`dt=1/30`・EKF の設定・`ekf_noise`・体格の比。`hybrid-raw` の出力は `kpts3d_raw_<stamp>_retri[_sN].csv` に改名
- 盤の向きは校正の meta の `"checkerboard_short_axis"`（USB の `calib.py:52-58` と同じ形に `up_label_runtime`・`tilt_deg`・`spread_deg`・`samples`・`method` を足す）。
  A は `app.hybrid.gravity_board.board_up_runtime(calibration.meta)` だけを使う
- 回の値: `cycle_work_<stamp>.csv` は既存の `work_j`（符号付き、dt は直す）を残し、末尾に `work_pos_j`・`work_neg_j`・`w1rm_j`・`score` を足す。
  `frames_<stamp>.csv` の末尾に `dyn_active`。meta.json に `subject_id`・`one_rm_kg`・`body_mass_kg`・`forearm_len_m`・`w1rm_j`・`gauge_bands_j`・
  `gravity{source,label,vector,tilt_deg}`・`ekf{enabled,origin,path,scale_ratio}`・`output_schema_version`
- 混成だけが読む設定（名前は `HYBRID_` で始め、GUI が渡す USB 向けの既定が漏れないようにする）: `HYBRID_EKF_PROFILE`、`HYBRID_GRAVITY_BOARD`（既定 1）・
  `HYBRID_GRAVITY_BOARD_TIMEOUT_S`（30）、`HYBRID_POSE_MODEL`・`HYBRID_POSE_MIN_DET/_MIN_PRESENCE/_MIN_TRACK`（0.5）・`HYBRID_POSE_INPUT_SCALE`（1.0）・
  `HYBRID_POSE_ROI`（0）、`HYBRID_DYN_GATE`（1）、`ONE_RM_CSV`。EKF のスカラー（`EKF_ENABLE`・`EKF_Q_ACC`・`EKF_R`・`EKF_GATE_STD`・`EKF_ROBUST_GATE`・
  `EKF_MAX_GAP_S`・`EKF_BPF_*`）、`SUBJECT_ID`・`DEMO_MONO_GAUGE_ON`・`OFFLINE_WRIST_CAPTURE`・`E_*` は USB と共用。**CURATED への追加は私が段取り 0 の後に
  1 コミットで入れてから worktree を作る**（A・B は settings.py を触らない）
- 13 は pyobjc を入れない（OpenCV の AVFoundation 経路は露出・WB・焦点を設定できない、Camo で番号がずれる、.app の同梱が増える）。
  `apply_camera_controls` を best effort でかけ、`set` の結果・`get` の値・起動時の実測 fps を 1 行で報告して meta に残す
- 14 は移植するが既定は今と同じ（lite / VIDEO / 0.5 / 縮小 1.0 / ROI なし）。ROI をオンにしたら IMAGE モード。横の切り出しは座標を戻す純粋な関数だけ。
  USB の横の切り出しは座標を戻していない（`master_research_code.py:2818`・`:2942`、確定）→ KNOWN_ISSUES に記録（本体は触らない）

### B のタスク（テスト先行、この順。各タスクは失敗するテスト → 実装 → 全テスト → コミット）

| # | 内容 | 主なファイル |
|---|---|---|
| B1 | `hybrid-raw` の出力名を `_retri` 付きに（計測中の生 CSV を上書きしない） | `tools/verify_run.py` |
| B2 | 盤の向きの純粋な関数（短辺は rows<cols なら y、上向きに符号を揃える、`board_axes_cam0`・`to_runtime`・`UprightCollector`・`board_up_runtime`） | 新 `app/hybrid/gravity_board.py`（**できたら私経由で A に知らせる**） |
| B3 | `calibration_io.update_meta(directory, **fields)`（原子的に書き直す） | `app/hybrid/calibration_io.py` |
| B4 | 校正の最後の「盤を立てて静止」: 保存を先にし、その後 `run_board_up`（Enter/n で省略、q・停止・時間切れは記録せず 0、傾き 10° 超で警告、30° 超の標本は捨てる、2 s ごとに標準出力へ進み具合） | `app/runners/hybrid_calibrate.py`、`tests/test_hybrid_calibrate_runner.py` に追加 |
| B5 | `check` の混成の拡張（`_ekf_stats` の中身を切り出して `round(t/dt)` で格子を合わせる、帯と回ごとの W_pos・到達回数・スコア、被験者・1RM・重力の出どころ・盤の傾き。新しい版の記録があるときだけ検査を足す＝古い記録は合格のまま。`HYBRID_EKF_NOTE` の書き換え） | `tools/verify_run.py`、新 `tests/test_hybrid_check_extended.py` |
| B6 | `SEED_FILES` に `m_max_all_merged.csv` | `app/core/workspace.py` |
| B7 | `tune_ekf`: `--out <フォルダ>` の修正、混成の収録は `hybrid/ekf_profiles/` へ書いて `HYBRID_EKF_PROFILE` を案内、dt が 1/30 から 5% を超えたら警告 | `app/runners/tune_ekf.py`、`app/hybrid/paths.py`（`ekf_profile_root`・`latest_raw_capture_dir`） |
| B8 | 解析ページの EKF の項目: 始まりの場所を最新の生 CSV のフォルダに、ファイルの絞り込み、説明文 | `app/shell/page_analyze.py` |
| B9 | Mac カメラ: `apply_camera_controls`（解像度・FOURCC は除く）＋報告＋実測 fps、`camera.controls` | `app/hybrid/mac_camera.py` |
| B10 | ROI の純粋な関数（本体 :1875-1969 と同じ式、全体の正規化座標へ戻す、横の切り出しの範囲） | 新 `app/hybrid/pose_roi.py` |
| B11 | `PoseOptions.from_env`（`HYBRID_POSE_*`）と `PoseDetector(options=)`、既定は今と同じ、GUI の既定が漏れないテスト、起動時に 1 行 | `app/hybrid/pose_detector.py` |
| B12 | オフライン用の書き出し `verify_run export-offline`（`joint_<ID>`、1 始まりの `cycle_index`、30 Hz 格子、被験者番号の入ったファイル名、次のコマンドの案内） | `tools/verify_run.py` |
| B13 | 比べる道具 `tools/compare_pose_options.py` を被験者 7・9 の USB 録画で実行し表を報告（既定は変えない） | 新 `tools/compare_pose_options.py` |

B が削ってよい順: B13 → B12 → B11 の ROI の配線 → B10 → B9 → B8 → B7 の周辺 → B5 の副次の節 → B4。削らない: B1・B2・B3・B5 の中核・B6。

### A の追加の決定（計画担当 A の発見から）

- **混成の EKF は、`HYBRID_EKF_PROFILE` が空なら同梱の既定値**（q=0.122、r=2.59e-5）を使う（GUI が渡す `EKF_Q_ACC/R=1e-3` だと合成の押し上げで肘の W_pos が +52%、
  同梱値なら +10%）。出どころは meta とサイドカーに残し、朝の最初の計測の `kpts3d_raw` から `tune_ekf` でプロファイルを作るよう案内する
- **関所は削らない**（関所なしで雑音 5 mm だと肘の W_pos が 1 回 72〜160 J になり W_0.85=56.9 J を超えて過負荷と誤表示）
- サイクルは `PushCycleDetector` を使わず新しい状態機械 `app/hybrid/rep_detector.py`（開く: 高さ＋2 cm か上向き 0.10 m/s が 2 フレーム、閉じる: 基準＋1 cm 以内が
  3 フレーム、最小の持ち上げ 3 cm 未満は捨てる、先読み 5 フレーム）。関所＝回の区切り
- 抜けは NaN での予測（dt=1/30 を missing 回）。0.5 s を超える抜けは EKF を作り直し、100 ms を超える抜けの後は `LinkVectorCalculator` と storage を作り直す
- 見積もり（65 kg・被験者 00・前腕 0.25 m）: 肘 W_0.70/W_0.85/W_1RM = 47.47/56.89/66.31 J、手首 9.65/11.58/13.52 J。13 cm の持ち上げで肘 22.0 J・
  手首 3.85 J（S≈0.33・0.28、どちらも不足）。**弧は動くが普通の押し上げでは帯に届かない見込み**（過去の実データの S と同じ桁）→ 朝に伝える
- ゲージの行は `sys.stdout.write(<1 行>)` の 1 回で書く（UI 側の求め。print は本体と改行を別に書き、受信スレッドの print と混ざりうる）

### A のタスク（テスト先行。A1 と A2 は同じ worktree `../master_research-core` で持ち分を分け、`git commit -- <パス>` で自分のパスだけコミット）

**A1（新しい純粋なモジュール、互いに独立）**

| # | 内容 | ファイル |
|---|---|---|
| T1 | `thresholds`: `subject_index("00")→0`、`load_one_rm`（`elbow_{s}_outer`・`wrist_{s}`、"none"/NaN は None）、`part_bands`（`theoretical_1rm_work(joint, M, L_side, c·m)`、前腕 0.15〜0.40 m の外は None）、`classify`（lo≤v<hi 目標帯、v≥hi 過負荷）。上の見積もりの数値を固定 | 新 `app/gauge/thresholds.py` |
| T2 | `GaugeTracker`（add・set_now・close_rep・discard_rep・set_bands・set_link・values・snapshot、ロック 1 つ）と `GaugeTicker`（0.1 s ごとに 1 回の write）。protocol の取り込みまでは偽の encode、取り込み後は `pytest.importorskip` の契約テスト | 新 `app/gauge/tracker.py` |
| T3 | `energy_pipeline.py` を本体の今の版に入れ替え（`EnergyFilterConfig.from_env`・`OnlineF0Estimator`・`fc_scheduler`・`AdaptiveCutoff`・`compute_cycle_energy_filtered(..., fc_override, config)`、`angle_between` は残す、`E_LPF_NATIVE_ON` の既定は 0）＋本体の 6 定義を AST で抜き出して rtol 1e-12 で突き合わせるテスト | `energy_pipeline.py` |
| T4 | `RepDetector`（上の状態機械。13 cm×3 回で 3 回、座って σ3 mm を 60 s で 0 回、1.5 cm は捨てる、大きな dt、30 s で閉じる） | 新 `app/hybrid/rep_detector.py` |
| T5 | `choose_gravity(trunk_ups, board_up_runtime, *, magnitude, mode, candidates, preferred, ambiguity)`（盤があれば `nearest_axis` に吸着・符号は体幹、盤と体幹が直交なら体幹に戻して理由、無ければ今の `estimate_gravity`）、`read_board_up(meta)`（`meta["checkerboard_short_axis"]["vector_runtime"]`、無ければ `vector_cam0` を変換）、`candidate_axes` | 新 `app/hybrid/gravity.py` |
| T13 | デモ `DemoGauge`（3D の肩の高さと肘角、0.02/0.10 m・8/45°→0.30/0.80、+0.025/−0.035、比 0.80 を帯の中央へ） | 新 `app/hybrid/demo_gauge.py` |

**A2（配線。A1 のモジュールが入りしだい使う）**

| # | 内容 | 依存 |
|---|---|---|
| T6 | `RepAccumulator`（新 `app/hybrid/rep_work.py`、Σ P_i·dt_i、100 ms 超は積まない、先読みの輪、上限つき）を今の `_accumulate_cycle` に入れて dt の不具合を直す | なし |
| T7 | EKF の配線（`EkfSettings.from_env`、`HYBRID_EKF_PROFILE`、NaN の予測、作り直し、`points_raw` と `points_3d` を分ける、出どころ）＋抜けの後の作り直し | なし |
| T8 | 先頭の窓（30 組）で体格の検査（範囲外は `ImplausibleBodyScale` → 終了コード 3）・プロファイルの掛け直し・慣性・重力（T5）・上向き・基準の高さ・前腕長・帯（T1）→ tracker | T1・T2・T5・T7 |
| T9 | 関所とサイクル（T4）、`dyn_active`、閉じている間はトルクを記録し仕事と tracker には積まない | T4・T8 |
| T10 | 回の確定: `cycle_work`（符号付き＋W±・w1rm・score）、肘の濾波 E±（T3、`cycle_energy`）、`tracker.close_rep` | T3・T9 |
| T11 | Recorder の出力（`kpts3d_raw_<stamp>.csv` 1/30 s 格子・NaN 行・サイドカー、`aim_torque_vec_<stamp>_s2_g<label>.csv`、`cycle_energy_`、`gauge_energy_<stamp>.csv/.json`、`frames_` の末尾の列、meta の鍵、npy） | T10 |
| T12 | `hybrid_measure`: 設定の読み込み・1RM の表・tracker・`GaugeTicker` を毎周回（`sys.stdout.write`、終わりに force）、メインループを通す初めてのテスト（偽のカメラ・推定・PhoneLink で @@GAUGE の行を読む） | T2・T11 |
| T13 の配線 | `DEMO_MONO_GAUGE_ON=1` で source="demo" | T12 |

壊れる既存のテスト（`test_network_measure` のサイクル・基準の系、`test_hybrid_measure` の NaN、`test_e2e_mock_phones` の `cycle_count>0`、
`test_hybrid_verification` の hybrid-raw）は、押し上げの部品（新 `tests/hybrid_pushup.py`）と新しい仕組みのテストに置き換え、docstring に経緯を残す。

A が削ってよい順: T13 → npy → 適応 fc と AST の突き合わせ → `aim_torque_vec` → `gauge_energy` の CSV → 盤からの重力 → 先読み。
削らない: EKF・関所とサイクル・dt・閾値・tracker と行・被験者と 1RM・体格の検査・`kpts3d_raw`。

### 実装担当の起動（段取り 0 の後）

- A1・A2 は `../master_research-core`、B は `../master_research-tools` で、`superpowers:subagent-driven-development` の流儀（1 タスク＝失敗するテスト→実装→
  全テスト→`git commit -- <パス>`）。依頼文には、この計画ファイルのパス、持ち分、契約、テストの実行のしかた（主の venv の絶対パス）、
  日本語の docstring とコミットメッセージ、`master_research_code.py` を import しない・変えない、を書く
- 私は再生・合成・画面の道具（段取り 4）を主のツリーで作りながら、A・B・UI の節目を取り込む。使用量の上限に当たったら、残りのタスクを削る順に従って組み直す

## 検証

- 全テスト `.venv/bin/python -m pytest -q tests`（`tests` 必須）、`python -m py_compile master_research_code.py`、`uvx --quiet pyflakes <変えたファイル>`、
  スマホは今回変えない
- 再生の検証（段取り 7）:
  - R1: 09-23 の記録を 0 s から → 終了コード 3 と理由が 5 s 以内（体格の検査）
  - R2: 20〜90 s を実時間で → 終了コード 0・`stop_reason=stop_request`、左腕 |τ| の 95% < 60 N·m、右腕 99% < 200 N·m（前回は 100 万 N·m）、
    左の W_pos/W_1RM が 0.1〜3、@@GAUGE が約 10 Hz・512 B 未満、`on_pairs` の 95% < 10 ms・最大 < 60 ms、100 ms 超の抜けで仕事が増えない、`check` の構造が合格
  - R3: 合成の押し上げ 10 回 → rep 10±1、回ごとに now が 0 に戻り prev が直前の回、帯が `theoretical_1rm_work` で別に計算した値と一致、
    振幅を変えた 2 本で不足・目標帯・過負荷を跨ぐ、押し上げの開始から now が増えるまで 0.3 s 以内
  - R4: 再生の途中で停止ファイル → 2 s 以内に終わり meta が complete・stop_request
- 画面: `tools/gauge_view.py` の grab を Read で目視（回数・針・帯・状態・「再生」の印・文字の重なり・人物の切れ目）
- 朝のユーザーの確認: 置く → 校正（最後に盤を立てる）→ 被験者番号 00・体重を入れて開始 → 2 s 静止 → 押し上げ 5 回 → 回数が 1 回ずつ増え、
  now が回ごとに戻り、帯を跨ぐと状態が変わる → 停止 → `verify_run check`
- 実機でしか分からないこと（朝に伝える）: 置き方、盤を立てる手間、実際の押し上げで回が閉じるか、実際の W_pos が W_0.70 に届く大きさか
  （届かなければ研究上の所見で不具合ではない。合成の見積もりでは肘は S≈0.26 で「不足」に留まりうる）、GPU の発熱後の fps、第 2 モニタ、.app の許可の取り直し

---

## 状態と引き継ぎ（2026-09-24 07:50、ローカルのセッションの持ち時間が切れるためクラウドへ）

ブランチ `murayama/fix-left-right-dynamics`（origin に push 済み、先端 36841ac）。全テスト 1376 passed / 1 skipped（macOS）。

**済み**（すべてこのブランチに取り込み済み）
- 段取り 0: 未コミットだった作業を 8 つにコミット、`m_max_all_merged.csv` を追跡、`HYBRID_*`・`ONE_RM_CSV` を CURATED に
- A1: `app/gauge/thresholds.py`（論文の W_0.70・W_0.85）、`app/gauge/tracker.py`（GaugeTracker・GaugeTicker）、`app/hybrid/gravity.py`、
  `app/hybrid/rep_detector.py`、`energy_pipeline.py` の入れ替え、`app/hybrid/demo_gauge.py`
- A2: dt の修正（`app/hybrid/rep_work.py`）、EKF（`app/hybrid/ekf.py`）、先頭の窓（体格の検査・重力・前腕長・帯）、関所と回の区切り、回の確定、
  出力（kpts3d_raw・aim_torque_vec・cycle_energy・gauge_energy・npy・meta）、骨の長さの見張り、`hybrid_measure` の毎周回の @@GAUGE、`HYBRID_REPLAY` の分岐
- B: `hybrid-raw` の改名、`app/hybrid/gravity_board.py` と校正の最後の盤を立てる段階、`calibration_io.update_meta`、SEED_FILES、Mac カメラの報告、
  `PoseOptions`（`HYBRID_POSE_*`、既定は今と同じ）、`check` の混成の拡張、`tune_ekf --out` の修正（B8・B10・B12・B13 と ROI の配線は未実装）
- 統合: `app/hybrid/replay.py`・`app/runners/hybrid_replay.py`（記録の再生）、`tools/synth_session.py`（合成の押し上げ）、`tools/gauge_view.py`
  （再生でゲージ窓を撮る）、`/code-review --fix`（子の側、11 件修正）、片方の肘が見えないときの窓の予備、計測画面の最小の配線（`app/shell/page_measure.py`）
- 検証: 実機の記録を先頭から流すと体格の検査で終了コード 3（R1）。合成の押し上げ 10 回で回 10・肘の W_pos 約 19.7 J・手首 約 6.9 J・帯（被験者 00）
  肘 47.7〜57.2 J・手首 9.7〜11.6 J（R3）、処理の中央値 5.8 ms。ゲージ窓の画像で回・帯・前回・目標帯・過負荷の表示を確認
- UI（別セッション）: protocol・model・scene・pictograms・widget・window・worker の gauge_frame・settings（GAUGE_SHOW_JOULES）まで取り込み済み

**残り（クラウドへ）**
1. `origin/murayama/subject-gauge` を取り込む（UI 側は Task 13〜16 をクラウドで仕上げ中。完了の合図は件名 `docs(gauge): 実装の報告書（統合担当）` のコミット）。
   `app/shell/page_measure.py` は UI 側の版を取る（こちらの最小の配線は置き換わる。ゲージ窓を開いて gauge_frame をつなぐことが UI 側の版にあるか確かめる）。
   CURATED・`tests/test_cross_platform.py` の一覧は両方の行を残す
2. 取り込んだ UI の部分に `/code-review --fix`、今夜の差分全体（`64590fd..HEAD`）に `/simplify`。全テスト
3. 報告書 `docs/superpowers/reports/2026-09-24-hybrid-port-report.md` を書いて push（完了の合図は件名 `docs(hybrid): USB 経路の機能の移植の報告書` のコミット）
4. Mac でしかできないこと（報告書に書く）: .app の作り直し（`./packaging/build_macos.sh`）、GUI で Mac＋Pixel を選んで開始しゲージ窓が出るか、朝の実機の手順（`docs/hybrid_field_run.md` §0）
