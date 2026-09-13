# 設計: EKF の自己チューニング

| | |
|---|---|
| 状態 | **未着手**（計画のみ。コードは 1 行も変更していない） |
| 更新 | 2026-09-13（現状のコードに合わせて再計画） |
| 基準コミット | `3a6a48c`（本文の行番号はこの時点のもの。**ずれていたらシンボル名で探すこと**） |
| 起点ブランチ | `murayama/app-framework` |
| 想定ブランチ | `murayama/ekf-self-tuning` |

## 再開のしかた

このファイルを読ませて「この設計どおりに実装して」と指示すれば続きから始められる。
**「実装順序」の S1 から順に進めること。** S6 だけは実機での収録
（間引きなしと 4Hz 間引きの 2 設定）が要る。

---

## 改訂履歴

| 版 | 日付 | 内容 |
|---|---|---|
| 1 | 2026-09-08 | 初版 |
| 2 | 2026-09-08 | 設計レビューの指摘を反映。初版の単位の誤認を訂正 |
| 3 | 2026-09-13 | 現状のコードに合わせて再計画 |

### 版 2: 初版の単位の誤認

初版は **EKF の入力を cm と誤認していた**。実際は、`_triangulate_transform_batch` の**内部**で
`* 0.01` と**軸の入れ替え**が済んでおり、EKF が受け取るのはその出力（m 系）。
実行時に書かれた `*_forearm_*.npy` のベクトル長の中央値は 0.08〜0.19 で、前腕の実長 0.25 m と整合する。

初版の推定に使った `Adjusted 3D Pose/kpts3d_9_20250925_201442.csv` は
**別スクリプトが書いた cm 系・軸入れ替え前**のファイルで、実行時の EKF 入力ではない。

**したがって初版の絶対値はすべて無効。** ただし**比と広がりは単位に依らないので生きている**。

### 版 3: 現状反映

版 2 の本文は、dt 修正（8d40c8d）直前の a08b4d4 時点のコードを見て書かれていた。その後に次が変わった。

- **欠陥 2（`dt = 0.3` が渡る）は 8d40c8d で解消**。EKF には `_DYN_DT` が渡る。
  版 2 の修正案 `_ekf_dt = 1.0 / fps` は間引きを考えておらず、今のコードでは誤り
- `pose_keypoints` が 12 → 16 点になった（8e20586、手の 17〜20）。系列は 36 → **48**
- NumPy 2（2.5.3）の `TypeError` は `EKF_VECTORIZED=0` だけでなく、`run_ekf` を使うもの全体に及ぶ
- `_vel_filt` / `_acc_filt` は下流で使われていない。力学に流れるのは EKF 後の**位置だけ**
- dt 不一致の扱いを「例外」から「合うプロファイルを探し、無ければ既定値で続けて記録」に変えた（決定 6）
- 生 CSV を EKF の**手前**で書くので、`EKF_ENABLE=0` で収録させる必要がなくなった（決定 8）
- 実装順序を組み直した: 生 CSV を先頭へ、実測（旧 Step 0）を推定器の後ろへ、配線をロバストゲートの前へ
- 細部の訂正: `IMPORT_SAFE_MODULES` には `extended_kalman_filter` が既に入っている／既存テストは 247 件／
  「`EKF_VECTORIZED` を消すと `Settings.get` が KeyError」は誤り／テストの手本は `test_dynamics_dt.py`／
  `rot_trans_c1.dat` の由来は断定できない

---

## Context

`master_research_code.py` の `LandmarkEKF`（`:512-660`）は、逆動力学に渡す位置を改善していない。
絡み合った欠陥がある。**どれか 1 つだけ直しても改善しない。**

EKF は USB カメラ経路（`master_research_code.py`）専用で、スマホ経路（`app/runners/network_measure.py`）は使っていない。
EKF の出力のうち下流に流れるのは**位置だけ**で（`kpts_3d.append`（`:3298`）→ `calculate_link_vectors(..., _DYN_DT)`（`:3386`））、
速度は `link_vector_calculator_module.py` が位置の差分から計算し直している。

### 欠陥 1: `q_acc` / `r` が単一定数で、しかも桁が合っていない

既定は `EKF_Q_ACC=1e-3`, `EKF_R=1e-3`（`master_research_code.py:172-173`、m 系）。

`Adjusted 3D Pose/` の CSV 36 系列を最尤推定し、m 系に換算した参考値:

| | 推定 `q_acc` | 推定 `r` (σ) |
|---|---|---|
| 最小 | 1.19e-2 | 4.40e-6 m² (2.10 mm) |
| **中央値** | **1.22e-1** | **2.59e-5 m² (5.09 mm)** |
| 最大 | 2.44e0 | 1.89e-3 m² (43.5 mm) |

> この表は 12 点時代の 36 系列ぶんで、手の 4 点（17〜20）を含まない。

現行既定との差: `r` は中央値の **39 倍**（測定を信用しなさすぎ）、`q_acc` は **1/122**
（動きを固すぎると仮定）。**`q/r` 比は現行 1 に対し推定 4699 で、約 4700 倍鈍い。**

**単一定数では原理的に直らない。** 幅は `r` が **432 倍**、`q_acc` が **205 倍**。
上肢（肩・肘・手首）は良く、車椅子に遮蔽される下肢が一貫して悪い。左右でも差が出る。
**この比と広がりは長さスケールに依らないので、単位の訂正後も有効。**

> **絶対値は S6 で取り直すこと。** 上の表は「幅がこれだけある」ことの根拠であって、
> そのまま同梱既定値にしてはならない。

### 欠陥 2: dt（`dt = 0.3` は解消済み。残りは 2 点）

`config.dt = 0.3` が EKF に渡っていた問題は 8d40c8d で解消し、
`landmark_ekf.step(transformed_p3ds, _DYN_DT)`（`:3294`）になった。
`_DYN_DT` は `config.resolve_dynamics_dt()` が間引き設定から算出する（`:2969`）。残る論点は次の 2 つ。

1. **dt は間引き設定で 8 倍変わる。** 素の既定（`RT_POSE_FIXED_HZ_ON=1`、4Hz 処理）では 8/30 = 0.267 s、
   GUI の既定（`CURATED` で `0`、設定画面で切り替えられる）では 1/30 = 0.0333 s。
   **較正した `(q, r)` は dt ごとに別物として扱う。** 白色ジャークの連続時間モデルなら dt に依らないはずだが、
   実測でイノベーションが白色でない（検証 2）ので、換算して使い回すのは当てにならない
2. **BPF の `fs` が dt と食い違う。** `LandmarkEKF(..., fs=fps)`（`:1906`）の `fps` は config の定数 30 で、
   しかも EKF の生成が `_src_fps`・`_DYN_DT` の算出（`:2950, :2969`）より前にある。素の既定では 8 倍ずれる。
   BPF は既定で無効なので、今は表に出ていない

版 2 で得た教訓は残す: **dt が合っていないと、正しい `(q, r)` を入れても指標上は正常に見えたまま
平滑化がほぼ効かなくなる**（版 2 の実測: dt=0.3 で棄却率 0.37%、正しい dt で 2.57%）。
失敗が静かなので、dt は必ず記録して照合する。

### 欠陥 3: ゲートが「棄却」なので死のスパイラルになる

`gate_std=3.0` で `|y| > 3√S` を捨てる（`_step_vectorized` の `:619-620`）。
捨てると補正されずさらにずれ、また掛かる。版 2 の実測では、現行既定で棄却率 **99.87%**（発散）。

正しい `(q, r)` なら棄却率は 2.57% まで落ちるので、**保険より先に較正が本体**。
ただし連続棄却は残るため、ゲートの形自体を変える（下記 実装 2）。

### 欠陥 4: 欠測中に無制限に外挿し、NaN を捏造値で埋める

一度初期化された系列は、欠測中も predict だけで進み、外挿値を返し続ける
（`_step_vectorized` の `:610-614`。NaN を返すのは未初期化の系列だけ、`:636`）。
しかも**下流は「データが無い」ことを知れない** — 直前で `nan_3d` を数えている（`:3289`）のに、EKF が直後に潰す。

`app/net/sync_buffer.py` は同じ問題を既に避けている（`max_gap_ms=100` を超える穴は補間せず、
サンプル自体を出さない）。EKF は行数を保つ必要があるので、NaN を返す形で同じ考え方を取る。

> 「45 フレームの全欠測」は `Adjusted 3D Pose/` の CSV での観測。
> 実行時経路での欠測長は S6 で測り直すこと。

### 欠陥 5: 再構成スケールが校正手法で変わる

- `calib.py:283-285` の `world_scaling` は **m を返す**。その後 `_triangulate_transform_batch` の `* 0.01`
  （`:2172`、native 経路は `:2164` の `scale=0.01`）が乗るので、
  `calib.py` で新規校正した利用者の実行時座標は **m の 1/100** になる
- `camera_parameters/rot_trans_c1.dat` の T のノルムはちょうど 1.0 で**スケール不定**。
  単位ベクトルを保存する経路は `pose_extrinsic_from_pose.py:159` の `t_unit = t / norm(t)` と
  `calib.py:650-654`（`baseline_m` なし）の 2 つあり、どちら由来かは断定できない

つまり「各利用者が自分の収録から作るテーブル」は**校正のやり直しで無効になる**。

**対策**: プロファイルを**スケール不変**にする。較正時に基準長 `L_cal`（肩–肘の中央値）を
記録し、実行時に先頭数百フレームから `L_run` を測って
`q_used = q_cal·(L_run/L_cal)²`, `r_used = r_cal·(L_run/L_cal)²` とする。
`L_run` が人体としてありえない範囲なら**起動時に失敗させる**
（`r` が小さすぎれば派手に発散するが、**大きすぎると静かに素通りする**ため）。

---

## 決定事項

| # | 項目 | 決定 |
|---|---|---|
| 1 | チューニング方式 | **オフライン最尤推定**（予測誤差分解） |
| 2 | テーブルの作り方 | **較正コマンドを用意し、各自の収録から作る** |
| 3 | 作業範囲 | **自己較正＋位置平滑化まで**。力学は触らない |
| 4 | 粒度 | 16 点 × 3 軸 = **48 系列**、各 `(q_acc, r, gate_std)` |
| 5 | プロファイルのキー | **MediaPipe のランドマーク ID（文字列）＋ 軸名**。位置添字は使わない |
| 6 | プロファイルの dt と実行時の `_DYN_DT` が違う | **dt が合うプロファイル（相対 5% 以内）を探す。無ければ同梱既定値で計測を続け、起動ログとサイドカー JSON に `source: "builtin"`, `reason: "dt_mismatch"` を残す**。計測は止めないが、精度が落ちた試技を後から見分けられる。**帰結として、同梱既定値も間引き設定ごと（最低 1/30 s と 8/30 s の 2 組）に持ち、近い dt のものを使う** |
| 7 | 実行時に BPF が有効 | **プロファイルは使わない**（決定 6 と同じく既定値＋`reason: "bpf_enabled"`）。BPF は既定で無効で、較正側で同じ BPF を再現する複雑さに見合わない |
| 8 | 較正入力の判定 | 生 CSV は **EKF の手前**で書くので、`EKF_ENABLE` や BPF の設定に依らない。判定はサイドカーの `stage: "pre_ekf"` |
| 9 | 行番号の書き方 | シンボル名を主にし、行番号は基準コミット時点の補助として書く |

---

## 実装

### 0. 較正入力を実行時経路から出す

`_triangulate_transform_batch` の直後（`:3288`、EKF の手前）で、`transformed_p3ds` を
`kpts3d_raw_{timestamp}.csv` に**1 フレームごとに追記して flush** する。

- 列: 処理フレーム番号、処理反復の時刻 `start_time`、ランドマーク ID ごとの x/y/z（丸めない）
- **終了時にまとめて書かない。** GUI の停止は `QProcess.terminate()`（`app/runners/worker.py:111`、
  macOS/Linux では SIGTERM）で、リポジトリに SIGTERM ハンドラが無い。Python の既定ではこのとき
  atexit も終了時の保存処理（`:4377-4391`）も走らないので、終了時保存では GUI から止めた試技の生 CSV が残らない
  （コードと Qt/Python の既定動作からの推論。実機未確認）
- 時刻列を持つのは、`RT_DELAY_SKIP_ON=1` のように間隔が変わる設定や、動画の巻き戻し
  （`LOOP_FILE_PLAYBACK`）を推定器が検出できるようにするため

併せて**起動時に**プロヴェナンスのサイドカー JSON を書く:
`stage: "pre_ekf"`、`_DYN_DT` とその由来文字列、`RT_POSE_FIXED_HZ_ON` / `RT_DELAY_SKIP_ON` / `SKIP_FRAMES`、
`_src_fps`、`EKF_*` 全部、ランドマーク ID の並び、camera_parameters のパス、git commit。

これで単位・軸・dt が実行時と**構造的に一致**し、欠陥 1・2・5 と軸の対応がまとめて解決する。

> 版 2 は「`EKF_ENABLE=0` で収録させる」ために `EKF_ENABLE` を `CURATED` に出す案だった
> （`as_env()` が既定値 `"1"` を必ず子プロセスに注入するため）。生 CSV を EKF の手前で書くことで不要になった。

### 1. `LandmarkEKF` を `extended_kalman_filter.py` へ移す

`master_research_code.py:512-660`。import 副作用のため単体テストが書けない。
移設先は `KNOWN_ISSUES.md §4-3` の修正案どおり。

**自由変数は 8 個だけ**（AST で確認し、その 8 個だけの名前空間で `exec` して動作確認済み）:
`EKFConfig, EKF_VECTORIZED, ExtendedKalman1D, _SCIPY_OK, butter, lfilter, lfilter_zi, np`。
`pose_keypoints` も `fps` も参照していない（`fps` は `fs` 引数、`dt` は `step` の引数で受ける）。**移設は安全。**

- `_SCIPY_OK` は `:251, :408, :431` でも使うので**元ファイルに残す**。移設先には独自の try/except ガードを新設する
  （`tests/test_cross_platform.py:34` の `IMPORT_SAFE_MODULES` に `extended_kalman_filter` が入っており、
  import で例外が出るとテストが落ちる）
- `EKF_VECTORIZED` は引数化するが、**env 変数自体は残す**。スキーマから消えると `as_env()` が渡さなくなり、
  GUI 経由でもシェルの値が漏れて効くようになる
- `master_research_code.py:35` の import は `LandmarkEKF` に置き換え、`ExtendedKalman1D` は不要になる

**`EKFConfig` は触らない。** `LandmarkEKF` 専用の設定型を新設する
（`q_acc: (N,)`, `r: (N,)`, `gate_std: (N,)`, `dt_expected`, `scale_ref`）。理由:

- `run_ekf` / `ExtendedKalmanND` / `tmp_filter_pose_torque.py:23` の契約を壊さない
- **`h_fn` をそもそも持たせない**のが §4-3 への最強の答え（設定できないものは無視できない）
- ndarray を dataclass に入れると `__eq__` が ValueError になる

### 2. 欠陥 2〜4 の修正

**NumPy 2 対応**: `ExtendedKalman1D.step` の `S = float(H @ self.P @ H.T + self.cfg.r)`
（`extended_kalman_filter.py:94`）は (1,1) 配列の `float()` で `TypeError` になる（NumPy 2.5.3 で再現）。
`EKF_VECTORIZED=0` の逐次パスだけでなく、`run_ekf` と `tmp_filter_pose_torque.py` も動かない。
スカラーを取り出す 1 行の修正で、逐次版とベクトル化版は外れ値・欠測・dt 2 通り・ゲート有無で
**最大相対差 1.4e-14** まで一致する（メモリ上で修正を当てて確認済み）。
併せて `_measure`（`:68`）が `h_fn` と `h_jac_fn` の片方だけだと黙って恒等観測に落ちる件を例外にする。

**F/Q の共有**: 状態遷移 F と過程雑音 Q の組み立てが `ExtendedKalman1D._predict`
（`extended_kalman_filter.py:51-63`）と `_step_vectorized`（`master_research_code.py:601-607`）に重複している。
推定器（実装 3）が 3 つ目を書くと、推定した q が実行時と違う形の Q に入り得るので、関数に切り出して 3 者で共有する。

**`dt` と BPF の `fs`**: dt は `_DYN_DT` のままでよい。EKF の生成（`:1902-1910`）を `_DYN_DT` の算出（`:2969`）の
後ろへ移し、`fs = 1 / _DYN_DT` を渡す（`landmark_ekf` を使うのは `:3292` だけなので移動は安全）。
プロファイルに較正時の dt を記録し、不一致時は決定 6 に従う。

**ゲート → 分散膨張型のロバスト更新**: `|y| > c√S` のとき棄却せず `S' = y²/c²`
（等価に `r' = r·(|y|/(c√S))²`）として更新する。Huber 型 M 推定に相当し、
連続的に劣化し、**デッドロックしない**のでカウンタも要らない。
`_step_vectorized` の update ブロック（`:613-633`）の数行。

> レビューは「P を膨らませて強制受理」を**却下**した。P はゲート判定そのものに
> 使われる（`:619-620`）ので自己参照ループになり、1 回のつもりが連鎖する。

**較正が先、ゲートは後。** この更新の補正量は `|y|` に反比例して縮むので、
`q` が小さすぎるままゲートだけ変えても追従不足は直らない。

**`gate_std` も系列別に較正する**。ゲート無しで推定した `r` は外れ値を吸収して過大に
なるため、固定 3.0 だと門が広くなる。正規化イノベーション `|y|/√S` の
99.7 パーセンタイルをプロファイルに書く。

**欠測の上限**: `max_gap` を超えたら **NaN を返し、次の観測で再初期化**する。
dt が設定で 8 倍変わるので、`max_gap` は**秒**で持ち dt で換算する。既定値は S6 の欠測長の分布から決める。
**NaN が力学計算とゲージに流れても落ちないことを確かめる**（今は EKF が外挿で埋めているので NaN が流れておらず、
`link_vector_calculator_module.py` の NaN の扱いは部分的）。

**広い `except`（`:3295-3297`）を狭める**。形状不整合などバグ由来の例外は再送出する。
今は例外時に三角測量の生の値が黙って下流に流れ、ログも 120 フレームに 1 回しか出ない
（`EKF_VECTORIZED=0` は NumPy 2 でこの経路に落ちて無音死している）。

### 3. `app/tuning/` — 推定器

```
app/tuning/ekf_likelihood.py   1系列ぶんの対数尤度（ゲート無しの素のKF）
app/tuning/ekf_estimate.py     初期値 → Nelder-Mead で (log q, log r)。生 CSV 1 本から 48 系列の表を出す報告 CLI
app/tuning/ekf_profile.py      読み書き・dt による探索・フォールバック・妥当性検査
app/runners/tune_ekf.py        生 CSV からプロファイルを書く CLI（GUI からも呼ぶ）
```

尤度は予測誤差分解 `ℓ = -½ Σ [log(2π S_k) + y_k²/S_k]`。
推定中はゲートを**完全に無効**にする（分散膨張も掛けない）。
欠測は predict のみ行い尤度項を落とす。F/Q は実行時 EKF と共有した関数を使う（実装 2）。

- **初期値**: `r₀ = Var(∇³z)/20` は歪む（実測で最大 8 倍過大、低ノイズ系列では
  符号反転もあった）。**ラグ 3 の自己共分散 `−γ₃(∇³z)` のほうが素直**。
  どちらにせよ**探索範囲は r₀ の ±2 桁以上取る**（狭いと「端に張り付き」が誤発火）。
  **r₀ をフォールバック値には使わない**
- 最適化は `scipy.optimize.minimize(method="Nelder-Mead")` を `(log₁₀ q, log₁₀ r)` 上で
- **所要時間**: 版 2 の実測で 36 系列 61 秒（3859 フレーム、逐次）。48 系列なら約 80 秒の見込み。並列化しない
- **出力に必ずラグ 1〜5 の正規化イノベーション自己相関を含める**（下記 検証 2）
- 時刻列から間隔のばらつきを調べ、`_DYN_DT` から外れた区間（`RT_DELAY_SKIP_ON`、動画の巻き戻し）は区切って扱う

**フォールバック階層**: 有効サンプル 300 未満、または端に張り付き → 同関節の他軸の
中央値 → 全系列の中央値 → **dt ごとの同梱既定値**。**適用した段階を系列ごとに記録する**
（`source: "fit" | "axis_median" | "global_median" | "builtin"`、`builtin` のときは `reason` も）。
4Hz 間引き（dt = 0.267 s）では、有効サンプル 300 に 80 秒以上（欠測分を足す）の収録が要る。

### 4. プロファイルのスキーマ

```jsonc
{
  "schema_version": 1,
  "frame": "runtime",              // _triangulate_transform_batch の軸入れ替え後
  "unit": "m",
  "dt": 0.03333, "fps": 30.0,      // dt は較正に使った生 CSV の _DYN_DT
  "bpf": {"low": 0.0, "high": 0.0, "order": 2},
  "scale_ref": {"pair": [12, 14], "median_len": 0.327},
  "source": {"csv": "...", "n_frames": 3859, "git": "...", "created": "..."},
  "series": {
    "16": {"x": {"q_acc": ..., "r": ..., "gate_std": ...,
                 "n_eff": ..., "nis": ..., "rho1": ..., "source": "fit"},
           "y": {...}, "z": {...}}
  }
}
```

- `schema_version` / `frame` / `unit` が合わなければ**例外**（壊れたファイル）
- `dt` が実行時の `_DYN_DT` と相対 5% 以上違えば、そのファイルは使わずに別のプロファイルを探す。
  `EKF_PROFILE` はファイルかディレクトリを受け、ディレクトリなら dt が合うものを選ぶ。
  合うものが無ければ決定 6 のとおり同梱既定値で動く
- 実行時に BPF が有効ならプロファイルは使わない（決定 7）
- `scale_ref` は自動補正。無い系列だけフォールバックし、**ファイル全体は捨てない**

### 5. 実行時への配線

- **優先順位を明示する。** `as_env()`（`app/core/settings.py:248-260`）は `EKF_Q_ACC=1e-3` / `EKF_R=1e-3` を
  **必ず**子プロセスに注入する（`app/entry.py:137-138` で `os.environ` を上書き）ので、
  「env → プロファイル」の順にすると **GUI 経由ではプロファイルが一生使われない**。
  **プロファイルが解決できたらプロファイルが勝つ**。スカラーは無い時のみ
- `CURATED`（`app/core/settings.py:63-148`）に `EKF_PROFILE`（`type="str"`）を追加する。
  env を足したら `tools/extract_env_schema.py` でスキーマを再生成する
  （走査対象は `master_research_code.py` と `config.py` だけ、`:33-36`。`settings_schema.json` は手編集しない）
- **`EKF_Q_ACC` / `EKF_R` は UI に出さない。** `app/shell/widgets.py:120` が
  `setDecimals(4)` なので、**推定値 r≈2.6e-5 は 0 に丸められる**
- 起動ログに系列別の `source` の内訳を出す
- UI は `app/shell/page_analyze.py` の `TASKS`（`:39-76`）に `AnalysisTask` を 1 項目足すだけ
  （入力パスが必須なので、CLI は生 CSV を 1 つ受ける）。
  新ページは作らない（`tests/test_shell_smoke.py:56-57` がページ数 3 をハードコード）
- 起動は `--role script --module app.runners.tune_ekf`。**`app/entry.py` は変更しない**
  （`resolve_module` がドット名を素通しする。`app/entry.py:95-98`）

---

## 実装順序

旧 Step 0（前提の確定）は推定器の後ろ（S6）へ移した。単位・軸・dt はコードを読んで確定済みで、
残る実測（48 系列の参考値、欠測長、手の点の質）は推定器が無いと取れず、S1〜S5 はどれもその値に依存しない。
生 CSV（S1）を先頭に置くので、以降の作業中に録ったデータがすべて実測の材料になる。

| # | 内容 | 主に触るファイル | 完了条件 | 実行時の挙動 |
|---|---|---|---|---|
| S1 | 生 CSV のフレームごとの追記と、起動時のサイドカー JSON（実装 0） | `master_research_code.py` | 途中で止めても CSV が残る／生と EKF 後で NaN 数が違う／時刻列から実測の間隔が読める | 変えない |
| S2 | NumPy 2 対応と `h_fn`/`h_jac_fn` 片側の例外化（実装 2） | `extended_kalman_filter.py` | `run_ekf` が 2 ステップ以上回る／片側で例外 | 逐次パスが動くようになるだけ |
| S3 | 逐次とベクトル化の一致テスト（移設前。AST で抜き出して exec） | `tests/test_landmark_ekf.py`（新規） | 外れ値・欠測・dt 2 通りで相対差 1e-12 以下 | 変えない |
| S4 | `LandmarkEKF` の純粋な移設（実装 1、scipy ガード付き） | `extended_kalman_filter.py`、`master_research_code.py` | S3 を import に置き換えて緑のまま／`test_cross_platform` が緑 | 変えない |
| S5 | F/Q の共有関数化 → 推定器と報告 CLI（実装 3） | `extended_kalman_filter.py`、`app/tuning/ekf_likelihood.py`、`ekf_estimate.py` | S3 が緑のまま／合成データで既知の `(q, r)` を 1 桁以内で回収（dt = 1/30 と 0.267 の両方） | 変えない |
| **S6** | **実測（人手の収録。コード変更なし）**: 間引きなしと 4Hz 間引きの**両方**で収録し、S5 の CLI で推定。参考値・欠測長の分布・手の点の質・白色性・2 設定の差をこのメモに追記する。**端に張り付く系列が過半数、または白色性が大きく崩れるなら、S7 以降に進む前に設計を見直す** | このメモ | dt ごとの同梱既定値、`max_gap` の既定値、フォールバック閾値の根拠が揃う | — |
| S7 | プロファイルの読み書き・dt による探索・フォールバック（実装 3・4）と、プロファイルを書く CLI | `app/tuning/ekf_profile.py`、`app/runners/tune_ekf.py` | 壊れた JSON／dt 違いで別ファイルを選ぶ／合うもの無しで builtin＋`dt_mismatch`／BPF 有効で builtin／未知 ID／系列欠落／S6 の CSV からプロファイルが出る | 変えない |
| S8 | (N,) 配列化（下表の 6 箇所、専用の設定型） | `extended_kalman_filter.py` | 全系列同値でスカラー版と一致／系列別で逐次版と一致 | 同値なら変えない |
| S9 | 実行時への配線（実装 5 と、実装 2 の dt・`fs`・except） | `master_research_code.py`、`app/core/settings.py`、`settings_schema.json`（再生成） | 1 試技で RMS 差と系列別棄却率が S6 の期待範囲／間引きを切り替えても止まらず、`source` が記録に残る | **ここから変わる**（較正値＋従来のゲート） |
| S10 | ロバスト更新、系列別 `gate_std`、欠測上限（実装 2） | `extended_kalman_filter.py` | 外れ値 1 発で引きずられず復帰／連続棄却なし／長い欠測で NaN／NaN を受けても力学計算とゲージが落ちない | 変わる |
| S11 | `TASKS` に 1 項目（S7 の CLI を呼ぶ） | `app/shell/page_analyze.py` | GUI から生 CSV を選ぶとプロファイルが出る | — |

**依存関係**: S1 → S5（CSV の形式）／S2 → S3 → S4 → S5（F/Q の共有）／S4 → S8／
S1 のデータ＋S5 → S6 → S7（既定値）→ S9／S8 → S9 → S10 → S11。

- S1〜S8 は実行時の既定挙動を変えないので、どこでも中断できる
- 収録の予定が遅れる場合、**S8 は S6 を待たずに先行できる**（S7 の既定値と S9 以降は S6 待ち）
- **配線（S9）をロバストゲート（S10）より先にする。** 逆順だと、1e-3 の既定値のままゲートだけが変わる期間ができる
- S7 で CLI まで作るのは、S9 の配線を S6 の収録から作ったプロファイルで確かめるため

### (N,) 配列化で直すのは 6 箇所

| 場所 | 内容 |
|---|---|
| `LandmarkEKF.__init__` の逐次パス（`:542`） | 全系列が同じ cfg を共有。直さないと 2 実装が黙って食い違う |
| `_step_vectorized` の `_fq` キャッシュ（`:596-609`） | `(dt, F, Qbase)` に分け、**`Q` に `pi` の添字を付ける**（忘れると形は合って値だけ静かに間違う） |
| `:617` | `S = P[ui,0,0] + r[ui]` |
| `:633` | `r[si][:,None,None] * (...)` |
| `:619-620` | `gate_std` を系列別にするなら `gate[ui][ok] * ...` |
| `extended_kalman_filter.py:28-36, 111-112, 169` | `EKFConfig` に ndarray を入れない（上記 実装 1） |

**`_A` の使い回し（`:539`）は安全**（`A[:,:,0]` だけを書き換え、列 1,2 は単位行列のまま）。変更不要。

---

## 検証

| # | 検証 | 合格条件 |
|---|---|---|
| 1 | 合成データの回復 | 既知の `(q, r)` から **1 桁以内**で回収。**dt = 1/30 と 0.267 の両方** |
| 2 | **イノベーションの白色性** | ラグ 1〜5 の自己相関。`|ρ₁| > 0.3` で警告。**版 2 の実測で 0.21〜0.41 と白色でない**（等加速度＋白色観測誤差というモデルが誤指定。ML は r を過小・q を過大に見積もる＝dt 問題と同じ方向に効く） |
| 3 | ホールドアウト尤度 | 学習に使わない区間で現行既定より高い |
| 4 | 出力が観測レンジを超えない | 実データ 48 系列で EKF 出力が観測の範囲＋余裕に収まる |
| 5 | 逐次／ベクトル化の一致 | S3 で相対差 1e-12 以下。系列別値でも一致（S8） |
| 6 | 連続棄却が起きない | ロバスト更新により更新を連続スキップしない |
| 7 | 長い欠測で NaN が出る | 捏造値ではなく NaN が返り、それを受けても力学計算とゲージが落ちない |
| 8 | 較正入力の検証 | サイドカーに `stage: "pre_ekf"` の無い CSV を拒否 |
| 9 | 不一致の扱い | dt・BPF は「別プロファイルの探索 → 既定値＋`reason` の記録」。`schema_version` / `frame` / `unit` は例外 |
| 10 | スケール不変性 | `L_run/L_cal` が 100 倍違っても同じ平滑化になる |
| 11 | 非破壊 | `pytest tests` で既存 247 件（246 passed / 1 skipped）が通る |
| 12 | 実機 | 30fps を維持して完走（並列 grab 後の 33.3ms/ループ） |
| 13 | 間引きの切り替え | 設定画面で間引きを切り替えても計測が止まらず、`source` が記録に残る |
| 14 | 途中停止 | GUI から途中で止めても、生 CSV とサイドカーが残る |

**NIS を主要な合格条件にしない。** 学習区間の NIS = 1.000 は恒等式
（`(q,r)` を同時にスケールしても状態推定は変わらず尤度だけ動くので ML 解は必ず 1 に釘付け）。
検証区間の値のみ参考にする。

**`σ(∇²z)/σ(Δz)` の判定は 1.73 基準にしない。** 版 2 の実測（36 系列）で分布は
**1.256〜1.723**。1.73 は純白色雑音の極限で、SNR が良い系列ほど下がるため、
「1.7 に近いこと」を条件にすると良質な生データを弾く。**「1.0 未満なら拒否」**が妥当。
ただし判定の主役はサイドカー JSON（事実）であって統計ではない。

### テストの置き方

`tests/test_dynamics_dt.py` が手本。モジュール docstring に**なぜこのテストがあるか**を実測値と根拠つきで書き、
`class TestXxx:` でグループ化し、assert に日本語メッセージを付ける。
`tests/conftest.py` は Qt と matplotlib の環境変数を設定するだけでフィクスチャを提供しないので、
フィクスチャは各テストファイル内で定義する。

- `tests/test_cross_platform.py` の `IMPORT_SAFE_MODULES` に `app.tuning.*` を追加する（`extended_kalman_filter` は既にある）
- 移設前の一致テスト（S3）は、`master_research_code.py` を import できないので AST で `LandmarkEKF` を抜き出して exec する
- 全体の実行は `pytest tests`。リポジトリ直下で引数なしの `pytest` は、無関係なスクリプトまで収集して INTERNALERROR になる

---

## 触らないもの

- `utils_dynamic.py` の逆動力学、`KNOWN_ISSUES.md` §1/§2
- `_DYN_DT` の算出（`config.resolve_dynamics_dt`）と、`calculate_link_vectors`・エネルギー積分での利用
- `app/entry.py`、新しい GUI ページ
- `EKFConfig` / `run_ekf` / `ExtendedKalmanND` の既存契約（NumPy 2 対応と `h_fn` 片側の例外化を除く）
- スマホ経路（`app/runners/network_measure.py`）。EKF を使っていない
- `app/core/settings_schema.json` の手編集（再生成は `tools/extract_env_schema.py` で行う）

## スコープ外だが記録しておく別件

- **（優先度高）GUI から停止すると、終了時の CSV（`kpts3d_`・トルク）が書かれない可能性が高い。**
  `app/runners/worker.py:101-116` の docstring は「CSV の書き出しを待つ」と書くが、`terminate()` は SIGTERM で、
  リポジトリに SIGTERM ハンドラが無い。実機で確認し、必要ならハンドラでループを抜けるようにする。EKF とは独立に直すべき
- `[CAMDIAG]`（`master_research_code.py:3090-3113`）の `gap_ms` / `eff_fps` は
  フレーム間隔ではなく**診断ログの出力間隔**を測っている（`_cam0_last_ts` が診断ブロック内の `:3096`
  でしか更新されない）。カメラを疑ったとき誤診を招く
- `LandmarkEKF.__init__` の `lfilter_zi(b, a)`（`:554-555`）を `× x[0]` していないので、BPF 初段に過渡応答が乗る
- `kpts3d_{timestamp}.csv` の保存処理のコメントが「(12, 3)」のまま（`:4377`）。この CSV は EKF 後の値で、4 桁に丸めている
- `tests/test_network_measure.py:28-35` の docstring は `camera_parameters/*.dat` の並進を「cm 単位」と書くが、
  実ファイルの T のノルムは 1.0（欠陥 5）
- `app/core/settings_schema.json` の `lines` が、現在のソースより約 5 行古い
- `KNOWN_ISSUES.md` §4-3 の行番号 `:516` が古く、NumPy 2 で逐次側が動かない件が書かれていない
- `tmp_filter_pose_torque.py:93` は補間前のデータを渡しており、NaN が `None` に変換されないので欠測として扱われない
- 将来 PyInstaller で固める際、`runpy` 経由の解析スクリプトは hidden import が必要。
  `packaging/app.spec` はまだリポジトリに存在しない（`app/core/resources.py` と `requirements_dev.txt` は前提にしている）
