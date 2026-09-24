# プロジェクト全体のコードレビュー（2026-09-24）

対象: ブランチ `murayama/fix-left-right-dynamics`（レビューの土台 `fb4f3ae`）。範囲は「値に効くコード全部」:
計測アプリ `app/`、道具 `tools/`、本体 `master_research_code.py`、app と論文の再計算が使う直下のモジュール、Pixel のアプリ `mobile/`。
`tmp_*`・`plot_*` の使い捨ては範囲外。8 つの領域に分けて並行でレビューし、指摘はできる限り再現して確かめた
（確かめ用の台本は各担当の作業場所にあり、リポジトリには入れていない）。`KNOWN_ISSUES.md` に既にある問題は数えていない。

**扱いの方針**（ユーザーの決定）: 報告し、再現と試験で確かめられた不具合は直す。本体 `master_research_code.py` と
論文の値に効く式（共有の `config.py`・`push_up_model.py`・`utils*.py`・`link_vector_calculator_module.py`・`energy_pipeline.py`
と再計算のスクリプト）は**報告だけ**にして直さない。Pixel のアプリ（Kotlin）はこの環境に Android SDK が無くビルド・試験が
できないので報告だけ。

## 1. 結論

- **論文の値**: 本体（リアルタイム経路）の不具合は論文の値に効いていない（論文の値はオフラインの再計算で出ている）。
  一方、**再計算のスクリプトに論文の値を動かす指摘が 3 つある**（§3.1 の R1〜R3: 被験者 9 のサイクルの区切り、全員 60 kg、
  GCVSPL が GCV を使っていない）。どれも研究上の判断が要るので、直さずに報告する
- **USB 経路のライブ計測**（本体）: 値を黙って狂わせる高が 3 つ（盤の短辺からの重力の符号、解像度より前の切り出し、体重 60 kg 固定）。
  本体は直さない方針なので §3.2 に判断の材料をまとめた
- **混成の計測アプリ**: 朝の実機に効く高・中を中心に直した（§2。回の区切りの基準、見えなかった腕、Mac のカメラの照合、
  同期の保持時間、未来の時刻、接続表示、凍結版の校正など）
- **Pixel のアプリ**: 高なし。電文の形・座標・時計の式は PC と一致。中 2（送信の数え方、時刻の再同期）

## 2. 直したもの

（統合の後に、コミットと確かめ方を記入する）

## 3. 直さずに報告するもの（判断が要る）

### 3.1 論文の再計算のスクリプト

再計算の道筋: `pose_lowpass.py` → `compute_torque_from_pose.py`（`push_up_model`・`utils`・`utils_dynamic`・`config`）→
`auto_detect_cycles_pose_ranges.py`（`detect_cycles_joint11y_threshold`）→ 受領した `m_max_all_merged.csv` →
`compute_cycle_energy_elbow_wrist.py`。レビューでは §6-1 の最新表を小数 3 桁まで再現してから比べた。
上流の `Adjusted 3D Pose/*.csv` を作ったスクリプトはリポジトリに無い（被験者 9 だけ本体の kpts3d）。

| # | 重さ | 場所 | 内容 | 効く値 |
|---|---|---|---|---|
| R1 | 高 | `auto_detect_cycles_pose_ranges.py:89-136`・`:311`（範囲 500〜） | 被験者 9 で安静の平らな区間（谷の深さ 0.016 m）を 1 サイクルと数え、本物の持ち上げ（底 662）を落とす。山の間の谷の深さを確かめていない。合成でも再現 | 区切り直すと被験者 9 の右肘 0.529→0.544、右手首 0.524→0.574、左肘 0.560→0.615、左手首 0.262→0.271。7 名の右の平均 0.861→約 0.866 |
| R2 | 高 | `compute_cycle_energy_elbow_wrist.py:261,342`・`compute_torque_from_pose.py:131` | 全被験者の体重を 60 kg に固定。被験者ごとの体重を渡す口が無い。スコアはほぼ体重に比例（被験者 3 を 70 kg で ×1.153）。論文の分母は 65 kg だった（§6-6） | すべてのスコア（被験者ごとに 体重/60 倍程度）。被験者の体重の表がリポジトリに無く、実際のずれは未確認 |
| R3 | 高（論文系列が通っていれば） | `smooth_pose_gcvspl.py:76,96,49` | `gcvspline.gcv_spline` は存在せず必ず例外 → `UnivariateSpline(s=N×1e-3)` に黙って落ちる。強さが入力の単位で 100 倍変わる（被験者 7 の m と cm でスコア 5〜14 倍）。今の pandas では :49 で落ちて動かない | 論文 0.99 の GCVSPL 系列（経路は §6-6 で未確認） |
| R4 | 中 | `compute_cycle_energy_elbow_wrist.py:79-89`・`compute_cycle_noise_contrib.py:70-76` | 旧来名 `<stem>_torque_lpf.csv`（倍率 0.01）を今の名前より優先し、既定の置き場は旧来の出力先。meta の版・倍率を見ない（0.192→0.0019 に黙って） | 受領データを `output_data/` に写して既定で回したとき |
| R5 | 中 | `compute_cycle_energy_elbow_wrist.py:48-55,279-280` | `kpts3d_subjectN_*` の番号が読めず黙って飛ばす | GCVSPL 系列を回し直すと全員消える |
| R6 | 中 | `annotate_cycles_in_csv.py:134,95` | `cycle_index` を 0 から振る（スコアが最初のサイクルを捨てる）、|y| の中央値 > 1 で cm と誤認（m のデータで 0 サイクル） | 判断待ちの 3_1・4_0・5_stereo を足すとき |
| R7〜R10 | 低 | `pose_lowpass.py:96-103`（−1 を LPF 前に NaN にしない、今の入力は NaN で影響なし）、`compute_cycle_energy_elbow_wrist.py:309-322`（cm の 2 本で上向きの基準、該当 0 件）、`compute_wrist_cycle_work_from_npy.py:231-252`（道筋外。右手首に左の 1RM）、`stereo_triangulate_pose.py:219,324`（間引き率を残さない） | | |

確かめて問題なし: dt 一定の前提（20 fps でも 0.2%）、長さの倍率、DLL と numpy、左右、行の対応、5_1、サイクル内の欠測。
関係する既知: 分母の積分範囲（R-5 と §6-6 が食い違ったまま）。

**推奨**: R1 と R2 はスコアの表を作り直す前に決める（R1 は谷の深さの条件と 1 サイクル 1 持ち上げの検査、R2 は被験者ごとの体重の表）。
R3 は論文の GCVSPL 系列の生成経路（§6-6）を確かめてから。

### 3.2 本体（USB 経路のリアルタイム計測）と共有のモジュール

| # | 重さ | 場所 | 内容 |
|---|---|---|---|
| U1 | 高 | `master_research_code.py:1533-1546`・`calib.py:791-806` | 重力を盤の短辺のファイル（校正の全ビューの中央値、符号をそろえない）から決め、体幹からの推定より優先（既定 `GRAVITY_FROM_CHECKERBOARD_SHORT=1`、GUI の校正が毎回書く）。180° 逆に持つと E+ と E− が入れ替わり、縦長で肘の仕事 −86%。既存の記録はファイル名 `_gZ+`・`_gX±` が疑わしい |
| U2 | 高 | `master_research_code.py:1596,1760,2050-2077,2505-2507,2818` | 解像度の設定より前に横の切り出し範囲と初期 ROI を最初のフレームから決める。既定 640×480 のカメラで `CAM_WIDTH=1280` にすると画像の 41.6% しか推定に渡らず人が外れる（`[CAMDIAG]` は 1280x720 と出る） |
| U3 | 高 | `config.py:12,16-19`・`master_research_code.py:3128-3131,3186-3194` | 力学が体重を読まず常に 60 kg（`BODY_MASS_KG` はゲージの閾値と JSON だけ）。90 kg で約 33% 小さい。混成とは食い違う |
| U4 | 中 | `master_research_code.py:2643-2644` | Ctrl-C（SIGINT）で終了時の CSV がすべて失われる（SIGTERM だけ扱う） |
| U5 | 中 | `master_research_code.py:3402,2993-3000`・`utils.py:571-592` | `rise_to_rise` で最初の押し上げがサイクルの記録に入らない（N 回で N−1 件、`gauge_energy` と 1 ずれ） |
| U6 | 中 | `master_research_code.py:2679-2699,2778-2795` | 片方のカメラの grab が 1 回失敗しただけで「Video ended」終了コード 0 |
| U7 | 中 | `utils.py:87-96`・`video_io.py:57-94` | 入力がファイル（自動フォールバック・サンプル・デバイス名）だと別のカメラの校正 `Param_for_MYvideo` で三角測量（腕の長さ 139 倍） |
| U8 | 中 | `utils_dynamic.py:81` | `calculate_inertia_tensor` が長さ NaN を黙って NaN のテンソルに。USB で先頭 30 フレームに姿勢が取れないと以後その腕のトルク NaN（混成の側は §2 で呼ぶ側を直した） |
| U9 | 低〜中 | `energy_pipeline.py:132,164,172,211-213` | 適応カットオフの f0 推定が 0.3 Hz を分解できず 0.5 Hz（`E_FC_ADAPTIVE_ON=1` のときだけ） |
| U10 | 低 | `config.py:189`・`master_research_code.py:3109-3111,2708,3026-3029` | 固定 Hz の暖機・`SKIP_FRAMES` とバーストで処理間隔が dt と食い違う（既定外の設定） |
| U11 | 低 | `push_up_model.py:193-214`・`utils.py:477` | 肘が伸び切ると局所 y 軸が雑音で決まる（屈曲 1° で P ×0.60。実データで 0〜6% のフレーム、5_1 右は 44%）。論文・USB・混成 |
| U12 | 低 | `link_vector_calculator_module.py:113-129,148-151` | 欠測で ω を 0 にし ω̇ にスパイク（±0.14 N·m） |
| U13 | 低 | `master_research_code.py:1269-1301`・`Gauge_display.py:434-444` | USB のゲージの目標帯が描かれない（角度の式が塗りと違う） |
| U14 | 低 | `master_research_code.py:3809-3816,3849-3850` | `kpts3d` と `aim_torque` の `frame` の起点（29 行ずれ）と列名の意味が違い、`prepare_wrist_inputs.py` が別の前腕を黙って読む |
| U15 | 低 | `master_research_code.py:472-504` | デモ用ゲージが古い関節の並びと無い world ランドマークを参照（コードの既定 `DEMO_MONO_GAUGE_ON=1`） |
| U16 | 低 | `realtime_kalman_filter.py:120,73-74,159-177` | 共分散の式の誤り（分析スクリプトだけ） |
| U17 | 低・潜在 | `config.py:348-349` | 大腿のリンクの向きが規約と逆（今は力学に使っていない） |

確かめて問題なし: ニュートン・オイラーの式（C++ 版も）、慣性回帰式、重心と角速度、局所座標系の右手系と左右の鏡映、
重力の推定、`LandmarkEKF`、EKF の較正の式、点の並び、`energy_pipeline` の LPF と再標本化、`config` の係数、
`aim_torque` の列、停止ファイル・SIGTERM での書き出し、既定経路の積分の配管。

**推奨**: USB で本番の計測をするなら、少なくとも U1（`GRAVITY_FROM_CHECKERBOARD_SHORT=0` で避けられる）・U2（カメラの既定の
解像度を校正と同じにしておく）・U3 を先に直す。U3 は論文の再計算（R2）とも同じ根なので一緒に決める。

### 3.3 Pixel のアプリ（`mobile/`、読んで確かめた）

| # | 重さ | 場所 | 内容 |
|---|---|---|---|
| M1 | 中 | `net/SensorClient.kt:226-234` | 「送信」「破棄」は OkHttp の待ち行列に積めた数（`send()` は 16 MiB 超でしか false）。詰まりは遅延として PC の `dropped_late` に化け、README の「送信 < 人」は起きない。案: `queueSize()` を見て捨てる・表示 |
| M2 | 中 | `net/TimeSync.kt:66-72` | 30 s ごとの再同期が、往復時間が前回よりずっと悪くても時刻のずれを上書き → `t_capture_ns` が段状に数〜十数 ms 跳ぶ |
| M3 | 低〜中 | `camera/CameraSetup.kt:130-145` | 自動露出・WB を収束前の最初の要求からロック |
| M4 | 低〜中 | `MainActivity.kt:351-356,427-455`・`pose/StageRates.kt:110-114` | カメラが開けない表示が 1 s で上書き、カメラ 0 でヒント無し |
| M5〜M10 | 低 | | 最初の接続失敗の案内、再接続直後の負の「送信」、`uiMode` の構成変更で Activity 再生成、古い接続の通知、QR 読み直しでの再同期の予約の重複、推定器の失敗が黙る |

確かめて問題なし: 電文の形（`app/net/protocol.py` と一致）、座標の正規化・回転・鏡像、時計の式、フレームと寸法の対応、
再接続の状態機械、ライフサイクル（回転）、release 版（難読化なし）。
