# USB ステレオの録画と、§6-2・§6-3・§3-2 の確認

2026-09-23 に、USB カメラ 2 台の映像を録画し、その録画を計測（`master_research_code.py`）に読み込ませて
確かめる道具を足した。計測自身の録画は、間引きの後のフレームだけを fps 30 固定で書くので時間が詰まり、
処理し直しに使えない（KNOWN_ISSUES §3-4）。

- `tools/record_stereo.py`: 2 台の全フレームを、計測と同じカメラ設定（`app.core.camera_controls`）で録画する
- `tools/verify_run.py replay`: 録画を計測に読み込ませる。起動は GUI と同じ `python -m app --role realtime`、設定も GUI の設定を土台にする
- `tools/verify_run.py check`: 計測の出力フォルダを確かめる（ファイルの有無、行の対応、トルク、ゲージ、処理の速さ、EKF の RMS 差と棄却率）

以下の `python` はプロジェクトの仮想環境の Python を使う。Mac は `.venv/bin/python`、Windows は
`.venv\Scripts\python.exe`。**リポジトリ直下で実行する**。Windows では論文当時の証拠の `MAINCODE` を動かさず、
別のフォルダに clone した新しいコードで行う。

## 1. 録画する

先に校正（`calib.py`、GUI の校正ページ）を済ませ、`camera_parameters/` に `c0.dat`・`c1.dat`・`rot_trans_c0.dat`・
`rot_trans_c1.dat` がある状態にする。カメラ番号と解像度の既定は、校正と同じ `calibration_settings.yaml` から取る。

```sh
python -m tools.record_stereo --label S07 --duration 90
```

- 出力は `recordings/S07_<MMDD_HHMMSS>/`。`cam0_<ts>.avi` と `cam1_<ts>.avi`（MJPG）、校正ファイル 4 つのコピー、
  `frames.csv`（フレームごとの撮影時刻）、`meta.json` がそろう。このフォルダは `CALIB_BASE_DIR` にそのまま渡せる
- 止め方は q・ESC（プレビュー）、Ctrl-C、`--duration`、停止ファイル（`--stop-file`）のどれでもよい。どれでも動画を閉じる
- 解像度が校正と違うと止まる（違う解像度の映像を校正の行列で三角測量すると 3D が狂う）
- カメラは計測と同じ開き方（`video_io.open_capture_and_read_first`）で開き、`meta.json` にバックエンド名を残す
- Windows で 1280×720 を USB2 で撮ると、既定の YUY2 では 5〜10 fps に落ちやすい。録画と計測の両方で
  `CAM_FOURCC=MJPG` を環境変数で与える（計測も同じ `app.core.camera_controls` で当てる）
- 4 Hz 間引きで EKF を較正する（S6）には **80 秒以上** 要る（設計メモ :275）。n_eff が 300 に届かない系列は採用されない
- 終わりに実測の fps を出す。容器の 30 fps と 5% 以上ずれたら警告し、再生で `DT_SEC` を自動で渡す
- ファイルを小さくしたいときは `--codec mp4v`（.mp4）。ただし圧縮の劣化で S6 の雑音の推定が計測とずれうる

## 2. 録画を計測に読み込ませる（S6 の 2 設定）

同じ録画を「間引きなし」と「4 Hz 間引き」で処理する。`SUBJECT_ID` は 1RM の部位表（`m_max_part_<ID>.json`）を選ぶ。

```sh
python -m tools.verify_run replay --session recordings/S07_0923_213245 --subject 7 --fixed-hz 0
python -m tools.verify_run replay --session recordings/S07_0923_213245 --subject 7 --fixed-hz 1
```

- 出力は `<録画>/runs/full/` と `<録画>/runs/hz4/`。計測はこのフォルダの中で動く（リポジトリを汚さない）
- デモ表示（`DEMO_MONO_GAUGE_ON`・`DEMO_MONO_CAM0_ONLY`、コードの既定は 1）は 0 にする。1 のままだと力学が回らない
- 入力が開けないと本体は作業フォルダの別の録画（`cam*_output_*`）へ黙って切り替えるので、起動の前に動画と校正ファイルを
  確かめ、ログに `入力の読み込みに失敗` が出ていないことも check の構造の検査に入れた
- 起動ログの `reason=config: file paths`、`[CALIB] using CALIB_BASE_DIR`、`[DT] dt=0.03333s`（4 Hz なら `0.26667s`）、
  `[INERTIA]` の骨長、`[GRAVITY]` を見る
- 受け取った `cameras_raw/<試技>/` のように `meta.json` の無い録画は `--cam0 --cam1 --calib --out` で指定する
- 終わると自動で `check` にかけ、`verify_report.json` を書く

## 3. EKF の較正（S6）と受け入れ判定（S9b）

```sh
python -m app.tuning.ekf_estimate recordings/S07_.../runs/full/kpts3d_raw_<ts>.csv   # 推定の表（張り付き●、ρ1、n_eff）
python -m app.runners.tune_ekf recordings/S07_.../runs/full/kpts3d_raw_<ts>.csv       # ekf_profile_0.03333.json
python -m app.runners.tune_ekf recordings/S07_.../runs/hz4/kpts3d_raw_<ts>.csv        # ekf_profile_0.26667.json
```

- **端に張り付く系列が過半数、または |ρ1| > 0.3 なら、先へ進まず設計を見直す**（設計メモ :396）
- `tune_ekf --out` はファイルのパスを取る（フォルダを渡すと落ちる。既定は生 CSV の隣）
- S9b: プロファイルを付けて再生し、`check` の「§6-3 EKF」の RMS 差と棄却率を見る。期待範囲は S6 の結果で決める。
  `--ekf-profile` は絶対パスに直して渡し、check は「EKF: 較正プロファイルを使った」を構造の検査に入れる。
  GUI の設定にプロファイルがあるとき、無しで回すには `--ekf-profile ""`

```sh
python -m tools.verify_run replay --session recordings/S07_... --subject 7 --fixed-hz 0 --name s9b \
    --ekf-profile recordings/S07_.../runs/full/ekf_profile_0.03333.json
```

## 4. 停止で CSV が書かれるか（§3-2）

GUI なしでは、ループに入ってから指定の秒数で停止ファイルを置く。

```sh
python -m tools.verify_run replay --session recordings/S07_... --subject 7 --name stop --stop-after-sec 15
```

`[STOP] 停止要求を受けました` の後に kpts3d・aim_torque・gauge_energy が書かれ、check の構造の検査がすべて合格すればよい。

**GUI の停止ボタンは実機で確かめる**。計測ページの「出力先」（`~/Documents/WheelchairTorque/output_data`）に CSV が
書かれる。止めた後、GUI の出力欄をテキストに保存して（`[STOP] 停止要求を受けました` の行があるはず）次を実行する。
`--expect-stop` はログが要る（無いと「停止要求を受けた」を不合格にする。`[STOP] Reached MAX_FRAMES` は停止要求と数えない）。

```sh
python -m tools.verify_run check ~/Documents/WheelchairTorque/output_data --expect-stop --log gui_output.txt
```

## 5. 30 fps とゲージの値（§6-2）

**30 fps は実機のライブ計測でしか確かめられない**（再生は処理が遅くても dt が動画の fps で決まり、速さの上限しか分からない）。
GUI の既定（間引きなし）で計測し、`check` の「処理間隔: dt と実際の間隔が 20% 以内」を見る（生 CSV の `t` 列の間隔の中央値と dt を比べる）。
起動ログの `[DT][警告]` も同じことを見ている。解像度は校正と同じにする。本体は `CAM_WIDTH`・`CAM_HEIGHT` が無いと
カメラの既定の解像度で開く。これらは設定画面の項目ではないので、GUI を起動する前の環境変数で与える
（例: `CAM_WIDTH=1280 CAM_HEIGHT=720 python -m app`）。起動ログの `[CAMDIAG]` で実際の大きさを確かめる。

ゲージの値は `gauge_energy_<ts>_s2_g*.csv`（処理フレームごと）と同名の `.json`（閾値の帯）に残る。
`check` はサイクルごとの最大と帯（E_low・E_high）を並べ、トルクは |τ_y| の中央値・95%・最大を並べる。

## 試走の結果（2026-09-23、受け取った被験者 7 の動画、Mac M1）

受け取った `cameras_raw/7_20250925_184912/`（Windows で撮った 60 秒、1280×720、mp4）を `--cam0 --cam1 --calib` で再生した。

| 条件 | 処理フレーム | 処理の速さ | サイクル検出 | 構造の検査 |
|---|---|---|---|---|
| 間引きなし | 1819 | 35.1 fps | **0 回** | すべて合格 |
| 4 Hz 間引き | 253 | 31.1 fps | 8 回 | すべて合格 |
| 間引きなし、15 秒で停止ファイル | 496 | 33.6 fps | 0 回 | すべて合格（`[STOP]` の後に CSV） |

- 骨長は上腕 0.28/0.31 m・前腕 0.24/0.27 m、重力は Z+ が上（体幹の傾き 9°）と推定した。受け取った校正の fx≠fy（KNOWN_ISSUES §6-2）は、骨長には目立って効いていない
- トルク |τ_y| の 95%: 右手首 24.6、右肘 7.5、左手首 50.8、左肘 13.3 N·m（間引きなし）。中央値が 0 なのは、GUI の既定
  `RT_DYN_ON_RISE_ONLY=1` が立ち上がりを検出するまで τ を 0 にするため
- **間引きなしではサイクルを 1 回も検出しない**（KNOWN_ISSUES §6-9）。ゲージは帯（約 95〜128 J、旧来のゲージの式）に
  一度も届かず、4 Hz の 1 サイクルの正の仕事は 1〜12 J 程度
- EKF（S6 の材料）: 間引きなし・4 Hz とも、推定できた系列はほぼすべて探索範囲の端に張り付き、ρ1 は 0.77〜0.99。
  約 4 割の系列は q_acc の初期値が取れない。どちらのプロファイルも全 48 系列が同梱既定値（仮置き）になり、
  それで再生すると RMS 差の中央値 37 mm・最大 289 mm、棄却率の中央値 17%・最大 81%（KNOWN_ISSUES §6-3）。
  古い構成の mp4 の映像なので、判断は新しい構成の録画でやり直す

## 残る実機確認

- USB 2 台での録画ツールの実機確認（この試走は動画ファイルを入力にした。Mac では 2 台目の USB カメラが無い）
- GUI の停止ボタン（§3-2）と、ライブでの 30 fps（§6-2）
- 新しい構成の録画での S6 → S9b（§6-3）
