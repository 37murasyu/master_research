# 混成ステレオ（Mac＋Pixel 7a）での §6-2・§6-3・§3-2 の確認

実際の計測構成は、Mac の内蔵カメラと、同じ Wi-Fi にいる Pixel 7a 1 台（`app.hybrid`、`python -m app --role hybrid_measure`、
GUI の計測ページで入力を切り替える）。USB カメラ 2 台の経路（`docs/usb_stereo_verification.md`）とは記録の形が違うので、
2026-09-23 に検証の道具（`tools/verify_run.py`）を混成の記録にも対応させた。

Pixel の映像は、撮影要求に JPEG で応える方式で、毎秒 3〜4 枚（表示用 640 px、校正用は全解像度）しか届かない
（`app.hybrid.link` の PREVIEW・CALIBRATION）。30 fps で残るのは映像ではなく、次の記録になる。

| ファイル | 中身 |
|---|---|
| `landmarks2d_<stamp>.csv` | Mac（cam0）と Pixel（cam1）の 2D ランドマーク。撮影時刻は PC の時計に換算済み |
| `kpts3d_raw_<stamp>.csv`（とサイドカー `.json`） | 三角測量した 3D（m、**EKF の手前**）。同期バッファの 1/30 s の格子で、組が抜けた格子は NaN の行。**実行時の EKF の較正（`tune_ekf`）の入力** |
| `kpts3d_<stamp>.csv` | **EKF の後**の 3D（m。`EKF_ENABLE=0` なら三角測量のまま）。組が届いた格子だけの行 |
| `frames_<stamp>.csv` | 組ごとの時刻・格子の番号（`grid_index`）・回の区切り（`cycle_detected`・`rep`）・関所の状態 |
| `local_torque_<stamp>.csv`・`cycle_work_<stamp>.csv` | 局所トルク（N·m）と回ごとの仕事（W+・W−・W_1RM・スコア） |
| `gauge_energy_<stamp>.csv`・`cycle_energy_<stamp>.csv` | ゲージに出した値（毎フレーム）、肘の濾波 E± |
| `meta.json` | 校正、体重、被験者・1RM、EKF の設定と出どころ、重力、状態（`recording` → `complete`／`failed`）、止まった理由（`stop_reason`） |

出力先は `~/Documents/WheelchairTorque/hybrid/measure/<stamp>/`（GUI の計測ページの「出力先」）。
以下の `python` はリポジトリ直下で仮想環境の Python を使う（Mac は `.venv/bin/python`）。

## 1. 計測して確かめる（§6-2・§3-2）

校正（GUI の校正ページ、または `python -m app --role hybrid_calibrate`）の後、GUI の計測ページで入力を Mac＋Pixel にして
計測し、**停止ボタンで止める**。その後、次を実行する（`measure` フォルダを渡すと最新の回を選ぶ）。

```sh
python -m tools.verify_run check ~/Documents/WheelchairTorque/hybrid/measure --expect-stop
```

- **§3-2**: 「meta: 記録を正しく閉じた」（`status=complete`。kill されると `recording` のまま）、「meta: 停止要求で止まった」
  （`stop_reason=stop_request`。GUI の停止ボタン・停止ファイル・SIGTERM）、行の対応（kpts3d・frames・meta の frames）。
  止まった理由は 2026-09-23 から `meta.json` に残す（`key`＝q・ESC、`ctrl_c`、`failed`、`error`）
- **§6-2**: トルク |τ_y| の中央値・95%・最大（見込みは手首・肘 10〜40 N·m 台）、サイクルの検出回数とサイクルごとの仕事、
  **Mac と Pixel それぞれの実際の fps**。どちらかが 24 fps（30 fps の 8 割）を下回ると「速さ」の検査を不合格にする
- 組は同期バッファ（`app.net.sync_buffer`）が 2 台を 30 Hz の格子へ線形補間して作るので、組の速さはカメラの速さではない。
  `check` は「組のうち遅い方のカメラの実測に基づく割合」と「100 ms を超える穴で組が抜けた時間」も出す
- 結果は計測フォルダの `verify_report.json` にも残る

## 2. EKF の雑音の推定（§6-3 の S6）と S9b

混成の計測は 2026-09-24 から USB と同じ `LandmarkEKF` を同期バッファの格子（1/30 s）で回す（`app.hybrid.ekf`。頑健な門、
発散の見張り）。雑音は設定 `HYBRID_EKF_PROFILE` の較正プロファイルで、空なら同梱の既定値。`check` は EKF の前後の差（RMS）と
棄却率（S9b の材料）を、生 CSV と `kpts3d` を格子の番号（`grid_index`）で合わせて出す。

**実行時の EKF のプロファイルは、記録器が書いた `kpts3d_raw_<stamp>.csv`（`_retri` の付かないもの）から作る**:

```sh
python -m app.tuning.ekf_estimate <計測フォルダ>/kpts3d_raw_<stamp>.csv   # 推定の表（張り付き●、ρ1、n_eff）
python -m app.runners.tune_ekf <計測フォルダ>/kpts3d_raw_<stamp>.csv      # hybrid/ekf_profiles/ekf_profile_0.03333.json
```

- `tune_ekf` は記録器の生 CSV（サイドカーの `source` が `hybrid`）のプロファイルだけを `~/Documents/WheelchairTorque/hybrid/ekf_profiles/`
  に書き、設定 `HYBRID_EKF_PROFILE` に入れる値を出す（GUI の解析ページ「EKF の較正プロファイルを作る」も同じ）。`--out` にはファイルかフォルダを渡せる
- 格子の dt は 1/30 s。実行時は dt の相対差 5% 以内のプロファイルしか選ばない
- 判断の基準は USB と同じ（端に張り付く系列が過半数、または |ρ1| > 0.3 なら設計を見直す）
- S9b: 設定 `HYBRID_EKF_PROFILE` にプロファイル（かそのフォルダ）を入れてもう 1 試技し、`check` の「§6-3 EKF」の雑音の出どころが
  `profile` で、RMS 差と棄却率が期待の範囲かを見る

**比べる用: 実際の撮影時刻で三角測量し直した 3D（`hybrid-raw`）**。Pixel が 30 fps に届かない（CPU 推論では 10〜15 Hz）と、
30 Hz の格子の Pixel 側は大半が線形補間の点で、補間の区間は直線になる。これを推定にかけると「なめらかで雑音が小さい」系列に
見えて推定が狂うので、格子の推定と比べるために、記録した 2D から**遅い方のカメラの実際の撮影時刻で三角測量し直す**
（`app.hybrid.retriangulate`。速い方だけを線形補間し、100 ms を超える穴は埋めない。三角測量と歪み補正は計測と同じ）。
出力は `kpts3d_raw_<stamp>_retri*.csv` で、記録器の生 CSV は上書きしない。

```sh
python -m tools.verify_run hybrid-raw ~/Documents/WheelchairTorque/hybrid/measure           # kpts3d_raw_<stamp>_retri.csv（Pixel の撮影時刻）
python -m tools.verify_run hybrid-raw ~/Documents/WheelchairTorque/hybrid/measure --hz 4    # kpts3d_raw_<stamp>_retri_s3.csv（4 Hz 相当）
python -m tools.verify_run hybrid-raw ~/Documents/WheelchairTorque/hybrid/measure --grid    # kpts3d_raw_<stamp>_retri_grid.csv（格子の 3D）
python -m app.tuning.ekf_estimate <計測フォルダ>/kpts3d_raw_<stamp>_retri.csv
python -m app.tuning.ekf_estimate <計測フォルダ>/kpts3d_raw_<stamp>_retri_s3.csv
```

- 名前の `_s3` は間引き幅で、実際の速さで決まる（12 Hz なら 3 組おき、15 Hz なら `_s4`、30 Hz なら `_s8`）。画面の「生 CSV:」の行に出る
- サイドカーの `source` は `hybrid_retri`。`tune_ekf` にかけてもプロファイルは CSV の隣に書き、実行時の置き場には書かない
  （`--grid` の出力は dt が同じ 1/30 s なので、記録器の生 CSV から作ったプロファイルを上書きしてしまうため）
- dt は撮影時刻の間隔の中央値。間隔は揺れ、推定は dt 一定を前提にするので、揺れの幅（5〜95%）をサイドカーに残し、画面にも出す
- 注意: `_retri` の dt は遅い方のカメラの間隔（Pixel が 12 Hz なら約 1/12 s）で、混成の計測の格子 1/30 s・USB 経路の 1/30 s・8/30 s とは別物
- 4 Hz 相当で n_eff 300 を満たすには 80 秒以上の計測が要る（設計メモ :275）

## Pixel が 10〜15 Hz であることの影響（§6-2）

- **30 fps は保てていない**。組の Pixel 側の 6〜7 割は線形補間の点になる
- 力学は 30 Hz の組の差分で速度・加速度を取るので、補間の直線の区間では加速度が 0 に近く、Pixel の実測点で跳ねる。
  慣性の項（寄与は数 %）と、回の区切りの速さの条件（直近 5 フレームの肩の高さの傾き、`app.hybrid.rep_detector`）がこの影響を受ける
- Pixel の間隔が 100 ms（10 Hz）に近づくと、同期バッファの穴の上限（100 ms）に当たって組が抜ける。`check` の「組が抜けた時間」で見る
- 2026-09-24 に、Pixel の「GPU推論」をオンにすれば推論以降も約 30 fps 出ることを確かめた（`docs/hybrid_field_run.md`）。
  本計測は「30fps固定」「GPU推論」ともオン

## 試走（2026-09-23、Mac のみ）

Pixel が無いので、模擬 Pixel（`app.net.mock_sender.MockPhone`、WebSocket で本当に送る）と偽の Mac カメラで 8 秒回した。
本物の受信（`PhoneLink`）・計測（`MeasurementSession`）・記録器（`Recorder`）を通し、停止の要求で止めた形で閉じた。

- 構造の検査はすべて合格（`status=complete`、`stop_reason=stop_request`、kpts3d・frames・meta が 238 行ずつ）
- 速さは Mac 19.3 fps（偽カメラの回し方による）、Pixel 28.0 fps、組 30.0 fps（補間）
- `hybrid-raw` の間引きなし・8 組おきの両方ができ、`ekf_estimate` が表を出した
- 模擬 Pixel を 12 Hz にして 20 秒回すと、「速さ」の検査が不合格（Mac 22.4 fps、Pixel 11.7 fps）、組のうち実測に基づく割合 0.39。
  `hybrid-raw` は Pixel の撮影時刻で dt 0.086 s、`--hz 4` で 3 組おき（dt 0.257 s）になった
- サイクル検出は当時 USB と同じ検出器（左肩の y、1 フレームあたりの速さ）だった。2026-09-24 から混成の回の区切りは
  `app.hybrid.rep_detector.RepDetector`（肩の中点の重力の上向きの高さ）で、KNOWN_ISSUES §6-9 は USB 経路にだけ残る

## 残る実機確認

- 実際の Pixel 7a と Mac で、GUI の停止ボタン → `check --expect-stop`（§3-2）
- Mac・Pixel それぞれが 30 fps を保つか、トルクの大きさ（§6-2）
- 80 秒以上の計測で S6（記録器の `kpts3d_raw_<stamp>.csv` から `tune_ekf`）→ S9b（§6-3）
