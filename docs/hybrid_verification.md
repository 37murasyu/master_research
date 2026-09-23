# 混成ステレオ（Mac＋Pixel 7a）での §6-2・§6-3・§3-2 の確認

実際の計測構成は、Mac の内蔵カメラと、同じ Wi-Fi にいる Pixel 7a 1 台（`app.hybrid`、`python -m app --role hybrid_measure`、
GUI の計測ページで入力を切り替える）。USB カメラ 2 台の経路（`docs/usb_stereo_verification.md`）とは記録の形が違うので、
2026-09-23 に検証の道具（`tools/verify_run.py`）を混成の記録にも対応させた。

Pixel の映像は、撮影要求に JPEG で応える方式で、毎秒 3〜4 枚（表示用 640 px、校正用は全解像度）しか届かない
（`app.hybrid.link` の PREVIEW・CALIBRATION）。30 fps で残るのは映像ではなく、次の記録になる。

| ファイル | 中身 |
|---|---|
| `landmarks2d_<stamp>.csv` | Mac（cam0）と Pixel（cam1）の 2D ランドマーク。撮影時刻は PC の時計に換算済み |
| `kpts3d_<stamp>.csv` | 三角測量した 3D（m、EKF なし） |
| `frames_<stamp>.csv` | 組ごとの時刻とサイクル検出 |
| `local_torque_<stamp>.csv`・`cycle_work_<stamp>.csv` | 局所トルク（N·m）とサイクルごとの仕事（J） |
| `meta.json` | 校正、体重、状態（`recording` → `complete`／`failed`）、止まった理由（`stop_reason`） |

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

## 2. EKF の雑音の推定（§6-3 の S6）

混成の経路は EKF を使っていない（S9b の受け入れ判定は対象外）。ただし 3D は EKF の手前の値なので、生 CSV の形に直せば
S6 の推定がそのまま使える。

**計測中の 3D（`kpts3d_*`）は S6 に使わない**。実機の Pixel 7a は 10〜15 Hz しか出ない（2026-09-23 に hybrid ブランチで確認）ので、
30 Hz の格子の Pixel 側は大半が線形補間の点で、補間の区間は直線になる。これを推定にかけると「なめらかで雑音が小さい」系列に
見えて推定が狂う。`hybrid-raw` は既定で、記録した 2D から**遅い方のカメラ（Pixel）の実際の撮影時刻で三角測量し直す**
（`app.hybrid.retriangulate`。速い方の Mac だけを線形補間し、100 ms を超える穴は埋めない。三角測量と歪み補正は計測と同じ）。
混成の経路には間引きの設定が無いので、4 Hz 間引きに当たる 2 設定目は `--hz 4` で作る（実際の速さから間引き幅を決める。12 Hz なら 3 組おき）。

```sh
python -m tools.verify_run hybrid-raw ~/Documents/WheelchairTorque/hybrid/measure           # Pixel の撮影時刻（約 12 Hz）
python -m tools.verify_run hybrid-raw ~/Documents/WheelchairTorque/hybrid/measure --hz 4    # 4 Hz 相当
python -m app.tuning.ekf_estimate <計測フォルダ>/kpts3d_raw_<stamp>.csv
python -m app.runners.tune_ekf <計測フォルダ>/kpts3d_raw_<stamp>_s3.csv
```

- dt は撮影時刻の間隔の中央値。間隔は揺れ、推定は dt 一定を前提にするので、揺れの幅（5〜95%）をサイドカーに残し、画面にも出す
- 計測中の格子の 3D と比べたいときは `--grid`（`kpts3d_raw_<stamp>_grid.csv`）
- 注意: できるプロファイルの dt は Pixel の間隔（約 1/12 s）で、USB 経路の 1/30 s・8/30 s とは別物。USB の EKF にそのまま流用しない
- 4 Hz 相当で n_eff 300 を満たすには 80 秒以上の計測が要る（設計メモ :275）
- 判断の基準は USB と同じ（端に張り付く系列が過半数、または |ρ1| > 0.3 なら設計を見直す）

## Pixel が 10〜15 Hz であることの影響（§6-2）

- **30 fps は保てていない**。組の Pixel 側の 6〜7 割は線形補間の点になる
- 力学は 30 Hz の組の差分で速度・加速度を取るので、補間の直線の区間では加速度が 0 に近く、Pixel の実測点で跳ねる。
  慣性の項（寄与は数 %）とサイクル検出の「1 フレームあたりの速さ」（§6-9）がこの影響を受ける
- Pixel の間隔が 100 ms（10 Hz）に近づくと、同期バッファの穴の上限（100 ms）に当たって組が抜ける。`check` の「組が抜けた時間」で見る
- 直す方向（未着手）: Pixel 側の姿勢推定を速くする（`mobile/` の PoseAnalyzer、解析する画像の大きさ）、または格子を Pixel の速さに合わせる

## 試走（2026-09-23、Mac のみ）

Pixel が無いので、模擬 Pixel（`app.net.mock_sender.MockPhone`、WebSocket で本当に送る）と偽の Mac カメラで 8 秒回した。
本物の受信（`PhoneLink`）・計測（`MeasurementSession`）・記録器（`Recorder`）を通し、停止の要求で止めた形で閉じた。

- 構造の検査はすべて合格（`status=complete`、`stop_reason=stop_request`、kpts3d・frames・meta が 238 行ずつ）
- 速さは Mac 19.3 fps（偽カメラの回し方による）、Pixel 28.0 fps、組 30.0 fps（補間）
- `hybrid-raw` の間引きなし・8 組おきの両方ができ、`ekf_estimate` が表を出した
- 模擬 Pixel を 12 Hz にして 20 秒回すと、「速さ」の検査が不合格（Mac 22.4 fps、Pixel 11.7 fps）、組のうち実測に基づく割合 0.39。
  `hybrid-raw` は Pixel の撮影時刻で dt 0.086 s、`--hz 4` で 3 組おき（dt 0.257 s）になった
- サイクル検出は USB と同じ検出器（左肩の y、1 フレームあたりの速さ）なので、KNOWN_ISSUES §6-9 は混成にも当てはまる

## 残る実機確認

- 実際の Pixel 7a と Mac で、GUI の停止ボタン → `check --expect-stop`（§3-2）
- Mac・Pixel それぞれが 30 fps を保つか、トルクの大きさ（§6-2）
- 80 秒以上の計測で S6（§6-3）
