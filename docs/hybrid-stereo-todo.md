# 混成ステレオ（Mac 内蔵カメラ＋Pixel 7a）今後の作業（Copilot 向け）

最終更新: 2026-09-23 ／ ブランチ: `murayama/hybrid-stereo`（作業ツリー `../master_research-hybrid`）

このファイルは、Claude Code のセッションで立てて一部を実装した計画の引き継ぎ書。
計画の原本は作業者のローカル（`~/.claude/plans/tingly-chasing-fairy.md`）にあり、ここに要点を写してある。
**上から順に進めること**（段階の依存は各項目に書いた）。

---

## 0. 背景と決まっていること

- 計測の構成は **Pixel 7a 1 台＋MacBook Air M1 の内蔵カメラ（FaceTime HD 720p）**。
  スマホ経路（`app/net/`、`mobile/`）は「2 台ともスマホ」前提で作られていた。
- **Mac = cam0（ワールド原点。PC 側で MediaPipe）**、**Pixel = cam1（背面カメラ。Android アプリが推定して Wi-Fi 送信）**。QR は cam1 の 1 つだけ。
- Pixel の向き合わせは **Mac の画面でライブ表示**する。校正にも使う撮影要求（`capture_req` → Pixel が解析中の同じフレームを JPEG で返す `calib_frame`）を毎秒数回くり返し、「Mac カメラ｜Pixel」を骨格付きで並べる。
- 暫定策として scrcpy を入れてある: `ADB=~/Library/Android/sdk/platform-tools/adb scrcpy --no-audio --stay-awake --max-size 1280`

### 守ること（作業者の規約）

- `main` に直接コミットしない。ブランチは `murayama/<英小文字ケバブケース>`。コミットと push は作業者の指示があったときだけ。
- 各段階はテスト先行（失敗するテスト → 実装 → 全体テスト）。
- `master_research_code.py` は import できない。ロジックは import できるモジュールに置いてテストする。
- `calib.py` は import しない（import 時に boto3、関数内に `cv.imshow`・`quit()`・作業フォルダ相対の読み書きがある）。書式と検出条件だけ踏襲する。

### テストの回し方

```sh
cd ../master_research-hybrid
../master_research/.venv/bin/python -m pytest tests -q          # 本体の .venv を worktree の直下から使う
cd mobile && JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./gradlew :app:testDebugUnitTest
# PC → 端末の契約見本を変えたら: UPDATE_CONTRACT=1 で tests/test_protocol_contract.py を回して mobile/contract/pc_messages.json を作り直す
# 端末 → PC の見本（golden_messages.json）は gradle のテストが書き出す。差分をコミットに含める
```

2026-09-23 時点: Python 535 件成功・1 件スキップ、Kotlin 34 件成功。

---

## 1. 済んだこと

| コミット | 内容 |
|---|---|
| `349f823` | fix(mobile): QR 読み取りと計測でカメラ設定を分ける（`CameraPurpose`。QR 読み取りが RGBA を ML Kit に渡して落ちていた） |
| `cdd7345` | feat(net): 撮影要求・校正画像の電文とサーバ（phone-path の 431696e を取り込み） |
| `6db912c` | feat(hybrid): 受信層。`remote_roles`・後勝ち・`on_landmarks`・`inject`、同期バッファの片側溜まり修正、`app/hybrid/link.py`（`PhoneLink`・`CaptureScheduler`）、模擬端末の撮影応答 |

`app/hybrid/link.py` の使い方（段階 2 以降の土台）:

- `PhoneLink(remote_role="cam1", on_pairs=..., on_landmarks=..., on_hello=..., capture_mode=PREVIEW)` → `start()` / `stop()`
- メインスレッドから: `inject(LandmarkFrame(role="cam0", ...))`、`set_capture_mode(PREVIEW|CALIBRATION|OFF)`、`take_capture()`（新しい JPEG があれば `CalibrationFrame`）、`nearest_remote(t_ns, tolerance_ns)`（画像に重ねる cam1 の点）、`status()`（`LinkStatus`）、`url`（QR に入れる接続先）、`session`
- `on_pairs` / `on_landmarks` は**ループのスレッド**で呼ばれる。例外は数えるだけで受信は止めない。

---

## 2. 作業中（段階 3: Android の撮影応答）— 未コミット

**済み**（作業ツリーにある）:

- `Protocol.kt`: `hello(..., deviceId)`、`messageType`、`parseCaptureRequest`、`CaptureRequest`、`calibrationFrame(..., jpeg: ByteArray)`（base64 は okio。`android.util.Base64` は JVM テストで動かず、`java.util.Base64` は API 26 から）
- `capture/CaptureRequests.kt`（どのフレームで応えるか。テスト付き）、`capture/JpegResponder.kt`（縮小・複製は解析スレッド、圧縮と送信は専用 executor）
- `PoseAnalyzer`: `captureSink` を追加（`toBitmap()` の直後、推論の前。人が写っていなくても呼ぶ）
- `SensorClient`: 電文を種類で振り分け、`onCaptureRequested`、`sendCalibrationFrame`、PC に閉じられたら理由を ERROR で出す、接続ごとの世代番号で古い接続の通知を無視（QR を読み直すと古い `onClosed` が新しい接続を消していた）
- `CameraSetup`: 光学式手ブレ補正（OIS）を OFF
- テスト: `CaptureRequestsTest`、`PcMessagesContractTest`、`ProtocolTest` 追加分、`ContractSampleTest` に device_id 付き hello と calib_frame（golden 再生成済み）、Python の `test_protocol_contract.py` に 2 件

**残り（ここから再開）**: `MainActivity.kt` の配線。

1. `device_id`: `Settings.Secure.ANDROID_ID` を SHA-256 して先頭 8 バイトの 16 進（生の ID は送らない）。リリース鍵は固定なので入れ直しても変わらない。`client.connect(target, deviceName(), deviceId())`
2. `CaptureRequests()` と `JpegResponder(Executors.newSingleThreadExecutor(), send = { req, nanos, w, h, jpeg -> client.sendCalibrationFrame(req.id, nanos, w, h, jpeg) })` を持つ。`onDestroy` で executor を `shutdown()`
3. `override fun onCaptureRequested(request)` → `captureRequests.add(request, timeSync.deviceNanos())`
4. `startStreaming()` で `captureRequests.clear()` し、`PoseAnalyzer(this, captureSink = { bitmap, nanos -> val due = captureRequests.takeDue(nanos, timeSync.offsetNanos); if (due.isNotEmpty()) jpegResponder.offer(bitmap, nanos, due) }) { ... }`
5. 送った画像の枚数を `onProgress` の表示に足す（例「送信 N フレーム / 画像 M 枚」）

完了条件: gradle のユニットテストが緑、`./gradlew :app:assembleRelease` が通る。実機（段階 2 の後）で右パネルに Pixel の画像が 3 Hz 以上で出て骨格と一致し、表示 0 Hz と 4 Hz でランドマークの fps の落ち込みが 10% 以内。
インストールは `cd mobile && ./install.sh --build`（今と同じ release 版で入れる。debug 版は署名鍵が違い、ANDROID_ID も変わる）。

---

## 3. 残りの段階

### 段階 2: ライブ表示（段階 3 より先に実機で試せる。今のアプリは capture_req を捨てるだけなので害が無い）

新規:

- `app/hybrid/mac_camera.py`: `app.core.video_source.open_capture(index)` で開き、1280×720 を指定して数枚読んで実寸を確かめる（手順は `calib.py:78-113` と同じ。import はしない）。違えば分かる文言で例外。先頭 10 枚は捨てる（自動露出の安定待ち）。`read()` は `grab()` の**直後**に `time.monotonic_ns()` を取ってから `retrieve()`（Android の「analyze 開始時刻」と意味を揃える）。反転しない。
- `app/hybrid/pose_detector.py`: MediaPipe Tasks の **VIDEO モード**の薄いラッパ（`pose_runtime.PoseEstimator` は IMAGE モードで追跡・平滑化が効かない）。モデルは `pose_landmarker_lite.task`、閾値 0.5×3、`visibility` が無ければ 1.0（Android と同じ）。mediapipe は関数内で import、timestamp_ms は単調増加を強制。人が写っていなければ inject しない。
- `app/hybrid/display.py`（純関数）: 左右 640×360 の 2 面＋下に状態帯。骨格（`config.pose_keypoints` の点を強調）、未接続の間は右に cam1 の QR 画像（`segno` で描く。ターミナルの文字 QR は実機で読めなかった）と URL、盤の角点。日本語は `utils.draw_text_jp`。
- `app/hybrid/live.py`: `LiveSession.step()`。左 = Mac の映像と推論の骨格、右 = Pixel の最新 JPEG と `nearest_remote(t, ±20 ms)` の骨格。状態帯 = Mac fps、Pixel fps・機種・device_id、ペア/秒、位相差（平均/最大）、欠測破棄、各カメラで写っている要点の数。表示の出力先は差し込めるようにする（本番 `cv.imshow`、テストは記録用の偽物）。
- 入口 `app/runners/hybrid_preview.py`（CLI のみ）: q/Esc、窓の×（`WND_PROP_VISIBLE`）、`app.core.stop_request.StopRequest`、SIGINT で終わる。`main()` の先頭で stdout を行バッファにする。

テスト: 偽カメラ・偽推定器・`MockPhone` で `LiveSession` を回す／接続前の右パネルを `cv.QRCodeDetector` で読んで URL と一致／停止ファイルで抜ける。`tests/hybrid_fakes.py` に偽物をまとめる。`tests/test_cross_platform.py` の `IMPORT_SAFE_MODULES` に新モジュールを足す。

完了条件: テスト緑。実機で `python -m app.runners.hybrid_preview --camera 0`（Camo が 0 番を取っていれば 1）→ 左に Mac＋骨格、QR を読むと右に骨格。目安 Mac 25 fps 以上、Pixel 20 fps 以上、ペア 20/秒以上、位相差平均 20 ms 未満。初回は macOS の「受信接続を許可」に答える。

### 段階 4: 校正の純関数（段階 1〜3 と並行できる）

- `app/hybrid/checkerboard.py`: 縮小画像で `FAST_CHECK` → 見つかれば全解像度で `cornerSubPix`（criteria は `calib.py:279` と同じ EPS+MAX_ITER, 100, 0.001。窓は角点間隔×0.35 を 3〜11 px に丸める）。内部パラメータは `CALIB_FIX_K3`、ステレオは `stereoCalibrate(CALIB_FIX_INTRINSIC)`（imageSize は cam0）。ビューごとの誤差が中央値の 3 倍を超えたら除いて 1 回だけ推定し直す。三角測量したマス寸法（cm）を返す検算関数。テスト用の合成盤描画。盤は「内側交点の数が片方偶数・片方奇数」を起動時に検査（4×7 は OpenCV 5.0 で角点の並びが盤に固定されることを確認済み。並べ直しはしない）。
- `app/hybrid/calibration_io.py`: `~/Documents/WheelchairTorque/hybrid/calibration/<YYYYmmdd_HHMMSS>/` に `c0.dat c1.dat rot_trans_c0.dat rot_trans_c1.dat meta.json`。書式は `calib.py:354-373` と `:1095-1138`、c0 は R=I・T=0。**T は cm**（実行時の三角測量が `scale=0.01` で m に直すため。`calib.py` は m で書くので混ぜないこと）。`latest.json` で最新を指す。内部パラメータのキャッシュ（Pixel は device_id＋解像度、Mac は hw.model＋カメラ番号＋解像度）。読み込みは `utils.get_projection_matrix(i, False, base_dir=dir)`。meta には単位、盤、各カメラ（種類・機種・device_id・寸法・rms・内部パラメータの出どころ）、ステレオ（rms・ペア数・基線・マス寸法誤差）、日時。

テスト: 合成盤で f 2%・|T| 2%・R 1° 以内（1280×720 と 1920×1080 の組も）／180° 回しても並びが盤に固定／書いた 4 ファイルを `get_projection_matrix` で読むと K[R|T]（cm）と一致／マス寸法 3.0±0.05 cm。

### 段階 5: 校正ランナー（2・3・4 の後）

- `BoardCollector`（純粋なロジック）: Mac の灰色画像のリングを 1.5 秒ぶん持つ。Pixel の画像は届いたら保留し、リングが t+150 ms まで埋まってから判定。ペアの採用条件は (1) 時刻差 40 ms 以内の Mac フレームがあり両側で盤が見つかる、(2) 盤が静止（t±150 ms の Mac フレームで角点の最大移動 1.0 px 以下）、(3) 新しい姿勢（既採用との cam0 角点の平均距離が対角で正規化して 0.05 超）。片側だけ見つかればそのカメラの単体ビュー。既定の必要数は単体 15・ペア 12（キャッシュがあるカメラは単体不要。ただし今回のペアへの再投影誤差が 1.5 px を超えたら警告）。
- 入口 `app/runners/hybrid_calibrate.py`: 位相 A（向き合わせ。`PREVIEW`）→ Space で位相 B（盤集め。推論を止めて `CALIBRATION`）→ 数がそろえば自動推定 → 窓に rms・基線（cm）・マス寸法誤差（mm）・再投影の重ね描き → rms ≤ 1 px かつ誤差 ≤ 1 mm なら自動保存、そうでなければ s 保存 / r やり直し / q 中止。盤の寸法は `resources.resource_root()/calibration_settings.yaml`（4×7、3.0 cm）を既定に `--rows/--cols/--square-cm` で上書き。

完了条件: テスト緑。実機で rms ≤ 1 px、マス寸法誤差 ≤ 1 mm、基線が巻尺と 5% 以内。**天板を鉛直に開いておく**（重力 `axis` モードは cam0 が水平である前提）。

### 段階 6: 計測ランナー（2・4 の後。実データは 5 の後）

- `app/runners/network_measure.py` の `NetworkMeasurement` に省略可能な `lens`（役割ごとの K・歪み・寸法）を足し、`_pixel_keypoints` の後で `cv.undistortPoints`（P=K）。画像の外の点は NaN。既定 None で今の挙動のまま（`tests/test_path_consistency.py` はこのメソッドを上書きしているので無変更で緑のこと）。
- `app/hybrid/recorder.py`（ループのスレッド専用）: `~/Documents/WheelchairTorque/hybrid/measure/<ts>/` に `kpts3d_<ts>.csv`（USB と同じ列）、局所トルク、サイクル仕事、`frames_<ts>.csv`（frame, t_ns, t_s, cycle_detected）、`landmarks2d_<ts>.csv`（両ロールの生 2D）、校正 4 ファイルの写し、`meta.json`（開始時に書き終了時に更新）。1 秒ごとに flush。
- 入口 `app/runners/hybrid_measure.py`: `--calibration`（既定 latest）、`--camera`、`--body-mass`、`--gravity-mode axis|trunk`、`--preview-hz 2`、`--cam0-offset-ms 0`、`--port`。Mac の実寸が meta と違えば終了コード 2。Pixel は `on_hello` で device_id を照合し違えば 1008 で拒否。寸法が校正時と違うフレームは捨てて数え、30 連続で終了コード 3。停止は `StopRequest`／q／SIGINT。終了は「サーバ停止 → 残りを drain → process → close」の順（ループのスレッドで）。

完了条件: テスト緑（歪み入り 2D から 3D 誤差 0.5 cm 未満、偽カメラ＋台本の MockPhone で 3 秒流して kpts3d が真値 1 cm 以内・1.5 秒時点で CSV に行・停止で meta 完成・device_id 不一致を拒否）。実機で肩幅が巻尺 ±2 cm、押し中の前腕長の標準偏差 < 1.5 cm。

### 段階 7: GUI（5・6 の後）

- `app/entry.py`: `WORKER_MODULES` に `hybrid_calibrate` と `hybrid_measure`。`uses_stop_file(role)`（realtime と混成 2 役割）。
- `app/runners/worker.py`: 停止ファイルを置く条件（今は `self.role == "realtime"`）を `entry.uses_stop_file(self.role)` に。
- `app/shell/page_measure.py`・`page_calibrate.py`: 「入力: USB カメラ 2 台／Mac＋Pixel（混成）」コンボで `self._runner.role` を差し替え。実行中は無効、出力先の表示も切り替え。映像は子プロセスの OpenCV 窓（USB 経路と同じ）。

完了条件: テスト緑。GUI から開始・停止でき、2 秒以内に終わって CSV が残る。

### 最後に文書

- `mobile/README.md`: 1 台構成・混成の手順、macOS のビデオエフェクト（Center Stage など）を切る、天板を鉛直に、校正の撮り方（盤を大きく静止、焦点は無限遠固定なので離して撮る）。
- `KNOWN_ISSUES.md` に下の申し送りを追記。

---

## 4. 気をつけること（この作業で分かったこと）

- **close の理由は UTF-8 で 123 バイトまで**。超えると相手には 1008 ではなく 1011 が届き理由が消える。`app.net.server.close_reason()` を通す。
- **後勝ちで古い接続を閉じ終わるのを待たない**（相手が応答しないと close_timeout 10 秒の間、新しい端末の同期が止まる）。`_take_over` は閉じるのを裏のタスクにしてある。
- 同期バッファ（`SyncBuffer`）にロックは無い。push と drain はループのスレッドだけで行う（`PhoneLink.inject` 経由）。
- Android の撮影時刻は `analyze()` に入った時刻でセンサ時刻ではない。Mac 側も grab 直後で揃えるが、2 経路の一定の遅延差は残る（`--cam0-offset-ms` と生 2D の記録まで。推定は申し送り）。
- `estimate_gravity("axis")` は cam0（Mac）が水平である前提。
- `mobile/local.properties`・`keystore.properties`・`keystore/` は git 管理外。新しい worktree では本体からコピーする。鍵はバックアップ必須。

---

## 5. スコープ外（申し送り）

- 配布（PyInstaller）と CI
- `calib.py` の単位修正（m → cm）、USB 経路の歪み補正
- phone-path ブランチの 2f313fe（workdir・events・control）と `stop_request` の統合
- Android の自動再接続（今は校正→計測のたびに QR を読み直す）
- 遅延差の根本対策（Android は `imageInfo.timestamp`、Mac は AVFoundation の PTS）と遅延差の自動推定ツール（生 2D と校正から −100〜+100 ms を走査してエピポーラ誤差最小を探す）
- 重力の基準を「天板を鉛直に」以外で決めること（床に置いた盤、Pixel の加速度計）
- 混成経路の 3D に EKF・LPF をかけること、映像を Qt の中に取り込むこと、Mac の露出固定
- アプリの細かい不具合: 横向きで「切断」ボタンの右端が画面から切れる／QR 読み取りを始めると案内文が解像度の表示ですぐ上書きされる
