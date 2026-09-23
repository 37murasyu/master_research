# 車椅子計測センサ（Android）

スマホを無線の姿勢センサにするアプリ。端末側で MediaPipe を実行し、映像ではなく
**33 点のランドマークだけを時刻付きで PC へ送る**。1 フレーム約 600 バイト、
30fps でも 18 KB/s に収まり、PC 側は映像のデコードが不要。

PC 側の受信層は `../app/net/` にある。

## 動作条件

| | |
|---|---|
| Android | 7.0 (API 24) 以上 |
| CPU | **arm64-v8a のみ**（Pixel 7a を含む 2019 年以降の端末はすべて該当） |
| APK サイズ | 約 33MB |
| 通信 | PC と**同じ Wi-Fi** にいること |

32bit 端末や Intel 機で動かす必要が出たら、`app/build.gradle.kts` の
`abiFilters` を調整して再ビルドすること（全 ABI を含めると 79MB になる）。

## 初回の準備

### 1. 端末側でデバッグを有効にする

まず開発者になる:

1. 設定 → デバイス情報 → **ビルド番号を 7 回タップ**

以降は Wi-Fi と USB のどちらでもよい。**Wi-Fi のほうが手軽**（Android 11 以降）。

**Wi-Fi の場合（ケーブル不要・推奨）**

2. 設定 → システム → 開発者向けオプション → **ワイヤレスデバッグ**をオン
3. PC 側で初回だけペア設定する:

   ```sh
   cd mobile && ./install.sh --pair
   ```

   案内に従って端末で「ペア設定コードによるデバイスのペア設定」を開き、
   表示された 6 桁コードを入力する。以降は `./install.sh --wifi` だけでよい。

**USB の場合**

2. 設定 → システム → 開発者向けオプション → **USB デバッグ**をオン
3. USB で PC に繋ぐ（**充電専用ケーブルでは認識しない**）
4. 端末に出る「USB デバッグを許可しますか」で**許可**

### 2. リリース署名鍵を用意する

配布用の APK には固定の鍵で署名する。デバッグ鍵は PC ごとに異なるため、
別の PC でビルドした版を上書き更新できなくなる。

```sh
cd mobile
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
"$JAVA_HOME/bin/keytool" -genkeypair -v \
  -keystore keystore/release.jks -alias wheelchair-sensor \
  -keyalg RSA -keysize 4096 -validity 10950 \
  -dname "CN=Wheelchair Torque Sensor, O=Murayama, C=JP"
```

続いて `keystore.properties` を作る（`storeFile` はリポジトリからの相対パス）:

```properties
storeFile=keystore/release.jks
storePassword=<設定したパスワード>
keyAlias=wheelchair-sensor
keyPassword=<設定したパスワード>
```

> **鍵は必ずバックアップすること。** 失うと同じアプリとして更新版を配れなくなる
> （Android は署名が一致しない APK の上書きインストールを拒否する）。
> `keystore/` と `keystore.properties` は git 管理外にしてある。

鍵が無い環境でもビルド自体は通る（署名なし APK になり、インストールはできない）。

## インストール

```sh
cd mobile
./install.sh              # 繋がっている端末に入れる（USB / Wi-Fi どちらでも）
./install.sh --wifi       # Wi-Fi で接続してから入れる
./install.sh --pair       # Wi-Fi の初回ペア設定
./install.sh --debug      # デバッグ版（ログが多い）
./install.sh --build      # ビルドし直してから入れる
```

端末の ABI と API レベルを確認したうえで転送する。条件を満たさない場合は
理由を出して止まる。Wi-Fi 経由は USB より転送に時間がかかる（33MB）。

APK を手渡しで配る場合は `app/build/outputs/apk/release/app-release.apk` を
そのまま渡し、受け取り側で「提供元不明のアプリ」を許可してもらう。

## 使い方

### 1. PC 側で受信サーバを起動する

リポジトリのルートで:

```sh
.venv/bin/python -m app.net.server
```

cam0 と cam1 それぞれの接続先が **QR コードで表示される**。

### 2. 端末で QR を読み取る

アプリを開き「PCのQRコードを読み取る」→ カメラを QR に向ける。
**2 台目には cam1 の QR を読ませること**（役割は QR に含まれている）。

接続すると自動で時刻同期が走り、完了すると送信が始まる。
画面に往復遅延（RTT）と送信フレーム数が出る。

### 3. 計測

PC 側のサーバが受信状況を 1 秒ごとに表示する:

```
接続 2台 ['cam0', 'cam1']  受信 163  ペア 81  欠測破棄 0  位相差 1.6ms  不正 0
```

**位相差**が 2 台の時刻ずれ。数 ms なら三角測量に十分な精度。
1 フレーム（33ms）を大きく超えるようなら Wi-Fi の状態を疑う。

## カメラの設置と校正

このアプリは**オートフォーカス・自動露出・ホワイトバランス・手ブレ補正を
すべて固定**する。オートフォーカスが動くとレンズの焦点距離が変わり、
校正で求めたカメラ行列が実態と合わなくなるため。

したがって:

- **端末を固定してから校正すること**（三脚など）
- **校正後にカメラの位置・向きを変えたらやり直すこと**
- **機種ごとに個別の校正が必要**（内部パラメータが違う）

内部パラメータは機種ごとに一度、チェッカーボードを撮影して求める。
外部パラメータは設置のたびに求め直す。

## うまくいかないとき

| 症状 | 確認すること |
|---|---|
| `install.sh` が端末を見つけない | **USB**: 充電専用ケーブルでないか、USB デバッグがオンか、許可ダイアログに応じたか。**Wi-Fi**: ワイヤレスデバッグがオンか、同じ Wi-Fi にいるか、ペア設定済みか |
| `--pair` で端末が自動検出されない | ペア設定の画面（6 桁コードが出ている画面）を開いたままにする。閉じると広告が止まる。手入力でも可 |
| Wi-Fi 接続が切れる | 端末が省電力で Wi-Fi を切っている。充電しながら試す。`adb connect <IP:ポート>` で再接続 |
| インストールが署名エラーで失敗 | 別の鍵の版が入っている。`adb uninstall com.murayama.wheelchairsensor` してから再実行 |
| QR を読んでも繋がらない | PC と端末が同じ Wi-Fi にいるか。PC のファイアウォールがポートを塞いでいないか |
| 「時刻同期中…」から進まない | サーバが応答していない。PC 側のログを確認 |
| 位相差が大きい | Wi-Fi の混雑。2.4GHz なら 5GHz に変える |
| 姿勢が検出されない | 全身が画角に入っているか。明るさが足りているか |

## 開発

```sh
./gradlew :app:testDebugUnitTest   # ユニットテスト
./gradlew :app:assembleRelease     # リリース APK
```

`ContractSampleTest` は PC 側と突き合わせるための電文サンプルを
`contract/golden_messages.json` に書き出す。電文の形を変えたらこれを更新し、
PC 側の `tests/test_protocol_contract.py` も通ることを確認すること。
片側だけのテストでは、両者が同じものを想定していることを保証できない。

## Mac 内蔵カメラ＋Pixel 1 台の混成ステレオ

Mac を cam0（原点）、Pixel の背面カメラを cam1 とする。Mac と Pixel を同じ Wi-Fi に接続する。
Mac のビデオエフェクト（Center Stage、ポートレートなど）はオフにし、計測中はカメラの位置と向きを固定する。
重力 `axis` モードではカメラが水平である必要があるため、Mac の天板を鉛直に開く。

Android は release 版で更新する。debug 版は署名鍵と端末 ID が変わるため、校正を流用できない。

```sh
cd ../master_research-hybrid/mobile
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./install.sh --build
cd ..
../master_research/.venv/bin/python -m app.runners.hybrid_preview --camera 0
```

左右パネルの右側にある cam1 の QR を Pixel アプリで読む。Mac カメラが別の番号なら `--camera 1` などを指定する。
macOS に受信接続の確認が出たら許可する。左に Mac、右に Pixel の画像と同じ時刻の骨格が表示される。
終了は q / Esc / ウィンドウを閉じる / Ctrl-C。起動のたびに新しい QR を読み直す。

### 校正

```sh
../master_research/.venv/bin/python -m app.runners.hybrid_calibrate --camera 0
```

向き合わせ後、Space で盤集めに移る。既定は内側交点 4×7、1 マス 3.0 cm。
必要なら `--rows 4 --cols 7 --square-cm 3.0` で変更する（片方偶数、片方奇数が必須）。
盤全体を大きく写し、毎回静止させる。採用数が増えたら位置・距離・傾きを変える。
Pixel の焦点は無限遠固定なので、盤がぼけない距離まで離し、十分な照明を用意する。

単体は各 15 ビュー、ステレオは 12 ペアを集める。端末 ID・解像度が同じ内部パラメータの
キャッシュがあれば単体の撮り直しを省略する。再校正は `--no-cache`。
RMS ≤ 1 px、マス寸法誤差 ≤ 1 mm を満たせば自動保存する。
それ以外は結果を確認して s で保存、r でやり直し、q で中止する。
保存先は `~/Documents/WheelchairTorque/hybrid/calibration/`。
混成の並進 T は **cm**。既存 USB 校正のファイルと混在させない。

### 計測

```sh
../master_research/.venv/bin/python -m app.runners.hybrid_measure \
  --camera 0 --body-mass 60 --gravity-mode axis --preview-hz 2
```

既定で最新校正を使用する。別の校正は `--calibration <保存ディレクトリ>`。
校正した Pixel の端末 ID が違う場合は接続を拒否する。解像度不一致も破棄し、30 フレーム連続で終了する。
`--cam0-offset-ms` は Mac 側時刻に加える補正量（既定 0）。遅延差の自動推定は未実装。
表示負荷の比較には `--preview-hz 0` と `--preview-hz 4` を使う。

保存先は `~/Documents/WheelchairTorque/hybrid/measure/`。USB と同じ列の `kpts3d_*.csv`、
局所トルク、サイクル仕事、時刻、両側の生 2D、校正の写し、メタ情報を保存する。
CSV は 1 秒ごとに flush し、通常停止時は残ったペアを処理してから閉じる。
サイクル仕事はスマホ経路の関節仕事率の積分であり、USB の肘専用エネルギー計算との直接比較には使わない。

GUI では校正・計測画面の「入力」を「Mac＋Pixel（混成）」へ切り替えて開始する。
映像は別ウィンドウで開く。計測画面の CAM0・体重設定が反映される。
混成経路では USB 用の動画入力・EKF・LPF などの設定は使わない。

### 実機で残る確認

- Mac ≥ 25 fps、Pixel ≥ 20 fps、ペア ≥ 20/秒、平均位相差 < 20 ms。
- Pixel 表示 ≥ 3 Hz、表示 0 Hz と 4 Hz でランドマーク fps の低下 ≤ 10%。
- 校正 RMS ≤ 1 px、マス誤差 ≤ 1 mm、基線が巻尺の ±5%。
- 肩幅が巻尺の ±2 cm、押し中の前腕長標準偏差 < 1.5 cm。
- GUI 停止から 2 秒以内に終了して CSV が残ること。

自動テストは模擬端末と合成座標による検証であり、これらの実機精度・性能を保証するものではない。
