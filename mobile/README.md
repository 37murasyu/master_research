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
