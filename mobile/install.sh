#!/usr/bin/env bash
# Android 端末へアプリを入れる。
#
#   ./install.sh            リリース版を入れる（配布と同じもの）
#   ./install.sh --debug    デバッグ版を入れる（ログが多い）
#   ./install.sh --build    ビルドし直してから入れる
#
# 使う前に端末側で開発者向けオプションと USB デバッグを有効にすること。
# 手順は README.md を参照。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VARIANT="release"
FORCE_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --debug)   VARIANT="debug" ;;
    --release) VARIANT="release" ;;
    --build)   FORCE_BUILD=1 ;;
    *) echo "不明な引数: ${arg} （使えるのは --debug / --release / --build）" >&2; exit 1 ;;
  esac
done

die() { echo "エラー: $*" >&2; exit 1; }

# --- adb を探す -------------------------------------------------------------
ADB="$(command -v adb || true)"
if [ -z "$ADB" ]; then
  for candidate in "$HOME/Library/Android/sdk/platform-tools/adb" \
                   "$HOME/Android/Sdk/platform-tools/adb" \
                   "$ANDROID_HOME/platform-tools/adb"; do
    if [ -x "$candidate" ]; then ADB="$candidate"; break; fi
  done
fi
[ -n "$ADB" ] || die "adb が見つかりません。Android SDK の platform-tools を入れてください。"

# --- 端末を確認 -------------------------------------------------------------
"$ADB" start-server >/dev/null 2>&1 || true
DEVICE_LINES="$("$ADB" devices | tail -n +2 | grep -v '^$' || true)"

if [ -z "$DEVICE_LINES" ]; then
  die "端末が見つかりません。次を確認してください:
  1. USB ケーブルで PC と繋ぐ（充電専用ケーブルでは認識しません）
  2. 端末の 設定 > デバイス情報 > ビルド番号 を 7 回タップして開発者になる
  3. 設定 > システム > 開発者向けオプション > USB デバッグ をオン
  4. 接続時に端末に出る「USB デバッグを許可しますか」で許可"
fi

if echo "$DEVICE_LINES" | grep -q "unauthorized"; then
  die "端末が未承認です。端末の画面に出る「USB デバッグを許可しますか」で許可してください。"
fi

DEVICE_COUNT="$(echo "$DEVICE_LINES" | grep -c "device$" || true)"
[ "$DEVICE_COUNT" -ge 1 ] || die "接続可能な端末がありません:
$DEVICE_LINES"
if [ "$DEVICE_COUNT" -gt 1 ]; then
  die "端末が複数繋がっています。1 台だけにしてください:
$DEVICE_LINES"
fi

MODEL="$("$ADB" shell getprop ro.product.model | tr -d '\r')"
API="$("$ADB" shell getprop ro.build.version.sdk | tr -d '\r')"
ABI="$("$ADB" shell getprop ro.product.cpu.abi | tr -d '\r')"
echo "接続端末: ${MODEL}  Android API ${API}  ${ABI}"

# APK は arm64-v8a のみを含む（app/build.gradle.kts の abiFilters）
case "$ABI" in
  arm64-v8a) ;;
  *) die "この端末の ABI は ${ABI} ですが、APK は arm64-v8a のみです。
  app/build.gradle.kts の abiFilters を調整して再ビルドしてください。" ;;
esac
[ "$API" -ge 24 ] || die "Android 7.0 (API 24) 以上が必要です。この端末は API ${API} です。"

# --- APK を用意 -------------------------------------------------------------
APK="app/build/outputs/apk/${VARIANT}/app-${VARIANT}.apk"

if [ "$VARIANT" = "release" ] && [ ! -f keystore.properties ]; then
  die "リリース鍵がありません（keystore.properties）。
  --debug でデバッグ版を入れるか、README.md の手順で鍵を作ってください。"
fi

if [ "$FORCE_BUILD" = "1" ] || [ ! -f "$APK" ]; then
  echo "==> ${VARIANT} をビルドします"
  TASK="assemble$(echo "${VARIANT:0:1}" | tr '[:lower:]' '[:upper:]')${VARIANT:1}"
  # Android Studio 同梱の JDK を使う。java_home は認識しないことがある。
  STUDIO_JDK="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
  if [ -z "${JAVA_HOME:-}" ] && [ -d "$STUDIO_JDK" ]; then
    export JAVA_HOME="$STUDIO_JDK"
  fi
  ./gradlew ":app:${TASK}" --console=plain
fi

[ -f "$APK" ] || die "APK が見つかりません: ${APK}"
echo "==> ${APK} ($(du -h "$APK" | cut -f1)) を転送します"

# -r で上書き更新。署名が違うと失敗するので、その場合の対処も示す。
if ! "$ADB" install -r "$APK"; then
  echo "" >&2
  die "インストールに失敗しました。
  署名が違う版が既に入っている場合は、一度消してから入れ直してください:
    ${ADB} uninstall com.murayama.wheelchairsensor"
fi

echo ""
echo "完了しました。端末のアプリ一覧に「車椅子計測センサ」が入っています。"
echo ""
echo "使い方:"
echo "  1. PC 側で受信サーバを起動する（リポジトリのルートで）:"
echo "       .venv/bin/python -m app.net.server"
echo "  2. 表示された cam0 / cam1 の URL を QR コードにして端末で読み取る"
echo "  3. 2 台目には別の役割（cam1）の QR を読ませる"
