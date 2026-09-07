#!/usr/bin/env bash
# Android 端末へアプリを入れる。USB でも Wi-Fi でも入れられる。
#
#   ./install.sh              繋がっている端末に入れる（USB / Wi-Fi どちらでも）
#   ./install.sh --pair       Wi-Fi 接続の初回設定（ペア設定。USB は不要）
#   ./install.sh --wifi       Wi-Fi 経由で接続してから入れる
#   ./install.sh --debug      デバッグ版を入れる（ログが多い）
#   ./install.sh --build      ビルドし直してから入れる
#
# Wi-Fi で入れるには Android 11 以降が必要（Pixel 7a は該当）。
# 一度ペア設定すれば、以降は --wifi だけで繋がる。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PACKAGE="com.murayama.wheelchairsensor"

VARIANT="release"
FORCE_BUILD=0
MODE="auto"   # auto | wifi | pair

for arg in "$@"; do
  case "$arg" in
    --debug)   VARIANT="debug" ;;
    --release) VARIANT="release" ;;
    --build)   FORCE_BUILD=1 ;;
    --wifi)    MODE="wifi" ;;
    --pair)    MODE="pair" ;;
    --usb)     MODE="auto" ;;
    *) echo "不明な引数: ${arg}
使えるのは --debug / --release / --build / --wifi / --pair / --usb" >&2; exit 1 ;;
  esac
done

die() { echo "エラー: $*" >&2; exit 1; }

# --- adb を探す -------------------------------------------------------------
ADB="$(command -v adb || true)"
if [ -z "$ADB" ]; then
  for candidate in "$HOME/Library/Android/sdk/platform-tools/adb" \
                   "$HOME/Android/Sdk/platform-tools/adb" \
                   "${ANDROID_HOME:-}/platform-tools/adb"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then ADB="$candidate"; break; fi
  done
fi
[ -n "$ADB" ] || die "adb が見つかりません。Android SDK の platform-tools を入れてください。"
"$ADB" start-server >/dev/null 2>&1 || true

# --- mDNS で端末を探す ------------------------------------------------------
# Android 11 以降は、ワイヤレスデバッグを有効にすると自分を mDNS で広告する。
#   _adb-tls-pairing._tcp  ペア設定を待っている（6桁コードの画面を開いている間だけ）
#   _adb-tls-connect._tcp  ペア設定済みで接続を待っている
discover_service() {
  local service="$1"
  # 広告が出るまで少し待つ。端末側が画面を開いた直後は間に合わないことがある。
  local attempt
  for attempt in 1 2 3 4 5 6; do
    local found
    found="$("$ADB" mdns services 2>/dev/null | awk -v svc="$service" '$2 == svc {print $3; exit}')"
    if [ -n "$found" ]; then
      echo "$found"
      return 0
    fi
    sleep 1
  done
  return 1
}

# 既に Wi-Fi で繋がっている端末があるか（アドレスが host:port 形式のもの）
wireless_device_connected() {
  "$ADB" devices | tail -n +2 | grep -E '^[0-9a-fA-F.:]+:[0-9]+[[:space:]]+device$' >/dev/null 2>&1
}

# --- ペア設定（初回のみ。USB は不要） ---------------------------------------
do_pair() {
  cat <<'GUIDE'
Wi-Fi 接続の初回設定をします。端末で次の操作をしてください:

  1. 設定 → システム → 開発者向けオプション → ワイヤレスデバッグ をオン
  2. 「ペア設定コードによるデバイスのペア設定」をタップ
  3. 6 桁のコードと IP アドレス:ポート が表示された状態にする
     （この画面を閉じるとペア設定できません）

PC と端末が同じ Wi-Fi にいることも確認してください。
GUIDE
  read -r -p "上記の画面を開いたら Enter を押してください: " _

  echo "==> 端末を探しています…"
  local pair_addr
  if pair_addr="$(discover_service "_adb-tls-pairing._tcp")"; then
    echo "    見つかりました: ${pair_addr}"
  else
    echo "    自動検出できませんでした。画面に出ている値を入力してください。"
    read -r -p "    IPアドレスとポート (例 192.168.1.23:37000): " pair_addr
    [ -n "$pair_addr" ] || die "入力がありません。"
  fi

  echo "==> ペア設定します。画面の 6 桁コードを入力してください。"
  "$ADB" pair "$pair_addr" || die "ペア設定に失敗しました。
  コードの入力ミス、または画面を閉じてしまった可能性があります。
  もう一度 ./install.sh --pair を実行してください。"

  echo ""
  echo "ペア設定が完了しました。続けて接続します。"
  connect_wireless || die "接続できませんでした。
  端末の「ワイヤレスデバッグ」画面に出ている IP:ポート を確認し、
  次を実行してください: ${ADB} connect <IP:ポート>"
  echo "以降は ./install.sh --wifi で入れられます。"
}

# --- Wi-Fi 接続 -------------------------------------------------------------
connect_wireless() {
  if wireless_device_connected; then
    return 0
  fi

  echo "==> Wi-Fi で端末を探しています…"
  local connect_addr
  if connect_addr="$(discover_service "_adb-tls-connect._tcp")"; then
    echo "    見つかりました: ${connect_addr}"
    "$ADB" connect "$connect_addr" >/dev/null 2>&1 || true
  else
    return 1
  fi

  wireless_device_connected
}

# --- 端末の確認 -------------------------------------------------------------
require_device() {
  local lines
  lines="$("$ADB" devices | tail -n +2 | grep -v '^$' || true)"

  if [ -z "$lines" ]; then
    die "端末が見つかりません。

  Wi-Fi で繋ぐ場合（USB ケーブル不要・Android 11 以降）:
    ./install.sh --pair    ← 初回のみ
    ./install.sh --wifi    ← 2 回目以降

  USB で繋ぐ場合:
    1. USB ケーブルで PC と繋ぐ（充電専用ケーブルでは認識しません）
    2. 設定 → デバイス情報 → ビルド番号 を 7 回タップして開発者になる
    3. 設定 → システム → 開発者向けオプション → USB デバッグ をオン
    4. 接続時に端末に出る「USB デバッグを許可しますか」で許可"
  fi

  if echo "$lines" | grep -q "unauthorized"; then
    die "端末が未承認です。端末の画面に出る許可ダイアログで許可してください。"
  fi

  local count
  count="$(echo "$lines" | grep -c "device$" || true)"
  [ "$count" -ge 1 ] || die "接続可能な端末がありません:
$lines"
  if [ "$count" -gt 1 ]; then
    die "端末が複数繋がっています。1 台だけにしてください:
$lines
  Wi-Fi の接続を切るには: ${ADB} disconnect"
  fi
}

# ===========================================================================
if [ "$MODE" = "pair" ]; then
  do_pair
  exit 0
fi

if [ "$MODE" = "wifi" ]; then
  connect_wireless || die "Wi-Fi で端末に繋げませんでした。

  端末側で「ワイヤレスデバッグ」がオンになっているか確認してください。
  まだペア設定していない場合は先に:  ./install.sh --pair"
fi

require_device

MODEL="$("$ADB" shell getprop ro.product.model | tr -d '\r')"
API="$("$ADB" shell getprop ro.build.version.sdk | tr -d '\r')"
ABI="$("$ADB" shell getprop ro.product.cpu.abi | tr -d '\r')"
TRANSPORT="USB"
wireless_device_connected && TRANSPORT="Wi-Fi"
echo "接続端末: ${MODEL}  Android API ${API}  ${ABI}  (${TRANSPORT})"

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
if [ "$TRANSPORT" = "Wi-Fi" ]; then
  echo "    Wi-Fi 経由なので少し時間がかかります。"
fi

# -r で上書き更新。署名が違うと失敗するので、その場合の対処も示す。
if ! "$ADB" install -r "$APK"; then
  echo "" >&2
  die "インストールに失敗しました。
  署名が違う版が既に入っている場合は、一度消してから入れ直してください:
    ${ADB} uninstall ${PACKAGE}"
fi

echo ""
echo "完了しました。端末のアプリ一覧に「車椅子計測センサ」が入っています。"
echo ""
echo "使い方:"
echo "  1. PC 側で受信サーバを起動する（リポジトリのルートで）:"
echo "       .venv/bin/python -m app.net.server"
echo "  2. 表示された QR を端末のアプリで読み取る"
echo "  3. 2 台目には cam1 の QR を読ませる"
