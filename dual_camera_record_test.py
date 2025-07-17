"""
Dual camera simple recorder.

起動 -> 特殊コマンド "##START_DUAL_REC" を標準入力で受けるまで待機 -> 2台カメラ同時録画開始。
ESC キー押下で録画を終了し安全にリソース解放するテスト用スクリプト。

環境変数:
  CAM0, CAM1 : それぞれのカメラ ID (int) もしくは動画パス。未設定時は config.py の input_stream1/2 を利用。
  FPS        : 録画 FPS (デフォルト config.fps)
  OUTPUT_DIR : 保存先フォルダ (デフォルト config.save_dir)
  FOURCC     : FourCC 例 "mp4v" / "XVID" など (既定 mp4v)

出力:
  cam0_<timestamp>.mp4
  cam1_<timestamp>.mp4

注意:
  - 単純録画のみ。処理負荷軽減のため余計な解析は実施しない。
  - ESC 以外にもウィンドウを閉じた場合は終了します。
"""
from __future__ import annotations
import os
import sys
import time
# pylint: disable=no-member
import cv2 as cv
from datetime import datetime

try:
    # 既存設定の再利用
    from config import input_stream1, input_stream2, fps as CFG_FPS, save_dir as CFG_SAVE, timestamp as CFG_TS
except Exception:
    input_stream1 = 0
    input_stream2 = 1
    CFG_FPS = int(os.getenv('FPS', '30'))
    CFG_SAVE = os.getenv('OUTPUT_DIR', 'output_data')
    os.makedirs(CFG_SAVE, exist_ok=True)
    CFG_TS = datetime.now().strftime('%m%d_%H%M%S')

START_TOKEN = '##START_DUAL_REC'
print(f"[INFO] Dual camera recorder standby. Type '{START_TOKEN}' + Enter to start.")

try:
    user_line = input().strip()
    while user_line != START_TOKEN:
        print(f"[INFO] 未入力 or 不一致: '{user_line}'. 再入力してください ({START_TOKEN})")
        user_line = input().strip()
except EOFError:
    print('[WARN] 標準入力が利用できません。即時開始します。')

# 実行パラメータ
def _as_source(v: str, default):
    if v is None:
        return default
    if v.isdigit() and len(v) < 6:  # 短い整数文字列はカメラIDとみなす
        return int(v)
    return v  # パス

cam0_src = _as_source(os.getenv('CAM0', ''), input_stream1)
cam1_src = _as_source(os.getenv('CAM1', ''), input_stream2)
rec_fps = float(os.getenv('FPS', CFG_FPS))
out_dir = os.getenv('OUTPUT_DIR', CFG_SAVE)
os.makedirs(out_dir, exist_ok=True)

start_ts = datetime.now().strftime('%m%d_%H%M%S')
out0 = os.path.join(out_dir, f'cam0_{start_ts}.mp4')
out1 = os.path.join(out_dir, f'cam1_{start_ts}.mp4')

print(f"[INFO] Start recording -> cam0={cam0_src} cam1={cam1_src} fps={rec_fps}")
cap0 = cv.VideoCapture(cam0_src)
cap1 = cv.VideoCapture(cam1_src)
if not cap0.isOpened():
    print('[ERROR] cam0 open failed')
    sys.exit(1)
if not cap1.isOpened():
    print('[ERROR] cam1 open failed')
    sys.exit(1)

# 1フレーム取得してサイズ確定
ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()
if not (ret0 and ret1):
    print('[ERROR] 事前フレーム取得失敗')
    sys.exit(1)

h0, w0 = frame0.shape[:2]
h1, w1 = frame1.shape[:2]
print(f"[INFO] cam0 size={w0}x{h0} cam1 size={w1}x{h1}")

fourcc = cv.VideoWriter_fourcc(*os.getenv('FOURCC', 'mp4v'))
writer0 = cv.VideoWriter(out0, fourcc, rec_fps, (w0, h0))
writer1 = cv.VideoWriter(out1, fourcc, rec_fps, (w1, h1))
if not writer0.isOpened() or not writer1.isOpened():
    print('[ERROR] VideoWriter open failed')
    cap0.release(); cap1.release()
    sys.exit(1)

cv.namedWindow('REC_CAM0', cv.WINDOW_NORMAL)
cv.namedWindow('REC_CAM1', cv.WINDOW_NORMAL)
cv.moveWindow('REC_CAM0', 0, 0)
cv.moveWindow('REC_CAM1', w0 + 10, 0)

print('[INFO] Recording... ESC で終了')
frame_count = 0
start_time = time.perf_counter()

try:
    while True:
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not (r0 and r1):
            print('[INFO] カメラフレーム取得終了/失敗 -> 停止')
            break
        writer0.write(f0)
        writer1.write(f1)
        cv.imshow('REC_CAM0', f0)
        cv.imshow('REC_CAM1', f1)
        frame_count += 1
        k = cv.waitKey(1) & 0xFF
        if k == 27:  # ESC
            print('[INFO] ESC 受信 -> 停止')
            break
        # ウィンドウが閉じられた場合も終了
        if cv.getWindowProperty('REC_CAM0', cv.WND_PROP_VISIBLE) < 1 or cv.getWindowProperty('REC_CAM1', cv.WND_PROP_VISIBLE) < 1:
            print('[INFO] ウィンドウ閉じ検出 -> 停止')
            break
except KeyboardInterrupt:
    print('[INFO] KeyboardInterrupt 受信 -> 停止')
finally:
    duration = time.perf_counter() - start_time
    cap0.release(); cap1.release()
    writer0.release(); writer1.release()
    try:
        cv.destroyAllWindows()
    except Exception:
        pass
    fps_eff = frame_count / duration if duration > 0 else 0
    print(f"[INFO] 終了 frames={frame_count} elapsed={duration:.2f}s eff_fps={fps_eff:.1f}")
    print(f"[INFO] 保存: {out0}")
    print(f"[INFO] 保存: {out1}")
