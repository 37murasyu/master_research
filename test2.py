# pylint: disable=no-member
import cv2 as cv
import numpy as np

from config import win_main, win_second

# モニターの解像度を取得

# 白と黒の画像
white_frame = 255 * np.ones((300, 400, 3), dtype=np.uint8)
black_frame = np.zeros((300, 400, 3), dtype=np.uint8)

# ウィンドウ生成
cv.namedWindow(win_main, cv.WINDOW_NORMAL)
cv.namedWindow(win_second, cv.WINDOW_NORMAL)

# モニター①: (0, 0) ～ (1280, 720)
cv.moveWindow(win_main, 100, 100)  # メインの中ほどに表示

# モニター②: (1200, -1080) ～ (3120, 0)
cv.moveWindow(win_second, 1300, -900)  # セカンドの中ほどに表示（要調整）

# 表示ループ
while True:
    cv.imshow(win_main, white_frame)
    cv.imshow(win_second, black_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cv.destroyAllWindows()


import matplotlib

matplotlib.use("TkAgg")  # TkAggバックエンドを使用
import matplotlib.pyplot as plt

import tkinter
from config import win_main_point, win_second_point

# プロットを作成
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.canvas.manager.set_window_title("Matplotlib on Second Monitor")

# tkinter ウィンドウハンドルを取得
mgr = plt.get_current_fig_manager()
window = mgr.window  # これは tkinter.Tk() インスタンス

# 位置移動：セカンドモニターの中ほどへ（例：1200, -900）
x1, y1, x2, y2 = win_second_point
w = (x2 - x1) / 2
h = abs(y2 - y1)  # 高さの絶対値（マイナス方向なので）

window.geometry(f"{int(w)}x{h}+{x1}+{y1}")  # "{幅}x{高さ}+{X座標}+{Y座標}"

# window.wm_geometry("+1300+-900")  # X=1300, Y=-900（マイナスで上へ）

plt.show()
