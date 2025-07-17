import sys
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPalette, QColor

# 仕事量計算機能

def compute_link_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    BA = A - B
    BC = C - B
    dot = np.einsum('ij,ij->i', BA, BC)
    normBA = np.linalg.norm(BA, axis=1)
    normBC = np.linalg.norm(BC, axis=1)
    cos_angle = dot / (normBA * normBC)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    cos_angle = np.nan_to_num(cos_angle, nan=1.0)
    return np.arccos(cos_angle)


def compute_vector_angle(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    v = P - Q
    return np.arctan2(v[:,1], v[:,0])


def work_from_angle_step(torque: float, angle_prev: float, angle_curr: float) -> float:
    return torque * (angle_curr - angle_prev)

class WorkGaugeWidget(pg.PlotWidget):
    def __init__(self, max_work, safe_th, danger_th, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setBackground('k')
        self.max_work, self.safe_th, self.danger_th = max_work, safe_th, danger_th
        self.work = 0.0
        self.setAspectLocked(True)
        self.hideAxis('bottom'); self.hideAxis('left')
        self.showGrid(False, False)
        self.radius = 1.0
        self.start_angle = 1.25 * np.pi
        self.total_span = -3 * np.pi / 2
        self.n_steps = 200
        theta_bg = np.linspace(
            self.start_angle,
            self.start_angle + self.total_span,
            self.n_steps
        )
        x_bg = self.radius * np.cos(theta_bg)
        y_bg = self.radius * np.sin(theta_bg)
        pen_bg = pg.mkPen(color=(200,200,200), width=20, cap=Qt.RoundCap)
        self.plot(x_bg, y_bg, pen=pen_bg)
        self.curve = pg.PlotCurveItem(pen=pg.mkPen(width=20, cap=Qt.RoundCap))
        self.addItem(self.curve)

    def update_gauge(self):
        pct = self.work / self.max_work if self.max_work > 0 else 0
        pct = min(max(np.nan_to_num(pct, nan=0.0), 0.0), 1.0)
        n = max(int(self.n_steps * abs(pct)), 2)
        theta = np.linspace(
            self.start_angle,
            self.start_angle + self.total_span * pct,
            n
        )
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        # 色設定: 無負荷=青, 安全域=緑, 危険域=赤
        color = (0, 0, 200)
        if self.work >= self.safe_th:
            color = (0, 200, 0)
        if self.work >= self.danger_th:
            color = (200, 0, 0)
        self.curve.setData(x, y, pen=pg.mkPen(color=color, width=20, cap=Qt.RoundCap))

    def add_work(self, dw):
        if np.isfinite(dw):
            self.work += dw
            self.update_gauge()

    def reset(self):
        self.work = 0
        self.update_gauge()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('6部位仕事量モニタ')
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor('black'))
        self.setPalette(pal)

        self.parts = [
            ('左肩', 'both_shoulder_L_torque_local_z', (1,0,2)),
            ('右肩', 'both_shoulder_R_torque_local_z', (0,1,3)),
            ('左上腕', 'up_arm_l_torque_local_z', (0,2,4)),
            ('右上腕', 'upper_arm_R_torque_local_z', (1,3,5)),
            ('左前腕', 'forearm_L_torque_local_z', (2,4)),
            ('右前腕', 'forearm_R_torque_local_z', (3,5)),
        ]

        central = QtWidgets.QWidget()
        central.setStyleSheet('background-color:black;')
        grid = QtWidgets.QGridLayout(central)

        df_t = pd.read_csv(r"C:\Users\villa\Desktop\master_Research\ONLYlocal161234_takizawa_with_cycles_ver2.csv")
        df_p = pd.read_csv(r"C:\Users\villa\My project (2)\Assets\GoTounity\output_data\1_滝沢_正しいkpts3d_0407_161234.csv")

        self.gauges = {}
        self.theta = {}
        for idx, (name, col, jidx) in enumerate(self.parts):
            tor = df_t[col].values
            arr_p = df_p.values
            if len(jidx) == 3:
                A = arr_p[:, jidx[0]*3:(jidx[0]*3+3)]
                B = arr_p[:, jidx[1]*3:(jidx[1]*3+3)]
                C = arr_p[:, jidx[2]*3:(jidx[2]*3+3)]
                th_full = compute_link_angle(A, B, C)
            else:
                P = arr_p[:, jidx[0]*3:(jidx[0]*3+3)]
                Q = arr_p[:, jidx[1]*3:(jidx[1]*3+3)]
                th_full = compute_vector_angle(P, Q)
            skip = len(th_full) - len(tor)
            theta_arr = th_full[skip:]
            self.theta[name] = theta_arr

            # nanを除去して仕事量算出
            delta_theta = np.nan_to_num(np.diff(theta_arr))
            work_steps = tor[1:] * delta_theta
            total_work = np.nansum(np.abs(work_steps))
            max_work = total_work * 1.3/12
            safe_th = 0.3 * max_work
            danger_th = 0.9 * max_work

            gw = WorkGaugeWidget(max_work, safe_th, danger_th)
            lbl = QtWidgets.QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet('color:white; font-size:24pt;')
            box = QtWidgets.QVBoxLayout()
            box.addWidget(gw)
            box.addWidget(lbl)
            w = QtWidgets.QWidget()
            w.setLayout(box)
            w.setStyleSheet('background-color:black;')
            row, col_pos = divmod(idx, 2)
            grid.addWidget(w, row, col_pos*2)

            setattr(self, f'torque_{name}', tor)
            setattr(self, f'cycle_{name}', df_t['cycle'].values)
            self.gauges[name] = gw

        pix = QPixmap(r"C:\Users\villa\My project (2)\Assets\GoTounity\wheelchair_user.png")
        img = QtWidgets.QLabel()
        img.setStyleSheet('background-color:black;')
        img.setPixmap(pix.scaled(400,400,Qt.KeepAspectRatio))
        grid.addWidget(img,1,1)

        self.setCentralWidget(central)
        self.idx = 1
        self.dt = 0.3
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_step)
        self.timer.start(int(self.dt*1000))

    def update_step(self):
        for name, _, _ in self.parts:
            tor = getattr(self, f'torque_{name}')
            cyc = getattr(self, f'cycle_{name}')
            th = self.theta[name]
            if cyc[self.idx] != cyc[self.idx-1]:
                self.gauges[name].reset()
            if self.idx < len(th):
                dw = work_from_angle_step(tor[self.idx], th[self.idx-1], th[self.idx])
                self.gauges[name].add_work(dw)
        self.idx += 1
        if self.idx >= len(next(iter(self.theta.values()))):
            self.idx = 1

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(800, 800)
    w.show()
    sys.exit(app.exec_())
