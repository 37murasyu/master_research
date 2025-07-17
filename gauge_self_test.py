"""GaugeDisplay のロジック部分セルフテスト。
GUIなし環境でも角度計算としきい値挙動を検証する。

実行例:
    python gauge_self_test.py
"""
from Gauge_display import GaugeDisplay
import json, tempfile, os, math, random

CONFIG_TEMPLATE = {
    "image_extent": [0, 10, 0, 10],
    "xlim": [0, 10],
    "ylim": [0, 10],
    "image_scale": 1.0,
    "gauges": [
        {"center": [2, 5], "label": "Right Upper Arm"},
        {"center": [5, 5], "label": "Right Forearm"},
        {"center": [8, 5], "label": "Left Upper Arm"},
        {"center": [5, 8], "label": "Left Forearm"}
    ],
    # energy thresholds keyed by internal part_keys mapping -> elbow_R, wrist_R, elbow_L, wrist_L
    "energy_thresholds": {
        "elbow_R": [8.0, 16.0],
        "wrist_R": [5.0, 12.0],
        "elbow_L": [8.0, 16.0],
        "wrist_L": [5.0, 12.0]
    }
}

# 仮統計 (平均0, 分散1)
stats = {k: (0.0, 1.0) for k in ["wrist_R","elbow_R","wrist_L","elbow_L"]}


def run_basic():
    with tempfile.TemporaryDirectory() as td:
        cfg_path = os.path.join(td, 'conf.json')
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(CONFIG_TEMPLATE, f)
        g = GaugeDisplay(cfg_path, stats, image_path="wheelchair_user.png", debug=False)
        # フィルタ (上腕・前腕 4つ想定)
        g.filter_parts(['wrist_R','elbow_R','wrist_L','elbow_L'])
        # 角度初期
        a0 = g.get_angles()
        assert len(a0) == 4
        # 0 -> 全部 180 近傍 (しきい区分により 180)
        if any(abs(x-180.0) > 1e-6 for x in a0):
            print('[FAIL] 初期角度が 180 でない', a0)
        # エネルギーしきい値 Low 付近
        test_imp = {"elbow_R": CONFIG_TEMPLATE['energy_thresholds']['elbow_R'][0]*0.5}
        g.update_impulses(test_imp)
        a1 = g.get_angles()
        if not (170.0 > a1[g.part_keys.index('elbow_R')] > 150.0):
            print('[WARN] elbow_R 角度が想定範囲外 (low中間)', a1)
        # High 付近超過
        test_imp2 = {"elbow_R": CONFIG_TEMPLATE['energy_thresholds']['elbow_R'][1]*1.2}
        g.update_impulses(test_imp2)
        a2 = g.get_angles()
        if not (abs(a2[g.part_keys.index('elbow_R')] - 60.0) < 1e-6):
            print('[FAIL] elbow_R high 超過でも 60° に張り付かない', a2)
        # ランダム試験
        for _ in range(50):
            vals = {k: random.uniform(0, 20.0) for k in g.part_keys}
            g.update_impulses(vals)
            angs = g.get_angles()
            if any(not (0.0 <= x <= 180.0) for x in angs):
                print('[FAIL] 角度範囲外検出', angs)
                break
        print('[OK] 基本セルフテスト完了')


if __name__ == '__main__':
    run_basic()
