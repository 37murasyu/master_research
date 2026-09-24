"""``tools/compare_dt.py`` の注記が、同じ出力の表の比と食い違わないことを固定する。

**なぜこのテストがあるか。**

角速度は R-2 で標準形 ``ω = (r × ṙ)/|r|²`` に直したので、``ω ∝ 1/dt``・``ω̇ ∝ 1/dt²``（``tests/test_dynamics_dt.py`` の
``TestScalingLaws``）。表はこの今の式で計算した比（dt 0.3 → 1/30 なら ×9・×81）を出すのに、注記とコメントは直す前の
別式の ``1/dt²``・``1/dt³``（×81・×729）のままで、読む人が表と注記のどちらを信じればよいか分からなかった。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from tools import compare_dt

N, JOINTS, FPS = 90, 16, 30.0


def _swinging_body(path):
    """全関節を原点まわりに揺らす（角速度が時間で変わるので角加速度も 0 でない）。単位は m。"""
    t = np.arange(N) / FPS
    theta = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    base = np.array([[0.1 * j, 0.02 * (j % 3), 0.05 * (j % 4)] for j in range(JOINTS)])   # 前腕 R は約 0.23 m
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    points = np.stack([c * base[:, 0] - s * base[:, 1], s * base[:, 0] + c * base[:, 1],
                       np.broadcast_to(base[:, 2], (N, JOINTS))], axis=-1)
    table = pd.DataFrame(points.reshape(N, -1), columns=[f"joint_{j}_{a}" for j in range(JOINTS) for a in "xyz"])
    table.insert(0, "frame", range(N))
    table.to_csv(path, index=False)
    return path


def test_the_note_matches_the_ratios_in_the_table(tmp_path, capsys):
    path = _swinging_body(tmp_path / "kpts3d_test.csv")
    assert compare_dt.main(["--input", str(path), "--old-dt", "0.3", "--new-dt", str(1 / 30)]) == 0
    out = capsys.readouterr().out
    ratio = {label: float(re.search(rf"\| forearm_R \| {label} \|[^\n]*\| ×([\d.,]+) \|", out).group(1).replace(",", ""))
             for label in ("角速度", "角加速度")}
    assert ratio == {"角速度": 9.0, "角加速度": 81.0}
    assert "角速度が `1/dt`、角加速度が `1/dt²`" in out, "注記が表の比（×9・×81）と食い違う"
