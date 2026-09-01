import glob
import math
import os

rows = []
files = glob.glob(r"C:/Users/villa/Desktop/master_Research/**/c1.dat", recursive=True)

for p in files:
    try:
        K = []
        D = []
        mode = None
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                low = ln.lower()
                if low.startswith("intrinsic"):
                    mode = "K"
                    continue
                if low.startswith("distortion"):
                    mode = "D"
                    continue
                vals = list(map(float, ln.split()))
                if mode == "K":
                    K.append(vals)
                elif mode == "D":
                    D.extend(vals)
        if not D:
            D = [0, 0, 0, 0, 0]
        k1, k2, p1, p2, k3 = (D + [0, 0, 0, 0, 0])[:5]
        radial = lambda r: 1 + k1 * r * r + k2 * r ** 4 + k3 * r ** 6
        r05 = radial(0.5)
        r1 = radial(1.0)
        rows.append((p, k1, k2, k3, p1, p2, r05, r1))
    except Exception as e:
        rows.append((p, "ERR", str(e), "", "", "", "", ""))

rows.sort()
for r in rows:
    print(r)
