import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


def compute_impulse(torque_series, dt=0.3):
    torque_array = np.array(torque_series)
    positive_impulse = np.sum(torque_array[torque_array > 0]) * dt
    negative_impulse = np.sum(torque_array[torque_array < 0]) * dt
    return positive_impulse, negative_impulse


def compute_cycle_impulses(df, target_column):
    cycle_ids = sorted(df["cycle"].dropna().unique())
    impulse_data = []
    for cycle in cycle_ids:
        cycle_data = df[df["cycle"] == cycle][target_column]
        pos_imp, neg_imp = compute_impulse(cycle_data)
        impulse_data.append(
            {"cycle": cycle, "positive_impulse": pos_imp, "negative_impulse": neg_imp}
        )
    return pd.DataFrame(impulse_data)


# ✅ 正しい動作（standard）
df_standard = pd.read_csv("output_data/local161234_takizawa_with_cycles.csv")
df_standard_impulses = compute_cycle_impulses(df_standard, "wrist_local_z")

# ✅ 間違った動作（test）
df_test = pd.read_csv("output_data/local161802_with_cycle.csv")
df_test_impulses = compute_cycle_impulses(df_test, "wrist_local_z")

# ✅ 負の力積のみを抽出
standard_neg = df_standard_impulses["negative_impulse"].values
test_neg = df_test_impulses["negative_impulse"].values

# ✅ Welchのt検定を実行
t_stat, p_val = ttest_ind(standard_neg, test_neg, equal_var=False)

# ✅ 結果表示
print("=== Welchのt検定結果 ===")
print(f"標準のサイクル数: {len(standard_neg)}")
print(f"テストのサイクル数: {len(test_neg)}")
print(f"t値 = {t_stat:.3f}")
print(f"p値 = {p_val:.4f}")
if p_val < 0.05:
    print("→ 有意差あり（負の力積に統計的な差があります）")
else:
    print("→ 有意差なし（統計的な差は認められません）")

# ✅ CSV保存（必要なら）
df_standard_impulses.to_csv("output_data/impulses_standard.csv", index=False)
df_test_impulses.to_csv("output_data/impulses_test.csv", index=False)


# データ準備
data = [standard_neg, test_neg]
labels = ["Standard", "Test"]

# ✅ 図1: 箱ひげ図
plt.figure(figsize=(8, 5))
plt.boxplot(data, labels=labels, showmeans=True)
plt.title("Negative Impulse per Cycle (Boxplot)")
plt.ylabel("Negative Impulse (Nm·s)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 図2: 平均＋標準偏差の棒グラフ
means = [np.mean(d) for d in data]
stds = [np.std(d) for d in data]

plt.figure(figsize=(8, 5))
plt.bar(labels, means, yerr=stds, capsize=10, color=["skyblue", "salmon"])
plt.title("Negative Impulse per Cycle (Mean ± SD)")
plt.ylabel("Negative Impulse (Nm·s)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()
