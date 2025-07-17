import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.decomposition import PCA


from tslearn.metrics import dtw

# DTW専用のCSV読み込み
# df_dtw_analysis = pd.read_csv("output_data\\local161802_with_cycle.csv")
# pd.read_csv("output_data\\local161234_takizawa_with_cycles.csv")
# ✅ CSVファイルを読み込み（正しい動作: standard, 誤った動作: test）
df_standard = pd.read_csv("output_data/local161234_takizawa_with_cycles.csv")
df_test = pd.read_csv("output_data/local161802_with_cycle.csv")
# 使用するローカル手首トルク成分（Q, R, S）
dtw_wrist_cols = ["wrist_local_x", "wrist_local_y", "wrist_local_z"]

# ✅ サイクルごとのデータ抽出
standard_cycles = {
    i: df_standard[df_standard["cycle"] == i][dtw_wrist_cols].dropna().values
    for i in range(12)
}
test_cycles = {
    i: df_test[df_test["cycle"] == i][dtw_wrist_cols].dropna().values for i in range(12)
}

# ✅ 基準：standard の cycle 1
standard_reference_cycle = standard_cycles[1]

# リストとして記録
standard_vs_standard_dtw = []
standard_vs_test_dtw = []

# ✅ standard 内の比較
print("standard内比較：Cycle 1 と Cycle 2〜11 の DTW 類似度")
for i in range(2, 12):
    distance = dtw(standard_reference_cycle, standard_cycles[i])
    standard_vs_standard_dtw.append({"cycle": i, "dtw_distance": distance})
    print(f"standard: Cycle 1 vs {i} → DTW Distance = {distance:.2f}")

# ✅ testデータとの比較
print("\nstandard vs test比較：standard Cycle 1 と test Cycle 1〜11 の DTW 類似度")
for i in range(1, 12):
    distance = dtw(standard_reference_cycle, test_cycles[i])
    standard_vs_test_dtw.append({"cycle": i, "dtw_distance": distance})
    print(f"standard: Cycle 1 vs test Cycle {i} → DTW Distance = {distance:.2f}")

# ✅ DataFrameにまとめる（外部キー = cycle）
df_merged = pd.DataFrame({"cycle": range(1, 12)})
df_merged = df_merged.merge(
    pd.DataFrame(standard_vs_standard_dtw, columns=["cycle", "standard_vs_standard"]),
    on="cycle",
    how="left",
)
df_merged = df_merged.merge(
    pd.DataFrame(standard_vs_test_dtw, columns=["cycle", "standard_vs_test"]),
    on="cycle",
    how="left",
)

# ✅ 出力保存
df_merged.to_csv("output_data/dtw_comparison.csv", index=False)

# %%
"""
# CSVファイルの読み込み
df1 = pd.read_csv("output_data\\local161234_takizawa_with_cycles.csv")
df2 = pd.read_csv("output_data\\local162237.csv")

# elbowの3軸を抽出
elbow_cols = ["wrist_local_x", "wrist_local_y", "wrist_local_z"]
elbow_data_1 = df1[elbow_cols].values
elbow_data_2 = df2[elbow_cols].values

# 12サイクルに分割
n_cycles = 12
cycle_len1 = len(elbow_data_1) // n_cycles
cycle_len2 = len(elbow_data_2) // n_cycles

cycles_1 = [
    elbow_data_1[i * cycle_len1 : (i + 1) * cycle_len1] for i in range(n_cycles)
]
cycles_2 = [
    elbow_data_2[i * cycle_len2 : (i + 1) * cycle_len2] for i in range(n_cycles)
]

# 時系列データを整形
all_cycles = cycles_1 + cycles_2
max_len = max(len(c) for c in all_cycles)
X = TimeSeriesResampler(sz=max_len).fit_transform(to_time_series_dataset(all_cycles))

# 時系列クラスタリング (KMeans, DTW距離)
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
labels = model.fit_predict(X)

# %%

# 可視化（PCAで2次元に次元削減）
X_flat = X.reshape(X.shape[0], -1)
X_pca = PCA(n_components=2).fit_transform(X_flat)

plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    idx = labels == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {label}")
plt.title("Elbow Torque Cycle Clustering")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
# 定量スコア：各クラスタに属するサイクルの割合
# from collections import Counter

# print("Cluster Distribution:", Counter(labels))
# %%
"""
# クラスタごとの平均波形プロット
for cluster_id in range(3):
    cluster_members = X[labels == cluster_id]
    mean_waveform = np.mean(cluster_members, axis=0)

    plt.figure(figsize=(10, 5))
    for axis_idx, axis_name in enumerate(["elbow_x", "elbow_y", "elbow_z"]):
        plt.plot(mean_waveform[:, axis_idx], label=axis_name)

    plt.title(f"Cluster {cluster_id} - Mean Torque Waveform")
    plt.xlabel("Time")
    plt.ylabel("Torque")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""
