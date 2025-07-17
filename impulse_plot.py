import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: 日本語表示対応
try:
    import japanize_matplotlib  # pylint: disable=unused-import
except ImportError:
    pass


def main():
    # 結合済み統計CSVファイルパス
    stats_csv = "output_data\\all_sessions_stats_combined_grouped.csv"
    # 各セッションのサイクル数（滝沢1～柳瀬4）
    # 実データのサイクル数に合わせて設定してください
    cycle_counts = [10, 12, 11, 9, 10, 12, 11, 9]

    # 統計データ読み込み
    df = pd.read_csv(stats_csv)

    # 対象カラム
    col_neg = "both_shoulder_L_torque_local_z_neg_impulse"
    col_pos = "both_shoulder_L_torque_local_z_pos_impulse"

    # セッションキーとラベル
    sessions = [
        "cycle_impulses_new_cycles",
        "0407_161615",
        "0407_161802",
        "0407_161944",
        "0407_162237",
        "0407_162449",
        "0407_162703",
        "0407_162947",
    ]
    labels = [
        "滝沢1(正常)",
        "滝沢2(早い)",
        "滝沢3(前傾)",
        "滝沢4(ネジリ)",
        "柳瀬1(正常)",
        "柳瀬2(早い)",
        "柳瀬3(前傾)",
        "柳瀬4(ネジリ)",
    ]

    # 平均値と標準偏差を収集
    means = []
    stds = []
    for idx, sess in enumerate(sessions):
        if idx in [0, 1, 2, 5, 3, 4, 6, 7]:
            # 滝沢1-3, 柳瀬4: 負インパルス
            m = df.loc[df["column"] == col_neg, f"{sess}_mean"].item()
            s = df.loc[df["column"] == col_neg, f"{sess}_std"].item()
        else:
            # それ以外: 正インパルス符号反転
            m = -df.loc[df["column"] == col_pos, f"{sess}_mean"].item()
            s = df.loc[df["column"] == col_pos, f"{sess}_std"].item()
        means.append(m)
        stds.append(s)

    # 95%信頼区間を計算
    n = np.array(cycle_counts)
    standard_errors = np.array(stds) / np.sqrt(n)
    cis = 1.96 * standard_errors

    # プロット
    x = np.arange(len(sessions))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, align="center", label="平均値")

    # 標準偏差のエラーバー (黒)
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=5,
        label="標準偏差",
    )
    # 95%信頼区間のエラーバー (灰色)
    ax.errorbar(
        x,
        means,
        yerr=cis,
        fmt="none",
        ecolor="gray",
        elinewidth=1,
        capsize=5,
        label="95% 信頼区間",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("セッション")
    ax.set_ylabel("平均インパルス")
    ax.set_title("プッシュアップ１サイクル中の左肩z軸トルク積のバラツキ")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
