# 適応的LPFフィルタ (Adaptive fc-Cutoff) 実装完了 & Self-Verification ガイド

## 実装内容概要

リアルタイムで肘角度の支配周波数 (f0) を推定し、LPFの遮断周波数 (fc) を適応的に追跡する機能を追加しました。

### 追加された設定 (環境変数)

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `E_FC_ADAPTIVE_ON` | 0 | 適応モード有効化 (0=固定fc, 1=適応fc) |
| `E_FC_MIN` | 2.1 | fc最小値 [Hz] |
| `E_FC_MAX` | 6.0 | fc最大値 [Hz] |
| `E_FC_K` | 6.0 | f0→fc変換係数 (fc = K × f0) |
| `E_F0_WIN_SEC` | 4.0 | 周波数推定の分析窓長 [秒] |
| `E_FC_EMA_BETA` | 0.15 | fc平滑度係数 [0-1] |
| `E_FC_UPDATE_HZ` | 1.0 | fc更新レート [Hz] |
| `E_F0_FMIN` | 0.3 | 最小検出周波数 [Hz] |
| `E_F0_SNR_THRESHOLD` | 3.0 | f0信頼度閾値 [dB] |

## A/B テスト実施方法

### 前提条件
- テスト用ビデオ (例: `test_video.mp4`)
- 同じビデオで2回実行: **固定モード (baseline)** と **適応モード**

### ステップ 1: 固定モード (Baseline)

```bash
# 環境変数をデフォルト（固定fc=1.2Hz）に設定して実行
E_FC_ADAPTIVE_ON=0 python master_research_code.py --input test_video.mp4 --output baseline_fixed.npy
```

**出力例:**
- エネルギー値: `baseline_fixed.npy`
- ログ: cycle detection のたびに E+ 値をプリント
- デバッグ出力：通常のE_DEBUGで確認

### ステップ 2: 適応モード (Adaptive fc tracking)

```bash
# 適応モードを有効化 (同じビデオを使用)
E_FC_ADAPTIVE_ON=1 E_FC_MIN=2.1 E_FC_MAX=6.0 E_FC_K=6.0 E_FC_UPDATE_HZ=1 E_DEBUG=1 python master_research_code.py --input test_video.mp4 --output adaptive_dynamic.npy
```

**予想される出力:**
- エネルギー値: `adaptive_dynamic.npy`
- ログ出力例:
  ```
  [ADAPTIVE_FC] f0=2.45Hz conf=8.5dB fc=2.468Hz
  [ADAPTIVE_FC] f0=2.42Hz conf=7.3dB fc=2.451Hz
  [EPIPE] elbow_R E+= 3.2541 info={'status': 'ok', ...}
  ```

### ステップ 3: 結果比較 (Python/Jupyter)

```python
import numpy as np
import matplotlib.pyplot as plt

# データロード
eng_fixed = np.load('baseline_fixed.npy', allow_pickle=True)
eng_adaptive = np.load('adaptive_dynamic.npy', allow_pickle=True)

# 比較プロット
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Fixed mode
axes[0].plot(eng_fixed['energy_R'], label='Fixed fc=1.2Hz', alpha=0.7)
axes[0].set_title('Baseline: Fixed LPF Cutoff')
axes[0].set_ylabel('Cycle Energy [J]')
axes[0].grid(True)
axes[0].legend()

# Adaptive mode
axes[1].plot(eng_adaptive['energy_R'], label='Adaptive fc(t)', alpha=0.7, color='green')
axes[1].set_title('Adaptive: Dynamic f0 Tracking')
axes[1].set_ylabel('Cycle Energy [J]')
axes[1].set_xlabel('Cycle Number')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig('ab_test_comparison.png', dpi=150)
plt.show()

# 統計比較
print(f"Fixed Energy:   mean={eng_fixed['energy_R'].mean():.4f}, std={eng_fixed['energy_R'].std():.4f}")
print(f"Adaptive Energy: mean={eng_adaptive['energy_R'].mean():.4f}, std={eng_adaptive['energy_R'].std():.4f}")
```

## 検証ポイント

### ✓ 期待される行動

1. **fc 値の安定性**: ログで表示される fc 値が（1秒ごとに）ジッタなく平滑に変化すること
   - 例: 2.10 → 2.45 → 2.47 → 2.48 （スムーズな上昇）
   - 悪い例: 2.10 → 5.80 → 1.50 （ジッタが大きい）

2. **信頼度スコア**: 十分なサンプル数が確保されているサイクルで `conf_db > 3.0` であること
   - サイクル初期は conf が低い可能性あり（OK）
   - 安定フェーズで conf < 0 は問題の可能性

3. **エネルギー値の滑らかさ**: 
   - 固定 fc と比較して、適応 fc の方が外れ値（spikes）が少ないはず
   - ただし劇的な差が出ない場合もある（fc の差が小さい場合）

4. **CPU オーバーヘッド**: 適応モード実行がベースラインより著しく遅くないこと
   - Welch/FFT は 1Hz 頻度の実行なので影響小さい

### ⚠️ トラブルシューティング

| 症状 | 考えられる原因 | 対応 |
|------|-------------|------|
| f0 が常に 0 | FFT ウィンドウが短すぎる | `E_F0_WIN_SEC` を 5-10 に増加 |
| conf_db が常に < 0 | ノイズが多い | `E_F0_SNR_THRESHOLD` を 2.0 に低下 |
| fc が範囲外 | E_FC_MIN/MAX 設定ミス | `E_FC_MIN <= fc_computed <= E_FC_MAX` を確認 |
| エネルギー値が NaN | theta/tau データ不具合 | `E_DEBUG=1` でバッファ内容確認 |

## コード統合点

### 1. 設定ブロック
- **File**: `master_research_code.py` lines 127-139
- **内容**: 適応的LPF環境変数のセット

### 2. クラス定義
- **File**: `master_research_code.py` lines 145-221
- **内容**: `OnlineF0Estimator` クラス (窓付きFFT f0推定)

### 3. スケジューラ関数
- **File**: `master_research_code.py` lines 223-251
- **内容**: `_fc_scheduler` 関数 (fc_raw → EMA平滑 → bounded)

### 4. 状態初期化
- **File**: `master_research_code.py` lines 145-150
- **内容**: グローバル状態 `_f0_estimator`, `_fc_current`, `_fc_update_counter`

### 5. メインループ統合
- **File**: `master_research_code.py` lines 2691-2721
- **内容**: theta 累積後, estimator.step() + 周期的 fc 更新

### 6. エネルギー計算呼び出し
- **File**: `master_research_code.py` line 2828
- **内容**: `compute_cycle_energy_filtered(..., fc_override=_fc_current)`

## 後方互換性

デフォルト設定 (`E_FC_ADAPTIVE_ON=0`) では既存の固定 fc 動作を保持。
新機能の導入による既存動作への影響なし。

## 今後の拡張案

1. **マルチ周波数追跡**: θ と τ 両方から f0 を推定して統合
2. **Subject-specific fc calibration**: オンライン学習で E_FC_K を自動調整
3. **Real-time plot**: fc(t) の推移をリアルタイムで表示
4. **CSV logging**: fc, f0, conf を per-cycle CSV に記録
