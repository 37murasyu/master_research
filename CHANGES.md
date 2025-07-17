# 変更履歴とリファクタリング詳細

対象: `master_research_code.py`（および関連モジュールの利用方法）  
更新日: 2025-09-12

## 目的
- 繰り返しコードの削減（辞書/ループ/内包表記の活用）
- 実行時安定性の向上（副作用の削減・ガードの追加・リソース解放）
- 設定と実測値の役割分離（`dt` vs 実測 `frame_dt`）
- 可読性と保守性の向上（行数・重複・条件式の簡素化）

## 主要変更点（Before → After）

### 1) フレーム間引き（SKIP_FRAMES の副作用排除）
- Before: グローバル `SKIP_FRAMES` をループ内で増加させて判定
- After: `skip_counter` と `skip_mod` を導入し、ローカルに間引き制御（設定値は不変）

### 2) 実測 `dt` と設定 `dt` の分離
- Before: ループ末尾で `dt`（設定）を実測で上書き
- After: `frame_dt` に実測時間を格納（表示専用）、積分や計算は設定の `dt` を継続使用

### 3) MediaPipe リソースの明示解放
- Before: 明示的に `close()` 未呼び出し
- After: `getattr(..., "close")` で存在確認の上、`pose0.close() / pose1.close()` を呼ぶ

### 4) トリミング範囲の安全化と簡素化
- Before: `if x_end - x_start < 1` 判定
- After: `if x_end <= x_start` に単純化。未検出時はフル幅にフォールバック

### 5) `file_mode` 判定の一行化
- Before: if/else で2行
- After: `file_mode = not (input_stream1 == 0 and input_stream2 == 1)`

### 6) LinkVectorCalculator 生成の一括化
- Before: for で逐次生成
- After: 辞書内包表記で一括生成

```python
calculators = {
    part: LinkVectorCalculator(s["start"], s["end"])
    for part, s in part_calculations.items()
}
```

### 7) 部位データの取得を辞書化
- Before: `upper_arm_R_data = ...` 等、個別変数8本
- After: `part_data = {name: storage.get_data(name) for name in part_names_internal}`

### 8) M/F 計算のループ化（`run_specs` 導入）
- Before: `calculate_M_and_F(...)` を左右・複数部位で逐次呼び出し
- After: 仕様リスト（`right_specs` / `left_specs`）を `run_specs` に渡して一括取得

```python
def run_specs(specs):
    Ms, Fs, Parts = [], [], []
    for I, mass, data_seq, kwargs in specs:
        M, F, Part = calculate_M_and_F(I, mass, data_seq, g, **kwargs)
        Ms.append(M); Fs.append(F); Parts.append(Part)
    return Ms, Fs, Parts
```

- 右: `condition=1`、左: `condition=0` を含む仕様を配列で定義

### 9) ローカルトルク算出の辞書化（重複削減）
- `links` と `globals_map` をキー一致の辞書にして、`locals_map` を内包表記で生成
- `storage.add_torque` もキー列挙でループ化

```python
links = { ... }
globals_map = { ... }
locals_map = {k: compute_local_torque(globals_map[k], links[k]) for k in globals_map}
for key in ["wrist_R", "elbow_R", "shoulder_R", "wrist_L", "elbow_L", "shoulder_L"]:
    storage.add_torque(key, locals_map[key])
```

### 10) 表示ラベルのマップ化と `part_keys` の活用
- Before: `labels = ["右手首", ...]` と `zip(labels, part_keys)`
- After: `jp_labels` マップ + `for i, key in enumerate(part_keys)` で描画（インパルス表示）

### 11) 変数シャドー解消と小規模整理
- `run_specs` の `data` → `data_seq` に改名
- テキスト読み込み用 `data` → `line` に改名
- 未使用の一時変数やコメントを削除

## 動作影響
- 機能的変更は最小限。数値計算の挙動は設定 `dt` 依存で維持
- 実測フレーム時間は `frame_dt` で出力（`FPS` 表示の安定化）
- フレーム間引きの副作用がなくなり、再現性・保守性が向上
- ラベル/描画/CSV 出力の順序は従来の `part_keys` と整合（既存保存形式を維持）

## 既知の前提と依存
- `config.py` の各設定値（`dt, fps, input_stream*, pose_keypoints, part_keys, ...`）
- ファイル入出力:
  - `rm_path`（1RM%参照）
  - `folder_path/max_value.txt`（閾値計算）
  - `supervision_stats.csv`（非監修モードのゲージ参照）
  - 出力: `save_dir/kpts3d_*.csv`, `save_dir/aim_torque_vec_*.csv`

## 実行方法
```pwsh
# 必要なら仮想環境を有効化
# .\.venv\Scripts\Activate.ps1

# 依存関係（初回のみ）
# pip install -r requirements.txt

# 実行
python .\master_research_code.py
```
- 起動時にモードを選択（監修=1／非監修=0）

## 最低限のテスト観点
- 2カメラ/動画入力で初期化が完了し、数フレーム処理できる
- 非監修モードでゲージ画像が表示/更新される
- ループ終了後、CSV が出力される
- `FPS` が表示され、`ESC` 長押しで終了する

## リスク/注意点
- `part_keys` のキー集合が UI/保存処理と一致している前提
- `supervision_stats.csv` が存在しない場合、非監修モード起動で例外
- Mediapipe リソース解放は `close()` 実装がある前提（なければ無視）

## 今後の拡張（任意）
- 設定 `dt` 未指定時のフォールバックとして、`frame_dt` の移動平均を積分に併用
- `kpts_cam0/kpts_cam1` 等の未使用変数/コードの完全削除
- 3フレーム目での慣性テンソル初期化ロジックの関数化
- `part_names`（CSV 出力順）を `part_keys` に統一し、重複を排除
- 処理プロファイルの採取（`example.prof` の更新）

## 変更マッピング（要求対応）
- 繰り返しの辞書/ループ化: 完了
- 安定化（副作用回避・リソース解放・ガード）: 完了
- 実測 `dt` の分離: 完了
- 追加の再構成（設定と保存周りの統一）: 一部提案（未実装）

---
メンテナンス/レビュー時は、本ファイルと `master_research_code.py` を併読してください。必要に応じて、`config.py` のキー定義（`part_keys` など）との整合も確認してください。
