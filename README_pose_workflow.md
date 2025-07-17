# Pose Workflow (MediaPipe Extraction + Sequence Comparison)

本ドキュメントは `video_pose_extractor.py` と `pose_sequence_comparison.py` を用いて MP4 動画から 3D 姿勢ランドマーク (MediaPipe Pose) を抽出し，2 系列の類似度を距離行列法 δD 指標で比較する手順をまとめたものです。

## 1. 依存パッケージ

```bash
pip install -r requirements.txt
```

(既に mediapipe / opencv-python / numpy が入っていれば不要)

## 2. 3Dランドマーク抽出

30Hz の MP4 が `cameras_raw/` にあると仮定:

```bash
python video_pose_extractor.py --input-dir ..\\cameras_raw --output-dir output_data\\poses --pattern "cam0_*.mp4" --limit 1 --verbose
python video_pose_extractor.py --input-dir ..\\cameras_raw --output-dir output_data\\poses --pattern "cam1_*.mp4" --limit 1 --verbose
```

生成ファイル例:

```text
output_data/poses/cam0_output_0925_062008_pose.npy   (T0,33,3)
output_data/poses/cam1_output_0925_062008_pose.npy   (T1,33,3)
```

オプション:

- `--stride N` : フレーム間引き
- `--max-frames M` : 上限フレーム
- `--save-visibility` : visibility配列 (T,33)
- `--save-2d` : 2D正規化ランドマーク (T,33,2)
- `--compare other.npy` : 直近抽出と簡易 δD

## 3. シーケンス比較 (δD)

```bash
python pose_sequence_comparison.py output_data/poses/cam0_output_0925_062008_pose.npy output_data/poses/cam1_output_0925_062008_pose.npy --dtw --save-csv delta_cam0_cam1.csv --out result_cam0_cam1.npz --plot
```

主なオプション:

- `--norm-mode {fro,mean,max,none}`: 距離行列正規化
- `--dtw` : Dynamic Time Warping で非線形アライメント
- `--alt-metric` : δD_alt (2倍スケール)
- `--ascii-output` : 文字化け回避
- `--stride` : フレーム間引き

## 4. δD 指標の意味

距離構造 (関節間距離) のフレーム毎差分を正規化して 0 に近いほど類似。DTW 使用で時間ずれ補正。

## 5. 典型ワークフローまとめ

1. 動画取得 (cam0, cam1 など)
2. 各動画のランドマーク抽出
3. 出力 npy を比較: δD 時系列 + 統計
4. 必要に応じて可視化 (プロット / cost matrix 拡張予定)

## 6. 今後の拡張アイデア

- Procrustes 射影による形状整合
- cost matrix / DTW パスの可視化
- JSON 出力 (--json)
- バンド制約付きDTW (--band)
- 欠損補間 (--interpolate)

## 7. トラブルシューティング

| 現象 | 対処 |
|------|------|
| mediapipe import 失敗 | `pip install --upgrade mediapipe` |
| GPU 利用したい | MediaPipe Pose は現状 CPU 依存 (一部構成除く) |
| 文字化け | `--ascii-output` |
| メモリ不足 (巨大動画) | `--stride` で間引き or FFmpegで短縮 |

## 8. ライセンス

内部研究用途を想定 (元 MediaPipe は Apache 2.0)。

---

何か追加要望があれば README 拡張可能です。

## 9. ステレオ再構成 (Triangulation)

キャリブ付き 2 カメラ動画から直接 3D を復元したい場合:

```bash
python stereo_triangulate_pose.py --input-dir ..\\cameras_raw\\9_20250925_201442 --stride 1 --verbose --save-2d --save-failed-mask
```

生成 (例):
```
.../9_20250925_201442/stereo_pose.npy              (T,33,3)
.../9_20250925_201442/stereo_pose_cam0_2d.npy      (T,33,2)
.../9_20250925_201442/stereo_pose_cam1_2d.npy      (T,33,2)
.../9_20250925_201442/stereo_pose_failed_mask.npy  (T,)  True=全欠測
```

その後，別手法(単眼worldランドマーク)との比較:
```bash
python pose_sequence_comparison.py mono_pose.npy ..\\cameras_raw\\9_20250925_201442\\stereo_pose.npy --dtw --save-csv mono_vs_stereo.csv
```

メモ:

## 10. オフライン局所トルク再計算

リアルタイム処理後に `output_data/` 内へ保存された以下 2 種の CSV から、リンク方向に基づく局所トルクベクトルを再生成するユーティリティ `compute_local_torque_offline.py` を用意しています。

必要ファイル (ID = 例 0407_161615):

```
output_data/aim_torque_vec_0407_161615.csv   # グローバルトルク系列
output_data/kpts3d_0407_161615.csv           # 3Dキーポイント (先頭にメタ 6 行)
```

実行例:

```bash
python compute_local_torque_offline.py 0407_161615 --base-dir output_data
```

生成物:

```text
output_data/local_torque_0407_161615.csv
```

### スキーマ自動判別

トルク CSV は 2 つのスキーマに対応:

1. 左右別 6 パート (18列): `wrist_R_x,...,shoulder_L_z`
2. 単一+体幹 5 パート (15列): `wrist_x, elbow_x, shoulder_x, core_x, hip_x` (各 xyz)

スクリプトは列名を走査して自動判別し、対応するリンク方向で `compute_local_torque` を適用します。左右別スキーマがある場合は右腕リンク (`*_R`) を wrist/elbow/shoulder のベース方向として使用。リンク長ゼロ/NaN の場合はフォールバックとしてグローバル値をそのまま出力します。

### オプション

| オプション | 説明 |
|-------------|------|
| `--skip-kpts-head N` | kpts3d CSV の冒頭メタ行数 (既定 6) |
| `--out path.csv` | 出力パスを明示 (既定: `base-dir/local_torque_<ID>.csv`) |
| `--base-dir DIR` | 入出力ディレクトリ (既定: `output_data`) |

### 処理手順概要

1. kpts3d CSV を読み `joint_i_x/y/z` 列を自動収集 (少なくとも 6 関節必要)
2. スキップ行除去後 (既定 6)、(T,J,3) に reshape
3. トルク CSV の列スキーマを判別し (P パート)、(T, 3P) を取得
4. 各フレームでリンクベクトルを構築 (`wrist_R = joint_4 - joint_2` など)
5. `compute_local_torque(global_torque, link_vector)` を各パートへ適用 (失敗時フォールバック)
6. `frame` 列は 0..T-1 の連番で再割当 (kpts3d スキップ後の相対フレーム)
7. CSV 保存

### 既知の注意点

- kpts3d とトルクでフレーム数が異なる場合、短い方に合わせて切り詰め
- link ベクトルは最低 6 関節 (0..5) を前提 (不足時全フォールバック)
- さらなるパート (e.g., forearm pronation) が必要ならスキーマ拡張が容易

拡張要望 (例: JSON 出力、欠損補間、左右別スキーマから単一スキーマへの再マッピング) があればお知らせください。
