"""pose_sequence_comparison.py
=================================
2系列の3D姿勢推定データ (frames x joints x 3) の時系列的な一致度を
関節間距離行列 (pairwise joint distance matrix) を用いて評価するスクリプト。

主指標: フレーム毎の正規化距離差 δD(t)

定義:
  各フレーム t における 3D関節座標を A_t, B_t ∈ R^{J×3} とする。
  そこから関節間ユークリッド距離行列 D_t^A, D_t^B ∈ R^{J×J} を
	(D_t^A)_{ij} = || A_t[i] - A_t[j] ||_2
  として構成する (B も同様)。

  スケール(大きさ)差の影響を減らすため正規化行列 \tilde{D}_t^A, \tilde{D}_t^B を作る。
  正規化方法はオプション --norm-mode で選択できる:
	- none:   正規化なし (\tilde{D}=D)
	- fro:    Frobenius ノルムで割る: \tilde{D} = D / ||D||_F
	- mean:   上三角( i<j )の平均で割る: \tilde{D} = D / mean_{i<j} D_{ij}
	- max:    上三角の最大値で割る: D / max_{i<j} D_{ij}

  本スクリプトの既定は fro。

  δD(t) は以下の (0〜1程度に収まる) 対称な比率型指標:

	  δD(t) =  || \tilde{D}_t^A − \tilde{D}_t^B ||_F  /  ( ||\tilde{D}_t^A||_F + ||\tilde{D}_t^B||_F )

  もし両者が完全一致 (同一距離構造) なら δD(t)=0。
  完全に異なる場合 1 に近づく (理論上 >1 になる構成は通常の距離正規化下では起きない)。

  追加の代替案 (式実装は --alt-metric で有効化):
	  δD_alt(t) = 2 * || \tilde{D}_t^A − \tilde{D}_t^B ||_F  /  ( ||\tilde{D}_t^A||_F + ||\tilde{D}_t^B||_F )
  こちらは 0〜2 を取りうる単純スケールで，差を強調したい場合に利用。

シーケンス長が異なる場合:
  --dtw オプションを付けると Dynamic Time Warping (DTW) により
  フレーム時系列の非線形アライメントを行い，パス上の平均 δD を算出。

入出力:
  入力ファイル: .npy / .npz / .csv (自動判別)
	形状: (T, J, 3) もしくは (T, 3J) のいずれか。後者なら reshape する。
  出力: 指標CSV, npz (オプション) / 統計サマリ stdout。

使用例:
  python pose_sequence_comparison.py seqA.npy seqB.npy \
	  --dtw --save-csv deltaD.csv --out result.npz --norm-mode fro

  デモ (ランダム生成 + ノイズ + 時間伸縮):
  python pose_sequence_comparison.py --demo

依存: numpy (必須), (任意) matplotlib 可視化 (--plot で利用; 未インストールなら自動スキップ)

注意:
  - NaN を含むフレームはスキップ (アライメント利用時は高コスト化)。
  - 計算量: DTW は O(T_A * T_B * J^2)。長大シーケンスでは --stride で間引き推奨。

著者: Auto-generated helper
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


# ------------------------------------------------------------
# データ構造
# ------------------------------------------------------------
@dataclass
class PoseSequence:
	data: np.ndarray  # shape (T, J, 3)
	name: str = ""

	@property
	def T(self) -> int:
		return self.data.shape[0]

	@property
	def J(self) -> int:
		return self.data.shape[1]


# ------------------------------------------------------------
# ロード関連
# ------------------------------------------------------------
def load_pose_file(path: str) -> PoseSequence:
	"""ファイルパスから姿勢シーケンスを読み込む。

	対応形式:
	  - .npy : そのまま np.load
	  - .npz : 最初の配列 (もしくは 'data' キー) を利用
	  - .csv : comma / tab 区切りを自動判別 (簡易)

	受理する形状:
	  - (T, J, 3)
	  - (T, 3J) → reshape(T, J, 3)

	Returns
	-------
	PoseSequence
	"""
	ext = os.path.splitext(path)[1].lower()
	if ext == ".npy":
		arr = np.load(path)
	elif ext == ".npz":
		npz = np.load(path)
		if "data" in npz.files:
			arr = npz["data"]
		else:
			# 最初のキーを使う
			arr = npz[npz.files[0]]
	elif ext == ".csv":
		arr_list = []
		with open(path, "r", newline="") as f:
			sample = f.readline()
			delim = "," if sample.count(",") >= sample.count("\t") else "\t"
			f.seek(0)
			reader = csv.reader(f, delimiter=delim)
			for row in reader:
				if not row:
					continue
				arr_list.append([float(x) for x in row])
		arr = np.asarray(arr_list, dtype=np.float32)
	else:
		raise ValueError(f"Unsupported extension: {ext}")

	if arr.ndim == 3:
		if arr.shape[2] != 3:
			raise ValueError(f"Last dimension must be 3 (x,y,z); got {arr.shape}")
		data = arr.astype(np.float32)
	elif arr.ndim == 2:
		T, C = arr.shape
		if C % 3 != 0:
			raise ValueError(f"2D array second dimension must be multiple of 3; got {C}")
		J = C // 3
		data = arr.reshape(T, J, 3).astype(np.float32)
	else:
		raise ValueError(f"Unsupported array ndim={arr.ndim}")

	return PoseSequence(data=data, name=os.path.basename(path))


# ------------------------------------------------------------
# 距離行列計算
# ------------------------------------------------------------
def pairwise_distance_matrix(frame: np.ndarray) -> np.ndarray:
	"""あるフレーム (J,3) から関節間距離行列 (J,J) を計算。

	Broadcasting により (x_i - x_j)^2 を求める。
	NaN が含まれる場合は結果を NaN に保持 (後段で処理)。
	"""
	# frame: (J,3)
	diff = frame[:, None, :] - frame[None, :, :]  # (J,J,3)
	dist2 = np.sum(diff * diff, axis=-1)  # (J,J)
	return np.sqrt(dist2, dtype=np.float32)


def normalize_distance_matrix(D: np.ndarray, mode: str = "fro") -> np.ndarray:
	if mode == "none":
		return D
	# 上三角のみ抽出 (自己距離=0 を除外)
	iu = np.triu_indices_from(D, k=1)
	vals = D[iu]
	# NaN 除外
	vals = vals[~np.isnan(vals)]
	if vals.size == 0:
		return D  # 全部 NaN ならそのまま
	if mode == "fro":
		denom = np.linalg.norm(vals)
	elif mode == "mean":
		denom = np.mean(vals)
	elif mode == "max":
		denom = np.max(vals)
	else:
		raise ValueError(f"Unknown norm-mode: {mode}")
	if denom == 0 or np.isnan(denom):
		return D
	return D / denom


# ------------------------------------------------------------
# δD 計算
# ------------------------------------------------------------
def delta_D(Da: np.ndarray, Db: np.ndarray, alt: bool = False) -> float:
	"""正規化済距離行列2つから δD を算出。

	Parameters
	----------
	Da, Db : (J,J) ndarray
		正規化済み距離行列。NaN を含む場合は共有部分のみ利用。
	alt : bool
		True の場合は倍スケール δD_alt = 2 * num / denom
	"""
	# 共通で有効な要素のみ
	mask = ~(np.isnan(Da) | np.isnan(Db))
	if not np.any(mask):
		return float("nan")
	A = Da[mask]
	B = Db[mask]
	diff = A - B
	num = np.linalg.norm(diff)
	denom = np.linalg.norm(A) + np.linalg.norm(B)
	if denom == 0:
		return 0.0
	base = num / denom
	if alt:
		return 2.0 * base
	return base


# ------------------------------------------------------------
# DTW 実装 (フレーム距離: δD)
# ------------------------------------------------------------
def compute_dtw(cost_fn, n1: int, n2: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
	"""標準的な O(n1*n2) DTW 実装。

	Parameters
	----------
	cost_fn : callable (i,j)-> float
		フレーム i (A), j (B) のコスト (非負) を返す関数。
	n1, n2 : int
		シーケンス長。

	Returns
	-------
	cost_matrix : (n1,n2) ndarray
		各セルでの累積コスト。
	path : list[(i,j)]
		復元された最適アライメントパス (逆順でなく時間順)。
	"""
	C = np.full((n1, n2), np.inf, dtype=np.float32)
	back = np.empty((n1, n2, 2), dtype=np.int16)

	# 初期化
	C[0, 0] = cost_fn(0, 0)
	back[0, 0] = [-1, -1]
	for i in range(1, n1):
		c = cost_fn(i, 0)
		C[i, 0] = C[i - 1, 0] + c
		back[i, 0] = [i - 1, 0]
	for j in range(1, n2):
		c = cost_fn(0, j)
		C[0, j] = C[0, j - 1] + c
		back[0, j] = [0, j - 1]

	# DP
	for i in range(1, n1):
		for j in range(1, n2):
			c = cost_fn(i, j)
			# 3遷移: (i-1,j), (i,j-1), (i-1,j-1)
			prevs = (C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
			k = int(np.argmin(prevs))
			if k == 0:
				pi, pj = i - 1, j
			elif k == 1:
				pi, pj = i, j - 1
			else:
				pi, pj = i - 1, j - 1
			C[i, j] = c + prevs[k]
			back[i, j] = [pi, pj]

	# パス復元
	path_rev: List[Tuple[int, int]] = []
	i, j = n1 - 1, n2 - 1
	while i >= 0 and j >= 0:
		path_rev.append((i, j))
		bi, bj = back[i, j]
		if bi < 0:
			break
		i, j = bi, bj
	path = list(reversed(path_rev))
	return C, path


# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
def compute_sequence_delta(
	seqA: PoseSequence,
	seqB: PoseSequence,
	norm_mode: str = "fro",
	dtw: bool = False,
	stride: int = 1,
	alt_metric: bool = False,
	progress: bool = True,
) -> Dict[str, Any]:
	"""2系列の δD(t) (あるいは DTW パス上の δD) を計算する。

	Returns 辞書には以下を含む:
	  - per_frame_deltaD : ndarray (N_aligned,)
	  - stats : 平均/中央値/分位点 など
	  - dtw_path (optional)
	  - dtw_cost_matrix (optional)
	"""
	A = seqA.data[::stride]
	B = seqB.data[::stride]
	T1, J1, _ = A.shape
	T2, J2, _ = B.shape
	if J1 != J2:
		raise ValueError(f"Joint count mismatch: {J1} vs {J2}")
	J = J1

	# 事前に距離行列群を構築
	def build_distance_stack(X: np.ndarray) -> np.ndarray:
		mats = []
		for t, frame in enumerate(X):
			if np.isnan(frame).any():
				D = np.full((J, J), np.nan, dtype=np.float32)
			else:
				D = pairwise_distance_matrix(frame)
			Dn = normalize_distance_matrix(D, norm_mode)
			mats.append(Dn)
		return np.stack(mats, axis=0)  # (T,J,J)

	if progress:
		print("Building distance matrices ...", file=sys.stderr)
	DAs = build_distance_stack(A)
	DBs = build_distance_stack(B)

	if not dtw:
		if progress:
			print("Computing per-frame δD (no DTW) ...", file=sys.stderr)
		T = min(T1, T2)
		deltas = np.zeros(T, dtype=np.float32)
		for t in range(T):
			deltas[t] = delta_D(DAs[t], DBs[t], alt=alt_metric)
		path = [(t, t) for t in range(T)]
		cost_matrix = None
	else:
		if progress:
			print("Computing DTW ...", file=sys.stderr)

		def cost_fn(i: int, j: int) -> float:
			return float(delta_D(DAs[i], DBs[j], alt=alt_metric))

		cost_matrix, path = compute_dtw(cost_fn, T1, T2)
		deltas = np.array([cost_fn(i, j) for (i, j) in path], dtype=np.float32)

	# 統計
	valid = deltas[~np.isnan(deltas)]
	stats = {
		"count": int(valid.size),
		"mean": float(np.mean(valid)) if valid.size else math.nan,
		"median": float(np.median(valid)) if valid.size else math.nan,
		"std": float(np.std(valid)) if valid.size else math.nan,
		"p10": float(np.percentile(valid, 10)) if valid.size else math.nan,
		"p90": float(np.percentile(valid, 90)) if valid.size else math.nan,
		"min": float(np.min(valid)) if valid.size else math.nan,
		"max": float(np.max(valid)) if valid.size else math.nan,
	}
	if dtw and valid.size:
		stats["dtw_total_cost"] = float(cost_matrix[-1, -1])
		stats["dtw_average_cost"] = float(np.mean(deltas))
		stats["dtw_path_length"] = int(len(path))

	return {
		"per_frame_deltaD": deltas,
		"path": path,
		"stats": stats,
		"dtw_cost_matrix": cost_matrix,
		"norm_mode": norm_mode,
		"alt_metric": alt_metric,
	}


# ------------------------------------------------------------
# 可視化 (任意)
# ------------------------------------------------------------
def try_plot(deltas: np.ndarray, title: str = "δD over time") -> None:
	try:
		import matplotlib.pyplot as plt
	except Exception:  # noqa: BLE001
		print("[plot] matplotlib 未インストールのためスキップ", file=sys.stderr)
		return
	plt.figure(figsize=(8, 3))
	plt.plot(deltas, lw=1.0)
	plt.xlabel("Aligned Frame Index")
	plt.ylabel("δD")
	plt.title(title)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.show()


# ------------------------------------------------------------
# デモデータ生成
# ------------------------------------------------------------
def generate_demo(T: int = 120, J: int = 17, noise: float = 0.01, stretch: float = 1.15) -> Tuple[PoseSequence, PoseSequence]:
	"""擬似的な回転 + ノイズ + 時間伸縮を加えた2系列を生成。"""
	rng = np.random.default_rng(42)
	# ベース (円運動 + ランダム初期ポーズ)
	base = rng.normal(size=(J, 3)).astype(np.float32) * 0.2
	th = np.linspace(0, 2 * np.pi, T, endpoint=False)
	motion = np.stack([
		np.cos(th),
		np.sin(th),
		np.sin(2 * th) * 0.5,
	], axis=-1)  # (T,3)
	seqA = base[None, :, :] + motion[:, None, :] * 0.3
	# Sequence B: 時間伸長 + 回転 + ノイズ
	T2 = int(T * stretch)
	th2 = np.linspace(0, 2 * np.pi, T2, endpoint=False)
	motion2 = np.stack([
		np.cos(th2 + 0.1),
		np.sin(th2 + 0.1),
		np.sin(2 * th2 + 0.2) * 0.5,
	], axis=-1)
	seqB = base[None, :, :] + motion2[:, None, :] * 0.3
	# 小回転 (Z軸) 行列
	ang = 0.15
	R = np.array([
		[np.cos(ang), -np.sin(ang), 0.0],
		[np.sin(ang), np.cos(ang), 0.0],
		[0.0, 0.0, 1.0],
	], dtype=np.float32)
	seqB = seqB @ R.T
	# ノイズ
	seqA += rng.normal(scale=noise, size=seqA.shape)
	seqB += rng.normal(scale=noise * 1.5, size=seqB.shape)
	return PoseSequence(seqA, name="demoA"), PoseSequence(seqB, name="demoB")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="3D姿勢シーケンス比較 (距離行列 + δD 指標, 任意DTW)",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument("seqA", nargs="?", help="1つ目のシーケンスファイル (.npy/.npz/.csv)")
	p.add_argument("seqB", nargs="?", help="2つ目のシーケンスファイル (.npy/.npz/.csv)")
	p.add_argument("--dtw", action="store_true", help="DTW による非線形アライメントを有効化")
	p.add_argument("--norm-mode", choices=["none", "fro", "mean", "max"], default="fro", help="距離行列正規化方法")
	p.add_argument("--stride", type=int, default=1, help="フレーム間引き間隔 (>=1)")
	p.add_argument("--alt-metric", action="store_true", help="代替スケール δD_alt (2倍) を使用")
	p.add_argument("--save-csv", type=str, default=None, help="per-frame δD を CSV 保存するパス")
	p.add_argument("--out", type=str, default=None, help="結果一式を npz で保存するパス")
	p.add_argument("--plot", action="store_true", help="δD 時系列を matplotlib で表示")
	p.add_argument("--demo", action="store_true", help="デモデータで動作確認 (ファイル指定不要)")
	p.add_argument("--quiet", action="store_true", help="進捗出力を抑制")
	p.add_argument("--ascii-output", action="store_true", help="端末文字化け回避のため δ など非ASCII文字を使わない出力にする")
	return p


def save_csv(path: str, arr: np.ndarray) -> None:
	with open(path, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["index", "deltaD"])
		for i, v in enumerate(arr):
			w.writerow([i, float(v)])


def main(argv: Optional[List[str]] = None) -> int:
	parser = build_argparser()
	args = parser.parse_args(argv)

	if args.demo:
		seqA, seqB = generate_demo()
	else:
		if not args.seqA or not args.seqB:
			parser.error("seqA と seqB を指定するか --demo を使用してください")
		seqA = load_pose_file(args.seqA)
		seqB = load_pose_file(args.seqB)

	if args.stride < 1:
		parser.error("--stride は 1 以上")

	start = time.time()
	result = compute_sequence_delta(
		seqA,
		seqB,
		norm_mode=args.norm_mode,
		dtw=args.dtw,
		stride=args.stride,
		alt_metric=args.alt_metric,
		progress=not args.quiet,
	)
	elapsed = time.time() - start

	deltas = result["per_frame_deltaD"]
	stats = result["stats"]

	def print_stats(stats_dict: Dict[str, Any], elapsed_sec: float, ascii_only: bool = False) -> None:
		# 表示順序を制御
		order = [
			"count",
			"mean",
			"median",
			"std",
			"min",
			"p10",
			"p90",
			"max",
			"dtw_total_cost",
			"dtw_average_cost",
			"dtw_path_length",
		]
		title = "===== deltaD stats =====" if ascii_only else "===== δD 統計 ====="
		print(title)
		for k in order:
			if k in stats_dict:
				v = stats_dict[k]
				if isinstance(v, float):
					print(f"{k:>16s}: {v:.8f}")
				else:
					print(f"{k:>16s}: {v}")
		print(f"{'elapsed_sec':>16s}: {elapsed_sec:.3f}")

	print_stats(stats, elapsed, ascii_only=args.ascii_output)

	if args.save_csv:
		save_csv(args.save_csv, deltas)
		print(f"[saved] per-frame δD -> {args.save_csv}")
	if args.out:
		np.savez_compressed(
			args.out,
			deltas=deltas,
			stats=stats,
			path=np.asarray(result["path"], dtype=np.int32),
			norm_mode=result["norm_mode"],
			alt_metric=result["alt_metric"],
		)
		print(f"[saved] result npz -> {args.out}")
	if args.plot:
		try_plot(deltas)
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())

