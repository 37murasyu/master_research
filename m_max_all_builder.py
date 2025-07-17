"""全被験者分 1RM (最大推定重量) 一括算出スクリプト

概要:
  Google 公開スプレッドシートから全行を取得し、被験者(=B列)ごとに以下部位の回数・重量を読み出し 1RM 推定値を計算して CSV 出力する。

  対象部位と列マッピング (0始まり index):
    wrist_L: (C=2 reps, D=3 weight)
    wrist_R: (E=4 reps, F=5 weight)
    elbow_L: (K=10 reps, L=11 weight)
    elbow_R: (M=12 reps, N=13 weight)

  1RM 計算: max_weight = weight / (percent * 0.01)
  percent は rm_method.csv の行 (反復回数が一致) の 1列目 (1RM%) を利用。

出力 CSV 既定: m_max_all.csv
  ヘッダ: subject_id,wrist_L,wrist_R,elbow_L,elbow_R
  数値は小数第3位まで (round(x,3))。計算できない/データ欠損は空欄。

環境変数:
  SHEET_ID   : スプレッドシート ID (省略時デフォルト)
  SHEET_GID  : gid 指定が必要なら設定
  OUTPUT_CSV : 出力ファイル名 (省略時 m_max_all.csv)
  RM_CSV     : rm_method.csv のパス (省略時 ./rm_method.csv)

終了コード:
  0: 正常終了
  4: スプレッドシート取得失敗 (ネットワーク等)
  5: rm_method.csv 未読込 (ファイル不存在)

注意:
  rm_method.csv に回数が無い場合はその部位は空欄にし警告ログを出す (プロセスは継続)。

使用例:
  python m_max_all_builder.py
  SHEET_ID=xxxxx OUTPUT_CSV=out.csv python m_max_all_builder.py
"""
from __future__ import annotations
import csv
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Optional

DEFAULT_SHEET_ID = "1B4XhsxISwVwJsoGwsZ33jbE30w-Qckt1r5-C1uwZ24s"  # 既存スクリプトと同じ
PART_COLS: Dict[str, Tuple[int, int]] = {
    'wrist_L': (2, 3),
    'wrist_R': (4, 5),
    'elbow_L': (10, 11),
    'elbow_R': (12, 13),
}


def fetch_sheet_csv(sheet_id: str, gid: Optional[str] = None, timeout: float = 10.0) -> List[List[str]]:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid:
        url += f"&gid={gid}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            data = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        print(f"[ERR] スプレッドシート取得失敗: {e}")
        sys.exit(4)
    rows: List[List[str]] = []
    for line in data.splitlines():
        rows.append(next(csv.reader([line])))
    return rows


def load_rm_method(path: str) -> Dict[int, float]:
    if not os.path.isfile(path):
        print(f"[ERR] rm_method.csv が存在しません: {path}")
        sys.exit(5)
    mapping: Dict[int, float] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                reps_val = int(float(row[1]))
                percent = float(row[0])
            except ValueError:
                continue
            # 同じ reps が複数行あれば最初を優先
            if reps_val not in mapping:
                mapping[reps_val] = percent
    return mapping


def safe_int(value: str) -> Optional[int]:
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def safe_float(value: str) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def compute_1rm(reps: int, weight: float, rm_map: Dict[int, float]) -> Optional[float]:
    percent = rm_map.get(reps)
    if percent is None or percent <= 0:
        return None
    return weight / (percent * 0.01)


def process_all(rows: List[List[str]], rm_map: Dict[int, float]) -> List[Dict[str, Optional[float]]]:
    results: List[Dict[str, Optional[float]]] = []
    if not rows:
        return results
    # 先頭行ヘッダ想定
    for r in rows[1:]:
        if len(r) < 2:
            continue
        subject = (r[1] or "").strip()
        if subject == "":
            continue
        row_result: Dict[str, Optional[float]] = {"subject_id": subject}
        for part, (idx_reps, idx_w) in PART_COLS.items():
            if max(idx_reps, idx_w) >= len(r):
                row_result[part] = None
                continue
            reps_val = safe_int(r[idx_reps])
            weight_val = safe_float(r[idx_w])
            if reps_val is None or weight_val is None:
                row_result[part] = None
                continue
            est = compute_1rm(reps_val, weight_val, rm_map)
            if est is None:
                print(f"[WARN] reps={reps_val} の 1RM% 未定義 -> subject {subject} part {part}")
            row_result[part] = round(est, 3) if est is not None else None
        results.append(row_result)
    return results


def write_csv(path: str, rows: List[Dict[str, Optional[float]]]):
    fieldnames = ["subject_id", *PART_COLS.keys()]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k)
                if v is None:
                    out[k] = ""
                else:
                    # 数値のみ 3 桁フォーマット。subject_id など文字列はそのまま。
                    if isinstance(v, (int, float)):
                        out[k] = f"{v:.3f}"
                    else:
                        out[k] = str(v)
            writer.writerow(out)


def main():
    sheet_id = os.getenv("SHEET_ID", DEFAULT_SHEET_ID)
    gid = os.getenv("SHEET_GID") or None
    output_csv = os.getenv("OUTPUT_CSV", "m_max_all.csv")
    rm_csv = os.getenv("RM_CSV", "rm_method.csv")

    print(f"[INFO] SHEET_ID={sheet_id} gid={gid} output={output_csv} rm_csv={rm_csv}")

    rows = fetch_sheet_csv(sheet_id, gid)
    if not rows:
        print("[ERR] スプレッドシートが空です")
        sys.exit(4)

    rm_map = load_rm_method(rm_csv)
    print(f"[INFO] rm_method ロード: {len(rm_map)} 件 (反復回数→%)")

    results = process_all(rows, rm_map)
    print(f"[INFO] 被験者数: {len(results)} 件")

    write_csv(output_csv, results)
    print(f"[OK] 書き出し完了 -> {output_csv}")

    # 先頭数件を JSON 風に表示 (確認用)
    preview = results[:5]
    print("--- PREVIEW (上位5件) ---")
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
