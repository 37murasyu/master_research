"""m_max_part_(被験者番号).json 自動生成スクリプト

手順:
 1. 被験者番号(Subject ID) を入力 (例: S001)。
 2. 公開 Google スプレッドシート (対象 URL) の B 列で一致する行を探索。
 3. 一致行から各部位ごとの「回数・重さ」を抽出して 1RM(最大重量) を算出:
     - wrist_L: C 列(回数), D 列(重さ)  ← 従来どおり
     - wrist_R: E 列(回数), F 列(重さ)
     - elbow_L: K 列(回数), L 列(重さ)
     - elbow_R: M 列(回数), N 列(重さ)
 4. ローカル rm_method.csv を読み込み、B 列(反復回数) が一致する行の A 列 (1RM%) を取得。
 5. 最大重量(1RM) = 重さ / (1RM% * 0.01) を各部位で計算し、m_max_part_(被験者番号).json に記録。
 6. 既存 m_max_part_(被験者番号).json があればマージ (他キー保持)。無ければテンプレ作成。
 7. JSON を保存し結果表示。

前提:
 - 対象スプレッドシートが "公開 (誰でも閲覧可)" になっていること。
 - 公開 CSV エンドポイントを利用: https://docs.google.com/spreadsheets/d/<ID>/export?format=csv
 - 指定シートが最初のワークシートの場合 (gid 指定不要)。必要なら ?gid=XXXX を URL に追加。

環境変数 (任意):
  SHEET_ID : スプレッドシート ID (省略時デフォルト使用)
  SHEET_GID: シート GID (あれば付与) 例: 0
  SUBJECT_ID: 対話入力を省略して被験者番号を直接指定
    OUTPUT_JSON: 出力先ファイル名 (既定: m_max_part_<SUBJECT_ID>.json)
    TARGET_KEY: 後方互換用。単一部位のみ更新したい場合に使用 (既定: 空=無視)

エラー処理:
 - 行が見つからない: 終了コード 2
 - reps が rm_method.csv に無い: 終了コード 3
 - ネットワーク障害: 終了コード 4
"""
from __future__ import annotations
import csv
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict

DEFAULT_SHEET_ID = "1B4XhsxISwVwJsoGwsZ33jbE30w-Qckt1r5-C1uwZ24s"  # 指定URLから抽出


def fetch_sheet_csv(sheet_id: str, gid: str | None = None, timeout: float = 10.0) -> list[list[str]]:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid:
        url += f"&gid={gid}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310 (trusted Google domain)
            data = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:  # network issues
        print(f"[ERR] スプレッドシート取得失敗: {e}")
        sys.exit(4)
    rows: list[list[str]] = []
    for line in data.splitlines():
        rows.append(next(csv.reader([line])))
    return rows


def find_subject_row(rows: list[list[str]], subject_id: str) -> Optional[list[str]]:
    """B列で被験者IDが一致する行を返す。見つからなければ None。"""
    for r in rows[1:]:  # 先頭行はヘッダー想定
        if len(r) < 2:
            continue
        sid = (r[1] or "").strip()
        if sid == subject_id:
            return r
    return None


def lookup_1rm_percent(rm_csv_path: str, reps: int) -> float:
    with open(rm_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # ヘッダー想定: ['1RM%', '反復回数'] だが柔軟に扱う
        best = None
        for row in reader:
            if len(row) < 2:
                continue
            try:
                reps_val = int(float(row[1]))
            except ValueError:
                continue
            if reps_val == reps:
                try:
                    best = float(row[0])
                    break
                except ValueError:
                    pass
        if best is None:
            print(f"[ERR] rm_method.csv に反復回数 {reps} が見つかりません")
            sys.exit(3)
        return best


def load_or_init_json(path: str) -> dict:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] 既存 JSON 読み込み失敗 -> 初期化: {e}")
    # 既定初期値
    return {
        "elbow_R": 6.0,
        "wrist_R": 4.0,
        "elbow_L": 6.0,
        "wrist_L": 4.0
    }


def main():
    sheet_id = os.getenv("SHEET_ID", DEFAULT_SHEET_ID)
    gid = os.getenv("SHEET_GID") or None
    target_key = os.getenv("TARGET_KEY")  # 後方互換: 指定時は単一部位のみ更新
    # SUBJECT_ID が決まってから出力パス既定値を組み立てる
    output_json_env = os.getenv("OUTPUT_JSON")

    subject_id = os.getenv("SUBJECT_ID")
    if not subject_id:
        try:
            subject_id = input("被験者番号を入力してください: ").strip()
        except EOFError:
            print("[ERR] SUBJECT_ID 未指定 & 標準入力不可")
            sys.exit(1)
    if not subject_id:
        print("[ERR] 被験者番号が空です")
        sys.exit(1)

    # 出力先の最終決定（環境変数があれば優先）
    output_json = output_json_env or f"m_max_part_{subject_id}.json"

    print(f"[INFO] SHEET_ID={sheet_id} SUBJECT_ID={subject_id}")
    print(f"[INFO] 出力ファイル: {output_json}")
    rows = fetch_sheet_csv(sheet_id, gid)
    if not rows:
        print("[ERR] スプレッドシートが空です")
        sys.exit(4)

    row = find_subject_row(rows, subject_id)
    if row is None:
        print(f"[ERR] 被験者 {subject_id} がシートに見つかりません")
        sys.exit(2)
    # 列インデックス (0始まり): A=0, B=1, C=2, D=3, E=4, F=5, ..., K=10, L=11, M=12, N=13
    part_cols: Dict[str, Tuple[int, int]] = {
        'wrist_L': (2, 3),  # C: reps, D: weight (従来どおり)
        'wrist_R': (4, 5),   # E: reps, F: weight
        'elbow_L': (10, 11), # K: reps, L: weight
        'elbow_R': (12, 13), # M: reps, N: weight
    }

    # 後方互換: target_key が指定されている場合のみ、そのキーに対して C/D を使う旧仕様もサポート
    if target_key:
        part_cols = {target_key: (2, 3)}

    data = load_or_init_json(output_json)
    updated: Dict[str, float] = {}

    for key, (idx_reps, idx_w) in part_cols.items():
        if max(idx_reps, idx_w) >= len(row):
            continue
        reps_raw = (row[idx_reps] or '').strip()
        w_raw = (row[idx_w] or '').strip()
        if not reps_raw or not w_raw:
            continue
        try:
            reps_val = int(float(reps_raw))
            weight_val = float(w_raw)
        except ValueError:
            continue
        rm_percent = lookup_1rm_percent("rm_method.csv", reps_val)
        max_weight = weight_val / (rm_percent * 0.01)  # 1RM 推定
        data[key] = round(max_weight, 3)
        updated[key] = data[key]

    if not updated:
        print("[WARN] 更新対象データが見つかりませんでした (空/非数値/列不足)")
    else:
        print("[INFO] 更新値:")
        for k, v in updated.items():
            print(f"  - {k}: {v} kg")

    # 保存
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 更新保存 -> {output_json}")

    # 追加表示
    print("--- 最終 JSON ---")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
