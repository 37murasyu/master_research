# pylint: disable=no-member
import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 対象キーポイント (既存 config.py の pose_keypoints に合わせる)
POSE_KEYPOINTS = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]

# 出力ディレクトリ
SAVE_DIR = "output_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_world_subset(results):
    """mediapipe の world_landmarks から POSE_KEYPOINTS 対応座標を抽出。
    未検出時は NaN。
    Returns list[[x,y,z]] 長さ len(POSE_KEYPOINTS)
    """
    if not results.pose_world_landmarks:
        return [[np.nan, np.nan, np.nan] for _ in POSE_KEYPOINTS]
    lm_list = results.pose_world_landmarks.landmark
    out = []
    for idx in POSE_KEYPOINTS:
        if idx < len(lm_list):
            lm = lm_list[idx]
            out.append([lm.x, lm.y, lm.z])
        else:
            out.append([np.nan, np.nan, np.nan])
    return out

def compute_bone_lengths(pts):
    """簡易精度指標: 特定ペア間距離を計算し骨長の変動(CV)を後で評価。
    ここでは上肢/体幹近辺いくつか。
    """
    # インデックス: 0:16(右手首) 1:14(右肘) 2:12(右肩) 3:11(左肩) 4:13(左肘) 5:15(左手首)
    pairs = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    lengths = []
    for a,b in pairs:
        pa, pb = np.array(pts[a]), np.array(pts[b])
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            lengths.append(np.nan)
        else:
            lengths.append(float(np.linalg.norm(pa-pb)))
    return lengths

def open_camera(requested_index: int | None, scan_max: int = 5):
    """カメラオープン補助。
    requested_index が指定されればその index のみを試行。
    未指定なら 0..scan_max-1 を順に試して最初に成功するものを返す。
    戻り値: (cap, used_index) / 失敗時 (None, None)
    """
    indices = [requested_index] if requested_index is not None else list(range(scan_max))
    for idx in indices:
        cap = cv.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        # フレームを1枚読んで実際に取得できるか確認
        ret, _ = cap.read()
        if not ret:
            cap.release()
            continue
        print(f"✅ カメラ index {idx} を使用します")
        return cap, idx
    print("❌ 利用可能なカメラが見つかりませんでした")
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(description="Mono 3D (mediapipe world) テスト")
    parser.add_argument("--cam", type=int, default=None, help="使用するカメラインデックス (未指定なら自動検出: 0→1→...)")
    parser.add_argument("--scan-max", type=int, default=5, help="自動検出で走査する最大インデックス (0..scan_max-1)")
    parser.add_argument("--no-plot", action="store_true", help="Matplotlib 3D プロットを無効化 (安定性/速度検証用)")
    parser.add_argument("--max-frames", type=int, default=0, help="指定フレーム数で自動終了 (0=無制限)")
    return parser.parse_args()


def main():
    args = parse_args()
    cap, used_index = open_camera(args.cam, scan_max=args.scan_max)
    if cap is None:
        return
    print(f"使用カメラ: index={used_index}  (内蔵カメラを優先したい場合は通常 index=0 です。外部USBを避けたい場合は --cam で別インデックスを指定してください)")

    mp_pose = mp.solutions.pose
    # Plot 有効時のみ初期化
    if not args.no_plot:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        scatter = ax.scatter([], [], [], c='b', s=30)
        ax.set_title('Mono 3D Landmarks (Mediapipe world)')
    else:
        fig = None
        ax = None
        scatter = None

    # 軸範囲は起動後最初の数フレームで自動調整、以降固定
    auto_range_frames = 30
    xyz_min = np.array([ np.inf,  np.inf,  np.inf])
    xyz_max = np.array([-np.inf, -np.inf, -np.inf])

    records = []
    bone_history = []  # 骨長変動評価用

    start_all = time.perf_counter()
    frame_idx = 0
    print("▶ 単眼3D検証開始: 'q' キーで終了 (選択カメラ index =", used_index, ")")
    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    print('映像入力終了または取得失敗')
                    break
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                pts = extract_world_subset(results)
                records.append(pts)
                bones = compute_bone_lengths(pts)
                bone_history.append(bones)

                # 可視化更新 (プロット有効時のみ)
                if scatter is not None:
                    arr = np.array(pts)
                    xs, ys, zs = arr[:,0], arr[:,1], arr[:,2]
                    scatter._offsets3d = (xs, ys, zs)

                    # 自動範囲決定 (プロット有効時のみ)
                    if frame_idx < auto_range_frames:
                        valid = arr[~np.isnan(arr).any(axis=1)]
                        if valid.size > 0:
                            xyz_min = np.minimum(xyz_min, valid.min(axis=0))
                            xyz_max = np.maximum(xyz_max, valid.max(axis=0))
                            rng = xyz_max - xyz_min
                            rng[rng == 0] = 1.0
                            center = (xyz_max + xyz_min)/2
                            if not np.all(np.isfinite(center)):
                                center = np.array([0.0, 0.0, 0.0])
                            span = float(np.max(rng)*0.6)
                            if not np.isfinite(span) or span <= 0:
                                span = 0.5
                        else:
                            center = np.array([0.0, 0.0, 0.0])
                            span = 0.5
                        ax.set_xlim(center[0]-span, center[0]+span)
                        ax.set_ylim(center[1]-span, center[1]+span)
                        ax.set_zlim(center[2]-span, center[2]+span)
                    elif frame_idx == auto_range_frames:
                        print("軸範囲固定:", ax.get_xlim(), ax.get_ylim(), ax.get_zlim())

                    plt.pause(0.001)

                cv.putText(frame, f"Frame: {frame_idx}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                dt_frame = time.perf_counter() - t0
                cv.putText(frame, f"Proc(ms): {dt_frame*1000:.1f}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
                cv.imshow('Mono3D_Cam0', frame)

                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                frame_idx += 1
                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    print(f"指定 max-frames({args.max_frames}) 到達で終了")
                    break
    finally:
        # cleanup
        cap.release()
        cv.destroyAllWindows()
        if fig is not None:
            plt.close(fig)

    total_time = time.perf_counter() - start_all
    if frame_idx > 0:
        print(f"平均処理時間: {total_time/frame_idx*1000:.2f} ms/フレーム, FPS≈ {frame_idx/total_time:.2f}")

    # 骨長変動 (CV) を簡易指標として算出
    bone_arr = np.array(bone_history)  # shape: (F, num_bones)
    bone_stats = []
    for j in range(bone_arr.shape[1]):
        col = bone_arr[:,j]
        col_valid = col[~np.isnan(col)]
        if col_valid.size < 5:
            mu = np.nan
            sigma = np.nan
            cvv = np.nan
        else:
            mu = float(np.mean(col_valid))
            sigma = float(np.std(col_valid))
            cvv = sigma/mu if abs(mu) > 1e-9 else np.nan
        bone_stats.append((j, mu, sigma, cvv))
    print("骨長変動 (index, mean, std, CV):", bone_stats)

    # CSV 保存
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    mono_rows = []
    for f_idx, pts in enumerate(records):
        row = {"frame": f_idx}
        for j,(x,y,z) in enumerate(pts):
            row[f"joint_{j}_x"] = round(float(x),5) if x==x else None
            row[f"joint_{j}_y"] = round(float(y),5) if y==y else None
            row[f"joint_{j}_z"] = round(float(z),5) if z==z else None
        mono_rows.append(row)
    df = pd.DataFrame(mono_rows)
    out_path = os.path.join(SAVE_DIR, f"mono3d_world_test_{timestamp}.csv")
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ 単眼3Dテスト CSV 出力: {out_path}")

    # 骨長統計も保存
    stat_rows = [{"bone_index": b, "mean": m, "std": s, "cv": c} for b,m,s,c in bone_stats]
    df_stat = pd.DataFrame(stat_rows)
    stat_path = os.path.join(SAVE_DIR, f"mono3d_bonestats_{timestamp}.csv")
    df_stat.to_csv(stat_path, index=False, encoding='utf-8-sig')
    print(f"✅ 骨長統計 CSV 出力: {stat_path}")

    # (既に finally で解放済) 重複解放は安全だが明示しない

if __name__ == "__main__":
    main()
