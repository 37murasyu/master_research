# %%


import glob
import os
import time
import argparse
import json

# pylint: disable=no-member
import cv2 as cv
import numpy as np
import wmi
import yaml

# import sys
from scipy import linalg
from botocore.exceptions import NoCredentialsError, BotoCoreError

# from config import (PADDING, dt, folder_path, fps, frame_shape, g,
#                    input_stream1, input_stream2, m1, m2, pose_keypoints,
#                    rm_path, save_dir, timestamp, w)
from JpText import putText_jp
from lambda_receive import invoke

RMSE = 0
# This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}
skip_manual_confirmation = True


def _save_checkerboard_short_axis(vector_cam0: np.ndarray, rows: int, columns: int, out_path: str = "camera_parameters/checkerboard_short_axis.json"):
    try:
        v = np.asarray(vector_cam0, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
        if (not np.all(np.isfinite(v))) or n < 1e-12:
            return
        v = v / n

        # master_research_code.py と同じ座標変換（raw -> transformed）
        v_runtime = np.array([-v[0], -v[2], -v[1]], dtype=float)
        nr = float(np.linalg.norm(v_runtime))
        if nr > 1e-12 and np.isfinite(nr):
            v_runtime = v_runtime / nr

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "rows": int(rows),
            "columns": int(columns),
            "short_side_axis_board": "x" if rows <= columns else "y",
            "vector_cam0": [float(v[0]), float(v[1]), float(v[2])],
            "vector_runtime": [float(v_runtime[0]), float(v_runtime[1]), float(v_runtime[2])],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] saved checkerboard short-side axis -> {out_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] checkerboard short-side axis save failed: {e}")


def _intrinsic_file_key(camera_name):
    if camera_name == "camera0":
        return "c0"
    if camera_name == "camera1":
        return "c1"
    return camera_name


def _get_setting(key, default=None):
    return calibration_settings.get(key, default)


def _set_and_verify_resolution(cap, desired_w, desired_h, label="cam", warmup_frames=5):
    """
    Try to set resolution, then read a few frames to confirm the actual output size.
    Warns if the camera cannot deliver the requested resolution.
    Returns (actual_w, actual_h).
    """
    # Read once to know the default
    ok0, f0 = cap.read()
    if ok0 and f0 is not None:
        print(f"[INFO] {label} default resolution: {f0.shape[1]}x{f0.shape[0]}")
    else:
        print(f"[WARN] {label} default read failed; will try to set resolution anyway")

    cap.set(3, desired_w)
    cap.set(4, desired_h)

    actual_w = actual_h = None
    for _ in range(max(1, warmup_frames)):
        ok, frame = cap.read()
        if ok and frame is not None:
            actual_w = frame.shape[1]
            actual_h = frame.shape[0]
        else:
            continue
    if actual_w is None:
        print(f"[WARN] {label} failed to read frame after resolution set")
        return desired_w, desired_h

    if actual_w != desired_w or actual_h != desired_h:
        print(
            f"[WARN] {label} requested {desired_w}x{desired_h} but got {actual_w}x{actual_h} (using actual)"
        )
    else:
        print(f"[INFO] {label} resolution set OK: {actual_w}x{actual_h}")

    return actual_w, actual_h


# Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):

    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)

    # print('Triangulated point: ')
    # print(Vh[3,0:3]/Vh[3,3])
    return Vh[3, 0:3] / Vh[3, 3]


# Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):

    global calibration_settings

    if not os.path.exists(filename):
        print("File does not exist:", filename)
        quit()

    # print('Using for calibration settings: ', filename)

    # Try UTF-8 first (YAML with Japanese comments), then UTF-8 BOM, then locale fallback.
    encodings = ["utf-8", "utf-8-sig", "cp932"]
    last_error = None
    for enc in encodings:
        try:
            with open(filename, encoding=enc) as f:
                calibration_settings = yaml.safe_load(f)
            break
        except UnicodeDecodeError as e:  # noqa: PERF203
            last_error = e
            continue
    else:  # no break
        raise UnicodeDecodeError(
            "", b"", 0, 0, f"Failed to decode settings file with encodings {encodings}: {last_error}"
        )

    # rudimentary check to make sure correct file was loaded
    if "camera0" not in calibration_settings.keys():
        # print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()


# Open camera stream and save frames
def save_frames_single_camera(camera_name):

    # create frames directory
    if not os.path.exists("frames"):
        os.mkdir("frames")

    # get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    number_to_save = calibration_settings["mono_calibration_frames"]
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]

    # open video stream (prefer DSHOW to avoid MSMF stream failures)
    cap = None
    for be in [cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY]:
        cap = cv.VideoCapture(camera_device_id, be)
        if cap is not None and cap.isOpened():
            print(f"[INFO] Opened {camera_name} index={camera_device_id} backend={be}")
            break
        if cap:
            cap.release()
            cap = None
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"{camera_name} (index {camera_device_id}) をオープンできません。USB接続/他アプリ占有を確認してください。")

    # change resolution (with verification)
    width, height = _set_and_verify_resolution(cap, width, height, label=camera_name)

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:

        ret, frame = cap.read()
        if ret == False:
            # if no video data is received, can't calibrate the camera, so exit.
            print("No video data received from camera. Exiting...")
            quit()

        frame_small = cv.resize(frame, None, fx=1 / view_resize, fy=1 / view_resize)

        if not start:
            cv.putText(
                frame_small,
                "Press SPACEBAR to start collection frames",
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

        if start:
            cooldown -= 1
            cv.putText(
                frame_small,
                "Cooldown: " + str(cooldown),
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                frame_small,
                "Num frames: " + str(saved_count),
                (50, 100),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )

            # save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join(
                    "frames", camera_name + "_" + str(saved_count) + ".png"
                )
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow("frame_small", frame_small)
        k = cv.waitKey(1)

        if k == 27:
            # if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            # Press spacebar to start data collection
            start = True

        # break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()


# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):

    # NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(images_prefix)

    # read all frames
    images = [cv.imread(imname, 1) for imname in images_names]
    if not images:
        raise RuntimeError(f"No images found for intrinsic calibration (prefix={images_prefix}).")

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    # checkerboard_box_size_scale は cm 単位で設定されているため m に換算してから使う
    world_scaling_cm = calibration_settings["checkerboard_box_size_scale"]
    world_scaling = world_scaling_cm * 0.01

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp
    short_axis_board = np.array([1.0, 0.0, 0.0], dtype=np.float64) if rows <= columns else np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space
    short_axis_samples_cam0 = []

    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.putText(
                frame,
                'If detected points are poor, press "s" to skip this sample',
                (25, 25),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

            cv.imshow("img", frame)
            if not skip_manual_confirmation:

                k = cv.waitKey(0)

                if k & 0xFF == ord("s"):
                    print("skipping")
                    continue
            else:
                k = -1  # 自動スキップ

            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )
    print("rmse:", ret)
    # print('camera matrix:\n', cmtx)
    # print('distortion coeffs:', dist)

    return cmtx, dist


# save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    # create folder if it does not exist
    if not os.path.exists("camera_parameters"):
        os.mkdir("camera_parameters")

    file_key = _intrinsic_file_key(camera_name)
    out_filename = os.path.join("camera_parameters", file_key + ".dat")
    outf = open(out_filename, "w")

    outf.write("intrinsic:\n")
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + " ")
        outf.write("\n")

    outf.write("distortion:\n")
    for en in distortion_coefs[0]:
        outf.write(str(en) + " ")
    outf.write("\n")


# open both cameras and take calibration frames
def save_frames_two_cams(camera0_name, camera1_name):

    # create frames directory
    if not os.path.exists("frames_pair"):
        os.mkdir("frames_pair")

    # settings for taking data
    view_resize = calibration_settings["view_resize"]
    cooldown_time = calibration_settings["cooldown"]
    number_to_save = calibration_settings["stereo_calibration_frames"]

    def _open_cam(index, label):
        preferred_backends = [cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY]
        last_cap = None
        for be in preferred_backends:
            cap = cv.VideoCapture(index, be)
            if cap is not None and cap.isOpened():
                print(f"[INFO] Opened {label} index={index} backend={be}")
                return cap
            if cap:
                cap.release()
        print(f"[ERROR] {label} (index {index}) をオープンできません。")
        return last_cap

    cam0_index = calibration_settings[camera0_name]
    cam1_index = calibration_settings[camera1_name]
    cap0 = _open_cam(cam0_index, camera0_name)
    cap1 = _open_cam(cam1_index, camera1_name)
    if cap0 is None or not cap0.isOpened() or cap1 is None or not cap1.isOpened():
        raise RuntimeError("カメラオープン失敗。インデックス/他アプリ占有/USB 接続を確認してください。")

    # Warm-up & resolution set/verify
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    width0, height0 = _set_and_verify_resolution(cap0, width, height, label=camera0_name)
    width1, height1 = _set_and_verify_resolution(cap1, width, height, label=camera1_name)
    if (width0, height0) != (width1, height1):
        print(f"[WARN] Resolution mismatch between cams: {camera0_name}={width0}x{height0}, {camera1_name}={width1}x{height1}")

    cooldown = cooldown_time
    start = False
    saved_count = 0
    while True:

        ret0, frame0 = cap0.read(); ret1, frame1 = cap1.read()
        if not ret0 or frame0 is None:
            print(f"[WARN] {camera0_name} からフレーム取得失敗。再試行します。")
            fail_retry = 0
            recovered = False
            while fail_retry < 5 and not recovered:
                fail_retry += 1
                ret0, frame0 = cap0.read()
                if ret0 and frame0 is not None:
                    recovered = True
            if not recovered:
                raise RuntimeError(f"{camera0_name} が継続的にフレームを返しません。ケーブル/インデックス/他プロセス確認。")
        if not ret1 or frame1 is None:
            print(f"[WARN] {camera1_name} からフレーム取得失敗。再試行します。")
            fail_retry = 0
            recovered = False
            while fail_retry < 5 and not recovered:
                fail_retry += 1
                ret1, frame1 = cap1.read()
                if ret1 and frame1 is not None:
                    recovered = True
            if not recovered:
                raise RuntimeError(f"{camera1_name} が継続的にフレームを返しません。ケーブル/インデックス/他プロセス確認。")

        frame0_small = cv.resize(
            frame0, None, fx=1.0 / view_resize, fy=1.0 / view_resize
        )
        frame1_small = cv.resize(
            frame1, None, fx=1.0 / view_resize, fy=1.0 / view_resize
        )

        if not start:
            # cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            frame0_small = putText_jp(
                frame0_small,
                "カメラが模様・肩・両腕・太ももを適切に捉えていることを確認してください",
                (50, 50),
                24,
                (0, 0, 255),
                20,
            )

            frame0_small = putText_jp(
                frame0_small,
                "スペースキーを押してください、5秒後に始めます",
                (50, 100),
                24,
                (0, 0, 255),
                20,
            )

            frame1_small = putText_jp(
                frame1_small,
                "撮影中には体を動かしたり、小物をもってください",
                (50, 150),
                24,
                (0, 0, 255),
                20,
            )

        if start:

            # 現在の時間を計算
            # elapsed_time = time.time() - start_time
            # remaining_time = countdown_duration - int(elapsed_time)

            """
            if remaining_time > 0:
                # カウントダウンのテキストをフレームに描画
                cv.putText(
                    frame0_small,
                    str(remaining_time),
                    (50, 50),
                    cv.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )
            """

            cooldown -= 1
            cv.putText(
                frame0_small,
                "Cooldown: " + str(cooldown),
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                frame0_small,
                "Num frames: " + str(saved_count),
                (50, 100),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )

            cv.putText(
                frame1_small,
                "Cooldown: " + str(cooldown),
                (50, 50),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                frame1_small,
                "Num frames: " + str(saved_count),
                (50, 100),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )

            # save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join(
                    "frames_pair", camera0_name + "_" + str(saved_count) + ".png"
                )
                cv.imwrite(savename, frame0)

                savename = os.path.join(
                    "frames_pair", camera1_name + "_" + str(saved_count) + ".png"
                )
                cv.imwrite(savename, frame1)

                saved_count += 1
                cooldown = cooldown_time

        cv.imshow("frame0_small", frame0_small)
        cv.imshow("frame1_small", frame1_small)
        k = cv.waitKey(1)

        if k == 27:
            # if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            # Press spacebar to start data collection
            start = True
            start_time = time.time()
            countdown_duration = 5  # カウントダウンの秒数

        # break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()


# ---- Feature-based extrinsic estimation (checkerboard不要) ----

def _create_feature_detector(name="orb", nfeatures=None):
    name = (name or "orb").lower()
    if name == "sift":
        if hasattr(cv, "SIFT_create"):
            return cv.SIFT_create()
        print("[WARN] SIFT unavailable. Falling back to ORB.")
    return cv.ORB_create(nfeatures or 2000)


def _match_descriptors(desc1, desc2, use_lowe=True, ratio=0.75):
    if desc1 is None or desc2 is None:
        return []
    is_float = desc1.dtype == np.float32 or desc1.dtype == np.float64
    if is_float:
        index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
        search_params = dict(checks=32)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=not use_lowe)
    if use_lowe:
        knn = matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good
    return matcher.match(desc1, desc2)


def _epipolar_rmse(F, pts1, pts2):
    if F is None or len(pts1) == 0:
        return float("inf")
    pts1_h = cv.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    pts2_h = cv.convertPointsToHomogeneous(pts2).reshape(-1, 3)
    Fx1 = (F @ pts1_h.T).T
    Ftx2 = (F.T @ pts2_h.T).T
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    num = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
    d = num / np.sqrt(denom + 1e-12)
    return float(np.sqrt(np.mean(d * d)))


def _estimate_extrinsic_feature(imgL, imgR, K0, D0, K1, D1, *, detector="orb", ratio=0.75, ransac_thresh=1.0, baseline_m=None, nfeatures=None):
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    det = _create_feature_detector(detector, nfeatures=nfeatures)
    kp1, des1 = det.detectAndCompute(grayL, None)
    kp2, des2 = det.detectAndCompute(grayR, None)
    matches = _match_descriptors(des1, des2, use_lowe=True, ratio=ratio)
    matches = sorted(matches, key=lambda m: m.distance)
    if len(matches) < 8:
        raise RuntimeError(f"特徴対応が不足しています: {len(matches)} < 8")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 歪みを補正した座標に変換（正規化座標に近づける）
    pts1_ud = cv.undistortPoints(pts1, K0, D0)
    pts2_ud = cv.undistortPoints(pts2, K1, D1)
    pts1_ud = pts1_ud.reshape(-1, 2)
    pts2_ud = pts2_ud.reshape(-1, 2)

    # Fundamental を推定してから Essential を構成（Kが異なる場合に安全）
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, ransac_thresh, 0.999)
    if F is None or maskF is None:
        raise RuntimeError("Fundamental 行列の推定に失敗しました")
    inlier_mask = maskF.ravel() == 1
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]
    if len(pts1_in) < 8:
        raise RuntimeError(f"インライア不足: {len(pts1_in)} < 8")

    E = K1.T @ F @ K0
    retval, R, t, pose_mask = cv.recoverPose(E, pts1_in, pts2_in, K0, K1)

    # スケールは recoverPose 内部で正規化される。指定があれば基線長でスケーリング。
    t_unit = t / (np.linalg.norm(t) + 1e-12)
    if baseline_m is not None:
        t = t_unit * float(baseline_m)
    else:
        t = t_unit

    rmse = _epipolar_rmse(F, pts1_in.reshape(-1, 2), pts2_in.reshape(-1, 2))
    meta = {
        "matches": len(matches),
        "inliers": int(np.sum(inlier_mask)),
        "recover_inliers": int(pose_mask.sum()) if pose_mask is not None else None,
    }
    return R, t, rmse, meta


# open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1, *, extrinsic_mode="checkerboard", feature_params=None, baseline_m=None):
    # read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    # open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    # frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    if extrinsic_mode == "feature":
        params = feature_params or {}
        detector = params.get("detector", "orb")
        ratio = float(params.get("ratio", 0.75))
        ransac_thresh = float(params.get("ransac_thresh", 1.0))
        nfeatures = params.get("nfeatures")
        per_pair_results = []
        for f0, f1 in zip(c0_images, c1_images):
            try:
                R, T, rmse, meta = _estimate_extrinsic_feature(
                    f0,
                    f1,
                    mtx0,
                    dist0,
                    mtx1,
                    dist1,
                    detector=detector,
                    ratio=ratio,
                    ransac_thresh=ransac_thresh,
                    baseline_m=baseline_m,
                    nfeatures=nfeatures,
                )
                per_pair_results.append((R, T, rmse, meta))
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] feature 外部推定失敗: {e}")
        if not per_pair_results:
            raise RuntimeError("feature ベース外部推定が全ペアで失敗しました")
        # 最良RMSEのペアを採用（簡易集約）
        per_pair_results.sort(key=lambda x: x[2])
        best_R, best_T, best_rmse, meta = per_pair_results[0]
        print(f"[INFO] feature外部推定: matches={meta['matches']} inliers={meta['inliers']} rmse={best_rmse:.4f}")
        return best_R, best_T, best_rmse

    # ---- Checkerboard (従来) ----
    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # calibration pattern settings
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    # 設定は cm 単位。外部パラメータのスケールを正しくするため m に換算して使用。
    world_scaling_cm = calibration_settings["checkerboard_box_size_scale"]
    world_scaling = world_scaling_cm * 0.01

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp
    short_axis_board = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if rows <= columns
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space
    short_axis_samples_cam0 = []

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0, 0].astype(np.int32)
            p0_c2 = corners2[0, 0].astype(np.int32)

            cv.putText(
                frame0,
                "O",
                (p0_c1[0], p0_c1[1]),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.drawChessboardCorners(frame0, (rows, columns), corners1, c_ret1)
            cv.imshow("img", frame0)

            cv.putText(
                frame1,
                "O",
                (p0_c2[0], p0_c2[1]),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.drawChessboardCorners(frame1, (rows, columns), corners2, c_ret2)
            cv.imshow("img2", frame1)
            if not skip_manual_confirmation:
                k = cv.waitKey(0)
                if k & 0xFF == ord("s"):
                    continue
            else:
                k = -1

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

            # camera0 画像で短辺方向ベクトルを推定して保持
            try:
                ok_pnp, rvec, tvec = cv.solvePnP(objp, corners1, mtx0, dist0)
                if ok_pnp:
                    R_cb, _ = cv.Rodrigues(rvec)
                    v_cam0 = (R_cb @ short_axis_board.reshape(3, 1)).reshape(3)
                    nv = float(np.linalg.norm(v_cam0))
                    if np.all(np.isfinite(v_cam0)) and nv > 1e-12:
                        short_axis_samples_cam0.append(v_cam0 / nv)
            except Exception:
                pass

    if len(objpoints) == 0:
        raise RuntimeError("チェッカーボード検出が1枚も成功しませんでした。パターンが映っているか、rows/cols設定を確認してください。")

    if short_axis_samples_cam0:
        v_med = np.median(np.stack(short_axis_samples_cam0, axis=0), axis=0)
        _save_checkerboard_short_axis(v_med, rows, columns)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    data = {
        "objpoints": [pts.tolist() for pts in objpoints],
        "imgpoints_left": [pts.reshape(-1, 2).tolist() for pts in imgpoints_left],
        "imgpoints_right": [pts.reshape(-1, 2).tolist() for pts in imgpoints_right],
        "mtx0": mtx0.tolist(),
        "dist0": dist0.tolist(),
        "mtx1": mtx1.tolist(),
        "dist1": dist1.tolist(),
        "image_size": [width, height],
        "criteria": [3, 100, 0.001],
    }

    try:
        R, T, ret, dist0, dist1 = invoke(data, Test_mode=False)
        print("rmse (remote): ", ret)
        RMSE = ret
        cv.destroyAllWindows()
        return R, T, RMSE
    except (NoCredentialsError, BotoCoreError) as e:
        print(f"[WARN] Remote invoke 失敗 (認証/接続): {e}. OpenCV stereoCalibrate にフォールバックします。")
    except Exception as e:  # noqa: E722
        print(f"[WARN] Remote invoke 失敗: {e}. OpenCV stereoCalibrate にフォールバックします。")

    # ---- ローカルフォールバック ----
    flags = cv.CALIB_FIX_INTRINSIC
    # OpenCV stereoCalibrate は (R,T,E,F) を返す
    ret, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx0,
        dist0,
        mtx1,
        dist1,
        (width, height),
        criteria=criteria,
        flags=flags,
    )
    print("rmse (local):", ret)
    RMSE = ret
    cv.destroyAllWindows()
    return R, T, RMSE


# Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1

    return P


# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P


# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(
    camera0_name, camera0_data, camera1_name, camera1_data, RMSE, _zshift=50.0
):

    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    # define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    z_shift = np.array([0.0, 0.0, _zshift]).reshape((1, 3))
    # increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    # project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    # Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.0])

        # project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera0.append(uv)

        # project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera1.append(uv)

    # these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    # open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    # set camera resolutions
    width = calibration_settings["frame_width"]
    height = calibration_settings["frame_height"]
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            # print('Video stream not returning frame data')
            quit()

        # follow RGB colors to indicate XYZ axes respectively
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        # draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)

        # draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)
        if RMSE < 1:
            cv.putText(
                frame0,
                "Calibration is Perfect.",
                (25, 25),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.putText(
                frame1,
                "Calibration is Perfect.",
                (25, 25),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )

        elif RMSE > 1 and RMSE < 3:
            cv.putText(
                frame0,
                "Calibration is Good.",
                (25, 25),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
            cv.putText(
                frame1,
                "Calibration is Good.",
                (25, 25),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                1,
            )
        else:
            cv.putText(
                frame0,
                "Calibration is Poor. Try again.",
                (25, 25),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            cv.putText(
                frame1,
                "Calibration is Poor. Try again.",
                (25, 25),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1,
            )
        cv.imshow("frame0", frame0)
        cv.imshow("frame1", frame1)

        k = cv.waitKey(1)
        if k == 27:
            break

    cv.destroyAllWindows()


def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    # calibration pattern settings
    rows = calibration_settings["checkerboard_rows"]
    columns = calibration_settings["checkerboard_columns"]
    # 設定値は cm。solvePnP で正しいワールドスケールを得るため m に換算。
    world_scaling_cm = calibration_settings["checkerboard_box_size_scale"]
    world_scaling = world_scaling_cm * 0.01

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
    cv.putText(
        frame,
        "If you don't see detected points, try with a different image",
        (50, 50),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
        1,
    )
    cv.imshow("img", frame)
    if not skip_manual_confirmation:
        cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _ = cv.Rodrigues(rvec)  # rvec is Rotation matrix in Rodrigues vector form

    return R, tvec


def get_cam1_to_world_transforms(
    cmtx0, dist0, R_W0, T_W0, cmtx1, dist1, R_01, T_01, image_path0, image_path1
):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32"
    ).reshape((4, 1, 3))
    # axes colors are RGB format to indicate XYZ axes.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    # project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4, 2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow("frame0", frame0)
    cv.imshow("frame1", frame1)
    if not skip_manual_confirmation:
        cv.waitKey(0)

    return R_W1, T_W1


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=""):

    # create folder if it does not exist
    if not os.path.exists("camera_parameters"):
        os.mkdir("camera_parameters")

    camera0_rot_trans_filename = os.path.join(
        "camera_parameters", prefix + "rot_trans_c0.dat"
    )
    outf = open(camera0_rot_trans_filename, "w")

    outf.write("R:\n")
    for l in R0:
        for en in l:
            outf.write(str(en) + " ")
        outf.write("\n")

    outf.write("T:\n")
    for l in T0:
        for en in l:
            outf.write(str(en) + " ")
        outf.write("\n")
    outf.close()

    # R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join(
        "camera_parameters", prefix + "rot_trans_c1.dat"
    )
    outf = open(camera1_rot_trans_filename, "w")

    outf.write("R:\n")
    for l in R1:
        for en in l:
            outf.write(str(en) + " ")
        outf.write("\n")

    outf.write("T:\n")
    for l in T1:
        for en in l:
            outf.write(str(en) + " ")
        outf.write("\n")
    outf.close()

    return R0, T0, R1, T1


def load_camera_mapping(filepath="camera_parameters/calibration_map.yaml"):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    return {"camera_mappings": {}}


def save_camera_mapping(mapping, filepath="camera_parameters/calibration_map.yaml"):
    with open(filepath, "w") as f:
        yaml.safe_dump(mapping, f)


def is_camera_calibrated(mapping, label, current_device_name):
    return (
        mapping.get("camera_mappings", {}).get(label, {}).get("device_name")
        == current_device_name
    )


def _intrinsic_path(camera_name):
    file_key = _intrinsic_file_key(camera_name)
    return os.path.join("camera_parameters", f"{file_key}.dat")


def load_camera_intrinsics(camera_name):
    path = _intrinsic_path(camera_name)
    with open(path, "r") as f:
        lines = f.readlines()
    camera_matrix, distortion = [], []
    mode = None
    for line in lines:
        if "intrinsic" in line:
            mode = "intrinsic"
            continue
        elif "distortion" in line:
            mode = "distortion"
            continue
        values = list(map(float, line.strip().split()))
        if mode == "intrinsic":
            camera_matrix.append(values)
        elif mode == "distortion":
            distortion.append(values)
    return np.array(camera_matrix), np.array([distortion[0]])


def process_camera_intrinsics(label, current_device_name, image_prefix):
    mapping = load_camera_mapping()
    if is_camera_calibrated(mapping, label, current_device_name):
        print(
            f"{label} ({current_device_name}) は校正済みです。ファイルを再利用します。"
        )
        cmtx, dist = load_camera_intrinsics(label)
    else:
        print(f"{label} ({current_device_name}) をキャリブレーションします。")
        cmtx, dist = calibrate_camera_for_intrinsic_parameters(image_prefix)
        save_camera_intrinsics(cmtx, dist, label)
        mapping["camera_mappings"][label] = {"device_name": current_device_name}
        save_camera_mapping(mapping)
    return cmtx, dist


def enumerate_camera_device_names_windows():
    w = wmi.WMI()
    cameras = []
    for item in w.Win32_PnPEntity():
        if item.Name and "camera" in item.Name.lower():
            cameras.append(item.Name)
    return cameras


def probe_cameras(max_index=10):
    print(f"[INFO] Probing camera indices 0..{max_index-1}")
    available = []
    for i in range(max_index):
        cap = cv.VideoCapture(i, cv.CAP_DSHOW)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"  Index {i}: OK (resolution={frame.shape[1]}x{frame.shape[0]})")
            available.append(i)
        else:
            print(f"  Index {i}: (no frame)")
        cap.release()
    if not available:
        print("[WARN] 取得可能なカメラが見つかりません。USB 接続/他アプリ占有を確認してください。")
    return available


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo calibration helper")
    parser.add_argument("legacy_settings", nargs="?", help="(旧形式) 最初の位置引数として設定YAMLパスを指定可能")
    parser.add_argument("--settings", default="calibration_settings.yaml", help="設定YAMLファイルパス (--settings が優先)")
    parser.add_argument("--cam-pair", default=None, help="使用するカメラインデックスを '0,2' のように指定。指定時 calibration_settings の camera0/camera1 を上書き")
    parser.add_argument("--skip-save-frames", action="store_true", help="Step3 のペアフレーム保存をスキップ (既存 frames_pair 利用)")
    parser.add_argument("--skip-check", action="store_true", help="最終的な check_calibration 表示をスキップ")
    parser.add_argument("--skip-gauge-demo", action="store_true", help="キャリブレーション完了後のデモ用ゲージ UI 表示をスキップ")
    parser.add_argument("--force-recapture", action="store_true", help="既存 frames_pair ディレクトリを削除して再キャプチャを強制")
    parser.add_argument("--expected-stereo-frames", type=int, default=None, help="ステレオキャプチャ期待枚数 (calibration_settings.stereo_calibration_frames を上書き)")
    parser.add_argument("--probe-cams", action="store_true", help="利用可能なカメラインデックスをスキャンして終了")
    parser.add_argument("--run-intrinsics", action="store_true", help="camera0/camera1 の内部パラメータ推定も実行 (既存データがない場合は自動で実行)")
    parser.add_argument("--skip-mono-capture", action="store_true", help="単体キャリブレーションの撮影をスキップし既存 frames ディレクトリを利用")
    parser.add_argument("--force-mono-recapture", action="store_true", help="単体キャリブレーション用 frames ディレクトリを削除して再撮影")
    parser.add_argument("--extrinsic-mode", choices=["checkerboard", "feature"], default=None, help="外部推定モード: checkerboard または feature (自然特徴)")
    parser.add_argument("--feature-detector", choices=["orb", "sift"], default=None, help="feature モード時の特徴量 (デフォルト orb)")
    parser.add_argument("--feature-ratio", type=float, default=None, help="Lowe ratio (feature) デフォルト 0.75")
    parser.add_argument("--feature-ransac-thresh", type=float, default=None, help="findFundamentalMat の RANSAC しきい値(px) デフォルト 1.0")
    parser.add_argument("--baseline-m", type=float, default=None, help="ベースライン長[m] を指定すると t をスケーリング")
    parser.add_argument("--feature-nfeatures", type=int, default=None, help="ORB の特徴点数上限 (デフォルト 2000)")
    args = parser.parse_args()

    # Determine which settings path to use (CLI precedence: --settings > positional)
    settings_path = args.settings
    if args.legacy_settings and args.legacy_settings.strip().lower().endswith(('.yml', '.yaml')):
        # Only override if user didn't explicitly change --settings
        if settings_path == 'calibration_settings.yaml':
            settings_path = args.legacy_settings
    print(f"[INFO] Using settings file: {settings_path}")
    parse_calibration_settings_file(settings_path)

    # Override camera indices if requested
    if args.cam_pair:
        try:
            c0_idx, c1_idx = [int(s) for s in args.cam_pair.split(',')]
            calibration_settings['camera0'] = c0_idx
            calibration_settings['camera1'] = c1_idx
            print(f"[INFO] Overridden camera indices -> camera0={c0_idx}, camera1={c1_idx}")
        except ValueError:
            print("[WARN] --cam-pair の形式が不正です。例: --cam-pair 0,2")

    # Expected frames override
    if args.expected_stereo_frames is not None:
        calibration_settings['stereo_calibration_frames'] = args.expected_stereo_frames
        print(f"[INFO] Overridden expected stereo frames -> {args.expected_stereo_frames}")

    # Extrinsic mode / feature params
    extrinsic_mode = args.extrinsic_mode or calibration_settings.get('extrinsic_mode', 'checkerboard')
    feature_params = {
        'detector': args.feature_detector or calibration_settings.get('feature_detector', 'orb'),
        'ratio': args.feature_ratio or calibration_settings.get('feature_ratio', 0.75),
        'ransac_thresh': args.feature_ransac_thresh or calibration_settings.get('feature_ransac_thresh', 1.0),
        'nfeatures': args.feature_nfeatures or calibration_settings.get('feature_nfeatures', None),
    }
    print(f"[INFO] Extrinsic mode: {extrinsic_mode}")

    intrinsic_results = {}

    def _intrinsics_exist():
        return os.path.exists(_intrinsic_path("camera0")) and os.path.exists(
            _intrinsic_path("camera1")
        )

    def _remove_old_frames_pair():
        import shutil, time
        target = 'frames_pair'
        if not os.path.isdir(target):
            return
        print('[INFO] Removing old frames_pair directory for fresh capture (--force-recapture)')
        # Windows でロックされている場合に備えてリトライ + リネームフォールバック
        for attempt in range(3):
            try:
                shutil.rmtree(target)
                return
            except PermissionError:
                print(f'[WARN] 削除ロック (PermissionError) attempt={attempt+1}. 0.5秒待機して再試行します。')
                time.sleep(0.5)
        # リネームフォールバック
        backup_name = f"{target}_old_{int(time.time())}"
        try:
            os.rename(target, backup_name)
            print(f"[INFO] 削除できないため一時リネーム: {backup_name}")
            # さらにバックグラウンドで削除試行
            try:
                shutil.rmtree(backup_name)
            except PermissionError:
                print(f"[WARN] 一時フォルダ {backup_name} の削除も失敗。手動で削除してください。")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] frames_pair の削除/リネームに失敗: {e}. エクスプローラで開かれていないか、画像が他アプリで使用中でないか確認してください。")

    def _remove_old_frames_single():
        import shutil, time
        target = 'frames'
        if not os.path.isdir(target):
            return
        print('[INFO] Removing old frames directory for fresh single-camera capture (--force-mono-recapture)')
        for attempt in range(3):
            try:
                shutil.rmtree(target)
                return
            except PermissionError:
                print(f'[WARN] 削除ロック (PermissionError) attempt={attempt+1}. 0.5秒待機して再試行します。')
                time.sleep(0.5)
        backup_name = f"{target}_old_{int(time.time())}"
        try:
            os.rename(target, backup_name)
            print(f"[INFO] 削除できないため一時リネーム: {backup_name}")
            try:
                shutil.rmtree(backup_name)
            except PermissionError:
                print(f"[WARN] 一時フォルダ {backup_name} の削除も失敗。手動で削除してください。")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] frames ディレクトリの削除/リネームに失敗: {e}.")

    def _validate_frames_pair():
        expected = calibration_settings.get('stereo_calibration_frames', 0)
        c0_files = sorted(glob.glob(os.path.join('frames_pair', 'camera0*')))
        c1_files = sorted(glob.glob(os.path.join('frames_pair', 'camera1*')))
        c0_n = len(c0_files)
        c1_n = len(c1_files)
        if c0_n == 0 or c1_n == 0:
            raise RuntimeError('キャリブレーション用フレームが 0 枚です。環境光/パターン表示/カメラ角度を確認してください。')
        if c0_n != c1_n:
            raise RuntimeError(f'左右枚数不一致: left={c0_n}, right={c1_n} / 同期に問題があります。')
        if expected and c0_n != expected:
            print(f"[WARN] 期待枚数 {expected} に対し取得 {c0_n} 枚。設定値と異なります。")
        else:
            print(f"[INFO] Stereo frame count OK: {c0_n} ペア")

    # Step1 / Step2 (単体キャリブレーション) はコメントのまま（必要時に有効化）

    if args.probe_cams:
        probe_cameras()
        raise SystemExit(0)

    need_intrinsics = args.run_intrinsics or not _intrinsics_exist()
    if need_intrinsics:
        if not args.run_intrinsics:
            print('[INFO] 既存の内部パラメータが見つからないため自動で推定します (--run-intrinsics 指定不要)')
        if args.force_mono_recapture:
            _remove_old_frames_single()
        mono_prefix_c0 = os.path.join('frames', 'camera0_*.png')
        mono_prefix_c1 = os.path.join('frames', 'camera1_*.png')
        if args.skip_mono_capture:
            print('[INFO] Skipping single-camera capture (using existing frames directory)')
            if not glob.glob(mono_prefix_c0) or not glob.glob(mono_prefix_c1):
                raise RuntimeError('単体キャリブレーション用の画像が見つかりません。--skip-mono-capture を外すか、frames ディレクトリに画像を用意してください。')
        else:
            save_frames_single_camera('camera0')
            save_frames_single_camera('camera1')
        cmtx0_mono, dist0_mono = calibrate_camera_for_intrinsic_parameters(mono_prefix_c0)
        save_camera_intrinsics(cmtx0_mono, dist0_mono, 'camera0')
        intrinsic_results['camera0'] = (cmtx0_mono, dist0_mono)
        cmtx1_mono, dist1_mono = calibrate_camera_for_intrinsic_parameters(mono_prefix_c1)
        save_camera_intrinsics(cmtx1_mono, dist1_mono, 'camera1')
        intrinsic_results['camera1'] = (cmtx1_mono, dist1_mono)
    else:
        print('[INFO] 内部パラメータが既に存在するためスキップします (--run-intrinsics で再推定可能)')

    # Step3. Save calibration frames for both cameras simultaneously
    if not args.skip_save_frames:
        if args.force_recapture:
            _remove_old_frames_pair()
        save_frames_two_cams("camera0", "camera1")
        try:
            _validate_frames_pair()
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] フレーム検証失敗: {e}")
            raise
    else:
        print("[INFO] Skipping frame capture (using existing frames_pair)")

    # Step4. Stereo calibration
    frames_prefix_c0 = os.path.join("frames_pair", "camera0*")
    frames_prefix_c1 = os.path.join("frames_pair", "camera1*")
    if 'camera0' in intrinsic_results:
        cmtx0, dist0 = intrinsic_results['camera0']
    else:
        cmtx0, dist0 = load_camera_intrinsics("camera0")
    if 'camera1' in intrinsic_results:
        cmtx1, dist1 = intrinsic_results['camera1']
    else:
        cmtx1, dist1 = load_camera_intrinsics("camera1")
    R, T, RMSE = stereo_calibrate(
        cmtx0,
        dist0,
        cmtx1,
        dist1,
        frames_prefix_c0,
        frames_prefix_c1,
        extrinsic_mode=extrinsic_mode,
        feature_params=feature_params,
        baseline_m=args.baseline_m,
    )

    # Step5. Save extrinsic (camera0 as world)
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
    save_extrinsic_calibration_parameters(R0, T0, R, T)
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R, T]
    if not args.skip_check:
        check_calibration("camera0", camera0_data, "camera1", camera1_data, RMSE, _zshift=60.0)
    else:
        print("[INFO] Skipping visual check")

    if not args.skip_gauge_demo:
        try:
            import time as _t
            print("[INFO] Gauge demo will launch in 2 seconds...")
            _t.sleep(2.0)
            try:
                from run_gauge_demo import main as gauge_demo_main
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] run_gauge_demo 読み込みに失敗しました: {e}")
            else:
                gauge_demo_main()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Gauge demo 起動に失敗しました: {e}")

# %%
