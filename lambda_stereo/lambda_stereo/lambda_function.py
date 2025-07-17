import json
import numpy as np

# pylint: disable=no-member
import cv2 as cv


def lambda_handler(event, context):
    try:
        objpoints = [np.array(pts, dtype=np.float32) for pts in event["objpoints"]]
        imgpoints_left = [
            np.array(pts, dtype=np.float32) for pts in event["imgpoints_left"]
        ]
        imgpoints_right = [
            np.array(pts, dtype=np.float32) for pts in event["imgpoints_right"]
        ]
        mtx0 = np.array(event["mtx0"], dtype=np.float64)
        dist0 = np.array(event["dist0"], dtype=np.float64)
        mtx1 = np.array(event["mtx1"], dtype=np.float64)
        dist1 = np.array(event["dist1"], dtype=np.float64)
        image_size = tuple(event["image_size"])

        criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            event["criteria"][1],
            event["criteria"][2],
        )

        flags = cv.CALIB_FIX_INTRINSIC

        ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            mtx0,
            dist0,
            mtx1,
            dist1,
            image_size,
            criteria=criteria,
            flags=flags,
        )

        return {
            "statusCode": 200,
            "R": R.tolist(),
            "T": T.tolist(),
            "rmse": ret,
            "dist0": dist0.tolist(),
            "dist1": dist1.tolist(),
        }

    except Exception as e:
        return {"statusCode": 500, "error": str(e)}
