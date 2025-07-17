import json
import boto3
import numpy as np


def load_test_data():
    return {
        "objpoints": [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]],
        "imgpoints_left": [[[100, 200], [150, 200], [100, 250], [150, 250]]],
        "imgpoints_right": [[[105, 205], [155, 205], [105, 255], [155, 255]]],
        "mtx0": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        "dist0": [[0.1, -0.05, 0.001, 0.001, 0.0]],
        "mtx1": [[505, 0, 315], [0, 505, 235], [0, 0, 1]],
        "dist1": [[0.09, -0.04, 0.001, 0.001, 0.0]],
        "image_size": [640, 480],
        "criteria": [3, 100, 0.001],
    }


def invoke(data, Test_mode=True):
    client = boto3.client("lambda", region_name="ap-northeast-1")
    if Test_mode:
        payload = load_test_data()
    else:
        payload = data
    response = client.invoke(
        FunctionName="stereo-calib-StereoCalibrateFunction-miSjAHEXwnsz",
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )
    result = json.loads(response["Payload"].read())
    print("Raw result:", result)
    if result["statusCode"] == 200:
        R = np.array(result["R"])
        T = np.array(result["T"])
        RMSE = result["rmse"]
        dist0 = np.array(result["dist0"])
        dist1 = np.array(result["dist1"])
        print("R:", R)
        print("T:", T)
        print("RMSE:", RMSE)
        print("dist0:", dist0)
        print("dist1:", dist1)
        return R, T, RMSE, dist0, dist1
    else:
        raise RuntimeError("Lambda Error: " + result.get("error", "Unknown error"))


if __name__ == "__main__":
    invoke(data=None)
