import json
import os
import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# 環境変数で挙動を制御
#   CALIB_REMOTE_DISABLE=1  -> 強制的にリモート呼び出しをスキップ
#   CALIB_REMOTE_VERBOSE=1  -> 追加デバッグ情報を表示
#   CALIB_AWS_PROFILE=prof  -> 指定プロファイルで boto3 セッション生成
#   CALIB_LAMBDA_NAME=xxxx  -> Lambda 関数名を上書き

DEFAULT_REGION = "ap-northeast-1"
DEFAULT_LAMBDA_NAME = "stereo-calib-StereoCalibrateFunction-miSjAHEXwnsz"


def _log_verbose(msg: str):
    if os.getenv("CALIB_REMOTE_VERBOSE"):
        print(f"[VERBOSE] {msg}")


def _get_session():
    profile = os.getenv("CALIB_AWS_PROFILE")
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    if profile:
        _log_verbose(f"Using AWS profile='{profile}', region='{region}'")
        return boto3.Session(profile_name=profile, region_name=region)
    return boto3.Session(region_name=region)


def _credentials_present(session=None):
    try:
        session = session or _get_session()
        creds = session.get_credentials()
        return creds is not None
    except Exception as e:  # noqa: BLE001
        _log_verbose(f"Credentials check exception: {e}")
        return False


def check_aws_identity():
    """STS で呼び出し元を確認 (デバッグ用)。"""
    try:
        session = _get_session()
        if not _credentials_present(session):
            print("[INFO] AWS 認証情報が見つかりません。ローカルフォールバックを使用します。")
            return None
        sts = session.client("sts")
        ident = sts.get_caller_identity()
        print(f"[INFO] AWS Caller Identity: {ident}")
        return ident
    except NoCredentialsError:
        print("[INFO] NoCredentialsError: 認証情報が取得できません。")
    except PartialCredentialsError:
        print("[WARN] PartialCredentialsError: 一部欠損した認証情報。環境変数や credentials ファイルを確認してください。")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] STS 確認中にエラー: {e}")
    return None


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
    """Lambda を呼び出しステレオキャリブ結果を取得。

    認証情報が無い場合や失敗時には NoCredentialsError / RuntimeError を投げるので
    呼び出し側でフォールバック処理を行ってください。
    """

    if os.getenv("CALIB_REMOTE_DISABLE"):
        raise NoCredentialsError()

    session = _get_session()
    if not _credentials_present(session):
        raise NoCredentialsError()

    lambda_name = os.getenv("CALIB_LAMBDA_NAME", DEFAULT_LAMBDA_NAME)
    client = session.client("lambda")

    if Test_mode:
        payload = load_test_data()
    else:
        payload = data

    _log_verbose(f"Invoking Lambda: {lambda_name} (payload keys={list(payload.keys())})")

    try:
        response = client.invoke(
            FunctionName=lambda_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
    except (NoCredentialsError, PartialCredentialsError):
        raise
    except client.exceptions.ResourceNotFoundException as e:  # type: ignore[attr-defined]
        raise RuntimeError(f"Lambda 関数が存在しません: {lambda_name}: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Lambda 呼び出し失敗: {e}") from e

    raw_payload = response.get("Payload")
    if raw_payload is None:
        raise RuntimeError("Lambda 応答に Payload が含まれていません。")
    result = json.loads(raw_payload.read())
    print("Raw result:", result)
    if isinstance(result, dict) and result.get("statusCode") == 200:
        try:
            R = np.array(result["R"], dtype=float)
            T = np.array(result["T"], dtype=float)
            RMSE = float(result["rmse"]) if "rmse" in result else float(result.get("RMSE", "nan"))
            dist0 = np.array(result.get("dist0", []), dtype=float)
            dist1 = np.array(result.get("dist1", []), dtype=float)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"結果デコード失敗: {e}") from e
        print("R:", R)
        print("T:", T)
        print("RMSE:", RMSE)
        print("dist0:", dist0)
        print("dist1:", dist1)
        return R, T, RMSE, dist0, dist1
    raise RuntimeError("Lambda Error: " + str(result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lambda remote stereo calib 呼び出しテスト")
    parser.add_argument("--local-test", action="store_true", help="テストデータで Lambda を呼び出す")
    parser.add_argument("--check-identity", action="store_true", help="STS get_caller_identity を実行")
    args = parser.parse_args()

    if args.check_identity:
        check_aws_identity()

    try:
        R, T, RMSE, dist0, dist1 = invoke(data=None, Test_mode=args.local_test)
        print("OK: remote result received")
    except NoCredentialsError:
        print("[ERROR] 認証情報が見つかりません。以下を確認してください:\n"
              " 1. AWS CLI がインストール済みか (aws --version)\n"
              " 2. 'aws configure' でアクセスキーを設定したか、または SSO プロファイルを設定したか\n"
              " 3. PowerShell で AWS_PROFILE / 環境変数 (AWS_ACCESS_KEY_ID など) を設定したか\n"
              " 4. ~/.aws/credentials に正しいエントリがあるか\n"
              " 5. 環境変数 CALIB_REMOTE_DISABLE=1 が設定されていないか")
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 失敗: {e}")
