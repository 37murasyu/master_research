import json


def lambda_handler(event, context):
    x = event.get("x")
    y = event.get("y")

    if x is None or y is None:
        return {"statusCode": 400, "body": "Missing 'x' or 'y'"}

    result = x + y

    return {"statusCode": 200, "body": json.dumps({"result": result})}
