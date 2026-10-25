"""Lambda 함수 URL 배포(상시 무료 등급만: Lambda·IAM·CloudWatch Logs. API Gateway·ECR·S3 안 씀). 멱등.
사용: python3 scripts/deploy_lambda.py [profile] [region]   — 키 값은 출력하지 않는다."""
import json, sys, time
import boto3
from botocore.exceptions import ClientError

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "dayafter"
REGION = sys.argv[2] if len(sys.argv) > 2 else "ap-northeast-2"
NAME, ROLE = "label-guard", "label-guard-lambda-role"
s = boto3.Session(profile_name=PROFILE, region_name=REGION)
iam, lam, logs = s.client("iam"), s.client("lambda"), s.client("logs")


def role_arn():
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
    except ClientError:
        arn = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust), Description="label-guard Lambda (logs only)")["Role"]["Arn"]
        iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
        time.sleep(10)
    return arn


def layer(name, path):
    r = lam.publish_layer_version(LayerName=name, Content={"ZipFile": open(path, "rb").read()}, CompatibleRuntimes=["python3.12"],
                                  CompatibleArchitectures=["x86_64"], Description=name)
    return r["LayerVersionArn"]


def main():
    arn = role_arn()
    layers = [layer("label-guard-cv2", "dist/layer-cv2.zip"), layer("label-guard-libs-numpy", "dist/layer-libs-numpy.zip")]
    code = open("dist/function.zip", "rb").read()
    cfg = dict(Runtime="python3.12", Role=arn, Handler="label_guard.infrastructure.lambda_handler.lambda_handler", Layers=layers,
               MemorySize=2048, Timeout=60, Architectures=["x86_64"], Environment={"Variables": {"PYTHONPATH": "/var/task:/opt/python"}})
    try:
        lam.get_function(FunctionName=NAME)
        lam.update_function_code(FunctionName=NAME, ZipFile=code)
        lam.get_waiter("function_updated").wait(FunctionName=NAME)
        lam.update_function_configuration(FunctionName=NAME, **{k: v for k, v in cfg.items() if k != "Architectures"})
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        lam.create_function(FunctionName=NAME, Code={"ZipFile": code}, Description="OpenCV label guard", **cfg)
    lam.get_waiter("function_active_v2").wait(FunctionName=NAME)
    lam.put_function_concurrency(FunctionName=NAME, ReservedConcurrentExecutions=5)   # 남용 상한(비용 보호)
    try:
        logs.create_log_group(logGroupName=f"/aws/lambda/{NAME}")
    except ClientError:
        pass
    logs.put_retention_policy(logGroupName=f"/aws/lambda/{NAME}", retentionInDays=7)
    try:
        url = lam.get_function_url_config(FunctionName=NAME)["FunctionUrl"]
    except ClientError:
        url = lam.create_function_url_config(FunctionName=NAME, AuthType="NONE")["FunctionUrl"]
    for sid, act, extra in (("public-url", "lambda:InvokeFunctionUrl", {"FunctionUrlAuthType": "NONE"}), ("public-invoke", "lambda:InvokeFunction", {"InvokedViaFunctionUrl": True})):
        try:
            lam.add_permission(FunctionName=NAME, StatementId=sid, Action=act, Principal="*", **extra)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceConflictException":
                raise
    print(url)


if __name__ == "__main__":
    main()
