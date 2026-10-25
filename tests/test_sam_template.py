"""template.yaml = 현재 배포와 같은 정의인지 실측 스냅샷(tests/fixtures/deployed-config.json)과 대조한다. 오프라인."""
import json
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


class _L(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):  # !Ref !Sub !GetAtt 등 CloudFormation 단축 태그
    if isinstance(node, yaml.ScalarNode):
        return {suffix: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {suffix: loader.construct_sequence(node)}
    return {suffix: loader.construct_mapping(node)}


_L.add_multi_constructor("!", _tag)
T = yaml.load((ROOT / "template.yaml").read_text(), Loader=_L)
D = json.loads((ROOT / "tests/fixtures/deployed-config.json").read_text())
R = T["Resources"]


def _fn():
    fns = [r for r in R.values() if r["Type"] == "AWS::Serverless::Function"]
    assert len(fns) == 1
    return fns[0]["Properties"]


def test_function_matches_deployed():
    p, d = _fn(), D["function"]
    for k in ("Runtime", "Handler", "MemorySize", "Timeout", "Architectures"):
        assert p[k] == d[k], k
    assert p["ReservedConcurrentExecutions"] == d["ReservedConcurrentExecutions"]
    assert p["EphemeralStorage"]["Size"] == d["EphemeralStorage"]
    assert p["Environment"]["Variables"] == d["Environment"]
    assert len(p["Layers"]) == len(d["Layers"])


def test_layers_match_deployed_names():
    names = sorted(r["Properties"]["LayerName"] for r in R.values() if r["Type"] == "AWS::Serverless::LayerVersion")
    assert names == sorted(D["function"]["Layers"])


def test_function_url_public_buffered():
    u = _fn()["FunctionUrlConfig"]
    assert u["AuthType"] == D["url"]["AuthType"] and u["InvokeMode"] == D["url"]["InvokeMode"]


def test_role_least_privilege_logs_only():
    p = _fn()
    assert p["Policies"] == ["AWSLambdaBasicExecutionRole"]
    assert D["role"]["InlinePolicies"] == 0 and len(D["role"]["AttachedPolicies"]) == 1
    assert not any(r["Type"] == "AWS::IAM::Role" for r in R.values()), "역할은 SAM이 만든다(관리형 정책 1개)"


def test_log_retention_matches():
    lg = [r for r in R.values() if r["Type"] == "AWS::Logs::LogGroup"]
    assert len(lg) == 1 and lg[0]["Properties"]["RetentionInDays"] == D["log_retention_days"]


def test_alarms_errors_and_throttles_defined_without_actions():
    al = {r["Properties"]["MetricName"]: r["Properties"] for r in R.values() if r["Type"] == "AWS::CloudWatch::Alarm"}
    assert set(al) == {"Errors", "Throttles"}
    for p in al.values():
        assert p["Namespace"] == "AWS/Lambda" and p["Threshold"] >= 1 and p["TreatMissingData"] == "notBreaching"
        assert "AlarmActions" not in p  # 알림 수신처(SNS)는 계정 행위라 정의하지 않는다


def test_no_stack_secrets_or_account_ids():
    txt = (ROOT / "template.yaml").read_text()
    assert "211125379804" not in txt and "AKIA" not in txt
