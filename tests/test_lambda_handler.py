"""AWS Lambda 함수 URL 핸들러 — 가짜 어댑터, 네트워크 0 (AC-8·AC-9)."""
import base64, json
import cv2, numpy as np
from label_guard.domain.model import PhotoQuality, ReadLine
from label_guard.infrastructure.lambda_handler import make_handler


class FakeAnalyzer:
    def analyze(self, image): return PhotoQuality(500, 0.0, 10, 0.0), image, {"regions_before": 3, "regions_after": 4}


class FakeReader:
    def __init__(self, lines): self.lines, self.seen = lines, []
    def read(self, image): self.seen.append(image.shape); return self.lines


def png_b64(w=64, h=48):
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 200, np.uint8)); return base64.b64encode(buf.tobytes()).decode()


def post(body_b64, qs=None, b64=True):
    return {"requestContext": {"http": {"method": "POST"}}, "body": body_b64, "isBase64Encoded": b64,
            "queryStringParameters": qs or {}, "headers": {"content-type": "image/png"}}


def test_get_returns_keyboard_usable_upload_page():
    r = make_handler(FakeAnalyzer(), FakeReader([]))({"requestContext": {"http": {"method": "GET"}}}, None)
    assert r["statusCode"] == 200 and "text/html" in r["headers"]["content-type"]
    assert 'type="file"' in r["body"] and "<label" in r["body"]


def test_post_image_returns_warning_with_trace_and_latency():
    h = make_handler(FakeAnalyzer(), FakeReader([ReadLine("CONTAINS MILK", 0.9)]))
    r = h(post(png_b64(), {"profile": "milk,soy", "lang": "ko"}), None)   # 기본은 영어(test_english_ui), 한국어는 토글
    d = json.loads(r["body"])
    assert r["statusCode"] == 200 and d["kind"] == "allergy_warning"
    assert d["evidence"][0]["group"] == "milk" and d["trace"] and d["latency_ms"] >= 0
    assert "최종 확인은 항상 사람" in d["message"]


def test_default_profile_is_all_groups_and_unknown_group_rejected():
    h = make_handler(FakeAnalyzer(), FakeReader([ReadLine("WHEAT FLOUR", 0.9)]))
    assert json.loads(h(post(png_b64()), None)["body"])["kind"] == "allergy_warning"
    r = h(post(png_b64(), {"profile": "milk,plutonium"}), None)
    assert r["statusCode"] == 400


def test_garbage_body_is_400_and_oversize_is_413():
    h = make_handler(FakeAnalyzer(), FakeReader([]))
    assert h(post(base64.b64encode(b"not an image").decode()), None)["statusCode"] == 400
    assert h(post(base64.b64encode(b"x" * (7 * 1024 * 1024)).decode()), None)["statusCode"] == 413
    assert h(post(None), None)["statusCode"] == 400


def test_logs_have_metrics_but_no_image_bytes(capsys):
    b = png_b64()
    make_handler(FakeAnalyzer(), FakeReader([ReadLine("nothing", 0.9)]))(post(b), None)
    out = capsys.readouterr().out
    assert "latency_ms" in out and "kind" in out and b[:40] not in out


def test_reader_failure_degrades_to_human_check_not_500():
    class Boom:
        def read(self, image): raise RuntimeError("x")
    d = json.loads(make_handler(FakeAnalyzer(), Boom())(post(png_b64()), None)["body"])
    assert d["kind"] == "human_check"
