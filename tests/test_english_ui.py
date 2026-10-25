"""영어 기본 화면(심사위원 영어권) + 한국어 토글 + 판독 단계 태그 실제화."""
import base64, json, re
import cv2, numpy as np
from label_guard.domain.model import PhotoQuality, ReadLine, AllergyProfile, Thresholds, VerdictKind
from label_guard.application.judge import judge_photo
from label_guard.infrastructure.lambda_handler import make_handler

HANGUL = re.compile("[가-힣]")


class A:
    def __init__(self, q=None): self.q = q or PhotoQuality(500, 0.0, 10, 0.0)
    def analyze(self, image): return self.q, image, {"regions_before": 3, "regions_after": 4}


class R:
    runs_on_opencv = True
    def __init__(self, lines): self.lines = lines
    def read(self, image): return self.lines


def post(qs=None):
    ok, buf = cv2.imencode(".png", np.full((48, 64, 3), 200, np.uint8))
    return {"requestContext": {"http": {"method": "POST"}}, "body": base64.b64encode(buf.tobytes()).decode(),
            "isBase64Encoded": True, "queryStringParameters": qs or {}}


def get(qs=None):
    return {"requestContext": {"http": {"method": "GET"}}, "queryStringParameters": qs or {}}


def test_default_page_is_english_with_korean_toggle():
    b = make_handler(A(), R([]))(get(), None)["body"]
    assert '<html lang="en"' in b and "Label Reading Guard" in b and "lang=ko" in b
    visible = re.sub(r"<script.*?</script>", "", b, flags=re.S)
    assert len(HANGUL.findall(visible)) <= 4   # 토글 낱말 「한국어」만


def test_korean_page_on_toggle():
    b = make_handler(A(), R([]))(get({"lang": "ko"}), None)["body"]
    assert '<html lang="ko"' in b and "라벨 판독 가드" in b


def test_default_post_is_english_no_hangul_in_message_or_trace():
    d = json.loads(make_handler(A(), R([ReadLine("CONTAINS MILK", 0.9)]))(post({"profile": "milk"}), None)["body"])
    assert d["kind"] == "allergy_warning" and "final check" in d["message"].lower()
    assert not HANGUL.search(d["message"] + "".join(d["trace"]))


def test_retake_and_human_check_english():
    d = json.loads(make_handler(A(PhotoQuality(1, 0.0, 10, 0.0)), R([]))(post(), None)["body"])
    assert d["kind"] == "retake_request" and "retake" in d["message"].lower() and not HANGUL.search(d["message"] + "".join(d["trace"]))
    d = json.loads(make_handler(A(), R([]))(post(), None)["body"])
    assert d["kind"] == "human_check" and not HANGUL.search(d["message"])


def test_korean_post_on_toggle():
    d = json.loads(make_handler(A(), R([ReadLine("MILK", 0.9)]))(post({"lang": "ko", "profile": "milk"}), None)["body"])
    assert "최종 확인은 항상 사람" in d["message"]


def test_read_step_tagged_opencv_when_reader_runs_on_opencv_dnn():
    v = judge_photo(object(), AllergyProfile({"milk"}), A(), R([ReadLine("x", 0.9)]), Thresholds())
    assert any(s.stage == "read" and s.opencv for s in v.steps)
    assert any("[OpenCV5-DNN] read" in t for t in v.trace)
    assert any("read" in t and "OpenCV 5 DNN" in t for t in v.trace_en)
    class V:  # macOS Vision 판독기는 OpenCV 아님
        def read(self, image): return [ReadLine("x", 0.9)]
    v2 = judge_photo(object(), AllergyProfile({"milk"}), A(), V(), Thresholds())
    assert any(s.stage == "read" and not s.opencv for s in v2.steps)


def test_rule_steps_tagged_rules():
    v = judge_photo(object(), AllergyProfile({"milk"}), A(), R([ReadLine("x", 0.9)]), Thresholds())
    assert any(t.startswith("[rules] match") for t in v.trace_en) and not any("non-OpenCV] match" in t for t in v.trace_en)


def test_page_has_english_429_message():
    b = make_handler(A(), R([]))(get(), None)["body"]
    assert "429" in b and "busy" in b.lower()


def _retake(q, qs=None):
    return json.loads(make_handler(A(q), R([]))(post(qs), None)["body"])


def test_retake_guidance_per_reason_english():
    d = _retake(PhotoQuality(1, 0.0, 10, 0.0))
    assert "10-15 cm" in d["message"] and "steady" in d["message"].lower() and d["guidance"] and d["guidance"][0]["reason"] == "흐림"
    d = _retake(PhotoQuality(500, 0.9, 10, 0.0))
    assert "tilt" in d["message"].lower() and "light" in d["message"].lower()
    d = _retake(PhotoQuality(500, 0.0, 0, 0.0), None)  # min_regions 기본 0이라 사진 품질 통과 -> 이유 없음
    assert d["kind"] != "retake_request"


def test_retake_guidance_two_reasons_both_tips():
    d = _retake(PhotoQuality(1, 0.9, 10, 0.0))
    assert len(d["guidance"]) == 2 and "10-15 cm" in d["message"] and "tilt" in d["message"].lower()
    assert not HANGUL.search(d["message"])


def test_retake_guidance_korean_toggle():
    d = _retake(PhotoQuality(1, 0.9, 10, 0.0), {"lang": "ko"})
    assert "10~15cm" in d["message"] and "비스듬히" in d["message"]


def test_regions_reason_gets_framing_tip():
    h = make_handler(A(PhotoQuality(500, 0.0, 0, 0.0)), R([]), Thresholds(min_regions=1))
    d = json.loads(h(post(), None)["body"])
    assert d["kind"] == "retake_request" and "whole" in d["message"].lower() and "straight" in d["message"].lower()
