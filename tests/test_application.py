"""UC-1 오케스트레이션 — 가짜 어댑터."""
from label_guard.domain.model import PhotoQuality, ReadLine, AllergyProfile, VerdictKind, Thresholds
from label_guard.application.judge import judge_photo

class FakeAnalyzer:
    def __init__(self, quality, rectified_regions=None): self.quality=quality; self.calls=0
    def analyze(self, image): self.calls+=1; return self.quality, image, {"regions_before": 4, "regions_after": 9}

class FakeReader:
    def __init__(self, lines): self.lines=lines; self.calls=0
    def read(self, image): self.calls+=1; return self.lines

OK = PhotoQuality(blur_score=500, glare_ratio=0.0, text_regions=10, skew_deg=3)

def test_retake_skips_reader():  # AC-1: 판독 줄을 만들지 않음
    bad = PhotoQuality(blur_score=1, glare_ratio=0, text_regions=10, skew_deg=0)
    r = FakeReader([ReadLine("milk", 0.9)])
    v = judge_photo(object(), AllergyProfile({"milk"}), FakeAnalyzer(bad), r, Thresholds())
    assert v.kind is VerdictKind.RETAKE_REQUEST and r.calls == 0

def test_warning_path_calls_reader_once():
    r = FakeReader([ReadLine("contains milk", 0.9)])
    v = judge_photo(object(), AllergyProfile({"milk"}), FakeAnalyzer(OK), r, Thresholds())
    assert v.kind is VerdictKind.ALLERGY_WARNING and r.calls == 1

def test_rectification_counts_recorded_in_trace():  # AC-3
    v = judge_photo(object(), AllergyProfile({"milk"}), FakeAnalyzer(OK), FakeReader([ReadLine("x",0.9)]), Thresholds())
    assert any("보정 전 4" in s and "보정 후 9" in s for s in v.trace)

def test_reader_exception_fails_safe_to_human_check():
    class Boom:
        def read(self, image): raise RuntimeError("x")
    v = judge_photo(object(), AllergyProfile({"milk"}), FakeAnalyzer(OK), Boom(), Thresholds())
    assert v.kind is VerdictKind.HUMAN_CHECK

def test_opencv_only_ratio_from_steps():  # AC-11
    from label_guard.application.judge import opencv_only_ratio
    retake = judge_photo(object(), AllergyProfile({"milk"}), FakeAnalyzer(PhotoQuality(1,0,10,0)), FakeReader([]), Thresholds())
    warn = judge_photo(object(), AllergyProfile({"milk"}), FakeAnalyzer(OK), FakeReader([ReadLine("milk",0.9)]), Thresholds())
    assert retake.decided_by_opencv is True and warn.decided_by_opencv is False
    assert opencv_only_ratio([retake, warn]) == 0.5
