"""AC-1,2,4,5,6 도메인 시험 — OpenCV·파일 I/O 0."""
from label_guard.domain.model import (
    PhotoQuality, ReadLine, AllergyProfile, VerdictKind, Thresholds)
from label_guard.domain.allergens import match_allergens
from label_guard.domain.decide import decide, assess_quality

T = Thresholds()

def q(blur=500.0, glare=0.0, regions=20, skew=0.0):
    return PhotoQuality(blur_score=blur, glare_ratio=glare, text_regions=regions, skew_deg=skew)

def lines(*texts, conf=0.9):
    return [ReadLine(text=t, confidence=conf) for t in texts]

def test_blur_below_threshold_requests_retake_without_reading():  # AC-1
    v = decide(q(blur=T.min_blur - 1), None, AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.RETAKE_REQUEST
    assert "흐림" in v.reasons and v.evidence == ()

def test_glare_above_threshold_requests_retake_with_ratio():  # AC-2
    v = decide(q(glare=T.max_glare + 0.01), None, AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.RETAKE_REQUEST and "반사" in v.reasons
    assert any("반사" in s and "%" in s for s in v.trace)

def test_blur_and_glare_both_listed():
    v = decide(q(blur=1, glare=0.5), None, AllergyProfile({"milk"}), T)
    assert set(v.reasons) == {"흐림", "반사"}

def test_no_text_regions_requests_retake():
    assert "글자영역없음" in assess_quality(q(regions=0), Thresholds(min_regions=1))

def test_milk_synonym_gives_allergy_warning_with_evidence():  # AC-4
    v = decide(q(), lines("Zutaten: Magermilchpulver, Zucker", "Salz"), AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.ALLERGY_WARNING
    ev = v.evidence[0]
    assert ev.group == "milk" and ev.line_index == 0 and "milchpulver" in ev.term.lower() or "milch" in ev.term.lower()

def test_korean_synonym_matches():
    v = decide(q(), lines("원재료명: 유청분말, 설탕"), AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.ALLERGY_WARNING

def test_low_confidence_without_match_asks_human_not_none():  # AC-5
    v = decide(q(), lines("sugar", "salt", conf=0.3), AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.HUMAN_CHECK

def test_no_lines_asks_human():
    assert decide(q(), [], AllergyProfile({"milk"}), T).kind is VerdictKind.HUMAN_CHECK

def test_confident_no_match_is_no_warning_with_human_final_check():  # AC-6
    v = decide(q(), lines("sugar, salt, water"), AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.NO_WARNING
    assert "사람" in v.message and "라벨 원문" in v.message

def test_message_has_no_medical_judgement():
    for v in (decide(q(), lines("milk"), AllergyProfile({"milk"}), T),
              decide(q(), lines("sugar"), AllergyProfile({"milk"}), T)):
        for bad in ("섭취해도", "먹어도", "복용", "진단", "안전합니다"):
            assert bad not in v.message

def test_warning_even_when_confidence_low_fail_safe():
    v = decide(q(), lines("contains milk", conf=0.2), AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.ALLERGY_WARNING

def test_only_profile_groups_warn():
    v = decide(q(), lines("contains soy lecithin"), AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.NO_WARNING

def test_match_allergens_word_boundary_en_avoids_false_hits():
    # 'eggplant' must not trigger egg, 'peanut' triggers peanut group
    m = match_allergens(lines("eggplant, peanut oil"), AllergyProfile({"egg", "nut"}))
    assert [e.group for e in m] == ["nut"]

def test_negated_free_from_is_not_flagged_as_contains_is_ambiguous_to_human():
    # '무우유'/'milk free' 표기는 단정하지 않고 일단 경고(미탐 방지) — 근거 줄이 사람에게 보인다
    v = decide(q(), lines("milk free"), AllergyProfile({"milk"}), T)
    assert v.kind is VerdictKind.ALLERGY_WARNING

def test_trace_marks_opencv_stages_ac11():
    v = decide(q(), lines("sugar"), AllergyProfile({"milk"}), T)
    stages = [s.stage for s in v.steps]
    assert "quality" in stages and "match" in stages
    assert all(isinstance(s.opencv, bool) for s in v.steps)
