from typing import List, Optional, Sequence
from .allergens import match_allergens
from .model import (AllergyProfile, PhotoQuality, ReadLine, Step, Thresholds, Verdict, VerdictKind)

FINAL = "최종 확인은 항상 사람이 라벨 원문으로 해 주세요."
FINAL_EN = "The final check is always done by a person, on the original label."
_REASON_EN = {"흐림": "blur", "반사": "glare", "글자영역없음": "no text regions"}
# 재촬영 안내: 이유별로 「어떻게 찍을지」(저시력 사용자가 혼자 고칠 수 있게). (한국어, 영어)
RETAKE_TIPS = {
    "흐림": ("폰을 두 손으로 고정하고 라벨에서 10~15cm 떨어져 초점이 맞을 때까지 화면을 눌러 주세요.",
            "Hold the phone steady with both hands, about 10-15 cm from the label, and tap the screen until it is in focus."),
    "반사": ("조명 쪽에서 라벨을 비스듬히 기울여 번쩍이는 빛이 글자 위에 오지 않게 해 주세요.",
            "Tilt the label slightly away from the light so the shiny glare is not on the text."),
    "글자영역없음": ("라벨 전체가 화면에 들어오게 정면에서 똑바로 찍어 주세요.",
              "Shoot straight on so the whole label is in the frame."),
}


def retake_guidance(reasons):
    return tuple((r, *RETAKE_TIPS[r]) for r in reasons if r in RETAKE_TIPS)


def assess_quality(q: PhotoQuality, t: Thresholds) -> List[str]:
    reasons = []
    if q.blur_score < t.min_blur:
        reasons.append("흐림")
    if q.glare_ratio > t.max_glare:
        reasons.append("반사")
    if q.text_regions < t.min_regions:
        reasons.append("글자영역없음")
    return reasons


def quality_step(q: PhotoQuality, t: Thresholds) -> Step:
    return Step("quality", True,
                f"흐림 {q.blur_score:.1f}(하한 {t.min_blur:g}) · 반사 {q.glare_ratio*100:.1f}%(상한 {t.max_glare*100:g}%)"
                f" · 글자영역 {q.text_regions}개",
                f"blur {q.blur_score:.1f} (min {t.min_blur:g}) · glare {q.glare_ratio*100:.1f}% (max {t.max_glare*100:g}%)"
                f" · text regions {q.text_regions}")


def decide(q: PhotoQuality, lines: Optional[Sequence[ReadLine]], profile: AllergyProfile,
           t: Thresholds, extra_steps: Sequence[Step] = (), read_opencv: bool = False) -> Verdict:
    steps = list(extra_steps) + [quality_step(q, t)]
    reasons = assess_quality(q, t)
    if reasons:
        en = ", ".join(_REASON_EN.get(r, r) for r in reasons)
        steps.append(Step("decision", True, "품질 미달 -> 재촬영 요청: " + ", ".join(reasons),
                          "quality below threshold -> retake request: " + en))
        g = retake_guidance(reasons)
        return Verdict(VerdictKind.RETAKE_REQUEST,
                       "사진이 불충분합니다(" + ", ".join(reasons) + "). 성분표가 선명하게 나오도록 다시 찍어 주세요. " + " ".join(x[1] for x in g),
                       tuple(reasons), (), tuple(steps),
                       f"The photo is not good enough to read ({en}). Please retake it so the ingredient list is sharp and fully in frame. " + " ".join(x[2] for x in g))
    lines = list(lines or [])
    mean = sum(l.confidence for l in lines) / len(lines) if lines else 0.0
    steps.append(Step("read", read_opencv, f"판독 줄 {len(lines)}개 · 평균 신뢰도 {mean:.2f}",
                      f"{len(lines)} lines read · mean confidence {mean:.2f}" + (" (OpenCV 5 DNN: DB detector + CRNN recognizer)" if read_opencv else " (non-OpenCV reader)")))
    ev = match_allergens(lines, profile)
    steps.append(Step("match", False, f"프로필 {sorted(profile.groups)} -> 근거 {len(ev)}건",
                      f"profile {sorted(profile.groups)} -> {len(ev)} pieces of evidence"))
    if ev:
        what = ", ".join(f"{e.group}('{e.term}', {e.line_index + 1}번째 줄)" for e in ev)
        steps.append(Step("decision", False, "프로필 낱말 발견 -> 알레르기 경고", "profile term found -> allergy warning"))
        what_en = ", ".join(f"{e.group} ('{e.term}', line {e.line_index + 1})" for e in ev)
        return Verdict(VerdictKind.ALLERGY_WARNING, f"알레르기 경고: {what}. " + FINAL, (), tuple(ev), tuple(steps),
                       f"Allergy warning: {what_en}. " + FINAL_EN)
    if not lines or mean < t.min_confidence:
        steps.append(Step("decision", False, "판독 불확실 -> 사람 확인(없음을 단정하지 않음)",
                          "reading uncertain -> human check (never claims 'none')"))
        return Verdict(VerdictKind.HUMAN_CHECK,
                       "글자를 충분히 확실하게 읽지 못했습니다. 없다고 말할 수 없으니 라벨 원문을 직접 확인해 주세요.",
                       ("판독불확실",), (), tuple(steps),
                       "I could not read the text reliably enough. I cannot say the allergen is absent, so please check the original label yourself.")
    steps.append(Step("decision", False, "프로필 낱말 없음 -> 경고 없음", "no profile term found -> no warning"))
    return Verdict(VerdictKind.NO_WARNING,
                   "읽은 줄에서 지정한 항목은 발견되지 않았습니다. " + FINAL, (), (), tuple(steps),
                   "None of the selected items were found in the lines I read. " + FINAL_EN)
