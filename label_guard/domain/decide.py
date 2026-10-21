from typing import List, Optional, Sequence
from .allergens import match_allergens
from .model import (AllergyProfile, PhotoQuality, ReadLine, Step, Thresholds, Verdict, VerdictKind)

FINAL = "최종 확인은 항상 사람이 라벨 원문으로 해 주세요."


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
                f" · 글자영역 {q.text_regions}개")


def decide(q: PhotoQuality, lines: Optional[Sequence[ReadLine]], profile: AllergyProfile,
           t: Thresholds, extra_steps: Sequence[Step] = ()) -> Verdict:
    steps = list(extra_steps) + [quality_step(q, t)]
    reasons = assess_quality(q, t)
    if reasons:
        steps.append(Step("decision", True, "품질 미달 -> 재촬영 요청: " + ", ".join(reasons)))
        return Verdict(VerdictKind.RETAKE_REQUEST,
                       "사진이 불충분합니다(" + ", ".join(reasons) + "). 성분표가 선명하게 나오도록 다시 찍어 주세요.",
                       tuple(reasons), (), tuple(steps))
    lines = list(lines or [])
    mean = sum(l.confidence for l in lines) / len(lines) if lines else 0.0
    steps.append(Step("read", False, f"판독 줄 {len(lines)}개 · 평균 신뢰도 {mean:.2f}"))
    ev = match_allergens(lines, profile)
    steps.append(Step("match", False, f"프로필 {sorted(profile.groups)} -> 근거 {len(ev)}건"))
    if ev:
        what = ", ".join(f"{e.group}('{e.term}', {e.line_index + 1}번째 줄)" for e in ev)
        steps.append(Step("decision", False, "프로필 낱말 발견 -> 알레르기 경고"))
        return Verdict(VerdictKind.ALLERGY_WARNING, f"알레르기 경고: {what}. " + FINAL, (), tuple(ev), tuple(steps))
    if not lines or mean < t.min_confidence:
        steps.append(Step("decision", False, "판독 불확실 -> 사람 확인(없음을 단정하지 않음)"))
        return Verdict(VerdictKind.HUMAN_CHECK,
                       "글자를 충분히 확실하게 읽지 못했습니다. 없다고 말할 수 없으니 라벨 원문을 직접 확인해 주세요.",
                       ("판독불확실",), (), tuple(steps))
    steps.append(Step("decision", False, "프로필 낱말 없음 -> 경고 없음"))
    return Verdict(VerdictKind.NO_WARNING,
                   "읽은 줄에서 지정한 항목은 발견되지 않았습니다. " + FINAL, (), (), tuple(steps))
