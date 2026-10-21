from dataclasses import replace
from typing import Any, Sequence
from ..domain.decide import assess_quality, decide
from ..domain.model import AllergyProfile, Step, Thresholds, Verdict, VerdictKind


def judge_photo(image: Any, profile: AllergyProfile, analyzer, reader, t: Thresholds) -> Verdict:
    """UC-1: 품질 먼저 -> 미달이면 판독 없이 재촬영 요청."""
    quality, rectified, info = analyzer.analyze(image)
    rect = Step("rectify", True,
                f"원근 보정 글자 영역 보정 전 {info.get('regions_before', 0)} -> 보정 후 {info.get('regions_after', 0)}")
    if assess_quality(quality, t):
        return decide(quality, None, profile, t, [rect])
    try:
        lines = reader.read(rectified)
    except Exception as e:  # fail-safe: 모르면 없음이라 하지 않는다
        v = decide(quality, [], profile, t, [rect])
        return replace(v, steps=v.steps + (Step("read", False, f"판독기 오류 {type(e).__name__}"),))
    return decide(quality, lines, profile, t, [rect])


def opencv_only_ratio(verdicts: Sequence[Verdict]) -> float:
    return sum(v.decided_by_opencv for v in verdicts) / len(verdicts) if verdicts else 0.0
