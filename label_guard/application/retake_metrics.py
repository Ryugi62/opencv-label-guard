"""재촬영 쌍 지표(UC-3) — 순수 계산. 열화본 판정 vs 재촬영(원본) 판정."""
from dataclasses import dataclass
from typing import Dict, Sequence

ABSTAIN = ("retake_request", "human_check")
ANSWER = ("allergy_warning", "no_warning")


@dataclass(frozen=True)
class PairRow:
    degraded: str        # 열화 사진 판정 종류
    reshot: str          # 재촬영(원본) 판정 종류
    truth_pos: bool      # 정답: 프로필 항목이 실제로 있음
    forced_correct: bool # 기권 없이 억지로 답했다면 맞았을 것인가(열화 사진 기준)


def _acc(kind: str, pos: bool) -> bool:
    return (kind == "allergy_warning") == pos


def summarize_pairs(rows: Sequence[PairRow]) -> Dict[str, float]:
    n = len(rows)
    if not n:
        return {"n": 0}
    ab_d = [r for r in rows if r.degraded in ABSTAIN]
    ans_d = [r for r in rows if r.degraded in ANSWER]
    nan = float("nan")
    return {
        "n": n,
        "flips": sum(r.degraded != r.reshot for r in rows),
        "flip_rate": sum(r.degraded != r.reshot for r in rows) / n,
        "abstain_degraded": len(ab_d),
        "abstain_reshot": sum(r.reshot in ABSTAIN for r in rows),
        "abstain_rate_degraded": len(ab_d) / n,
        "abstain_rate_reshot": sum(r.reshot in ABSTAIN for r in rows) / n,
        "traces_abstain_to_answer": sum(r.degraded in ABSTAIN and r.reshot in ANSWER for r in rows),
        "abstain_useful_rate": sum(not r.forced_correct for r in ab_d) / len(ab_d) if ab_d else nan,
        "answered_accuracy_degraded": sum(_acc(r.degraded, r.truth_pos) for r in ans_d) / len(ans_d) if ans_d else nan,
        "answered_n_degraded": len(ans_d),
        "answered_accuracy_reshot": (lambda a: sum(_acc(r.reshot, r.truth_pos) for r in a) / len(a) if a else nan)(
            [r for r in rows if r.reshot in ANSWER]),
        "silent_miss_degraded": sum(r.degraded == "no_warning" and r.truth_pos for r in rows),
        "silent_miss_reshot": sum(r.reshot == "no_warning" and r.truth_pos for r in rows),
    }
