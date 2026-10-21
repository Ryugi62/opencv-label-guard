"""값 객체 — 프레임워크·I/O를 모른다."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Tuple


class VerdictKind(str, Enum):
    ALLERGY_WARNING = "allergy_warning"   # 알레르기 경고
    RETAKE_REQUEST = "retake_request"     # 재촬영 요청
    HUMAN_CHECK = "human_check"           # 사람 확인
    NO_WARNING = "no_warning"             # 경고 없음(최종 확인은 사람)


@dataclass(frozen=True)
class Thresholds:
    min_blur: float = 15.0       # 라플라시안 분산(폭 1000 정규화) 미만 = 흐림 — docs/eval 보정표 근거
    max_glare: float = 0.40      # 포화 반사 덩어리 비율 초과 = 반사
    min_regions: int = 0         # 글자 영역 최소 개수(0=끔: 고전 검출기가 판독기보다 적게 찾음)
    min_confidence: float = 0.6  # 판독 줄 평균 신뢰도 미만 = 사람 확인


@dataclass(frozen=True)
class PhotoQuality:
    blur_score: float
    glare_ratio: float
    text_regions: int
    skew_deg: float = 0.0


@dataclass(frozen=True)
class ReadLine:
    text: str
    confidence: float


@dataclass(frozen=True)
class AllergyProfile:
    groups: FrozenSet[str]

    def __init__(self, groups):
        object.__setattr__(self, "groups", frozenset(groups))


@dataclass(frozen=True)
class Evidence:
    group: str
    term: str
    line_index: int
    line_text: str


@dataclass(frozen=True)
class Step:
    stage: str
    opencv: bool   # 이 단계가 OpenCV 5 연산인가(AC-11)
    detail: str


@dataclass(frozen=True)
class Verdict:
    kind: VerdictKind
    message: str
    reasons: Tuple[str, ...] = ()
    evidence: Tuple[Evidence, ...] = ()
    steps: Tuple[Step, ...] = ()

    @property
    def trace(self) -> Tuple[str, ...]:
        return tuple(f"[{'OpenCV5' if s.opencv else '비OpenCV'}] {s.stage}: {s.detail}" for s in self.steps)

    @property
    def decided_by_opencv(self) -> bool:
        """판정을 가른 단계가 OpenCV 5 연산뿐인가: 재촬영 요청만 해당."""
        return self.kind is VerdictKind.RETAKE_REQUEST
