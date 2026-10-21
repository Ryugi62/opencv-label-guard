from typing import Any, Dict, List, Protocol, Tuple
from ..domain.model import PhotoQuality, ReadLine


class QualityAnalyzer(Protocol):
    def analyze(self, image: Any) -> Tuple[PhotoQuality, Any, Dict[str, int]]:
        """-> (사진 품질, 보정된 이미지, {'regions_before':n,'regions_after':m})"""


class TextReader(Protocol):
    def read(self, image: Any) -> List[ReadLine]: ...
