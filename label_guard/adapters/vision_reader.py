"""글자 판독 포트 구현 1: macOS Vision(온디바이스). 컴파일된 swift 바이너리를 부른다.
Lambda 이식용 대체 판독기는 같은 TextReader 포트 뒤에 둔다(다음 단계)."""
import json, os, subprocess, tempfile
from typing import Any, List
import cv2
from ..domain.model import ReadLine

DEFAULT_BIN = os.path.join(os.path.dirname(__file__), "..", "..", ".build", "ocr_vision")


class VisionReader:
    def __init__(self, binary: str = DEFAULT_BIN, timeout: int = 60):
        self.binary, self.timeout = os.path.abspath(binary), timeout

    def read(self, image: Any) -> List[ReadLine]:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(path, image)
            out = subprocess.run([self.binary, path], capture_output=True, text=True, timeout=self.timeout)
            if out.returncode != 0 or not out.stdout.strip():
                raise RuntimeError("vision failed")
            d = json.loads(out.stdout.strip().splitlines()[0])
            if "error" in d:
                raise RuntimeError(d["error"])
            return [ReadLine(text=l["text"], confidence=float(l.get("conf", 0.0))) for l in d["lines"]]
        finally:
            os.unlink(path)  # 이미지 비저장
