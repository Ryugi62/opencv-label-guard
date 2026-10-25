"""글자 판독 포트 구현 2: OpenCV 5 DNN만으로 도는 리눅스 판독기(Lambda 이식용).

DB 글자 검출(cv2.dnn_TextDetectionModel_DB) + 공개 사전 학습 인식 모델(PP-OCRv3 en, ONNX)을 cv2.dnn으로 실행한다.
학습 0, onnxruntime 불필요, macOS 의존 0. 인식 모델은 띄어쓰기를 못 내므로 글자 줄 잘라낸 그림의 열 잉크 분포로
단어를 나눠 단어별로 인식한다(단어 경계가 있어야 `soy`·`wheat` 같은 단어 시작 규칙이 산다)."""
from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from ..domain.model import ReadLine

REC_H = 48


def ctc_greedy(logits: np.ndarray, chars: Sequence[str]) -> Tuple[str, float]:
    """(T, C) 확률/로짓 -> (문자열, 평균 확률). 0번 클래스는 공백(blank), 연속 같은 클래스는 합친다."""
    ids = logits.argmax(1)
    out, probs, prev = [], [], -1
    for t, i in enumerate(ids):
        if i != 0 and i != prev and i < len(chars):
            out.append(chars[i]); probs.append(float(logits[t, i]))
        prev = i
    if not out:
        return "", 0.0
    return "".join(out), float(min(1.0, max(0.0, np.mean(probs))))


def word_spans(ink_cols: np.ndarray, height: int, gap_ratio: float = 0.35) -> List[Tuple[int, int]]:
    """열별 잉크 유무 -> 단어 구간 [(시작, 끝)). 높이 * gap_ratio 이상 빈 틈만 단어 경계."""
    xs = np.flatnonzero(ink_cols)
    if xs.size == 0:
        return []
    min_gap = height * gap_ratio
    spans, start, prev = [], int(xs[0]), int(xs[0])
    for x in xs[1:]:
        if x - prev - 1 >= min_gap:
            spans.append((start, prev + 1)); start = int(x)
        prev = int(x)
    spans.append((start, prev + 1))
    return spans


def pick_best(primary: Tuple[str, float], alt: Optional[Tuple[str, float]], sure: float = 0.8) -> Tuple[str, float]:
    """주 인식 결과가 확실(>= sure)하면 유지, 아니면 보조 인식(다른 문자 체계) 결과가 더 확실할 때만 교체."""
    if alt is None or (primary[0] and primary[1] >= sure):
        return primary
    return alt if alt[1] > primary[1] else primary


def load_dict(path: str) -> List[str]:
    """PP-OCR 사전 파일(줄당 1글자) -> [blank] + 글자들 + [공백]."""
    with open(path, encoding="utf-8") as f:
        return ["blank"] + f.read().split("\n") + [" "]


class DnnTextReader:
    runs_on_opencv = True   # cv2.dnn(DB 검출 + CRNN 인식)만 쓴다 — 트레이스에서 OpenCV 5 단계로 표시
    def __init__(self, det_path: str, rec_path: str, dict_path: str, det_size: int = 960,
                 max_lines: int = 200, binary_thr: float = 0.3, unclip: float = 1.6,
                 alt_rec_path: Optional[str] = None, alt_dict_path: Optional[str] = None, alt_h: int = 32, gap_ratio: float = 0.25):
        self.gap_ratio = gap_ratio
        self.det = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(det_path))
        self.det.setBinaryThreshold(binary_thr); self.det.setPolygonThreshold(0.5)
        self.det.setUnclipRatio(unclip); self.det.setMaxCandidates(max_lines)
        self.det_size = det_size
        self.det.setInputSize((det_size, det_size))
        self.det.setInputMean((123.675, 116.28, 103.53))
        self.det.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
        self.rec = cv2.dnn.readNet(rec_path)
        self.chars = load_dict(dict_path)
        self.alt = (cv2.dnn.readNet(alt_rec_path), load_dict(alt_dict_path), alt_h) if alt_rec_path and alt_dict_path else None

    # -- 검출 --
    def _boxes(self, image) -> List[np.ndarray]:
        L = self.det_size
        h, w = image.shape[:2]
        s = L / max(h, w)
        small = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        canvas = np.full((L, L, 3), 255, np.uint8)
        canvas[:small.shape[0], :small.shape[1]] = small
        boxes, _ = self.det.detect(canvas)
        out = [np.array(b, np.float32) / s for b in boxes]  # 원본 좌표
        out.sort(key=lambda b: (round(float(b[:, 1].mean()) / max(1.0, 0.5 * np.linalg.norm(b[0] - b[1]))), float(b[:, 0].mean())))
        return out

    # -- 인식 --
    @staticmethod
    def _run(net, chars, rec_h, crop) -> Tuple[str, float]:
        h, w = crop.shape[:2]
        tw = max(32, int(round(rec_h * w / max(h, 1))))
        img = cv2.resize(crop, (tw, rec_h), interpolation=cv2.INTER_AREA if h > rec_h else cv2.INTER_CUBIC)
        net.setInput(cv2.dnn.blobFromImage(img, 1 / 127.5, (tw, rec_h), (127.5, 127.5, 127.5), swapRB=True))
        o = net.forward()
        return ctc_greedy(o.reshape(o.shape[-2], o.shape[-1]), chars)

    def _recognize(self, crop) -> Tuple[str, float]:
        p = self._run(self.rec, self.chars, REC_H, crop)
        if self.alt is None or (p[0] and p[1] >= 0.8):
            return p
        return pick_best(p, self._run(self.alt[0], self.alt[1], self.alt[2], crop))

    def _line(self, image, box) -> Tuple[str, float]:
        w = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
        h = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
        if w < 4 or h < 4:
            return "", 0.0
        dst = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], np.float32)
        crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(box, dst), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (bw > 0).mean() > 0.5:   # 잉크 = 소수 쪽 화소
            bw = 255 - bw
        spans = word_spans((bw > 0).sum(0) >= max(1, int(0.04 * h)), h, self.gap_ratio)
        texts, confs = [], []
        for a, b in spans or [(0, w)]:
            pad = max(2, h // 8)
            word = crop[:, max(0, a - pad):min(w, b + pad)]
            t, c = self._recognize(word)
            if t:
                texts.append(t); confs.append((c, len(t)))
        if not texts:
            return "", 0.0
        n = sum(k for _, k in confs)
        return " ".join(texts), sum(c * k for c, k in confs) / n

    def read(self, image: Any) -> List[ReadLine]:
        lines = []
        for box in self._boxes(image):
            t, c = self._line(image, box)
            if t.strip():
                lines.append(ReadLine(text=t, confidence=c))
        return lines
