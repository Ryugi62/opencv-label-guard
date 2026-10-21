"""OpenCV 5 어댑터: 원근 보정 · 반사 · 흐림 · 글자 영역 검출. 사전 학습 모델 0, 고전 영상처리만."""
from __future__ import annotations
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
from ..domain.model import PhotoQuality

NORM_W = 1000


def load_image(path: str, max_side: int = 2400):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없음")
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def _norm(img):
    h, w = img.shape[:2]
    s = NORM_W / w
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def blur_score(img) -> float:
    """라플라시안 분산(폭 1000 정규화). 글자 영역이 있는 부분만이 아니라 전체 평균이므로 보수적."""
    g = cv2.cvtColor(_norm(img), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def glare_ratio(img) -> float:
    """3채널 모두 250 이상(포화)이고 15px 열림 연산에도 남는 큰 덩어리 비율."""
    n = _norm(img)
    mask = (n.min(axis=2) >= 250).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return float(mask.mean() / 255.0)


def text_boxes(img) -> List[Tuple[int, int, int, int]]:
    """모폴로지 기울기 + 가로 닫기 + 연결 성분으로 글자 줄 후보 상자."""
    g = cv2.cvtColor(_norm(img), cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 30 and 6 <= h <= 70 and w / h >= 2.0:
            out.append((x, y, w, h))
    return out


def skew_deg(img) -> float:
    pts = np.array([[x + w // 2, y + h // 2] for x, y, w, h in text_boxes(img)], dtype=np.float32)
    if len(pts) < 5:
        return 0.0
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    a = ang if rw >= rh else ang - 90
    a = (a + 90) % 180 - 90
    return float(a)


def _order(pts):
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def rectify(img):
    """가장 큰 사각형 윤곽(이미지 면적 25~95%)을 정면으로 편다. 못 찾으면 None."""
    h, w = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.dilate(cv2.Canny(g, 50, 150), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        a = cv2.contourArea(c) / (h * w)
        if a < 0.25 or a > 0.95:
            continue
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            best = _order(ap.reshape(4, 2).astype(np.float32)); break
    if best is None:
        return None
    tl, tr, br, bl = best
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 200 or H < 100:
        return None
    M = cv2.getPerspectiveTransform(best, np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32))
    out = cv2.warpPerspective(img, M, (W, H))
    m = max(2, int(0.02 * min(W, H)))  # 가장자리 배경 띠 제거
    return out[m:H - m, m:W - m]


class OpenCV5QualityAnalyzer:
    OPS = ("cv2.Laplacian", "cv2.morphologyEx", "cv2.Canny", "cv2.findContours", "cv2.getPerspectiveTransform",
           "cv2.warpPerspective", "cv2.minAreaRect", "cv2.threshold")

    def analyze(self, image: Any) -> Tuple[PhotoQuality, Any, Dict[str, int]]:
        before = len(text_boxes(image))
        warped = rectify(image)
        used = image
        after = before
        if warped is not None:
            n = len(text_boxes(warped))
            if n >= before:   # AC-3: 보정이 글자 영역을 줄이면 원본 유지
                used, after = warped, n
        q = PhotoQuality(blur_score=blur_score(used), glare_ratio=glare_ratio(used),
                         text_regions=after, skew_deg=skew_deg(used))
        return q, used, {"regions_before": before, "regions_after": after, "rectified": int(used is not image)}
