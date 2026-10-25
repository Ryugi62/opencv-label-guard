"""합성 열화(재촬영 전 사진 모사) — 결정적(난수 0). 실제 폰 촬영이 아니라 합성이며 보고서에 그렇게 표기한다."""
import cv2
import numpy as np

KINDS = ("blur", "glare", "tilt", "lowres")


def degrade(img, kind: str, level: int = 2):
    if kind not in KINDS:
        raise ValueError(f"알 수 없는 열화: {kind}")
    h, w = img.shape[:2]
    if kind == "blur":
        s = 1.5 * level * (w / 1000.0)
        return cv2.GaussianBlur(img, (0, 0), sigmaX=s)
    if kind == "lowres":
        f = 1.0 / (2 + 2 * level)
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "glare":
        out = img.copy()
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (int(w * 0.6), int(h * 0.45)), (int(w * 0.10 * level), int(h * 0.12 * level)), -20, 0, 360, 255, -1)
        core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        soft = cv2.GaussianBlur(mask, (0, 0), 0.03 * w).astype(np.float32) / 255.0
        out = (out * (1 - soft[..., None]) + 255 * soft[..., None]).astype(np.uint8)
        out[core > 0] = 255
        return out
    # tilt: 위쪽 가장자리를 안쪽으로 모아 원근 기울임 + 배경 회색
    k = 0.06 * level
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[w * k, h * k * 0.5], [w * (1 - k), 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(128, 128, 128))
