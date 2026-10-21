"""OpenCV 5 어댑터 — 합성 이미지 시험."""
import cv2, numpy as np
from label_guard.adapters.opencv5_quality import (OpenCV5QualityAnalyzer, blur_score, glare_ratio, text_boxes, rectify)

def label(w=900, h=500):
    img = np.full((h, w, 3), 235, np.uint8)
    for i in range(8):
        cv2.putText(img, "INGREDIENTS: MILK SUGAR WHEAT FLOUR SALT", (30, 50 + i * 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return img

def test_opencv_version_is_5():
    assert cv2.__version__.startswith("5.")

def test_blur_lowers_score():
    s = label()
    assert blur_score(cv2.GaussianBlur(s, (0, 0), 6)) < blur_score(s) / 5

def test_glare_blob_raises_ratio_but_white_paper_does_not():
    s = label()
    g = s.copy(); cv2.circle(g, (450, 250), 200, (255, 255, 255), -1)
    assert glare_ratio(g) > 0.12 > glare_ratio(s)

def test_text_regions_detected_on_synthetic_label():
    assert len(text_boxes(label())) >= 6

def test_perspective_rectification_does_not_reduce_regions():  # AC-3
    s = label(); canvas = np.full((800, 1200, 3), 60, np.uint8)
    src = np.float32([[0, 0], [899, 0], [899, 499], [0, 499]])
    dst = np.float32([[200, 150], [1000, 100], [1050, 650], [150, 600]])
    warped = cv2.warpPerspective(s, cv2.getPerspectiveTransform(src, dst), (1200, 800), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    q, used, info = OpenCV5QualityAnalyzer().analyze(warped)
    assert info["regions_after"] >= info["regions_before"] and info["rectified"] == 1
    assert q.text_regions >= 4
