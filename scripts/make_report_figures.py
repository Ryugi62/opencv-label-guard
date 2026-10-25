"""보고서 그림: OpenCV 5 연산별 입력 -> 출력 예시 PNG 쌍과 수치(figures.json).

스펙(1문단): 공개 라벨 사진 1장(또는 시험용 합성 라벨)에 판정 경로가 실제로 쓰는 연산을 그대로 돌려,
연산마다 입력 그림과 출력 그림을 저장하고 그 연산이 낸 숫자(흐림 점수·반사 비율·글자 영역 수·기울기·DNN 줄 수)를
figures.json에 남긴다. 열화 입력(기울임·흐림·반사)은 degrade.py 합성이며 파일 이름과 JSON에 synthetic 표기를 남긴다.
시험: tests/test_report_figures.py (합성 라벨, 가중치 없으면 DNN 쌍은 건너뜀).

사용: python scripts/make_report_figures.py data/images/0014100098423.jpg docs/figures
"""
from __future__ import annotations
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from label_guard.adapters.degrade import degrade  # noqa: E402
from label_guard.adapters.opencv5_quality import (_norm, blur_score, glare_ratio, rectify, skew_deg,  # noqa: E402
                                                 text_boxes)

THUMB_W = 520


def _thumb(img):
    h, w = img.shape[:2]
    s = THUMB_W / w
    return cv2.resize(img, (THUMB_W, max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def _on_canvas(img, pad_ratio=0.18):
    """라벨 사진을 어두운 탁자 위에 둔 것처럼 여백을 둔다(원근 보정 예시용, 합성)."""
    h, w = img.shape[:2]
    p = int(max(h, w) * pad_ratio)
    canvas = np.full((h + 2 * p, w + 2 * p, 3), 70, np.uint8)
    canvas[p:p + h, p:p + w] = img
    return canvas


def op_rectify(img):
    tilted = degrade(_on_canvas(img), "tilt", 3)
    warped = rectify(tilted)
    before = len(text_boxes(tilted))
    out = warped if warped is not None else tilted
    after = len(text_boxes(out))
    return tilted, out, {"rectified": warped is not None, "regions_before": before, "regions_after": after,
                         "ops": ["cv2.Canny", "cv2.findContours", "cv2.approxPolyDP", "cv2.getPerspectiveTransform", "cv2.warpPerspective"],
                         "input": "synthetic tilt (degrade.py tilt 3)"}


def op_blur(img):
    blurred = degrade(img, "blur", 2)
    g = cv2.cvtColor(_norm(blurred), cv2.COLOR_BGR2GRAY)
    lap = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F), alpha=4)
    g2 = cv2.cvtColor(_norm(img), cv2.COLOR_BGR2GRAY)
    lap_sharp = cv2.convertScaleAbs(cv2.Laplacian(g2, cv2.CV_64F), alpha=4)
    out = np.vstack([cv2.cvtColor(lap_sharp, cv2.COLOR_GRAY2BGR), np.full((12, lap.shape[1], 3), 255, np.uint8),
                     cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)])
    return blurred, out, {"blur_score_original": round(blur_score(img), 1), "blur_score_blurred": round(blur_score(blurred), 1),
                          "threshold": 15, "ops": ["cv2.cvtColor", "cv2.Laplacian"], "input": "synthetic blur (degrade.py blur 2)",
                          "output": "|Laplacian| of original (top) and blurred input (bottom)"}


def op_glare(img):
    glared = degrade(img, "glare", 3)
    n = _norm(glared)
    mask = (n.min(axis=2) >= 250).astype(np.uint8) * 255
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    vis = cv2.addWeighted(n, 0.45, np.full_like(n, 255), 0.55, 0)
    vis[opened > 0] = (40, 40, 220)
    return glared, vis, {"glare_ratio_original": round(glare_ratio(img), 3), "glare_ratio_input": round(glare_ratio(glared), 3),
                         "threshold": 0.4, "ops": ["cv2.morphologyEx(MORPH_OPEN)", "cv2.getStructuringElement"],
                         "input": "synthetic glare (degrade.py glare 3)", "output": "saturated blobs that survive a 15px opening (red)"}


def op_text_regions(img):
    n = _norm(img)
    vis = n.copy()
    boxes = text_boxes(img)
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 160, 0), 2)
    return img, vis, {"text_regions": len(boxes),
                      "ops": ["cv2.morphologyEx(MORPH_GRADIENT)", "cv2.threshold(OTSU)", "cv2.morphologyEx(MORPH_CLOSE)", "cv2.findContours", "cv2.boundingRect"],
                      "input": "original public label photo"}


def op_skew(img):
    tilted = degrade(img, "tilt", 2)
    n = _norm(tilted)
    vis = n.copy()
    boxes = text_boxes(tilted)
    pts = np.array([[x + w // 2, y + h // 2] for x, y, w, h in boxes], dtype=np.float32)
    for x, y in pts:
        cv2.circle(vis, (int(x), int(y)), 5, (0, 120, 255), -1)
    if len(pts) >= 5:
        box = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.int32)
        cv2.polylines(vis, [box], True, (200, 0, 160), 3)
    return tilted, vis, {"skew_deg": round(skew_deg(tilted), 1), "skew_deg_original": round(skew_deg(img), 1),
                         "ops": ["cv2.minAreaRect", "cv2.boxPoints"], "input": "synthetic tilt (degrade.py tilt 2)",
                         "output": "text-line centres (orange) and their min-area rectangle (magenta)"}


def op_dnn(img, models_dir):
    from label_guard.adapters.dnn_reader import DnnTextReader
    m = models_dir
    r = DnnTextReader(os.path.join(m, "text_detection_en_ppocrv3_2023may.onnx"), os.path.join(m, "en_PP-OCRv3_rec_infer.onnx"),
                      os.path.join(m, "en_ppocrv3_dict.txt"))
    boxes = r._boxes(img)
    vis = img.copy()
    for b in boxes:
        cv2.polylines(vis, [b.astype(np.int32)], True, (220, 90, 0), 2)
    lines = r.read(img)
    return img, vis, {"dnn_boxes": len(boxes), "read_lines": len(lines),
                      "sample_lines": [{"text": l.text, "confidence": round(l.confidence, 2)} for l in lines[:4]],
                      "ops": ["cv2.dnn_TextDetectionModel_DB", "cv2.dnn.readNet", "cv2.dnn.blobFromImage", "cv2.warpPerspective"],
                      "input": "original public label photo", "output": "DB text polygons; PP-OCRv3 recognition per word"}


def make(image_path, out_dir, models_dir=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {image_path}")
    os.makedirs(out_dir, exist_ok=True)
    ops = [("op1_rectify", op_rectify), ("op2_blur", op_blur), ("op3_glare", op_glare),
           ("op4_text_regions", op_text_regions), ("op5_skew", op_skew)]
    if models_dir and os.path.exists(os.path.join(models_dir, "text_detection_en_ppocrv3_2023may.onnx")):
        ops.append(("op6_dnn_text", lambda im: op_dnn(im, models_dir)))
    meta = {"source": os.path.basename(image_path), "opencv": cv2.__version__, "ops": {}}
    for name, fn in ops:
        a, b, info = fn(img)
        cv2.imwrite(os.path.join(out_dir, f"{name}_in.png"), _thumb(a))
        cv2.imwrite(os.path.join(out_dir, f"{name}_out.png"), _thumb(b))
        meta["ops"][name] = info
    with open(os.path.join(out_dir, "figures.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "images", "0014100098423.jpg")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "docs", "figures")
    print(json.dumps(make(src, out, os.path.join(root, "models")), ensure_ascii=False, indent=2))
