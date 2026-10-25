"""개인정보 점검(1회성 스크래치): 평가 이미지 104건에 OpenCV 5 YuNet 얼굴 검출을 돌려 얼굴 수를 센다.
사용: .venv/bin/python scripts/face_check.py <yunet.onnx>  (모델: opencv_zoo face_detection_yunet_2023mar.onnx)"""
import cv2, glob, sys
d = cv2.FaceDetectorYN.create(sys.argv[1], "", (320, 320), 0.8)
hits, n = [], 0
for f in sorted(glob.glob("data/images/*.jpg")):
    im = cv2.imread(f); s = 640 / max(im.shape[:2])
    im = cv2.resize(im, None, fx=s, fy=s) if s < 1 else im
    d.setInputSize((im.shape[1], im.shape[0])); _, r = d.detect(im); n += 1
    if r is not None and len(r): hits.append((f, len(r)))
print(f"검사 {n}건 · 얼굴 검출 이미지 {len(hits)}건 {hits}")
