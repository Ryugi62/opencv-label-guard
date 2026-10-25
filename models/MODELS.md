# 사전 학습 가중치 (학습 0, 레포에는 안 넣음 — scripts/download_models.py)
| 파일 | 출처 | 용도 | sha256 |
|---|---|---|---|
| text_detection_en_ppocrv3_2023may.onnx | huggingface.co/opencv/text_detection_ppocr (OpenCV Zoo, Apache-2.0) | DB 글자 영역 검출 | 03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587 |
| en_PP-OCRv3_rec_infer.onnx | huggingface.co/SWHL/RapidOCR PP-OCRv3 (PaddleOCR 변환, Apache-2.0) | 영문·라틴 글자 인식 | ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea |
| korean_mobile_v2.0_rec_infer.onnx | huggingface.co/SWHL/RapidOCR PP-OCRv1 (PaddleOCR 변환, Apache-2.0) | 한글 보조 인식(주 판독이 불확실한 단어만) | b6558500138b43b46a4941957fb8c918546dae5fb0e71718536f1883acc80faf |
| en_ppocrv3_dict.txt · ko_v2_dict.txt | 위 ONNX 메타데이터 `character` 항목에서 추출 | 글자 사전 | 레포 포함 |
