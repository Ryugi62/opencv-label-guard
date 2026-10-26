[English README](../README.md)


# opencv-label-guard

**라이브 데모(비로그인, 사진 1장 올리면 판정): https://hqadhnbyzbv5dyok3zdnnfpcae0tcozx.lambda-url.ap-northeast-2.on.aws/**

폰으로 찍은 식품 성분표 사진을 OpenCV 5가 읽을 수 있는 상태인지 판정(원근 보정·반사·흐림·글자 영역)하고, 결과에 따라 알레르기 경고 / 재촬영 요청 / 사람 확인으로 행동을 가르는 루프. 의료 판단은 하지 않으며 최종 확인은 항상 사람이 라벨 원문으로 한다. 설계는 [SPEC.md](../SPEC.md).

## 실행 (3줄)
```
python3.12 -m venv .venv && .venv/bin/pip install opencv-python-headless==5.0.0.93 pytest pyyaml && .venv/bin/python scripts/download_models.py
.venv/bin/python -m pytest -q          # 단위 시험 81개(SAM 템플릿 시험 7개 포함) (가중치 없으면 판독기 시험 2개는 건너뜀)
.venv/bin/python scripts/download_images.py && .venv/bin/python -m label_guard.adapters.evaluate linux   # 리눅스 판독기로 104건 평가
```
macOS 판독기(Vision)로 돌리려면 `swiftc -O label_guard/adapters/ocr_vision.swift -o .build/ocr_vision` 후 `-m label_guard.adapters.evaluate`.

## 구조
`domain/`(OpenCV·I/O 0) <- `application/`(포트·판정 오케스트레이션) <- `adapters/`(opencv5_quality, vision_reader[macOS], dnn_reader[리눅스·OpenCV DNN], evaluate) <- `infrastructure/`(lambda_handler).
판독 포트 `TextReader` 구현 2개: macOS Vision(온디바이스), OpenCV 5 DNN(DB 검출 + PP-OCRv3 인식, 학습 0). 배포는 [docs/deploy.md](deploy.md).

## 평가 (실측, 104건)
[docs/eval/eval.md](eval/eval.md) · [docs/eval/failures.md](eval/failures.md) · 판독기 비교 [docs/eval/reader-compare.md](eval/reader-compare.md) · 재촬영 [docs/eval/retake.md](eval/retake.md) · 엔드포인트 지연 [docs/eval/endpoint-latency.md](eval/endpoint-latency.md). 이미지 출처 [data/ATTRIBUTION.md](../data/ATTRIBUTION.md).
