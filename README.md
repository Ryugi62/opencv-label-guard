# opencv-label-guard

폰으로 찍은 식품 성분표 사진을 OpenCV 5가 읽을 수 있는 상태인지 판정(원근 보정·반사·흐림·글자 영역)하고, 결과에 따라 알레르기 경고 / 재촬영 요청 / 사람 확인으로 행동을 가르는 루프. 의료 판단은 하지 않으며 최종 확인은 항상 사람이 라벨 원문으로 한다. 설계는 [SPEC.md](SPEC.md).

## 실행 (3줄)
```
python3.12 -m venv .venv && .venv/bin/pip install opencv-python==5.0.0.93 pytest
.venv/bin/python -m pytest -q          # 단위 시험 30개
swiftc -O label_guard/adapters/ocr_vision.swift -o .build/ocr_vision && .venv/bin/python scripts/download_images.py && .venv/bin/python -m label_guard.adapters.evaluate
```

## 구조
`domain/`(OpenCV·I/O 0) <- `application/`(포트·판정 오케스트레이션) <- `adapters/`(opencv5_quality, vision_reader, evaluate) <- `infrastructure/`(다음 단계: AWS).
판독 포트 구현은 macOS Vision(온디바이스)이며 AWS 이식 시 같은 `TextReader` 포트 뒤에 다른 판독기를 둔다.

## 평가 (실측, 104건)
[docs/eval/eval.md](docs/eval/eval.md) · [docs/eval/failures.md](docs/eval/failures.md) · [results.json](docs/eval/results.json). 이미지 출처 [data/ATTRIBUTION.md](data/ATTRIBUTION.md).
