# AWS 배포 (Cloud delivery) — 상시 무료 등급 구성

공개 URL(비로그인): https://hqadhnbyzbv5dyok3zdnnfpcae0tcozx.lambda-url.ap-northeast-2.on.aws/
- `GET /` 업로드 화면(라벨·파일 입력·판정 버튼, 키보드만으로 조작) · `POST /?profile=milk,soy` 본문 = 이미지 바이트 → JSON(판정·근거·행동 로그·서버 처리 시간).

## 구성 (AWS 서비스 5개)
| 서비스 | 역할 | 과금 |
|---|---|---|
| Lambda (python3.12, x86_64, 2048MB, 60초) | 판정 엔진(OpenCV 5 품질 분석 + DNN 판독) 실행 | 상시 무료: 월 요청 100만·400,000 GB-초 |
| Lambda 함수 URL (AuthType NONE) | 공개 HTTPS 진입점(API Gateway 안 씀) | 무료(요청은 Lambda 요청으로 계산) |
| Lambda 계층 2개 | cv2(31MB) / 공유 라이브러리+numpy(46MB) — 직접 업로드 한도 50MB에 맞춰 분할, S3·ECR 안 씀 | 무료 |
| CloudWatch Logs | 지표 로그(판정 종류·지연 ms·그룹 수만, 이미지 0), 보존 7일 | 상시 무료 5GB |
| IAM 역할 1개 `label-guard-lambda-role` | 로그 쓰기만 | 무료 |

## 재현 (명령 2줄)
```
./scripts/build_lambda.sh                 # dist/ 에 zip 3개 (cv2 계층 / 라이브러리+numpy 계층 / 함수)
python3 scripts/deploy_lambda.py dayafter ap-northeast-2   # 계층 게시 · 함수 생성/갱신 · 함수 URL 출력 (멱등)
```
가중치(`models/*.onnx`)는 `scripts/download_models.py`로 받는다(출처·해시는 `models/MODELS.md`).

## 권한 최소화 (IAM 목록)
- 함수 실행 역할: `AWSLambdaBasicExecutionRole`(logs:CreateLogStream·PutLogEvents) 하나. S3·DB·네트워크 권한 0.
- 배포 주체(IAM 사용자)는 별도 — 함수는 아무 비밀값·키도 들고 있지 않다.
- 남용 방지: 예약 동시 실행 5(초과 요청은 429), 본문 6MB 상한(413), 이미지 아닌 본문 400.

## 월 비용 추정
- 요청 1건 = 약 3초 × 2GB = 6 GB-초(실측 청구 시간 3.0~3.5초). 상시 무료 400,000 GB-초 = 월 약 66,000건까지 0원.
- 월 1,000건(심사·시연 규모) = 6,000 GB-초 → 무료 한도의 1.5%, **0원**. 한도 초과분은 GB-초당 약 $0.0000166667 → 10만 건이라도 약 $4 미만. 기존 계정 상한 월 $10 이내.
- 기존 자원(EC2·`dayafter` 함수)은 건드리지 않았다.

## 지연 실측 (공개 URL, 2026-10-26, 폰 사진 크기 2000px JPEG 약 260KB)
30건 모두 HTTP 200. 클라이언트 왕복 p50 3.34s · p95 3.71s, 서버 처리 p50 3.20s · p95 3.54s, 콜드 스타트 1건 5.77s, 최대 메모리 377MB. 표: [docs/eval/endpoint-latency.md](eval/endpoint-latency.md) (`scripts/measure_endpoint.py`로 재현).

## 관측 지표
CloudWatch Logs 한 줄 `{"kind": ..., "latency_ms": ..., "groups": n}`와 Lambda `REPORT`(Duration·Max Memory Used·Init Duration). 알람은 만들지 않았다(무료 한도 1.5%라 실익 없음, 예약 동시성이 상한).
