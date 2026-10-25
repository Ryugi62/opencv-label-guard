# 공개 엔드포인트 지연 실측

- URL: https://hqadhnbyzbv5dyok3zdnnfpcae0tcozx.lambda-url.ap-northeast-2.on.aws/ (비로그인) · 측정 2026-10-26 05:33 KST · 요청 30건(사진 3장 x 10회, 따뜻한 상태), HTTP 200 비율 30/30
- 클라이언트 왕복(업로드 포함) p50 3.34s · p95 3.71s (SPEC S4: p95 <= 60s, 서버 p50 <= 10s)
- 서버 처리 p50 3.20s · p95 3.54s

| 언어 | 바코드(OFF) | 업로드 크기 | 판정 | 왕복 p50 | 왕복 최대 |
|---|---|---|---|---|---|
| en | 0014100098423 | 254KB | allergy_warning | 3.34s | 3.44s |
| ko | 2000000021345 | 492KB | allergy_warning | 3.67s | 3.80s |
| fr | 2935306002206 | 181KB | allergy_warning | 1.97s | 2.14s |

- 흐린 사진(가우시안 12) -> 판정 retake_request 사유 ['흐림'] · 왕복 0.17s (판독 건너뜀: 품질 관문이 OpenCV 단독으로 결정)
- 콜드 스타트(새 실행 환경) 1건: 왕복 5.77s, 서버 처리 3.6s, 초기화 0.75s (CloudWatch REPORT).
