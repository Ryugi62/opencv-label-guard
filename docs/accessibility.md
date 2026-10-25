# 접근성 점검 목록 (데모 화면, 키보드·대체 텍스트)
2026-10-26 실화면(공개 URL)에서 확인. 캡처: `demo-upload-warning.png`(경고)·`demo-upload-retake-white.png`(재촬영) — Jarvis 볼트 handoff/drop/opencv-ai-comp/.
| # | 점검 | 결과 |
|---|---|---|
| 1 | 마우스 없이 업로드·판정·결과 확인: Tab으로 체크박스 -> 파일 선택 -> 「판정하기」, Enter로 제출 | 통과(판정하기 버튼에 포커스 후 Enter로 제출 -> 결과 표시) |
| 2 | 모든 입력에 연결된 라벨(`label for=photo`, 체크박스는 label로 감쌈) | 통과 (시험 test_get_returns_keyboard_usable_upload_page) |
| 3 | 결과 영역 `role=status aria-live=polite` — 화면 낭독기에 결과 낭독 | 통과 |
| 4 | 제출 뒤 포커스가 「결과」 제목으로 이동(tabindex=-1) | 통과(캡처의 파란 포커스 링) |
| 5 | 포커스 표시 `:focus-visible` 3px 윤곽선 | 통과 |
| 6 | 이미지(img) 요소 0개 — 대체 텍스트 필요 없음, 결과는 전부 텍스트 | 통과 |
| 7 | 휴대폰 폭: viewport 메타 + 본문 max-width 40rem, 사진은 올리기 전 2000px로 줄임 | 통과(본문 6MB 한도 대비 260KB) |
| 8 | 색만으로 의미 전달 없음(판정은 글자) | 통과 |
한계: 화면 낭독기(VoiceOver) 실청취는 하지 않았다. 캡처에 쓴 사진은 캔버스로 그린 합성 라벨(실사진 3장 응답은 endpoint-latency.md).
