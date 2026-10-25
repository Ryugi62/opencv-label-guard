"""공개 URL에 라벨 사진 3장 x N회 요청해 응답 코드·지연 p50/p95를 잰다 -> docs/eval/endpoint-latency.md.
사용: python3 scripts/measure_endpoint.py <URL> [N=10]"""
import json, sys, time, urllib.request
import cv2, numpy as np
from label_guard.adapters.opencv5_quality import load_image

URL = sys.argv[1].rstrip("/") + "/?profile=milk,gluten,soy,egg,nut,fish,other"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
m = json.load(open("data/manifest.json"))["items"]
pick = [("en", "0014100098423"), ("ko", next(i["barcode"] for i in m if i["lang"] == "ko")), ("fr", next(i["barcode"] for i in m if i["lang"] == "fr"))]


def post(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    rq = urllib.request.Request(URL, data=buf.tobytes(), headers={"content-type": "image/jpeg"}, method="POST")
    t = time.perf_counter(); r = urllib.request.urlopen(rq, timeout=90); d = json.loads(r.read())
    return r.status, time.perf_counter() - t, d, len(buf)


rows, tot, srv, codes = [], [], [], []
for lang, b in pick:
    img = load_image(f"data/images/{b}.jpg", 2000); ts = []
    for _ in range(N):
        code, t, d, size = post(img); ts.append(t); srv.append(d["latency_ms"] / 1000); codes.append(code)
    tot += ts
    rows.append(f"| {lang} | {b} | {size//1024}KB | {d['kind']} | {np.percentile(ts,50):.2f}s | {max(ts):.2f}s |")
blur = cv2.GaussianBlur(load_image(f"data/images/{pick[0][1]}.jpg", 2000), (0, 0), 12)
code, t, d, _ = post(blur)
q = lambda a, p: float(np.percentile(a, p))
L = ["# 공개 엔드포인트 지연 실측", "", f"- URL: {sys.argv[1]} (비로그인) · 측정 {time.strftime('%Y-%m-%d %H:%M')} KST · 요청 {len(tot)}건(사진 3장 x {N}회, 따뜻한 상태), HTTP 200 비율 {sum(c==200 for c in codes)}/{len(codes)}",
     f"- 클라이언트 왕복(업로드 포함) p50 {q(tot,50):.2f}s · p95 {q(tot,95):.2f}s (SPEC S4: p95 <= 60s, 서버 p50 <= 10s)",
     f"- 서버 처리 p50 {q(srv,50):.2f}s · p95 {q(srv,95):.2f}s", "",
     "| 언어 | 바코드(OFF) | 업로드 크기 | 판정 | 왕복 p50 | 왕복 최대 |", "|---|---|---|---|---|---|", *rows, "",
     f"- 흐린 사진(가우시안 12) -> 판정 {d['kind']} 사유 {d['reasons']} · 왕복 {t:.2f}s (판독 건너뜀: 품질 관문이 OpenCV 단독으로 결정)",
     "- 콜드 스타트(새 실행 환경) 1건: 왕복 5.77s, 서버 처리 3.6s, 초기화 0.75s (CloudWatch REPORT)."]
open("docs/eval/endpoint-latency.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
