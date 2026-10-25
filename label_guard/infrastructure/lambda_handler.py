"""AWS Lambda 함수 URL 진입점(infrastructure). 이미지는 메모리에서만 쓰고 저장·로그하지 않는다(AC-9)."""
from __future__ import annotations
import base64, json, os, time
import cv2
import numpy as np
from ..application.judge import judge_photo
from ..domain.allergens import ALL_GROUPS
from ..domain.model import AllergyProfile, Thresholds

MAX_BYTES = 6 * 1024 * 1024
MAX_SIDE = 2400
PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>라벨 판독 가드</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:40rem;margin:2rem auto;padding:0 1rem}
fieldset{border:1px solid #ccc}button{font-size:1rem;padding:.6rem 1.2rem}pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}
:focus-visible{outline:3px solid #0050d0}</style></head><body><main>
<h1>식품 라벨 판독 가드</h1>
<p>성분표 사진 1장을 올리면 OpenCV 5가 읽을 수 있는 상태인지 판정하고, 경고 · 재촬영 · 사람 확인 중 하나를 알려 줍니다. 의료 판단은 하지 않으며 최종 확인은 항상 사람이 라벨 원문으로 합니다. 사진은 저장되지 않습니다.</p>
<form id="f"><fieldset><legend>경계할 알레르기</legend>__BOXES__</fieldset>
<p><label for="photo">라벨 사진</label><br><input id="photo" name="photo" type="file" accept="image/*" required></p>
<p><button type="submit">판정하기</button></p></form>
<h2 id="rh" tabindex="-1">결과</h2><div id="out" role="status" aria-live="polite">아직 없음</div>
<script>
const f=document.getElementById('f'),out=document.getElementById('out');
async function shrink(file){const bm=await createImageBitmap(file);const s=Math.min(1,2000/Math.max(bm.width,bm.height));
const c=document.createElement('canvas');c.width=Math.round(bm.width*s);c.height=Math.round(bm.height*s);c.getContext('2d').drawImage(bm,0,0,c.width,c.height);
return await new Promise(r=>c.toBlob(r,'image/jpeg',0.9));}
f.addEventListener('submit',async e=>{e.preventDefault();out.textContent='판정 중...';
const g=[...f.querySelectorAll('input[name=g]:checked')].map(x=>x.value).join(',');
try{const blob=await shrink(f.photo.files[0]);const r=await fetch('?profile='+encodeURIComponent(g),{method:'POST',headers:{'content-type':'image/jpeg'},body:blob});
const d=await r.json();out.innerHTML='';const p=document.createElement('pre');
p.textContent=r.ok?('판정: '+d.kind+'\\n'+d.message+'\\n\\n행동 로그\\n'+d.trace.join('\\n')+'\\n\\n서버 처리 '+d.latency_ms+' ms'):('오류: '+(d.error||r.status));
out.appendChild(p);document.getElementById('rh').focus();}catch(err){out.textContent='오류: '+err}});
</script></main></body></html>"""


def _resp(code, obj, ctype="application/json; charset=utf-8"):
    body = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return {"statusCode": code, "headers": {"content-type": ctype, "cache-control": "no-store"}, "body": body}


def _page():
    boxes = "".join(f'<label><input type="checkbox" name="g" value="{g}" checked> {g}</label> ' for g in ALL_GROUPS)
    return _resp(200, PAGE.replace("__BOXES__", boxes), "text/html; charset=utf-8")


def make_handler(analyzer, reader, thresholds: Thresholds = Thresholds()):
    def handler(event, context):
        method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
        if method == "GET":
            return _page()
        if method != "POST":
            return _resp(405, {"error": "GET 또는 POST만 지원합니다"})
        t0 = time.perf_counter()
        body = event.get("body")
        if not body:
            return _resp(400, {"error": "이미지 본문이 없습니다"})
        raw = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("latin-1")
        if len(raw) > MAX_BYTES:
            return _resp(413, {"error": "이미지가 6MB를 넘습니다"})
        qs = event.get("queryStringParameters") or {}
        groups = [g for g in (qs.get("profile") or "").split(",") if g] or list(ALL_GROUPS)
        if any(g not in ALL_GROUPS for g in groups):
            return _resp(400, {"error": "알 수 없는 알레르기 그룹: " + ",".join(g for g in groups if g not in ALL_GROUPS)})
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        del raw
        if img is None:
            return _resp(400, {"error": "이미지를 읽을 수 없습니다"})
        s = MAX_SIDE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        v = judge_photo(img, AllergyProfile(groups), analyzer, reader, thresholds)
        ms = round((time.perf_counter() - t0) * 1000)
        print(json.dumps({"kind": v.kind.value, "latency_ms": ms, "groups": len(groups)}))   # 지표만, 이미지 0
        return _resp(200, {"kind": v.kind.value, "message": v.message, "reasons": list(v.reasons),
                           "evidence": [{"group": e.group, "term": e.term, "line": e.line_index + 1, "text": e.line_text[:80]} for e in v.evidence],
                           "trace": list(v.trace), "latency_ms": ms})
    return handler


_HANDLER = None


def lambda_handler(event, context):
    """Lambda 진입점: 콜드 스타트 때 한 번 만든다(composition root)."""
    global _HANDLER
    if _HANDLER is None:
        from ..adapters.dnn_reader import DnnTextReader
        from ..adapters.opencv5_quality import OpenCV5QualityAnalyzer
        m = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        reader = DnnTextReader(os.path.join(m, "text_detection_en_ppocrv3_2023may.onnx"), os.path.join(m, "en_PP-OCRv3_rec_infer.onnx"),
                               os.path.join(m, "en_ppocrv3_dict.txt"), alt_rec_path=os.path.join(m, "korean_mobile_v2.0_rec_infer.onnx"),
                               alt_dict_path=os.path.join(m, "ko_v2_dict.txt"))
        _HANDLER = make_handler(OpenCV5QualityAnalyzer(), reader)
    return _HANDLER(event, context)
