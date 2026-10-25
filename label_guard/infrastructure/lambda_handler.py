"""AWS Lambda 함수 URL 진입점(infrastructure). 이미지는 메모리에서만 쓰고 저장·로그하지 않는다(AC-9)."""
from __future__ import annotations
import base64, json, os, time
import cv2
import numpy as np
from ..application.judge import judge_photo
from ..domain.allergens import ALL_GROUPS
from ..domain.decide import retake_guidance
from ..domain.model import AllergyProfile, Thresholds

MAX_BYTES = 6 * 1024 * 1024
MAX_SIDE = 2400
STYLE = """body{font:16px/1.5 system-ui,sans-serif;max-width:40rem;margin:2rem auto;padding:0 1rem}
fieldset{border:1px solid #ccc}button{font-size:1rem;padding:.6rem 1.2rem}pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}
:focus-visible{outline:3px solid #0050d0}.top{text-align:right}"""
TEXT = {
    "en": {"title": "Label Reading Guard", "h1": "Food Label Reading Guard", "toggle": "한국어", "toggle_to": "ko",
           "intro": "Upload one photo of an ingredient list. OpenCV 5 first judges whether the photo is readable, then tells you one of: allergy warning, retake request, or human check. It gives no medical advice, and a person always makes the final check on the original label. Photos are not stored.",
           "legend": "Allergens to watch for", "photo": "Label photo", "go": "Check photo", "result": "Result", "none": "Nothing yet",
           "busy": "Checking...", "verdict": "Verdict: ", "log": "Action log", "server": "Server time ", "err": "Error: ", "busy429": "The demo is busy (limited to 5 concurrent requests to stay inside the free tier). Please try again in a few seconds."},
    "ko": {"title": "라벨 판독 가드", "h1": "식품 라벨 판독 가드", "toggle": "English", "toggle_to": "en",
           "intro": "성분표 사진 1장을 올리면 OpenCV 5가 읽을 수 있는 상태인지 판정하고, 경고 · 재촬영 · 사람 확인 중 하나를 알려 줍니다. 의료 판단은 하지 않으며 최종 확인은 항상 사람이 라벨 원문으로 합니다. 사진은 저장되지 않습니다.",
           "legend": "경계할 알레르기", "photo": "라벨 사진", "go": "판정하기", "result": "결과", "none": "아직 없음",
           "busy": "판정 중...", "verdict": "판정: ", "log": "행동 로그", "server": "서버 처리 ", "err": "오류: ", "busy429": "지금 요청이 몰렸습니다(무료 등급 보호를 위해 동시 5건 제한). 몇 초 뒤 다시 시도해 주세요."},
}
PAGE = """<!doctype html><html lang="__LANG__"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__title__</title><style>__STYLE__</style></head><body><main>
<p class="top"><a href="?lang=__toggle_to__" lang="__toggle_to__">__toggle__</a></p>
<h1>__h1__</h1>
<p>__intro__</p>
<form id="f"><fieldset><legend>__legend__</legend>__BOXES__</fieldset>
<p><label for="photo">__photo__</label><br><input id="photo" name="photo" type="file" accept="image/*" required></p>
<p><button type="submit">__go__</button></p></form>
<h2 id="rh" tabindex="-1">__result__</h2><div id="out" role="status" aria-live="polite">__none__</div>
<script>
const f=document.getElementById('f'),out=document.getElementById('out'),L=__LANG_JSON__,T=__T_JSON__;
async function shrink(file){const bm=await createImageBitmap(file);const s=Math.min(1,2000/Math.max(bm.width,bm.height));
const c=document.createElement('canvas');c.width=Math.round(bm.width*s);c.height=Math.round(bm.height*s);c.getContext('2d').drawImage(bm,0,0,c.width,c.height);
return await new Promise(r=>c.toBlob(r,'image/jpeg',0.9));}
f.addEventListener('submit',async e=>{e.preventDefault();out.textContent=T.busy;
const g=[...f.querySelectorAll('input[name=g]:checked')].map(x=>x.value).join(',');
try{const blob=await shrink(f.photo.files[0]);const r=await fetch('?lang='+L+'&profile='+encodeURIComponent(g),{method:'POST',headers:{'content-type':'image/jpeg'},body:blob});
if(r.status===429){out.textContent=T.busy429;return}
const d=await r.json();out.innerHTML='';const p=document.createElement('pre');
p.textContent=r.ok?(T.verdict+d.kind+'\\n'+d.message+'\\n\\n'+T.log+'\\n'+d.trace.join('\\n')+'\\n\\n'+T.server+d.latency_ms+' ms'):(T.err+(d.error||r.status));
out.appendChild(p);document.getElementById('rh').focus();}catch(err){out.textContent=T.err+err}});
</script></main></body></html>"""


def _resp(code, obj, ctype="application/json; charset=utf-8"):
    body = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return {"statusCode": code, "headers": {"content-type": ctype, "cache-control": "no-store"}, "body": body}


def _lang(event):
    return "ko" if (event.get("queryStringParameters") or {}).get("lang") == "ko" else "en"   # 기본 영어


def _page(lang):
    t = TEXT[lang]
    boxes = "".join(f'<label><input type="checkbox" name="g" value="{g}" checked> {g}</label> ' for g in ALL_GROUPS)
    html = PAGE.replace("__STYLE__", STYLE).replace("__BOXES__", boxes).replace("__LANG__", lang)
    html = html.replace("__LANG_JSON__", json.dumps(lang)).replace("__T_JSON__", json.dumps(t, ensure_ascii=False))
    for k, v in t.items():
        html = html.replace(f"__{k}__", v)
    return _resp(200, html, "text/html; charset=utf-8")


def make_handler(analyzer, reader, thresholds: Thresholds = Thresholds()):
    def handler(event, context):
        method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
        lang = _lang(event)
        en = lang == "en"
        if method == "GET":
            return _page(lang)
        if method != "POST":
            return _resp(405, {"error": "Only GET or POST is supported" if en else "GET 또는 POST만 지원합니다"})
        t0 = time.perf_counter()
        body = event.get("body")
        if not body:
            return _resp(400, {"error": "No image in request body" if en else "이미지 본문이 없습니다"})
        raw = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("latin-1")
        if len(raw) > MAX_BYTES:
            return _resp(413, {"error": "Image is larger than 6 MB" if en else "이미지가 6MB를 넘습니다"})
        qs = event.get("queryStringParameters") or {}
        groups = [g for g in (qs.get("profile") or "").split(",") if g] or list(ALL_GROUPS)
        if any(g not in ALL_GROUPS for g in groups):
            return _resp(400, {"error": ("Unknown allergen group: " if en else "알 수 없는 알레르기 그룹: ") + ",".join(g for g in groups if g not in ALL_GROUPS)})
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        del raw
        if img is None:
            return _resp(400, {"error": "Could not decode the image" if en else "이미지를 읽을 수 없습니다"})
        s = MAX_SIDE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        v = judge_photo(img, AllergyProfile(groups), analyzer, reader, thresholds)
        ms = round((time.perf_counter() - t0) * 1000)
        print(json.dumps({"kind": v.kind.value, "latency_ms": ms, "groups": len(groups)}))   # 지표만, 이미지 0
        return _resp(200, {"kind": v.kind.value, "message": v.message_en if en else v.message, "reasons": list(v.reasons),
                           "guidance": [{"reason": r, "tip": (e_ if en else k_)} for r, k_, e_ in retake_guidance(v.reasons)],
                           "evidence": [{"group": e.group, "term": e.term, "line": e.line_index + 1, "text": e.line_text[:80]} for e in v.evidence],
                           "trace": list(v.trace_en if en else v.trace), "latency_ms": ms})
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
