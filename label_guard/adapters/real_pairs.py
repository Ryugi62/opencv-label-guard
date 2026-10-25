"""실제 폰 사진 쌍(Open Food Facts 서로 다른 업로드) 수집·평가 -> docs/eval/real-pairs.md.
같은 제품의 같은 언어 성분표를 서로 다른 사람이 찍어 올린 원본 업로드 쌍이다(합성 아님).
성분표 업로드 선별은 macOS Vision OCR이 읽은 글자와 제품 성분 문장의 낱말 겹침으로 한다 — 판정기(배포 DNN)와 다른 판독기다.
이미지는 레포에 저장하지 않는다(CC BY-SA, 상품 페이지 링크만)."""
from __future__ import annotations
import json, os, re, time, urllib.request, urllib.parse
from ..domain.allergens import ALL_GROUPS
from ..domain.decide import decide
from ..domain.model import AllergyProfile, PhotoQuality, ReadLine, Thresholds

UA = "opencv-label-guard-eval/0.1 (contest research)"
MIN_SIDE = 800
LANGS = ("en", "fr", "es", "de")
ABSTAIN = ("retake_request", "human_check")


def raw_url(code: str, imgid: int) -> str:
    p = f"{code[:3]}/{code[3:6]}/{code[6:9]}/{code[9:]}" if len(code) > 8 else code
    return f"https://images.openfoodfacts.org/images/products/{p}/{imgid}.jpg"


def candidates(images: dict) -> list:
    out = []
    for k, v in images.items():
        if not k.isdigit() or not isinstance(v, dict):
            continue
        up = str(v.get("uploader", ""))
        f = (v.get("sizes") or {}).get("full") or {}
        if up.startswith("org-") or min(f.get("w", 0), f.get("h", 0)) < MIN_SIDE:
            continue
        out.append({"imgid": int(k), "uploader": up, "uploaded_t": v.get("uploaded_t"), "w": f.get("w"), "h": f.get("h")})
    return sorted(out, key=lambda c: c["imgid"])


def _tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-zà-ÿœ]{4,}", s.lower())}


def overlap(ocr_text: str, truth: str) -> float:
    """제품 성분 문장 낱말(4자 이상) 중 OCR 글자에서 다시 나온 비율."""
    tt = _tokens(truth)
    return len(tt & _tokens(ocr_text)) / len(tt) if tt else 0.0


def valid_pair(a: dict, b: dict) -> bool:
    return a["code"] == b["code"] and a["imgid"] != b["imgid"]


def _worse(row, side):
    o = "b" if side == "a" else "a"
    return row[f"blur_{side}"] < row[f"blur_{o}"] or row[f"glare_{side}"] > row[f"glare_{o}"]


def summarize(rows: list) -> dict:
    n = len(rows)
    agree = sum(r["ka"] == r["kb"] for r in rows)
    one = [r for r in rows if (r["ka"] in ABSTAIN) != (r["kb"] in ABSTAIN)]
    worse = 0
    for r in one:
        side = "a" if r["ka"] in ABSTAIN else "b"
        worse += bool(_worse(r, side))
    both_answer_diff = sum(1 for r in rows if r["ka"] != r["kb"] and r["ka"] not in ABSTAIN and r["kb"] not in ABSTAIN)
    imgs = [(r["ka"], r.get("truth_a")) for r in rows] + [(r["kb"], r.get("truth_b")) for r in rows]
    ab = sum(k in ABSTAIN for k, _ in imgs)
    return {"n": n, "agree": agree, "agree_rate": agree / n if n else float("nan"),
            "one_sided_abstain": len(one), "one_sided_rate": len(one) / n if n else float("nan"),
            "both_answer_diff": both_answer_diff, "abstainer_worse": worse, "abstainer_worse_n": len(one),
            "abstain_images": ab, "images": len(imgs), "abstain_rate": ab / len(imgs) if imgs else float("nan")}


def http_json(url, delay=1.1, tries=4):
    for k in range(tries):
        time.sleep(delay * (k + 1))   # 실패하면 간격을 늘려 다시(서버 보호)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=40) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise


def http_bytes(url, delay=1.1):
    time.sleep(delay)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        return r.read()


def search_codes(pages, sort="unique_scans_n"):
    """인기순 바코드·언어 목록. 검색 API는 분당 10회 제한 -> 7초 간격, 무거운 images 필드는 제품 조회에서 받는다."""
    out = []
    for p in range(1, pages + 1):
        q = urllib.parse.urlencode({"sort_by": sort, "page_size": 100, "page": p, "fields": "code,lang"})
        try:
            d = http_json("https://world.openfoodfacts.org/api/v2/search?" + q, delay=7)
        except Exception:
            continue
        out += [(x["code"], x.get("lang")) for x in d.get("products", []) if x.get("lang") in LANGS]
    return out


def fetch_product(code):
    f = "code,lang,allergens_tags,images,product_name," + ",".join(f"ingredients_text_{l}" for l in LANGS)
    pr = http_json(f"https://world.openfoodfacts.org/api/v2/product/{code}?fields={f}", delay=1.1)["product"]
    lang = pr.get("lang")
    txt = pr.get(f"ingredients_text_{lang}") or ""
    cs = candidates(pr.get("images", {}))
    if lang in LANGS and len(txt) >= 60 and len(cs) >= 2:
        return {"code": pr["code"], "lang": lang, "name": pr.get("product_name", ""), "truth_text": txt,
                "allergens_tags": pr.get("allergens_tags", []), "cands": cs}
    return None


def quality_and_kind(path, analyzer, reader, load_image, t=Thresholds()):
    """배포 경로와 같은 순서: 품질 분석(원근 보정) -> 미달이면 판독 없이 재촬영 요청 -> 판독 -> 결정."""
    from ..application.judge import judge_photo
    img = load_image(path)
    q, _, _ = analyzer.analyze(img)
    v = judge_photo(img, AllergyProfile(ALL_GROUPS), analyzer, reader, t)
    return q, v


FORCED = Thresholds(min_blur=0.0, max_glare=2.0, min_regions=0, min_confidence=0.0)
MIN_OVERLAP = 0.35
MAX_TRY = 8


def judge_image(path, analyzer, reader, load_image, truth_pos, t=Thresholds()):
    """배포 경로 판정 + 억지로 답했다면(관문 끔) 맞았는지."""
    from ..application.judge import judge_photo
    img = load_image(path)
    q, used, _ = analyzer.analyze(img)
    v = judge_photo(img, AllergyProfile(ALL_GROUPS), analyzer, reader, t)
    forced = decide(q, [ReadLine(l.text, l.confidence) for l in reader.read(used)], AllergyProfile(ALL_GROUPS), FORCED)
    return {"kind": v.kind.value, "blur": q.blur_score, "glare": q.glare_ratio, "regions": q.text_regions,
            "forced_ok": (forced.kind.value == "allergy_warning") == truth_pos}


def pick_pair(qualified):
    """성분표로 확인된 업로드 중 업로더가 다른 두 장을 imgid 순으로(없으면 imgid 가장 작은 두 장)."""
    for i, a in enumerate(qualified):
        for b in qualified[i + 1:]:
            if a["uploader"] != b["uploader"]:
                return a, b
    return (qualified[0], qualified[1]) if len(qualified) >= 2 else None


def run(codes, target, workdir, cache_path, log=print):
    import cv2
    from ..domain.allergens import OFF_TAG_GROUP
    from .evaluate import linux_reader, truth_groups
    from .opencv5_quality import OpenCV5QualityAnalyzer, load_image
    from .vision_reader import VisionReader
    os.makedirs(workdir, exist_ok=True)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {"pairs": [], "skipped": {}}
    done = {p["code"] for p in cache["pairs"]} | set(cache["skipped"])
    an, dnn, vis = OpenCV5QualityAnalyzer(), linux_reader(), VisionReader()
    for code, _ in codes:
        if len(cache["pairs"]) >= target:
            break
        if code in done:
            continue
        try:
            pr = fetch_product(code)
            if pr is None:
                cache["skipped"][code] = "후보 업로드<2 또는 성분 문장 없음"; continue
            qual = []
            for c in pr["cands"][:MAX_TRY]:
                path = os.path.join(workdir, f"{code}_{c['imgid']}.jpg")
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    data = http_bytes(raw_url(code, c["imgid"]))
                    open(path, "wb").write(data)
                lines = vis.read(load_image(path))
                ov = overlap(" ".join(l.text for l in lines), pr["truth_text"])
                c = dict(c, path=path, overlap=round(ov, 3))
                if ov >= MIN_OVERLAP:
                    qual.append(c)
                if len(qual) >= 3:
                    break
            pair = pick_pair(qual)
            if not pair:
                cache["skipped"][code] = f"성분표로 확인된 업로드 {len(qual)}장(<2)"; continue
            tp = bool(truth_groups(pr["allergens_tags"]))
            a, b = pair
            ra, rb = (judge_image(x["path"], an, dnn, load_image, tp) for x in pair)
            assert valid_pair({"code": code, "imgid": a["imgid"]}, {"code": code, "imgid": b["imgid"]})
            cache["pairs"].append({"code": code, "name": pr["name"], "lang": pr["lang"], "truth_pos": tp,
                                   "a": {k: a[k] for k in ("imgid", "w", "h", "overlap")} | {"same_uploader": a["uploader"] == b["uploader"]},
                                   "b": {k: b[k] for k in ("imgid", "w", "h", "overlap")}, "ra": ra, "rb": rb})
            log(f"pair {len(cache['pairs'])} {code} {a['imgid']}/{b['imgid']} {ra['kind']}/{rb['kind']}")
        except Exception as e:
            cache["skipped"][code] = f"{type(e).__name__}: {str(e)[:80]}"
        json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
    json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
    return cache


def rows_from_cache(cache):
    out = []
    for p in cache["pairs"]:
        a, b = p["ra"], p["rb"]
        out.append({"code": p["code"], "ka": a["kind"], "kb": b["kind"], "blur_a": a["blur"], "blur_b": b["blur"],
                    "glare_a": a["glare"], "glare_b": b["glare"], "truth_a": p["truth_pos"], "truth_b": p["truth_pos"],
                    "fa": a["forced_ok"], "fb": b["forced_ok"]})
    return out


def abstain_accuracy(rows):
    """기권한 사진을 억지로 답하게 했다면 틀렸을 비율(= 유용한 기권)."""
    ab = [(r["fa"]) for r in rows if r["ka"] in ABSTAIN] + [(r["fb"]) for r in rows if r["kb"] in ABSTAIN]
    return (sum(1 for ok in ab if not ok) / len(ab) if ab else float("nan")), len(ab)


def answered_accuracy(rows):
    ok = n = 0
    for r in rows:
        for k, t in ((r["ka"], r["truth_a"]), (r["kb"], r["truth_b"])):
            if k not in ABSTAIN:
                n += 1; ok += (k == "allergy_warning") == bool(t)
    return (ok / n if n else float("nan")), n


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def write_report(cache, out="docs/eval/real-pairs.md"):
    rows = rows_from_cache(cache)
    s = summarize(rows)
    ua, na = abstain_accuracy(rows)
    aa, nn = answered_accuracy(rows)
    lo, hi = wilson(s["agree"], s["n"])
    same = sum(1 for p in cache["pairs"] if p["a"].get("same_uploader"))
    langs = {}
    for p in cache["pairs"]:
        langs[p["lang"]] = langs.get(p["lang"], 0) + 1
    lang_txt = " · ".join(f"{k} {v}" for k, v in sorted(langs.items(), key=lambda kv: -kv[1]))
    fo = [r["fa"] for r in rows] + [r["fb"] for r in rows]
    forced_acc = sum(fo) / len(fo) if fo else float("nan")
    VK = {"allergy_warning": "경고", "retake_request": "재촬영 요청", "human_check": "사람 확인", "no_warning": "경고 없음"}
    L = ["# 실제 폰 사진 쌍 — 같은 제품 성분표를 서로 다른 사람이 올린 원본 업로드", "",
         "- **합성이 아니다**: Open Food Facts에서 같은 제품·같은 언어 성분표를 **서로 다른 업로드(imgid)** 로 올린 원본 사진 2장을 한 쌍으로 묶었다. 업로드가 다르므로 촬영 시점·조명·각도가 다르다(업로더가 다른 쌍은 폰·촬영자도 다름 — 아래 요약의 업로더 구성 참고). 우리가 찍은 사진이 아니고, 한 사람이 같은 라벨을 다시 찍는 「재촬영」이 아니라 「같은 라벨의 다른 사진」이다.",
         "- 판정기 = 배포 판독기(OpenCV 5 DNN: DB 글자 검출 + PP-OCRv3 인식, 임계값 흐림 <15 · 반사 >0.4 · 신뢰도 <0.6, 재보정 없음). 프로필 = 전체 그룹. 정답 = 제품의 allergens_tags(이미지 단위 「알레르기 있음」).",
         "- **쌍 선별 방법(한계)**: 제품 후보는 OFF 인기순(스캔 수)에서 받았고, 업로드가 성분표인지는 macOS Vision OCR이 읽은 글자가 제품 성분 문장 낱말의 35% 이상과 겹치는지로 정했다(판정기와 다른 판독기). 따라서 Vision조차 못 읽는 심하게 흐린 사진은 쌍에서 빠져 **기권 비율이 실제보다 낮게 나온다**. 한 제품에서 한 쌍만 쓴다(독립).",
         "- 라이선스: 데이터 ODbL + DbCL, 사진 CC BY-SA(OFF terms-of-use 원문 확인, 2026-10-26). 사진은 레포에 두지 않고 상품 페이지 링크와 imgid만 적는다.", "",
         "## 1. 요약", "",
         "| 쌍 수 | 판정 일치율 | 한쪽만 기권 | 기권한 쪽이 더 나쁜 사진(흐림 낮음 또는 반사 높음) | 전체 기권율(사진) | 기권 정확도(유용한 기권) | 답한 사진 정확도 |",
         "|---|---|---|---|---|---|---|",
         f"| {s['n']} | {s['agree_rate']:.2f} ({s['agree']}/{s['n']}) | {s['one_sided_rate']:.2f} ({s['one_sided_abstain']}/{s['n']}) | "
         f"{s['abstainer_worse']}/{s['abstainer_worse_n']} | {s['abstain_rate']:.2f} ({s['abstain_images']}/{s['images']}) | "
         f"{ua:.2f} (기권 {na}장) | {aa:.2f} (n={nn}) |", "",
         f"- 판정 일치율 95% 신뢰구간(Wilson) {lo:.2f}~{hi:.2f}. 업로더 구성: 서로 다른 업로더 {len(cache['pairs']) - same}쌍 · 같은 업로더의 다른 사진 {same}쌍(다른 업로더 쌍을 못 찾은 제품). 언어 구성: {lang_txt}(인기순 후보에 프랑스어권 제품이 많았다).",
         f"- 전후 비교(실제 사진 {len(fo)}장): 관문 끄고 무조건 답한다면 정확도 {forced_acc:.2f} -> 관문 켜면 답한 사진 정확도 {aa:.2f} (기권 {s['abstain_images']}장은 답하지 않고 사람에게 넘김).",
         f"- 불일치 {s['n'] - s['agree']}쌍의 내역: 한쪽만 기권 {s['one_sided_abstain']}쌍(사람에게 넘겨 안전) · 둘 다 답했는데 서로 다름 {s['both_answer_diff']}쌍(이쪽이 위험 — 쌍별 표 X 행 참고).",
         "- 판정 일치 = 두 사진의 판정 종류(경고·재촬영 요청·사람 확인·경고 없음)가 같음. 판정 변화율 = 1 − 일치율.",
         "- 기권 = 재촬영 요청 + 사람 확인. 유용한 기권 = 기권한 사진을 품질 관문·신뢰도 하한을 끄고 억지로 답하게 했다면 틀렸을 비율. 「더 나쁜 사진」은 흐림 지표(라플라시안 분산)가 상대보다 낮거나 반사 비율이 상대보다 높은 쪽.", "",
         "## 2. 쌍별 결과", "",
         "| 상품 | 언어 | 정답 | 사진 A(imgid) 판정 | 흐림 · 반사 | 사진 B(imgid) 판정 | 흐림 · 반사 | 일치 |", "|---|---|---|---|---|---|---|---|"]
    for p in cache["pairs"]:
        a, b = p["ra"], p["rb"]
        L.append(f"| [{p['code']}](https://world.openfoodfacts.org/product/{p['code']}) | {p['lang']} | {'양성' if p['truth_pos'] else '음성'} | "
                 f"{VK[a['kind']]} ({p['a']['imgid']}) | {a['blur']:.0f} · {a['glare']*100:.1f}% | {VK[b['kind']]} ({p['b']['imgid']}) | {b['blur']:.0f} · {b['glare']*100:.1f}% | {'O' if a['kind'] == b['kind'] else 'X'} |")
    sk = cache.get("skipped", {})
    L += ["", "## 3. 제외·한계", "",
          f"- 후보 제품 {len(cache['pairs']) + len(sk)}개를 훑어 {len(cache['pairs'])}쌍을 얻었다. 제외 {len(sk)}개(성분표로 확인된 업로드가 2장 미만 · 후보 업로드 부족 · 다운로드 오류).",
          "- 언어는 영·불·스·독(알레르기 낱말 사전 범위, 인기순 후보). 한국어 제품 쌍은 이번 후보에 넣지 않았다(계획) — 한국어 라벨 결과로 읽으면 안 된다.",
          "- 표본이 작다(쌍 수 위). 비율의 불확실성이 크므로 「이 정도 범위」로만 읽는다."]
    open(out, "w").write("\n".join(L) + "\n")
    return {"summary": s, "abstain_accuracy": ua, "abstain_n": na, "answered_accuracy": aa, "answered_n": nn}
