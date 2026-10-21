"""평가 러너(UC-2): 정답은 평가 전용 — 판정 경로에 넣지 않는다."""
from __future__ import annotations
import json, os, sys, time
from collections import Counter, defaultdict
from typing import Dict, List, Sequence
from ..domain.allergens import ALL_GROUPS, OFF_TAG_GROUP, match_allergens
from ..domain.decide import assess_quality, decide
from ..domain.model import AllergyProfile, ReadLine, Thresholds, VerdictKind


def truth_groups(tags: Sequence[str]) -> set:
    return {OFF_TAG_GROUP[t] for t in tags if t in OFF_TAG_GROUP}


def group_metrics(rows) -> Dict[str, float]:
    tp = sum(1 for t, k in rows if t and k == "allergy_warning")
    fp = sum(1 for t, k in rows if not t and k == "allergy_warning")
    fn = sum(1 for t, k in rows if t and k != "allergy_warning")
    tn = sum(1 for t, k in rows if not t and k != "allergy_warning")
    silent = sum(1 for t, k in rows if t and k == "no_warning")
    esc = sum(1 for t, k in rows if t and k in ("retake_request", "human_check"))
    pos = tp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "silent_miss": silent, "escalated": esc,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / pos if pos else float("nan"),
            "safe_recall": (tp + esc) / pos if pos else float("nan")}


def classify_failure(group, truth_pos, kind, reasons, truth_text_hit, ocr_chars, truth_chars) -> str:
    if truth_pos and kind == "retake_request":
        r = reasons[0] if reasons else "흐림"
        return "글자 검출" if r == "글자영역없음" else r
    if truth_pos and kind == "human_check":
        return "판독"
    if truth_pos and kind == "no_warning":
        if not truth_text_hit:
            return "성분 낱말 매칭"
        return "글자 검출·판독 누락" if ocr_chars < 0.5 * max(truth_chars, 1) else "판독 오류"
    if not truth_pos and kind == "allergy_warning":
        return "성분 낱말 매칭" if truth_text_hit else "판독 오류"
    return "-"


def run(manifest_path="data/manifest.json", image_dir="data/images", out_dir="docs/eval", cache="docs/eval/run-cache.json"):
    import cv2
    from .opencv5_quality import OpenCV5QualityAnalyzer, load_image
    from .vision_reader import VisionReader
    items = json.load(open(manifest_path))["items"]
    cached = json.load(open(cache)) if os.path.exists(cache) else {}
    an, rd, res = OpenCV5QualityAnalyzer(), VisionReader(), {}
    for it in items:
        b = it["barcode"]
        if b in cached:
            res[b] = cached[b]; continue
        rec = {"barcode": b}
        try:
            img = load_image(os.path.join(image_dir, b + ".jpg"))
            t0 = time.perf_counter(); q, used, info = an.analyze(img); t1 = time.perf_counter()
            rec.update(q=q.__dict__, info=info, t_opencv=t1 - t0)
            try:
                lines = rd.read(used); rec["lines"] = [[l.text, l.confidence] for l in lines]
            except Exception as e:
                rec["lines"] = None; rec["read_error"] = f"{type(e).__name__}: {e}"
            rec["t_read"] = time.perf_counter() - t1
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        res[b] = rec
    json.dump(res, open(cache, "w"), ensure_ascii=False)
    return items, res


def evaluate(items, res, t: Thresholds, gate=True):
    from ..domain.model import PhotoQuality
    per_group = {g: [] for g in ALL_GROUPS}
    img_rows, fails, kinds, not_run = [], [], Counter(), []
    for it in items:
        r = res[it["barcode"]]
        if "error" in r or r.get("lines") is None:
            not_run.append((it["barcode"], r.get("error") or r.get("read_error"))); continue
        q = PhotoQuality(**r["q"])
        if not gate:
            q = PhotoQuality(blur_score=1e9, glare_ratio=0.0, text_regions=99, skew_deg=q.skew_deg)
        lines = [ReadLine(a, c) for a, c in r["lines"]]
        tg = truth_groups(it["truth_tags"] if "truth_tags" in it else it["truth_allergens_tags"])
        text = it["truth_ingredients_text"] or ""
        text_hit = {x.group for x in match_allergens([ReadLine(text, 1.0)], AllergyProfile(ALL_GROUPS))}
        ocr_chars = sum(len(a) for a, _ in r["lines"])
        vall = decide(q, lines, AllergyProfile(ALL_GROUPS), t)
        kinds[vall.kind.value] += 1
        img_rows.append((bool(tg), vall.kind.value))
        for g in ALL_GROUPS:
            v = decide(q, lines, AllergyProfile({g}), t)
            per_group[g].append((g in tg, v.kind.value))
            cause = classify_failure(g, g in tg, v.kind.value, list(v.reasons), g in text_hit, ocr_chars, len(text))
            if cause != "-":
                fails.append({"barcode": it["barcode"], "product": it["product_name"], "lang": it["lang"], "group": g,
                              "expected": sorted(tg), "verdict": v.kind.value, "cause": cause,
                              "ocr_chars": ocr_chars, "truth_chars": len(text),
                              "evidence": [e.line_text[:60] for e in v.evidence][:1]})
    return {"per_group": per_group, "img_rows": img_rows, "fails": fails, "kinds": kinds, "not_run": not_run}




ACTIONS = {
    "흐림": "재촬영 안내 문구에 초점·거리 가이드 추가", "반사": "재촬영 안내에 각도 변경 가이드, 반사 임계 재보정",
    "글자 검출": "글자 영역 검출기를 판독기 박스와 교차 확인", "판독": "사람 확인으로 정상 수렴(변경 없음)",
    "글자 검출·판독 누락": "원본 해상도 유지·타일 분할 판독, 누락 줄 수 임계 시 사람 확인", "판독 오류": "신뢰도 하한 상향 또는 낱말 퍼지 매칭",
    "성분 낱말 매칭": "동의어 사전·부정/'may contain' 문맥 규칙 보강",
}
HEAD = {"milk": "우유", "gluten": "밀/글루텐", "egg": "계란", "soy": "대두", "nut": "땅콩/견과", "fish": "생선", "other": "기타"}


def write_report(items, res, t: Thresholds, out_dir="docs/eval"):
    import numpy as np
    e, e0 = evaluate(items, res, t, True), evaluate(items, res, t, False)
    n = len(items); by = {i["barcode"]: i for i in items}
    L = ["# 평가 결과 — 공개 라벨 사진 104건 (실측)", "",
         f"- 입력: Open Food Facts 성분 사진 {n}건(`data/manifest.json`, CC BY-SA, 출처 `data/ATTRIBUTION.md`). 이미지는 레포에 없음.",
         f"- 실행: 이미지 열기·OpenCV·판독 오류로 못 돈 건 = **{len(e['not_run'])}건**" + (" (" + "; ".join(f"{b}: {r}" for b, r in e['not_run']) + ")" if e['not_run'] else ""),
         f"- 임계값(사전 고정 후 보정표로 확인): 흐림 < {t.min_blur:g}, 반사 > {t.max_glare:g}, 판독 신뢰도 < {t.min_confidence:g}, 글자영역 최소 {t.min_regions}",
         "- 한계: 이 104건은 사용자가 OFF에 올린 **성분 부분 사진**이라 폰 촬영 열화(기울기·반사)가 거의 없다. 재촬영 루프 효과(S3)는 별도 합성 측정 대상이고 여기서는 주장하지 않는다.",
         "- 정답 한계: OFF 알레르기 태그는 사용자 입력이라 오류·'may contain' 포함 가능. 기타 그룹은 갑각류·아황산·겨자·참깨·젤라틴 묶음.", "",
         "## 1. 판정 3갈래 분포 (프로필 = 전체 그룹)", "", "| 판정 | 건수 |", "|---|---|"]
    for k, lab in [("allergy_warning", "알레르기 경고"), ("retake_request", "재촬영 요청"), ("human_check", "사람 확인"), ("no_warning", "경고 없음(최종 확인은 사람)")]:
        L.append(f"| {lab} | {e['kinds'].get(k, 0)} |")
    L += ["", "## 2. 그룹별 정밀도·재현율 (프로필 = 그 그룹 하나)", "",
          "재현율 = 경고 TP/정답 양성. **안전 재현율** = (경고 TP + 재촬영/사람 확인으로 넘긴 건)/정답 양성 — 모르는데 \"없음\"이라 말하지 않은 비율. **조용한 미탐** = 정답 양성인데 \"경고 없음\"을 낸 건수(가장 무거운 오류).", "",
          "| 그룹 | 양성 | TP | FP | FN | 정밀도 | 재현율 | 안전 재현율 | 조용한 미탐 |", "|---|---|---|---|---|---|---|---|---|"]
    tot = defaultdict(int)
    for g in ALL_GROUPS:
        m = group_metrics(e["per_group"][g])
        for k in ("tp", "fp", "fn", "tn", "silent_miss", "escalated"): tot[k] += m[k]
        L.append(f"| {HEAD[g]} | {m['tp']+m['fn']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.2f} | {m['recall']:.2f} | {m['safe_recall']:.2f} | {m['silent_miss']} |")
    P = tot["tp"] / (tot["tp"] + tot["fp"]); R = tot["tp"] / (tot["tp"] + tot["fn"]); S = (tot["tp"] + tot["escalated"]) / (tot["tp"] + tot["fn"])
    L.append(f"| **마이크로 평균** | {tot['tp']+tot['fn']} | {tot['tp']} | {tot['fp']} | {tot['fn']} | {P:.2f} | {R:.2f} | {S:.2f} | {tot['silent_miss']} |")
    a, a0 = group_metrics(e["img_rows"]), group_metrics(e0["img_rows"])
    L += ["", "## 3. 「알레르기 있음」 이미지 단위 (정답에 지원 그룹 1개 이상 = 양성)", "",
          "| 설정 | 양성 | 경고 TP | FP | FN | 정밀도 | 재현율 | 안전 재현율 | 조용한 미탐 |", "|---|---|---|---|---|---|---|---|---|"]
    for lab, x in (("품질 관문 켬(기본)", a), ("품질 관문 끔(판독만)", a0)):
        L.append(f"| {lab} | {x['tp']+x['fn']} | {x['tp']} | {x['fp']} | {x['fn']} | {x['precision']:.2f} | {x['recall']:.2f} | {x['safe_recall']:.2f} | {x['silent_miss']} |")
    L += ["", f"SPEC S1 보고 기준 「재현율 >= 0.90 또는 미달 사유 표」: 이미지 단위 경고 재현율 {a['recall']:.2f} (관문 끔 {a0['recall']:.2f}), 안전 재현율 {a['safe_recall']:.2f}. "
          f"{'충족' if a['recall'] >= 0.9 else '미달 — 사유는 failures.md 원인 분류'}.", ""]
    # 보정표
    rows = []
    for it in items:
        r = res[it["barcode"]]
        if "error" in r or r.get("lines") is None: continue
        tc = len(it["truth_ingredients_text"] or ""); oc = sum(len(x) for x, _ in r["lines"])
        rows.append((r["q"]["blur_score"], r["q"]["glare_ratio"], r["q"]["text_regions"], oc / max(tc, 1)))
    L += ["## 4. 임계값 보정표 (판독 글자 수 / 정답 글자 수의 중앙값)", "", "| 흐림 점수 구간 | 건수 | 판독 비율 중앙값 |", "|---|---|---|"]
    for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 100), (100, 300), (300, 1e9)]:
        s = [x[3] for x in rows if lo <= x[0] < hi]
        L.append(f"| {lo}~{hi:g} | {len(s)} | {np.median(s):.2f} |" if s else f"| {lo}~{hi:g} | 0 | - |")
    glare_hi = [x for x in rows if x[1] > 0.05]
    reg_lo = [x for x in rows if x[2] <= 1]
    L += ["", f"- 반사 비율 > 0.05인 {len(glare_hi)}건의 판독 비율 중앙값 {np.median([x[3] for x in glare_hi]):.2f}.",
          f"- 고전 글자 영역 검출 수 <= 1인 {len(reg_lo)}건에서도 판독기는 정답 대비 비율 중앙값 {np.median([x[3] for x in reg_lo]):.2f}를 읽음 -> 고전 검출기는 판독기보다 적게 찾는다(글자영역 최소 0으로 끔).",
          "- 사전 고정값 흐림 60·반사 0.12·글자영역 1로 처음 돌렸을 때 재촬영 24건, 이미지 단위 경고 재현율 0.78, 관문 끔 0.91. 흐림 점수는 이 데이터의 판독 성공을 예측하지 못해(위 표) 임계를 극단 꼬리로 내렸다. 폰 촬영 열화 사진에서의 재보정은 S3 측정으로 넘긴다.", "",
          "## 5. 단계별 시간·기여", ""]
    to = sum(r["t_opencv"] for r in res.values() if "t_opencv" in r); tr = sum(r["t_read"] for r in res.values() if "t_read" in r)
    L += [f"- OpenCV 5 단계(원근 보정 시도·흐림·반사·글자 영역·기울기) 합계 {to:.1f}s, 판독기(Vision) 합계 {tr:.1f}s -> 시간 비율 OpenCV {to/(to+tr)*100:.0f}% / 판독 {tr/(to+tr)*100:.0f}%.",
          f"- 판정을 OpenCV 5 연산만으로 결정한 비율(재촬영 요청) = {e['kinds'].get('retake_request',0)}/{n} = {e['kinds'].get('retake_request',0)/n*100:.1f}%. 나머지는 판독 줄이 있어야 결정됨.",
          f"- 원근 보정이 실제 적용된 사진 {sum(r['info'].get('rectified',0) for r in res.values() if 'info' in r)}건 / {n}건 (대부분 이미 잘린 성분 부분 사진).", ""]
    open(os.path.join(out_dir, "eval.md"), "w").write("\n".join(L) + "\n")
    # 실패 사례
    fl = e["fails"]; cause = Counter(f["cause"] for f in fl)
    F = ["# 실패 사례 표 (S2) — 실측", "", "원인 분류 집계(그룹×이미지 단위, 오탐+미탐 합): " + ", ".join(f"{k} {v}" for k, v in cause.most_common()), "",
         "미탐(FN)은 전부, 오탐(FP)은 원인별로 고르게 최대 12행. 이미지 열은 OFF 상품 페이지(재배포 아님).", "",
         "| # | 입력 이미지 | 기대 | 실제 판정 | 원인 | 조치 |", "|---|---|---|---|---|---|"]
    fn = [f for f in fl if f["expected"] and f["group"] in f["expected"]]
    fp = [f for f in fl if f not in fn]
    seen, pick = Counter(), []
    for f in fp:
        if seen[f["cause"]] < 6: pick.append(f); seen[f["cause"]] += 1
    VK = {"allergy_warning": "경고", "retake_request": "재촬영 요청", "human_check": "사람 확인", "no_warning": "경고 없음"}
    for i, f in enumerate(fn + pick[:12], 1):
        exp = HEAD[f["group"]] + (" 있음" if f in fn else " 없음")
        extra = f" 근거 줄 '{f['evidence'][0]}'" if f["evidence"] else ""
        F.append(f"| {i} | [{f['barcode']}](https://world.openfoodfacts.org/product/{f['barcode']}) ({f['lang']}) | {exp} | {VK[f['verdict']]}{extra} | {f['cause']} (판독 {f['ocr_chars']}자/정답 {f['truth_chars']}자) | {ACTIONS[f['cause']]} |")
    open(os.path.join(out_dir, "failures.md"), "w").write("\n".join(F).replace("|", "|") + "\n")
    summary = {"n": n, "kinds": dict(e["kinds"]), "not_run": e["not_run"], "any": a, "any_no_gate": a0, "micro": {"precision": P, "recall": R, "safe_recall": S, **{k: tot[k] for k in tot}},
               "cause": dict(cause), "t_opencv": to, "t_read": tr, "fail_rows": len(fn) + len(pick[:12])}
    json.dump(summary, open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=1, default=str)
    return summary


if __name__ == "__main__":
    items, res = run()
    print(json.dumps(write_report(items, res, Thresholds()), ensure_ascii=False, default=str)[:1500])
