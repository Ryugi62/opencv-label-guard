"""UC-3 재촬영 비교 러너: 합성 열화 사진 판정 vs 재촬영(원본) 판정. 합성이며 실제 폰 촬영이 아니다."""
from __future__ import annotations
import json, os
from collections import Counter
from ..application.retake_metrics import PairRow, summarize_pairs
from ..domain.allergens import ALL_GROUPS
from ..domain.decide import decide
from ..domain.model import AllergyProfile, PhotoQuality, ReadLine, Thresholds
from .degrade import KINDS, degrade
from .evaluate import truth_groups

LEVELS = (1, 2, 3)
FORCED = Thresholds(min_blur=0.0, max_glare=2.0, min_regions=0, min_confidence=0.0)
KO = {"blur": "흐림", "glare": "반사", "tilt": "기울임", "lowres": "저해상도"}
VK = {"allergy_warning": "경고", "retake_request": "재촬영 요청", "human_check": "사람 확인", "no_warning": "경고 없음"}


def run(manifest="data/manifest.json", image_dir="data/images", base_cache="docs/eval/run-cache.json",
        cache="docs/eval/retake-cache.json"):
    from .opencv5_quality import OpenCV5QualityAnalyzer, load_image
    from .vision_reader import VisionReader
    items = json.load(open(manifest))["items"]
    cached = json.load(open(cache)) if os.path.exists(cache) else {}
    an, rd = OpenCV5QualityAnalyzer(), VisionReader()
    for it in items:
        b = it["barcode"]
        img = None
        for kind in KINDS:
            for lv in LEVELS:
                key = f"{b}|{kind}|{lv}"
                if key in cached:
                    continue
                try:
                    img = img if img is not None else load_image(os.path.join(image_dir, b + ".jpg"))
                    q, used, info = an.analyze(degrade(img, kind, lv))
                    try:
                        lines = [[l.text, l.confidence] for l in rd.read(used)]
                    except Exception as e:
                        lines = None
                    cached[key] = {"q": q.__dict__, "lines": lines}
                except Exception as e:
                    cached[key] = {"error": f"{type(e).__name__}: {e}"}
        json.dump(cached, open(cache, "w"), ensure_ascii=False)
    return items, json.load(open(base_cache)), cached


def build_rows(items, base, cached, t: Thresholds, kind=None, level=None):
    rows, ex = [], []
    prof = AllergyProfile(ALL_GROUPS)
    for it in items:
        b = it["barcode"]; r0 = base[b]
        if "error" in r0 or r0.get("lines") is None:
            continue
        tg = bool(truth_groups(it["truth_allergens_tags"] if "truth_allergens_tags" in it else it["truth_tags"]))
        reshot = decide(PhotoQuality(**r0["q"]), [ReadLine(a, c) for a, c in r0["lines"]], prof, t)
        for k in KINDS:
            for lv in LEVELS:
                if (kind and k != kind) or (level and lv != level):
                    continue
                c = cached.get(f"{b}|{k}|{lv}")
                if not c or "error" in c or c.get("lines") is None:
                    continue
                q, lines = PhotoQuality(**c["q"]), [ReadLine(a, x) for a, x in c["lines"]]
                d = decide(q, lines, prof, t)
                forced = decide(q, lines, prof, FORCED)
                row = PairRow(d.kind.value, reshot.kind.value, tg, (forced.kind.value == "allergy_warning") == tg)
                rows.append(row)
                ex.append((b, it["product_name"], k, lv, tg, d, reshot, forced))
    return rows, ex


def write_report(items, base, cached, t: Thresholds, out_dir="docs/eval"):
    L = ["# 재촬영 루프 효과 (S3 · UC-3) — 합성 열화 실측", "",
         "- **합성이다**: 공개 라벨 사진 104건에 결정적 열화(흐림·반사·기울임·저해상도 각 강도 1·2·3)를 가해 「재촬영 전」 사진을 만들고, 원본을 「재촬영 후」 사진으로 본다. 실제 폰 촬영·사람의 재촬영 행동이 아니다 — 폰 실사진 쌍은 아직 없음(계획).",
         "- 프로필 = 전체 그룹(이미지 단위 「알레르기 있음」). 임계값은 평가 보고와 동일(흐림 <%g, 반사 >%g, 신뢰도 <%g) — 이 열화로 다시 보정하지 않았다." % (t.min_blur, t.max_glare, t.min_confidence),
         "- 정의: 판정 변화율 = 열화본과 재촬영본의 판정 종류(경고·재촬영 요청·사람 확인·경고 없음)가 다른 쌍 / 전체 쌍. 기권 = 재촬영 요청 + 사람 확인. 유용한 기권 = 기권한 열화 사진 중 억지로 답했다면(품질 관문·신뢰도 하한 끔) 틀렸을 비율. 조용한 미탐 = 정답 양성인데 「경고 없음」.", ""]
    rows, _ = build_rows(items, base, cached, t)
    s = summarize_pairs(rows)
    L += ["## 1. 전체 (열화 4종 × 강도 3 × 104건)", "",
          "| 쌍 수 | 판정 변화율 | 기권율 열화 | 기권율 재촬영 | 기권→답 전환 | 유용한 기권 | 답한 건 정확도 열화 | 같은 쌍 재촬영 후 | 조용한 미탐 열화 | 재촬영 후 |",
          "|---|---|---|---|---|---|---|---|---|---|",
          f"| {s['n']} | {s['flip_rate']:.2f} | {s['abstain_rate_degraded']:.2f} | {s['abstain_rate_reshot']:.2f} | {s['traces_abstain_to_answer']} | "
          f"{s['abstain_useful_rate']:.2f} (기권 {s['abstain_degraded']}건) | {s['answered_accuracy_degraded']:.2f} (n={s['answered_n_degraded']}) | {s['answered_accuracy_reshot']:.2f} | {s['silent_miss_degraded']} | {s['silent_miss_reshot']} |", "",
          "## 2. 열화 종류·강도별", "",
          "| 열화 | 강도 | 쌍 | 판정 변화율 | 기권율(열화) | 유용한 기권 | 답한 건 정확도 | 조용한 미탐 |", "|---|---|---|---|---|---|---|---|"]
    out = {"overall": s, "by": {}}
    for k in KINDS:
        for lv in LEVELS:
            rs, _ = build_rows(items, base, cached, t, k, lv)
            m = summarize_pairs(rs)
            if not m["n"]:
                continue
            out["by"][f"{k}|{lv}"] = m
            L.append(f"| {KO[k]} | {lv} | {m['n']} | {m['flip_rate']:.2f} | {m['abstain_rate_degraded']:.2f} | {m['abstain_useful_rate']:.2f} (n={m['abstain_degraded']}) | {m['answered_accuracy_degraded']:.2f} | {m['silent_miss_degraded']} |")
    _, ex = build_rows(items, base, cached, t)
    tr = [e for e in ex if e[5].kind.value in ("retake_request", "human_check") and e[6].kind.value in ("allergy_warning", "no_warning")]
    fl = [e for e in ex if e[5].kind.value == "no_warning" and e[4]]
    L += ["", f"## 3. 재촬영 요청 → 행동이 바뀐 트레이스 (AC-10, 총 {len(tr)}건 중 최대 5건)", "",
          "| 상품 | 열화 | 열화 사진 판정 | 단계 근거 | 재촬영 후 판정 |", "|---|---|---|---|---|"]
    for b, nm, k, lv, tg, d, rs_, f in tr[:5]:
        L.append(f"| [{b}](https://world.openfoodfacts.org/product/{b}) | {KO[k]} {lv} | {VK[d.kind.value]} | {d.steps[-1].detail} | {VK[rs_.kind.value]} |")
    L += ["", f"## 4. 열화 상태에서 조용히 틀린 사례 (정답 양성인데 「경고 없음」, 총 {len(fl)}건 중 최대 5건) — 한계", "",
          "| 상품 | 열화 | 열화 판정 | 재촬영 후 판정 | 함의 |", "|---|---|---|---|---|"]
    for b, nm, k, lv, tg, d, rs_, f in fl[:5]:
        L.append(f"| [{b}](https://world.openfoodfacts.org/product/{b}) | {KO[k]} {lv} | {VK[d.kind.value]} | {VK[rs_.kind.value]} | 현재 임계가 이 열화를 못 걸러 냄 |")
    L.append("")
    open(os.path.join(out_dir, "retake.md"), "w").write("\n".join(L))
    out["traces"] = len(tr); out["silent_examples"] = len(fl)
    json.dump(out, open(os.path.join(out_dir, "retake-results.json"), "w"), ensure_ascii=False, indent=1, default=str)
    return out


if __name__ == "__main__":
    it, base, ca = run()
    print(json.dumps(write_report(it, base, ca, Thresholds())["overall"], ensure_ascii=False, default=str))
