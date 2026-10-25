"""재촬영 1248쌍을 배포와 같은 리눅스 판독기(OpenCV 5 DNN)로 다시 재고 macOS 값과 나란히 적는다 -> docs/eval/retake-linux.md.
합성 열화이며 실제 폰 촬영이 아니다. 캐시·결과에 판독기 종류를 찍는다(섞이면 거부)."""
from __future__ import annotations
import json, os
from ..domain.model import Thresholds
from . import evaluate as ev
from .degrade import KINDS
from .retake_eval import LEVELS, build_rows, KO
from ..application.retake_metrics import summarize_pairs

KIND = "opencv5-dnn"
HEAD = {"milk": "우유", "gluten": "밀/글루텐", "egg": "계란", "soy": "대두", "nut": "땅콩/견과", "fish": "생선", "other": "기타"}


def load_cache(path, reader_kind):
    if not os.path.exists(path):
        return {"_reader": reader_kind}
    c = json.load(open(path))
    if c.get("_reader") != reader_kind:
        raise ValueError(f"캐시 판독기 {c.get('_reader')} != {reader_kind}")
    return c


def run(manifest="data/manifest.json", image_dir="data/images", base_cache="docs/eval/linux/run-cache.json",
        cache="docs/eval/linux/retake-cache.json"):
    from .degrade import degrade
    from .opencv5_quality import OpenCV5QualityAnalyzer, load_image
    items = json.load(open(manifest))["items"]
    cached = load_cache(cache, KIND)
    an, rd = OpenCV5QualityAnalyzer(), ev.linux_reader()
    for n, it in enumerate(items):
        b, img = it["barcode"], None
        for kind in KINDS:
            for lv in LEVELS:
                key = f"{b}|{kind}|{lv}"
                if key in cached:
                    continue
                try:
                    img = img if img is not None else load_image(os.path.join(image_dir, b + ".jpg"))
                    q, used, _ = an.analyze(degrade(img, kind, lv))
                    try:
                        lines = [[l.text, l.confidence] for l in rd.read(used)]
                    except Exception:
                        lines = None
                    cached[key] = {"q": q.__dict__, "lines": lines}
                except Exception as e:
                    cached[key] = {"error": f"{type(e).__name__}: {e}"}
        json.dump(cached, open(cache, "w"), ensure_ascii=False)
    return items, json.load(open(base_cache)), cached


def summarize(items, base, cached, t):
    out = {"overall": summarize_pairs(build_rows(items, base, cached, t)[0]), "by": {}}
    for k in KINDS:
        for lv in LEVELS:
            m = summarize_pairs(build_rows(items, base, cached, t, k, lv)[0])
            if m["n"]:
                out["by"][f"{k}|{lv}"] = m
    return out


def _row(label, a, b, fmt="{:.2f}"):
    f = lambda v: fmt.format(v) if isinstance(v, float) else str(v)
    return f"| {label} | {f(a)} | {f(b)} |"


def write_outputs(mac, lin, reader_kind=KIND, out_dir="docs/eval", real=None):
    ma, li = mac["overall"], lin["overall"]
    L = ["# 재촬영 1248쌍 재측정 — 배포 판독기(리눅스 OpenCV 5 DNN) vs macOS Vision", "",
         f"- 판독기 종류(결과 파일에도 찍힘): **{reader_kind}** = 배포된 Lambda와 같은 DB 검출 + PP-OCRv3 인식, `cv2.dnn`만 사용. 임계값·판정 코드는 macOS 측정과 동일(재보정 없음).",
         "- **합성이다**: 공개 라벨 104건에 결정적 열화(흐림·반사·기울임·저해상도 × 강도 3)를 가한 쌍이며 실제 폰 재촬영이 아니다. 실제 폰 쌍은 계획.", "",
         "## 1. 전체 1248쌍", "", "| 지표 | macOS Vision | 리눅스 DNN (배포) |", "|---|---|---|",
         _row("쌍 수", ma["n"], li["n"]), _row("판정 변화율", ma["flip_rate"], li["flip_rate"]),
         _row("기권율 (열화 사진)", ma["abstain_rate_degraded"], li["abstain_rate_degraded"]),
         _row("기권율 (재촬영 후)", ma["abstain_rate_reshot"], li["abstain_rate_reshot"]),
         _row("기권→답 전환(건)", ma["traces_abstain_to_answer"], li["traces_abstain_to_answer"]),
         _row("유용한 기권", ma["abstain_useful_rate"], li["abstain_useful_rate"]),
         _row("기권 건수", ma["abstain_degraded"], li["abstain_degraded"]),
         _row("답한 건 정확도 (열화)", ma["answered_accuracy_degraded"], li["answered_accuracy_degraded"]),
         _row("답한 건 정확도 (재촬영 후)", ma["answered_accuracy_reshot"], li["answered_accuracy_reshot"]),
         _row("조용한 미탐 (열화)", ma["silent_miss_degraded"], li["silent_miss_degraded"]),
         _row("조용한 미탐 (재촬영 후)", ma["silent_miss_reshot"], li["silent_miss_reshot"]), "",
         "## 2. 열화 종류·강도별 (판정 변화율 / 기권율 / 유용한 기권)", "",
         "| 열화 | 강도 | macOS 변화율 | DNN 변화율 | macOS 기권율 | DNN 기권율 | macOS 유용 | DNN 유용 |", "|---|---|---|---|---|---|---|---|"]
    for key, m in lin["by"].items():
        k, lv = key.split("|"); a = mac["by"].get(key, {})
        g = lambda d, x: f"{d[x]:.2f}" if x in d else "-"
        L.append(f"| {KO[k]} | {lv} | {g(a,'flip_rate')} | {g(m,'flip_rate')} | {g(a,'abstain_rate_degraded')} | {g(m,'abstain_rate_degraded')} | {g(a,'abstain_useful_rate')} | {g(m,'abstain_useful_rate')} |")
    if real:
        L += ["", "## 3. 실사진 104장 — 품질 관문 켬/끔 (리눅스 DNN)", "",
              "| 관문 | 경고 재현율 | 안전 재현율 | 정밀도 | 조용한 미탐(이미지) | 재촬영·사람확인으로 걸러진 양성 |", "|---|---|---|---|---|---|"]
        for name, r in real["gate"].items():
            L.append(f"| {name} | {r['recall']:.2f} | {r['safe_recall']:.2f} | {r['precision']:.2f} | {r['silent_miss']} | {r['escalated']} |")
        L += ["", "성분 그룹별(그룹×이미지 단위, 양성 = 그 그룹 성분이 실제 있음)", "",
              "| 성분 | 양성 수 | 재현율 켬 | 재현율 끔 | 조용한 미탐 켬 | 조용한 미탐 끔 |", "|---|---|---|---|---|---|"]
        for g, r in real["groups"].items():
            L.append(f"| {HEAD.get(g, g)} | {r['pos']} | {r['on_recall']:.2f} | {r['off_recall']:.2f} | {r['on_silent']} | {r['off_silent']} |")
        tot = real["total"]
        L.append(f"| 합계 | {tot['pos']} | {tot['on_recall']:.2f} | {tot['off_recall']:.2f} | {tot['on_silent']} | {tot['off_silent']} |")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "retake-linux.md"), "w").write("\n".join(L) + "\n")
    json.dump({"reader": reader_kind, "macos": mac, "linux": lin, "real": real}, open(os.path.join(out_dir, "retake-linux-results.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    return out_dir


def real_photo_tables(items, res, t):
    on, off = ev.evaluate(items, res, t, gate=True), ev.evaluate(items, res, t, gate=False)
    gate = {}
    for name, e in (("켬(기본)", on), ("끔(판독만)", off)):
        gate[name] = ev.group_metrics(e["img_rows"])
    groups, tp = {}, {"pos": 0, "on_tp": 0, "off_tp": 0, "on_silent": 0, "off_silent": 0}
    for g in on["per_group"]:
        a, b = ev.group_metrics(on["per_group"][g]), ev.group_metrics(off["per_group"][g])
        pos = a["tp"] + a["fn"]
        groups[g] = {"pos": pos, "on_recall": a["recall"], "off_recall": b["recall"], "on_silent": a["silent_miss"], "off_silent": b["silent_miss"]}
        tp["pos"] += pos; tp["on_tp"] += a["tp"]; tp["off_tp"] += b["tp"]; tp["on_silent"] += a["silent_miss"]; tp["off_silent"] += b["silent_miss"]
    total = {"pos": tp["pos"], "on_recall": tp["on_tp"] / tp["pos"], "off_recall": tp["off_tp"] / tp["pos"], "on_silent": tp["on_silent"], "off_silent": tp["off_silent"]}
    return {"gate": gate, "groups": groups, "total": total}


if __name__ == "__main__":
    t = Thresholds()
    items, base, cached = run()
    mac_items = json.load(open("data/manifest.json"))["items"]
    mac = summarize(mac_items, json.load(open("docs/eval/run-cache.json")), json.load(open("docs/eval/retake-cache.json")), t)
    lin = summarize(items, base, cached, t)
    real = real_photo_tables(items, base, t)
    write_outputs(mac, lin, KIND, real=real)
    print(json.dumps({"lin": lin["overall"], "real": real["total"]}, ensure_ascii=False, default=str))
