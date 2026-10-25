"""임계값 조정 표본 / 검증 표본 분할 평가 -> docs/eval/holdout.md.
선정 기준(결과를 보기 전에 고정): 조정 표본의 이미지 단위 경고 재현율 최대, 단 정밀도 >= 0.75, 동률이면 조용한 미탐(이미지) 적은 쪽, 그다음 배포 기본값에 가까운 쪽."""
from __future__ import annotations
import itertools, json, os, random
from ..domain.model import Thresholds
from . import evaluate as ev

SEED = 20261026
MIN_PRECISION = 0.75
BLUR = (5, 10, 15, 30, 60)
GLARE = (0.12, 0.2, 0.3, 0.4, 0.6)
CONF = (0.4, 0.5, 0.6, 0.7, 0.8)


def split_ids(ids, seed=SEED):
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    half = len(ordered) // 2
    return sorted(ordered[:half]), sorted(ordered[half:])


def thresholds_grid():
    return [Thresholds(min_blur=b, max_glare=g, min_regions=0, min_confidence=c) for b, g, c in itertools.product(BLUR, GLARE, CONF)]


def measure(items, res, t):
    e = ev.evaluate(items, res, t, True)
    img = ev.group_metrics(e["img_rows"])
    tp = pos = silent = esc = fp = 0
    for g in e["per_group"]:
        m = ev.group_metrics(e["per_group"][g])
        tp += m["tp"]; pos += m["tp"] + m["fn"]; silent += m["silent_miss"]; esc += m["escalated"]; fp += m["fp"]
    return {"n": len(items), "img_pos": img["tp"] + img["fn"], "img_recall": img["recall"], "img_precision": img["precision"],
            "img_silent": img["silent_miss"], "img_safe_recall": img["safe_recall"],
            "grp_pos": pos, "grp_recall": tp / pos if pos else float("nan"), "grp_silent": silent,
            "grp_safe_recall": (tp + esc) / pos if pos else float("nan"), "grp_fp": fp}


def _dist(t, d=Thresholds()):
    return abs(t.min_blur - d.min_blur) / 60 + abs(t.max_glare - d.max_glare) + abs(t.min_confidence - d.min_confidence)


def tune(items, res, grid):
    rows = []
    for t in grid:
        m = measure(items, res, t)
        rows.append((t, m))
    ok = [(t, m) for t, m in rows if m["img_precision"] >= MIN_PRECISION] or rows
    best = min(ok, key=lambda r: (-round(r[1]["img_recall"], 6), r[1]["img_silent"], _dist(r[0])))
    return best[0], rows


def main(manifest="data/manifest.json", cache="docs/eval/linux/run-cache.json", out="docs/eval/holdout.md"):
    items = json.load(open(manifest))["items"]
    res = json.load(open(cache))
    tid, vid = split_ids([i["barcode"] for i in items])
    pick = lambda ids: [i for i in items if i["barcode"] in set(ids)]
    tune_items, val_items = pick(tid), pick(vid)
    assert not set(tid) & set(vid)
    best, rows = tune(tune_items, {k: res[k] for k in tid}, thresholds_grid())
    dflt = Thresholds()
    cells = {}
    for name, t in (("배포 기본값(전체 104장에서 고른 값)", dflt), ("조정 표본에서만 고른 값", best)):
        cells[name] = {"t": t, "tune": measure(tune_items, res, t), "val": measure(val_items, res, t)}
    json.dump({"seed": SEED, "tune_ids": tid, "val_ids": vid, "best": best.__dict__,
               "cells": {k: {"t": v["t"].__dict__, "tune": v["tune"], "val": v["val"]} for k, v in cells.items()}},
              open(os.path.join(os.path.dirname(out), "holdout-results.json"), "w"), ensure_ascii=False, indent=1)
    f = lambda x: f"{x:.2f}"
    L = ["# 임계값 검증 표본 — 조정 표본과 검증 표본을 나눠 다시 쟀다", "",
         f"- 입력: 공개 라벨 사진 {len(items)}장의 배포 판독기(OpenCV 5 DNN) 캐시(`docs/eval/linux/run-cache.json`). 고정 시드 {SEED}로 섞어 **조정 표본 {len(tid)}장 / 검증 표본 {len(vid)}장**, 겹침 0(시험 `tests/test_holdout.py`).",
         f"- 선정 기준(결과를 보기 전에 고정): 조정 표본에서 이미지 단위 경고 재현율 최대, 정밀도 ≥ {MIN_PRECISION}, 동률이면 조용한 미탐 적은 쪽, 그다음 배포 기본값에 가까운 쪽. 후보 {len(rows)}개 = 흐림 {BLUR} × 반사 {GLARE} × 신뢰도 {CONF}.",
         "- 배포 기본값(흐림 15·반사 0.4·신뢰도 0.6)은 처음 고정값 60·0.12를 이 104장 전체를 본 뒤 낮춘 것이라 **검증 표본도 이미 본 값**이다. 그래서 두 번째 줄(조정 표본에서만 고른 값)이 검증 표본에 대한 정직한 값이다.",
         f"- 검증 표본은 {len(vid)}장·정답 양성 이미지 {cells['조정 표본에서만 고른 값']['val']['img_pos']}장뿐이라 신뢰구간이 넓다. 방향만 읽는다.", "",
         "## 임계값과 두 표본 값 (이미지 단위: 정답에 지원 성분 그룹이 1개 이상 있으면 양성)", "",
         "| 임계값 | 흐림 | 반사 | 신뢰도 | 표본 | 장수 | 경고 재현율 | 정밀도 | 안전 재현율 | 조용한 미탐(이미지) | 그룹×이미지 재현율 | 조용한 미탐(그룹×이미지) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in cells.items():
        for lab in ("tune", "val"):
            m, t = c[lab], c["t"]
            L.append(f"| {name} | {t.min_blur:g} | {t.max_glare:g} | {t.min_confidence:g} | {'조정' if lab == 'tune' else '검증'} | {m['n']} | {f(m['img_recall'])} ({round(m['img_recall']*m['img_pos'])}/{m['img_pos']}) | {f(m['img_precision'])} | {f(m['img_safe_recall'])} | {m['img_silent']} | {f(m['grp_recall'])} ({round(m['grp_recall']*m['grp_pos'])}/{m['grp_pos']}) | {m['grp_silent']} |")
    v, tv = cells["조정 표본에서만 고른 값"]["val"], cells["조정 표본에서만 고른 값"]["tune"]
    L += ["", f"조정 표본 → 검증 표본: 경고 재현율 {f(tv['img_recall'])} → {f(v['img_recall'])}, 그룹×이미지 재현율 {f(tv['grp_recall'])} → {f(v['grp_recall'])}, 조용한 미탐(그룹×이미지) {tv['grp_silent']}건/{tv['n']}장 → {v['grp_silent']}건/{v['n']}장.", ""]
    open(out, "w").write("\n".join(L) + "\n")
    return cells


if __name__ == "__main__":
    c = main()
    print(json.dumps({k: {"t": v["t"].__dict__, "tune": v["tune"], "val": v["val"]} for k, v in c.items()}, ensure_ascii=False, indent=1))
