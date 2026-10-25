"""임계값 검증 표본 분할 — 결정적이고 겹침 0이어야 한다."""
import json
from label_guard.adapters.holdout import split_ids, SEED, tune, thresholds_grid
from label_guard.domain.model import Thresholds


def _ids():
    return [i["barcode"] for i in json.load(open("data/manifest.json"))["items"]]


def test_split_deterministic():
    a, b = split_ids(_ids(), SEED), split_ids(list(reversed(_ids())), SEED)
    assert a == b


def test_split_disjoint_and_complete():
    tune_ids, val_ids = split_ids(_ids(), SEED)
    assert not set(tune_ids) & set(val_ids)
    assert set(tune_ids) | set(val_ids) == set(_ids())
    assert len(tune_ids) == len(val_ids) == 52


def test_tune_uses_only_given_items():
    items = json.load(open("data/manifest.json"))["items"]
    res = json.load(open("docs/eval/linux/run-cache.json"))
    tid, vid = split_ids(_ids(), SEED)
    sub = [i for i in items if i["barcode"] in set(tid)]
    best, _ = tune(sub, {k: res[k] for k in tid}, thresholds_grid())
    assert isinstance(best, Thresholds)
