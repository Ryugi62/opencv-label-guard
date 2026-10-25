import json
from label_guard.adapters import retake_linux as rl


def test_reader_kind_is_stamped_in_cache_and_results(tmp_path):
    o = {k: 0.5 for k in ("flip_rate", "abstain_rate_degraded", "abstain_rate_reshot", "abstain_useful_rate",
                          "answered_accuracy_degraded", "answered_accuracy_reshot")}
    o.update(n=1, traces_abstain_to_answer=0, abstain_degraded=0, silent_miss_degraded=0, silent_miss_reshot=0)
    out = rl.write_outputs({"overall": o, "by": {}}, {"overall": o, "by": {}},
                           reader_kind="opencv5-dnn", out_dir=str(tmp_path))
    d = json.load(open(tmp_path / "retake-linux-results.json"))
    assert d["reader"] == "opencv5-dnn" and "opencv5-dnn" in open(tmp_path / "retake-linux.md").read()


def test_cache_with_other_reader_is_rejected(tmp_path):
    p = tmp_path / "c.json"; json.dump({"_reader": "macos-vision"}, open(p, "w"))
    try:
        rl.load_cache(str(p), "opencv5-dnn"); assert False
    except ValueError:
        pass
