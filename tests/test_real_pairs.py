from label_guard.adapters import real_pairs as rp


def test_raw_url_splits_barcode():
    assert rp.raw_url("3017620422003", 124) == "https://images.openfoodfacts.org/images/products/301/762/042/2003/124.jpg"
    assert rp.raw_url("12345678", 3) == "https://images.openfoodfacts.org/images/products/12345678/3.jpg"


def test_candidates_filter_size_org_and_tiny():
    images = {"1": {"uploader": "u1", "sizes": {"full": {"w": 1500, "h": 2000}}},
              "2": {"uploader": "org-ferrero", "sizes": {"full": {"w": 2000, "h": 2000}}},
              "3": {"uploader": "u3", "sizes": {"full": {"w": 400, "h": 380}}},
              "front_en": {"imgid": "1"}}
    assert [c["imgid"] for c in rp.candidates(images)] == [1]


def test_overlap_counts_truth_tokens():
    truth = "sugar, palm oil, hazelnuts 13%, skimmed milk powder 8.7%, cocoa 7.4%, lecithin"
    assert rp.overlap("SUGAR PALM OIL hazelnuts skimmed MILK powder cocoa lecithin", truth) >= 0.8
    assert rp.overlap("washing instructions keep dry", truth) < 0.2


def test_valid_pair_requires_same_code_different_imgid():
    a = {"code": "1", "imgid": 5}; b = {"code": "1", "imgid": 9}
    assert rp.valid_pair(a, b)
    assert not rp.valid_pair(a, {"code": "1", "imgid": 5})
    assert not rp.valid_pair(a, {"code": "2", "imgid": 9})


def test_summarize_pairs_agreement_and_one_sided_abstain():
    # (kind_a, kind_b, worse_side_by_quality)
    rows = [{"ka": "allergy_warning", "kb": "allergy_warning"},
            {"ka": "retake_request", "kb": "allergy_warning", "blur_a": 5.0, "blur_b": 90.0, "glare_a": 0.0, "glare_b": 0.0},
            {"ka": "no_warning", "kb": "human_check", "blur_a": 80.0, "blur_b": 95.0, "glare_a": 0.0, "glare_b": 0.0}]
    s = rp.summarize(rows)
    assert s["n"] == 3 and s["agree"] == 1
    assert s["one_sided_abstain"] == 2 and s["both_answer_diff"] == 0
    assert s["abstainer_worse"] == 1 and s["abstainer_worse_n"] == 2


def test_write_report_has_rates(tmp_path):
    mk = lambda k, b: {"kind": k, "blur": b, "glare": 0.0, "regions": 5, "forced_ok": k != "retake_request"}
    cache = {"pairs": [{"code": "1", "lang": "en", "truth_pos": True, "a": {"imgid": 1}, "b": {"imgid": 2},
                        "ra": mk("allergy_warning", 90), "rb": mk("retake_request", 3)}], "skipped": {"2": "x"}}
    out = tmp_path / "r.md"
    res = rp.write_report(cache, str(out))
    assert res["summary"]["one_sided_abstain"] == 1 and res["abstain_accuracy"] == 1.0
    t = out.read_text()
    assert "합성이 아니다" in t and "서로 다른 업로드" in t and "1/1" in t
