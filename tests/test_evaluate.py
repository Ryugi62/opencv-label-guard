from label_guard.adapters.evaluate import truth_groups, group_metrics, classify_failure

def test_truth_groups_maps_off_tags_and_ignores_junk():
    assert truth_groups(["en:milk", "en:peanuts", "en:nuts", "en:Well", "en:chicken"]) == {"milk", "nut"}

def test_group_metrics_counts_silent_miss_and_escalation():
    # (truth positive, verdict kind)
    rows = [(True, "allergy_warning"), (True, "no_warning"), (True, "human_check"), (False, "allergy_warning"), (False, "no_warning")]
    m = group_metrics(rows)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 2, 1)
    assert m["silent_miss"] == 1 and m["escalated"] == 1
    assert m["precision"] == 0.5 and abs(m["recall"] - 1 / 3) < 1e-9 and abs(m["safe_recall"] - 2 / 3) < 1e-9

def test_classify_retake_reason_maps_to_cause():
    assert classify_failure("milk", True, "retake_request", ["흐림"], truth_text_hit=True, ocr_chars=0, truth_chars=100) == "흐림"
    assert classify_failure("milk", True, "retake_request", ["글자영역없음"], True, 0, 100) == "글자 검출"

def test_classify_silent_miss_dictionary_vs_ocr():
    assert classify_failure("milk", True, "no_warning", [], truth_text_hit=False, ocr_chars=90, truth_chars=100) == "성분 낱말 매칭"
    assert classify_failure("milk", True, "no_warning", [], truth_text_hit=True, ocr_chars=20, truth_chars=100) == "글자 검출·판독 누락"
    assert classify_failure("milk", True, "no_warning", [], truth_text_hit=True, ocr_chars=95, truth_chars=100) == "판독 오류"

def test_classify_false_alarm():
    assert classify_failure("milk", False, "allergy_warning", [], truth_text_hit=True, ocr_chars=90, truth_chars=100) == "성분 낱말 매칭"
    assert classify_failure("milk", False, "allergy_warning", [], truth_text_hit=False, ocr_chars=90, truth_chars=100) == "판독 오류"
