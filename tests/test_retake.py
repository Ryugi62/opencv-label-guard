import numpy as np, cv2
from label_guard.application.retake_metrics import PairRow, summarize_pairs
from label_guard.adapters.degrade import degrade, KINDS
from label_guard.adapters.opencv5_quality import blur_score, glare_ratio


def _text_img():
    img = np.full((600, 900, 3), 255, np.uint8)
    for i in range(12):
        cv2.putText(img, "INGREDIENTS milk wheat soy", (20, 40 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img


def test_degrade_blur_lowers_sharpness_and_is_deterministic():
    a = _text_img()
    b1, b2 = degrade(a, "blur", 2), degrade(a, "blur", 2)
    assert blur_score(b1) < blur_score(a) * 0.5 and (b1 == b2).all()


def test_degrade_glare_raises_saturated_ratio():
    a = _text_img(); a[:] = 120
    assert glare_ratio(degrade(a, "glare", 2)) > glare_ratio(a) + 0.05


def test_degrade_kinds_keep_shape_valid_and_unknown_rejected():
    a = _text_img()
    for k in KINDS:
        assert degrade(a, k, 1).ndim == 3
    try:
        degrade(a, "nope", 1); assert False
    except ValueError:
        pass


def test_summarize_pairs_flip_abstain_and_traces():
    # degraded kind, reshot kind, truth positive, forced answer correct on degraded?
    rows = [
        PairRow("retake_request", "allergy_warning", True, True),
        PairRow("human_check", "no_warning", False, False),
        PairRow("allergy_warning", "allergy_warning", True, True),
        PairRow("no_warning", "allergy_warning", True, False),   # 조용한 미탐 -> 재촬영에서 바로잡힘
    ]
    s = summarize_pairs(rows)
    assert s["n"] == 4 and s["flips"] == 3 and abs(s["flip_rate"] - 0.75) < 1e-9
    assert s["abstain_degraded"] == 2 and s["abstain_reshot"] == 0
    assert s["traces_abstain_to_answer"] == 2
    # 기권 2건 중 강제 답이 틀렸을 건 = 1 -> 유용한 기권 비율 0.5
    assert abs(s["abstain_useful_rate"] - 0.5) < 1e-9
    # 답한 2건(경고 TP, 경고없음 미탐)의 정확도: allergy_warning&positive 맞음, no_warning&positive 틀림
    assert abs(s["answered_accuracy_degraded"] - 0.5) < 1e-9
    assert s["silent_miss_degraded"] == 1


def test_summarize_empty():
    assert summarize_pairs([])["n"] == 0
