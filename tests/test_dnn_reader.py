"""리눅스 판독기 어댑터(OpenCV 5 DNN: DB 검출 + CRNN/SVTR 인식) — TextReader 포트 시험."""
import os
import cv2, numpy as np, pytest
from label_guard.adapters.dnn_reader import ctc_greedy, word_spans, DnnTextReader
from label_guard.domain.model import ReadLine

M = os.path.join(os.path.dirname(__file__), "..", "models")
DET = os.path.join(M, "text_detection_en_ppocrv3_2023may.onnx")
REC = os.path.join(M, "en_PP-OCRv3_rec_infer.onnx")
DICT = os.path.join(M, "en_ppocrv3_dict.txt")
needs_models = pytest.mark.skipif(not (os.path.exists(DET) and os.path.exists(REC)), reason="models/ 가중치 없음(scripts/download_models.py)")


def test_ctc_greedy_collapses_repeats_and_blanks():
    chars = ["blank", "a", "b", " "]
    # 프레임: a a blank a b b -> "aab"
    logits = np.zeros((6, 4), np.float32)
    for t, c in enumerate([1, 1, 0, 1, 2, 2]):
        logits[t, c] = 5.0
    text, conf = ctc_greedy(logits, chars)
    assert text == "aab" and 0.0 < conf <= 1.0


def test_ctc_greedy_all_blank_is_empty_with_zero_conf():
    logits = np.zeros((4, 3), np.float32); logits[:, 0] = 3.0
    assert ctc_greedy(logits, ["blank", "a", " "]) == ("", 0.0)


def test_word_spans_splits_on_gaps_wider_than_ratio_of_height():
    ink = np.zeros(100, bool); ink[5:20] = True; ink[24:40] = True; ink[70:90] = True  # 틈 4px, 30px
    assert word_spans(ink, height=32, gap_ratio=0.35) == [(5, 40), (70, 90)]  # 틈 11.2px 이상만 단어 경계


def test_word_spans_empty_ink_returns_empty():
    assert word_spans(np.zeros(50, bool), height=32) == []


def _label():
    img = np.full((960, 960, 3), 255, np.uint8)
    for i, t in enumerate(["INGREDIENTS: MILK, SUGAR,", "WHEAT FLOUR, SOY LECITHIN", "CONTAINS EGG AND PEANUTS"]):
        cv2.putText(img, t, (40, 120 + i * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
    return img


@needs_models
def test_reader_returns_readlines_with_words_separated():
    lines = DnnTextReader(DET, REC, DICT).read(_label())
    assert lines and all(isinstance(l, ReadLine) and 0.0 <= l.confidence <= 1.0 for l in lines)
    joined = " ".join(l.text for l in lines).lower()
    for w in ("milk", "sugar", "wheat", "flour", "peanuts"):
        assert w in joined, joined


@needs_models
def test_reader_blank_image_gives_no_lines():
    assert DnnTextReader(DET, REC, DICT).read(np.full((400, 600, 3), 255, np.uint8)) == []


def test_pick_best_prefers_alt_only_when_primary_is_unsure_and_alt_is_better():
    from label_guard.adapters.dnn_reader import pick_best
    assert pick_best(("milk", 0.95), ("m1lk", 0.99)) == ("milk", 0.95)      # 주 판독이 확실하면 유지
    assert pick_best(("zzz", 0.5), ("우유", 0.9)) == ("우유", 0.9)           # 불확실 + 보조가 더 좋음
    assert pick_best(("zzz", 0.5), ("abc", 0.4)) == ("zzz", 0.5)             # 보조가 더 나쁘면 유지
    assert pick_best(("", 0.0), ("우유", 0.9)) == ("우유", 0.9)
    assert pick_best(("milk", 0.5), None) == ("milk", 0.5)
