"""보고서 그림 스크립트 — 연산별 입출력 PNG 쌍과 수치가 나온다(합성 라벨, 네트워크 0)."""
import importlib.util
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mrf", os.path.join(HERE, "..", "scripts", "make_report_figures.py"))
mrf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrf)


def _label(path):
    img = np.full((600, 1000, 3), 235, np.uint8)
    for i in range(9):
        cv2.putText(img, "INGREDIENTS: MILK SUGAR WHEAT FLOUR SALT", (30, 55 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2)
    cv2.imwrite(path, img)


def test_five_classical_op_pairs_with_numbers(tmp_path):
    src = str(tmp_path / "label.png")
    _label(src)
    meta = mrf.make(src, str(tmp_path / "fig"), models_dir=None)
    names = sorted(meta["ops"])
    assert names == ["op1_rectify", "op2_blur", "op3_glare", "op4_text_regions", "op5_skew"]
    for n in names:
        assert os.path.getsize(tmp_path / "fig" / f"{n}_in.png") > 0
        assert os.path.getsize(tmp_path / "fig" / f"{n}_out.png") > 0
        assert meta["ops"][n]["ops"]
    ops = meta["ops"]
    assert ops["op2_blur"]["blur_score_blurred"] < ops["op2_blur"]["blur_score_original"]
    assert ops["op3_glare"]["glare_ratio_input"] > ops["op3_glare"]["glare_ratio_original"]
    assert ops["op4_text_regions"]["text_regions"] >= 6
    assert ops["op1_rectify"]["rectified"] is True
    assert json.load(open(tmp_path / "fig" / "figures.json"))["opencv"].startswith("5.")
