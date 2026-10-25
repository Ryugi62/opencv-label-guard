"""공개 사전 학습 가중치를 models/에 받는다(학습 0). 사용: python3 scripts/download_models.py"""
import hashlib, os, urllib.request
HF = "https://huggingface.co/"
FILES = {
    "text_detection_en_ppocrv3_2023may.onnx": HF + "opencv/text_detection_ppocr/resolve/main/text_detection_en_ppocrv3_2023may.onnx",
    "en_PP-OCRv3_rec_infer.onnx": HF + "SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx",
    "korean_mobile_v2.0_rec_infer.onnx": HF + "SWHL/RapidOCR/resolve/main/PP-OCRv1/korean_mobile_v2.0_rec_infer.onnx",
}
os.makedirs("models", exist_ok=True)
for name, url in FILES.items():
    path = os.path.join("models", name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    print(name, hashlib.sha256(open(path, "rb").read()).hexdigest())
