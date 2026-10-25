# opencv-label-guard

Live demo (no login, upload one photo): https://hqadhnbyzbv5dyok3zdnnfpcae0tcozx.lambda-url.ap-northeast-2.on.aws/ · Demo video v2 (unlisted, 136 s, English captions): https://youtu.be/gn5Gw9fM5Zg

Label Guard judges whether a phone photo of a food ingredient label is good enough to read (perspective rectification, blur, glare, text regions with OpenCV 5), and that judgement decides the next action: ask for a retake, read it with an OpenCV 5 DNN text reader and warn about allergens, hand it to a person when the read is uncertain, or say "no warning". It is not medical advice: a person always makes the final check against the printed label. Design and acceptance criteria: [SPEC.md](SPEC.md). Korean README: [docs/README.ko.md](docs/README.ko.md).

## Install

Python 3.12. Pinned versions are in `requirements-lambda.txt` (`opencv-python-headless==5.0.0.93`, `numpy==2.5.4`).

```
python3.12 -m venv .venv
.venv/bin/pip install opencv-python-headless==5.0.0.93 pytest
.venv/bin/python scripts/download_models.py
```

## Reproduce

```
.venv/bin/python -m pytest -q        # 72 unit tests (2 reader tests are skipped when the model weights are missing)
.venv/bin/python scripts/download_images.py
.venv/bin/python -m label_guard.adapters.evaluate linux     # 104-photo evaluation with the deployed reader
.venv/bin/python scripts/measure_endpoint.py <live URL above> 10   # latency of the public endpoint
```

`template.yaml` is the same definition as the live deployment (function, layers, function URL, reserved concurrency 5, logs-only role); `tests/test_sam_template.py` checks it against a read-only snapshot of the deployed configuration (`tests/fixtures/deployed-config.json`, needs PyYAML) and `sam validate --lint` passes. Nothing is deployed from it here.
The CloudWatch alarms (Errors, Throttles) are defined only, with no notification target, and are created when the template is deployed (the live function was made by `scripts/deploy_lambda.py`, so a stack deploy needs a fresh function name or a resource import).
Run `./scripts/build_lambda.sh` before `sam validate` so `dist/*.zip` exist.

Layout: `domain/` (no OpenCV, no I/O) <- `application/` (ports, orchestration) <- `adapters/` (OpenCV 5 quality, OpenCV DNN reader, macOS Vision reader, evaluation) <- `infrastructure/` (AWS Lambda handler). Deployment: [docs/deploy.md](docs/deploy.md).

## Evaluation (measured; same values as the technical report)

| What | Result | Source |
|---|---|---|
| 104 public Open Food Facts photos, deployed reader (OpenCV 5 DNN), 55 positive photos | warning recall 0.87, precision 0.80, safe recall 0.91, 5 silent misses | [docs/eval/eval.md](docs/eval/eval.md) |
| Held-out check: thresholds picked on 52 photos, scored on the other 52 (28 positive; deployed values) | recall 0.86 (24/28), 2 silent photos; per group x photo recall 0.75 (30/40), 8 silent misses | [docs/eval/holdout.md](docs/eval/holdout.md) |
| Per ingredient group x photo (90 positives) | recall 0.74; fish 0.33, egg 0.56 | [docs/eval/retake-linux.md](docs/eval/retake-linux.md) |
| 1,248 synthetic retake pairs (blur, glare, tilt, low resolution x 3 strengths) | verdict change 0.38; abstention 0.38 -> 0.10 after retake; useful abstention 0.26 | [docs/eval/retake-linux.md](docs/eval/retake-linux.md) |
| 42 real pairs (two different Open Food Facts uploads of one label; not one person retaking) | verdict agreement 0.76 (32/42, 95% CI 0.61-0.87); one side abstains 0.14 (6/42); accuracy when answering 0.78 (59/76); 3 of 56 positive photos silently missed | [docs/eval/real-pairs.md](docs/eval/real-pairs.md) |
| Live endpoint, 30 requests (3 photos x 10) | 30/30 HTTP 200; round trip p50 3.36 s, p95 3.75 s; server p50 3.20 s; blurred photo returns a retake request in 0.2-0.4 s | [docs/eval/endpoint-latency.md](docs/eval/endpoint-latency.md) |
| Reader time per photo, M-series laptop (not Lambda) | OpenCV 5 DNN 0.40 s, macOS Vision 0.59 s | [docs/eval/reader-compare.md](docs/eval/reader-compare.md) |

## Limitations

- All 1,248 retake pairs are synthetic degradations; the only real photos are the 42 pairs above (mostly French labels, no Korean pairs). No same-person phone retakes were measured.
- Thresholds (blur 15, glare 0.4, confidence 0.6) were lowered after seeing the 104 photos; the 52/52 split check is small (28 held-out positives), so results may be optimistic.
- Useful abstention is only 0.26 and glare detection is weak; 83 silent misses remain among degraded synthetic photos (60 after retake). Allergen tags from Open Food Facts are crowd-sourced. No user study.
- Details: [docs/limits-and-responsible.md](docs/limits-and-responsible.md), [docs/eval/failures.md](docs/eval/failures.md).

## License

Code: MIT ([LICENSE](LICENSE)). Photos and data come from Open Food Facts (photos CC BY-SA, data ODbL + DbCL); no images are stored in this repository, only product-page links ([data/ATTRIBUTION.md](data/ATTRIBUTION.md)).
