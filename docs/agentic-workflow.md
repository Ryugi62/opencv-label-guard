# Agentic Vision evidence: perception, decision, action

The "agent" is a deterministic policy, not a language model. No generative model is called on the judging path.

| Stage | Component | Output |
|---|---|---|
| Perception | OpenCV 5 quality analysis (rectification, blur, glare, text regions) and OpenCV 5 DNN text reader | quality numbers, read lines with confidence |
| Decision | `label_guard/domain/decide.py` (explicit rules, thresholds blur 15, glare 0.4, confidence 0.6) | one of four verdicts |
| Action | the JSON response and page text | retake request with how-to, allergy warning with evidence lines, human check, or "no warning"; every response ends with "final check is by a person" |

## OpenCV 5 output changes a later action (live public URL, 2026-10-26)

| # | Photo | OpenCV 5 output | Later action | Source |
|---|---|---|---|---|
| 1 | 0014100098423 blurred | blur score below 15 | retake request, reader not called (0.41 s) | docs/eval/endpoint-latency.md, report 6.2 |
| 2 | same label, sharp | quality passes, 20 text polygons read | allergy warning (gluten, milk, soy) | same |
| 3 | 8801115381518 blurred then sharp | blur gate, then read | retake request (0.2 s), then allergy warning (milk) | same |
| 4 | 0888989930170 | quality passes, reader mean confidence 0.13 | human check: "I cannot say the allergen is absent" | docs/eval/live-human-check.json |
| 5 | 0078742297071 | quality passes, reader mean confidence below 0.6 | human check | docs/eval/live-human-check.json |

## Human control (approval point)

When reading is uncertain (mean confidence below 0.6 or no lines), the policy never says "none": it returns `human_check` and asks the person to read the original label. This is the request for human approval named in the rules. The photo is not stored.

## Failure handling

| Failure | What the system does | Where |
|---|---|---|
| Photo blurred, glared or without text regions | retake request naming the reason and how to fix it; reader skipped | `decide.py` |
| Reader raises an exception | verdict becomes `human_check` and the trace shows the reader error (fail-safe, never "none") | `application/judge.py` |
| Reader confidence too low or no lines read | `human_check` | `decide.py` |
| Image undecodable, missing, unknown allergen group | HTTP 400 with an English message | `infrastructure/lambda_handler.py` |
| Image above 6 MB | HTTP 413 | same |
| More than 5 concurrent requests | HTTP 429, the page shows "The demo is busy, try again in a few seconds" | same, `template.yaml` reserved concurrency 5 |

## Observability

Every response carries a trace tagged `[OpenCV5]`, `[OpenCV5-DNN]` or `[rules]` with the numbers behind each step and `latency_ms`; CloudWatch Logs keep one line per request (verdict kind, latency, group count; no image). Alarms for Errors and Throttles are defined in `template.yaml` but not created in the live account.

## Measured task success

Photo-level warning recall 0.87 on 104 public photos, retake pairs and real pairs in `docs/eval/`; unfavourable results (useful abstention 0.26, weak glare detection) are in `docs/limits-and-responsible.md`.
