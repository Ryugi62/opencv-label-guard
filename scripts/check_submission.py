"""Submission consistency checks (README numbers vs evaluation files, one metric definition, no internal notes).
Run: .venv/bin/python scripts/check_submission.py   (exit 0 = all consistent). Also used by tests/test_consistency.py."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SILENT_MISS_DEFINITION = "Silent miss = positive photo answered “no warning”"

# (README fragment that must appear, evaluation file that must contain the number)
NUMBER_CHECKS = [
    ("warning recall 0.87, precision 0.80, safe recall 0.91", "docs/eval/retake-linux.md", "0.87"),
    ("recall 0.86 (24/28)", "docs/eval/holdout.md", "24/28"),
    ("fish 0.33, egg 0.56", "docs/eval/retake-linux.md", "0.56"),
    ("verdict change 0.38", "docs/eval/retake-linux.md", "0.38"),
    ("useful abstention 0.26", "docs/eval/retake-linux.md", "0.26"),
    ("verdict agreement 0.76 (32/42", "docs/eval/real-pairs.md", "32/42"),
    ("round trip p50 3.36 s, p95 3.75 s", "docs/eval/endpoint-latency.md", "3.36"),
]

INTERNAL = re.compile(r"\[\[|projects/|메인 판단|Jarvis|TODO|FIXME|\[REPO_URL\]|\[메인|발송 전|회신 대기|handoff/")


def count_tests() -> int:
    return sum(len(re.findall(r"^\s*def test_", p.read_text(encoding="utf-8"), re.M)) for p in (ROOT / "tests").glob("test_*.py"))


def problems() -> list:
    out = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for frag, doc, needle in NUMBER_CHECKS:
        if frag not in readme:
            out.append(f"README lacks «{frag}»")
        if needle not in (ROOT / doc).read_text(encoding="utf-8"):
            out.append(f"{doc} lacks {needle} (README «{frag}»)")
    m = re.search(r"(\d+) unit tests", readme)
    if not m or int(m.group(1)) != count_tests():
        out.append(f"README test count {m.group(1) if m else None} != {count_tests()} test functions")
    if SILENT_MISS_DEFINITION not in readme:
        out.append("README lacks the silent-miss definition sentence")
    for p in [ROOT / "README.md", ROOT / "SPEC.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "docs/eval").glob("*.md"))]:
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if INTERNAL.search(ln):
                out.append(f"internal note in {p.relative_to(ROOT)}:{i}: {ln[:80]}")
    return out


if __name__ == "__main__":
    ps = problems()
    print("\n".join(ps) if ps else "consistent: 0 problems")
    sys.exit(1 if ps else 0)
