import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_submission as cs  # noqa: E402


def test_readme_numbers_match_evaluation_files_and_no_internal_notes():
    assert cs.problems() == []


def test_internal_note_pattern_catches_leftovers():
    assert cs.INTERNAL.search("see [REPO_URL] and [메인 결정]")
    assert not cs.INTERNAL.search("Live demo (no login)")
