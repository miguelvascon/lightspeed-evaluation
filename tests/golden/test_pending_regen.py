"""PENDING_REGEN parsing, skip, and the two-PR protocol."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.golden._compare import (
    PendingRegenError,
    apply_pending,
    parse_pending_regen,
    pending_protocol_error,
    pending_summary,
    rate_fixture,
)
from tests.golden.conftest import skip_pending_items

pytestmark = pytest.mark.golden

_LISTED = "ragas_paths RM-15\n"


def _node(case: str | None) -> Any:
    node = SimpleNamespace(case=case, added=[])

    def get_closest_marker(name: str) -> Any:
        if name != "golden_case" or node.case is None:
            return None
        return SimpleNamespace(args=(node.case,))

    def add_marker(marker: Any) -> None:
        node.added.append(marker)

    node.get_closest_marker = get_closest_marker
    node.add_marker = add_marker
    return node


def test_unlisted_diff_still_fails() -> None:
    """A diff on a case that is not listed stays FAIL."""
    rating = rate_fixture({"score": 0.0}, {"score": 1.0}, compare_png=True)
    assert rating == "FAIL"
    pending = {"ragas_paths": "RM-15"}
    assert apply_pending("nlp_metrics", rating, pending) == "FAIL"
    assert apply_pending("ragas_paths", rating, pending) == "PENDING"
    assert apply_pending("nlp_metrics", rating, {}) == "FAIL"


def test_listed_case_is_pending_and_skipped() -> None:
    """A listed case is PENDING and receives a skip marker. Others do not."""
    pending = parse_pending_regen(_LISTED)
    listed = _node("ragas_paths")
    other = _node("nlp_metrics")
    skip_pending_items([listed, other], pending)
    assert pending_summary(pending) == "PENDING: ragas_paths RM-15"
    assert len(listed.added) == 1
    assert listed.added[0].mark.kwargs["reason"] == "PENDING ragas_paths RM-15"
    assert other.added == []


def test_empty_file_is_a_noop() -> None:
    """The checked-in file lists nothing, so no case is skipped."""
    text = Path("tests/golden/PENDING_REGEN").read_text(encoding="utf-8")
    pending = parse_pending_regen(text)
    assert pending == {}
    assert pending_summary(pending) == "PENDING: none"
    node = _node("ragas_paths")
    skip_pending_items([node], pending)
    assert node.added == []
    assert pending_protocol_error("", "", ["src/lightspeed_evaluation/api.py"]) is None


@pytest.mark.parametrize(
    "text",
    [
        "ragas_paths",
        "ragas_paths rm-15",
        "ragas_paths RM-",
        "ragas_paths RM-0",
        "Ragas_paths RM-15",
        "ragas-paths RM-15",
        "ragas_paths RM-15 extra",
        "ragas_paths RM-15\nragas_paths RM-16\n",
    ],
)
def test_malformed_entries_are_rejected(text: str) -> None:
    """Reject a line that is not ``case_id RM-<number>``, and duplicates."""
    with pytest.raises(PendingRegenError):
        parse_pending_regen(text)


def test_protocol_requires_the_regen_pr_next() -> None:
    """A non-empty base blocks every PR except the one that empties the list."""
    base = _LISTED
    unrelated = ["docs/README.md"]
    assert pending_protocol_error(base, base, unrelated) is not None
    regen = [
        "tests/golden/PENDING_REGEN",
        "tests/golden/metrics/expected/ragas_paths/out.json",
    ]
    assert pending_protocol_error(base, "", regen) is None
    with_src = ["tests/golden/PENDING_REGEN", "src/lightspeed_evaluation/api.py"]
    assert pending_protocol_error(base, "", with_src) is not None
    assert pending_protocol_error(base, "nlp_metrics RM-16\n", regen) is not None
    assert pending_protocol_error("not a line\n", "", regen) is not None
