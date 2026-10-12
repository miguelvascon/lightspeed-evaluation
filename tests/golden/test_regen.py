"""Unit tests for the golden-regen guard."""

from __future__ import annotations

import pytest

from tests.golden._regen import regen_guard_blocks

pytestmark = pytest.mark.golden


def test_regen_guard_blocks_mixed_pr() -> None:
    """A PR that touches src and golden artifacts fails unless it is a main sync."""
    mixed = [
        "src/lightspeed_evaluation/api.py",
        "tests/golden/metrics/expected/case/out.json",
    ]
    assert regen_guard_blocks(mixed, []) is True
    assert regen_guard_blocks(mixed, ["main-sync"]) is False
    assert regen_guard_blocks(["src/lightspeed_evaluation/api.py"], []) is False
    assert (
        regen_guard_blocks(["tests/golden/metrics/cassettes/case.jsonl"], []) is False
    )
    assert (
        regen_guard_blocks(
            ["src/lightspeed_evaluation/api.py", "tests/golden/_compare.py"], []
        )
        is False
    )
    fixtures = [
        "src/lightspeed_evaluation/api.py",
        "tests/golden/agent/fixtures/http/body.json",
    ]
    assert regen_guard_blocks(fixtures, []) is True
