"""Golden-regen guard (SPEC-01 §9.2).

A pull request that edits ``src/`` and a golden expected file, cassette, or
fixture is rejected. The ``main-sync`` label is the documented exception.
"""

from __future__ import annotations

from collections.abc import Sequence

_ARTIFACTS = frozenset({"expected", "cassettes", "fixtures"})


def regen_guard_blocks(paths: Sequence[str], labels: Sequence[str]) -> bool:
    """True when a PR edits ``src/`` and golden expected, cassettes, or fixtures.

    Pull requests labeled ``main-sync`` are exempt (SPEC-01 §9.2).
    """
    if "main-sync" in labels:
        return False
    touched_src = False
    touched_golden = False
    for path in paths:
        if path == "src" or path.startswith("src/"):
            touched_src = True
        parts = path.split("/")
        if (
            len(parts) >= 3
            and parts[0] == "tests"
            and parts[1] == "golden"
            and _ARTIFACTS.intersection(parts)
        ):
            touched_golden = True
    return touched_src and touched_golden
