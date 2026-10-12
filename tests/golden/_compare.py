"""Exact compare plus the Q-2 fallback (SPEC-01 v2.2 §5.1 and §5.3).

``rate_fixture`` returns ``EXACT``, ``ORDER-ONLY``, or ``FAIL``.

``ORDER-ONLY`` means the pinned-order sequences differ and the sorted
sequences match. Bootstrap CI values and PNG bytes may differ in that case.
A fixture whose only difference is CI or PNG rates ``FAIL``. ``passes``
accepts ``ORDER-ONLY`` only when ``golden-order-only-ok`` is among the labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.golden._normalize import normalize_obj

ORDER_ONLY_LABEL = "golden-order-only-ok"
_MAP_KEYS = frozenset({"by_conversation", "by_metric", "by_tag"})
_GRAPH_KEYS = frozenset(
    {"bars", "boxes", "wedges", "heatmap", "texts", "lines", "patches", "collections"}
)
_ROW_KEYS = ("conversation_group_id", "conversation_id", "conversation_group")
_SECTION = re.compile(
    r"^(By Metric|By Conversation|By Tag):\n-+\n(.*?)(?=^(?:By Metric|By Conversation|By Tag):|\Z)",
    re.M | re.S,
)


def png_compare_enabled() -> bool:
    """True on the pinned Python 3.12 leg (``LS_GOLDEN_COMPARE_PNG=1``)."""
    return os.environ.get("LS_GOLDEN_COMPARE_PNG") == "1"


def active_labels() -> list[str]:
    """Labels honored by the golden job. CI sets ``LS_GOLDEN_ORDER_ONLY_OK``."""
    if os.environ.get("LS_GOLDEN_ORDER_ONLY_OK") == "1":
        return [ORDER_ONLY_LABEL]
    return []


def passes(rating: str, labels: Sequence[str]) -> bool:
    """EXACT passes. ORDER-ONLY passes only with the DS label. FAIL does not."""
    if rating == "EXACT":
        return True
    if rating == "ORDER-ONLY":
        return ORDER_ONLY_LABEL in labels
    return False


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=False, separators=(",", ":"), default=str)


def _drop(obj: Any, keys: frozenset[str]) -> Any:
    if isinstance(obj, dict):
        return {
            key: _drop(value, keys) for key, value in obj.items() if key not in keys
        }
    if isinstance(obj, list):
        return [_drop(item, keys) for item in obj]
    return obj


def _conv(row: dict[str, Any]) -> str | None:
    for key in _ROW_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def _sort_key(item: Any) -> str:
    return json.dumps(item, sort_keys=True, default=str)


def _canon_txt(text: str) -> str:
    if not any(
        header + ":" in text for header in ("By Metric", "By Conversation", "By Tag")
    ):
        return text

    def _section(match: re.Match[str]) -> str:
        title = match.group(1)
        chunks = [
            chunk.strip()
            for chunk in re.split(r"\n(?=\S)", match.group(2))
            if chunk.strip()
        ]
        chunks.sort(key=lambda chunk: chunk.split(":", 1)[0])
        body = "\n\n".join(chunks)
        return f"{title}:\n{'-' * len(title)}\n{body}\n\n"

    return _SECTION.sub(_section, text)


def _canon(obj: Any) -> Any:
    """Sort cross-conversation rows, keyed maps, TXT blocks, and graph categories."""
    if isinstance(obj, dict):
        built = {key: _canon(value) for key, value in obj.items()}
        for key in _MAP_KEYS:
            child = built.get(key)
            if isinstance(child, dict):
                built[key] = {name: child[name] for name in sorted(child)}
        for key in _GRAPH_KEYS:
            child = built.get(key)
            if isinstance(child, list):
                built[key] = sorted(child, key=_sort_key)
        return built
    if isinstance(obj, list):
        items = [_canon(item) for item in obj]
        if items and all(
            isinstance(item, dict) and _conv(item) is not None for item in items
        ):
            return sorted(items, key=lambda item: _conv(item) or "")
        return items
    if isinstance(obj, str):
        return _canon_txt(obj)
    return obj


def _prepared(obj: Any) -> Any:
    return normalize_obj(obj)


def rate_fixture(actual: Any, expected: Any, *, compare_png: bool | None = None) -> str:
    """Classify one fixture as ``EXACT``, ``ORDER-ONLY``, or ``FAIL``."""
    use_png = png_compare_enabled() if compare_png is None else compare_png
    left = _prepared(actual)
    right = _prepared(expected)
    if not use_png:
        left = _drop(left, frozenset({"png"}))
        right = _drop(right, frozenset({"png"}))
    if _dump(left) == _dump(right):
        return "EXACT"
    if use_png and _dump(_drop(left, frozenset({"png"}))) == _dump(
        _drop(right, frozenset({"png"}))
    ):
        return "FAIL"
    stripped = frozenset({"png", "confidence_interval"})
    left_s, right_s = _drop(left, stripped), _drop(right, stripped)
    if _dump(_canon(left_s)) != _dump(_canon(right_s)):
        return "FAIL"
    # Q-2: CI/PNG leftovers are excused only by an actual cross-conversation reorder.
    if _dump(left_s) == _dump(right_s):
        return "FAIL"
    return "ORDER-ONLY"


def order_only_digest(actual: Any, expected: Any) -> str:
    """Stable digest of the CI and PNG leftovers for the DS sign-off."""
    blob = _dump({"actual": _keep(actual), "expected": _keep(expected)})
    return hashlib.sha256(blob.encode()).hexdigest()


_CASE_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_RM = re.compile(r"RM-[1-9][0-9]*\Z")
_PENDING_PATH = "tests/golden/PENDING_REGEN"


def match_case_id(value: object) -> str | None:
    """Return ``value`` when it is a ``[a-z][a-z0-9_]*`` case id."""
    if isinstance(value, str) and _CASE_ID.match(value):
        return value
    return None


def _touches_src(paths: Sequence[str]) -> bool:
    return any(path == "src" or path.startswith("src/") for path in paths)


class PendingRegenError(ValueError):
    """A ``PENDING_REGEN`` line is not ``case_id RM-<number>``."""


def parse_pending_regen(text: str) -> dict[str, str]:
    """Parse ``case_id RM-<number>`` lines. Blank lines and comments are empty.

    An empty file is a no-op: no case is pending. A duplicate case id, a
    missing RM number, or any other shape raises ``PendingRegenError``.
    """
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not _CASE_ID.match(parts[0]) or not _RM.match(parts[1]):
            raise PendingRegenError(f"malformed PENDING_REGEN line: {raw!r}")
        case_id, rm_number = parts
        if case_id in found:
            raise PendingRegenError(f"duplicate PENDING_REGEN case: {case_id}")
        found[case_id] = rm_number
    return found


def apply_pending(case_id: str, rating: str, pending: Mapping[str, str]) -> str:
    """Return ``PENDING`` for a listed case. Every other rating stays as rated.

    An unlisted diff therefore stays ``FAIL``. An empty ``pending`` map leaves
    the rating unchanged.
    """
    if case_id in pending:
        return "PENDING"
    return rating


def pending_skip_reason(case_id: str, pending: Mapping[str, str]) -> str | None:
    """Skip reason for a listed case, or ``None`` when the case still blocks."""
    rm_number = pending.get(case_id)
    if rm_number is None:
        return None
    return f"PENDING {case_id} {rm_number}"


def pending_summary(pending: Mapping[str, str]) -> str:
    """Job-summary line. Empty input is ``PENDING: none``."""
    if not pending:
        return "PENDING: none"
    body = ", ".join(f"{case_id} {pending[case_id]}" for case_id in sorted(pending))
    return f"PENDING: {body}"


def pending_protocol_error(
    base_text: str,
    head_text: str,
    pr_paths: Sequence[str],
) -> str | None:
    """Enforce the SPEC-02 §4.3 pair while ``PENDING_REGEN`` is non-empty.

    ``pr_paths`` is the three-dot diff (what this PR itself changes). When the
    base file has entries, this PR must edit that file down to empty and must
    not change ``src/``. A base with no entries is a no-op, including a fix PR
    that adds lines. Malformed text returns the parse error.
    """
    try:
        base = parse_pending_regen(base_text)
        head = parse_pending_regen(head_text)
    except PendingRegenError as exc:
        return str(exc)
    if not base:
        return None
    if _PENDING_PATH not in pr_paths:
        return (
            "PENDING_REGEN is non-empty on main; the next merge must be "
            "the golden-regen PR that empties it"
        )
    if head:
        return "the golden-regen PR must empty PENDING_REGEN"
    if _touches_src(pr_paths):
        return "the golden-regen PR must not change src/"
    return None


def pending_review_warning(
    base_text: str, head_text: str, pr_paths: Sequence[str]
) -> str | None:
    """Return the DS-review warning when a ``src/`` PR adds PENDING entries."""
    base = parse_pending_regen(base_text)
    head = parse_pending_regen(head_text)
    added = {
        case_id: rm_number
        for case_id, rm_number in head.items()
        if base.get(case_id) != rm_number
    }
    if not added or not _touches_src(pr_paths):
        return None
    body = ", ".join(f"{case_id} {added[case_id]}" for case_id in sorted(added))
    return (
        "DS review required: this PR changes src/ and adds PENDING_REGEN "
        f"entries ({body}). The DS is not a GitHub reviewer; this warning "
        "is the review control."
    )


def publish_review_warning() -> None:
    """Print the DS warning and append the golden job-summary block."""
    warning = pending_review_warning(
        Path("/tmp/pending.base").read_text(encoding="utf-8"),
        Path("/tmp/pending.head").read_text(encoding="utf-8"),
        Path("/tmp/pr-changed.txt").read_text(encoding="utf-8").splitlines(),
    )
    if warning is None:
        return
    print(f"::warning::{warning}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write("## DS review required\n\n" + warning + "\n")


def reject_unknown_cases(pending: Mapping[str, str], known: Collection[str]) -> None:
    """Raise ``PendingRegenError`` when an id is not a known golden case."""
    unknown = sorted(set(pending) - set(known))
    if unknown:
        listed = ", ".join(unknown)
        raise PendingRegenError(
            f"unknown PENDING_REGEN case id(s): {listed}. "
            "Every id must be a collected case_id or a fixtures/ directory name"
        )


def _keep(obj: Any) -> Any:
    if isinstance(obj, dict):
        kept: dict[str, Any] = {}
        for key, value in obj.items():
            if key in {"confidence_interval", "png"}:
                kept[key] = value
            else:
                child = _keep(value)
                if child not in ({}, []):
                    kept[key] = child
        return kept
    if isinstance(obj, list):
        return [
            item for item in (_keep(value) for value in obj) if item not in ({}, [])
        ]
    return {}
