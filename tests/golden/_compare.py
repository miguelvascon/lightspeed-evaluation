"""Exact compare plus the Q-2 fallback (SPEC-01 v2.2 §5.1 and §5.3).

``rate_fixture`` returns ``EXACT``, ``ORDER-ONLY``, or ``FAIL``.

``ORDER-ONLY`` means the exact compare failed, the sorted fallback is clean,
and the only leftovers are bootstrap CI values and PNG bytes. A PNG-only
difference on an otherwise exact fixture is ``FAIL`` when PNG bytes are in
the comparison (the 3.12 leg). ``passes`` accepts ``ORDER-ONLY`` only when
``golden-order-only-ok`` is among the labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
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
    if _dump(_canon(_drop(left, stripped))) == _dump(_canon(_drop(right, stripped))):
        return "ORDER-ONLY"
    return "FAIL"


def order_only_digest(actual: Any, expected: Any) -> str:
    """Stable digest of the CI and PNG leftovers for the DS sign-off."""
    blob = _dump({"actual": _keep(actual), "expected": _keep(expected)})
    return hashlib.sha256(blob.encode()).hexdigest()


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
