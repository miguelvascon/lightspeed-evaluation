"""Q-2 ratings, normalizers, and the golden-regen guard."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.golden._compare import (
    ORDER_ONLY_LABEL,
    order_only_digest,
    passes,
    rate_fixture,
)

pytestmark = pytest.mark.golden


def _section(title: str, keys: tuple[str, ...]) -> str:
    body = "".join(f"{key}:\n  Pass: 1 (100.0%)\n\n" for key in keys)
    return f"{title}:\n{'-' * len(title)}\n{body}"


def _row(conv: str, turn: str, score: float, reason: str) -> dict[str, Any]:
    return {
        "conversation_group_id": conv,
        "turn_id": turn,
        "score": score,
        "reason": reason,
    }


def _base() -> dict[str, Any]:
    rows = [
        _row("c1", "t1", 1.0, "ok"),
        _row("c1", "t2", 0.5, "half"),
        _row("c2", "t1", 0.0, "no"),
    ]
    storage = [dict(row, id=index) for index, row in enumerate(rows, start=1)]
    return {
        "rows": rows,
        "storage_rows": storage,
        "summary": {
            "by_conversation": {"c1": {"pass": 2}, "c2": {"pass": 0}},
            "by_metric": {
                "keywords": {
                    "score_statistics": {
                        "mean": 0.5,
                        "confidence_interval": {"low": 0.2, "mean": 0.5, "high": 0.8},
                    }
                }
            },
            "by_tag": {"red": {"pass": 1}},
        },
        "txt": _section("By Metric", ("m_b", "m_a"))
        + _section("By Conversation", ("c2", "c1"))
        + _section("By Tag", ("tag_b", "tag_a")),
        "graph": {
            "bars": [{"key": "m_b", "value": 1}, {"key": "m_a", "value": 2}],
            "boxes": [{"key": "m_b", "value": 1}, {"key": "m_a", "value": 2}],
            "wedges": [{"key": "FAIL", "value": 1}, {"key": "PASS", "value": 2}],
            "heatmap": [{"key": "c2|m", "value": 0}, {"key": "c1|m", "value": 1}],
            "rcparams": {"axes.grid": True},
        },
        "png": "png-a",
        "requests": "POST /v1/chat/completions\n{}",
    }


def _reordered(source: dict[str, Any]) -> dict[str, Any]:
    actual = copy.deepcopy(source)
    actual["rows"] = [actual["rows"][2], actual["rows"][0], actual["rows"][1]]
    actual["storage_rows"] = [
        actual["storage_rows"][2],
        actual["storage_rows"][0],
        actual["storage_rows"][1],
    ]
    summary = actual["summary"]
    summary["by_conversation"] = {
        key: summary["by_conversation"][key]
        for key in reversed(summary["by_conversation"])
    }
    summary["by_metric"]["keywords"]["score_statistics"]["confidence_interval"] = {
        "low": 0.1,
        "mean": 0.4,
        "high": 0.7,
    }
    actual["txt"] = (
        _section("By Metric", ("m_a", "m_b"))
        + _section("By Conversation", ("c1", "c2"))
        + _section("By Tag", ("tag_a", "tag_b"))
    )
    for key in ("bars", "boxes", "wedges", "heatmap"):
        actual["graph"][key] = list(reversed(actual["graph"][key]))
    actual["png"] = "png-b"
    return actual


def test_identical_fixture_is_exact() -> None:
    """Byte-identical fixtures rate EXACT and do not need the DS label."""
    expected = _base()
    rating = rate_fixture(copy.deepcopy(expected), expected, compare_png=True)
    assert rating == "EXACT"
    assert passes(rating, []) is True


def test_cross_conversation_reorder_is_order_only() -> None:
    """Reordered rows, TXT, graphs, CI, and PNG bytes are ORDER-ONLY."""
    expected = _base()
    actual = _reordered(expected)
    rating = rate_fixture(actual, expected, compare_png=True)
    assert rating == "ORDER-ONLY"
    assert passes(rating, []) is False
    assert passes(rating, [ORDER_ONLY_LABEL]) is True
    digest = order_only_digest(actual, expected)
    assert digest == order_only_digest(actual, expected)
    assert len(digest) == 64


@pytest.mark.parametrize("mutate", ["within", "score", "reason", "requests"])
def test_real_change_fails_even_with_label(mutate: str) -> None:
    """Within-conversation order, score, reason, and request bytes stay FAIL."""
    expected = _base()
    actual = _reordered(expected)
    if mutate == "within":
        actual["rows"][1], actual["rows"][2] = actual["rows"][2], actual["rows"][1]
    elif mutate == "score":
        actual["rows"][0]["score"] = 0.25
    elif mutate == "reason":
        actual["rows"][0]["reason"] = "changed"
    else:
        actual["requests"] = actual["requests"] + "x"
    rating = rate_fixture(actual, expected, compare_png=True)
    assert rating == "FAIL"
    assert passes(rating, [ORDER_ONLY_LABEL]) is False


def test_png_only_diff_fails_on_the_pinned_leg() -> None:
    """PNG bytes alone are FAIL when compared, and ignored off the 3.12 leg."""
    expected = _base()
    actual = copy.deepcopy(expected)
    actual["png"] = "png-other"
    assert rate_fixture(actual, expected, compare_png=True) == "FAIL"
    assert (
        passes(rate_fixture(actual, expected, compare_png=True), [ORDER_ONLY_LABEL])
        is False
    )
    assert rate_fixture(actual, expected, compare_png=False) == "EXACT"
