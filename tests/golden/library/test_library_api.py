"""F9: ``evaluate_with_summary`` computes intervals only when asked."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lightspeed_evaluation import (
    EvaluationData,
    SystemConfig,
    TurnData,
    evaluate_with_summary,
)

pytestmark = pytest.mark.golden

_EXPECTED = Path(__file__).parent / "expected" / "seeded_intervals.json"


def _config() -> SystemConfig:
    raw: dict[str, Any] = {
        "core": {
            "max_threads": 1,
            "cache_enabled": False,
            "cache_base_dir": ".caches",
        },
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
        "agents": {"enabled": False},
        "logging": {"source_level": "WARNING", "package_level": "WARNING"},
        "default_turn_metrics_metadata": {
            "custom:keywords_eval": {"description": "keywords", "default": True}
        },
    }
    return SystemConfig.model_validate(raw)


def _data() -> list[EvaluationData]:
    shared = {
        "turn_id": "t1",
        "query": "What is a namespace?",
        "response": "A namespace provides isolation.",
        "turn_metrics": ["custom:keywords_eval"],
    }
    return [
        EvaluationData(
            conversation_group_id="alpha",
            turns=[
                TurnData(
                    expected_keywords=[["namespace", "isolation"]],
                    **shared,
                )
            ],
        ),
        EvaluationData(
            conversation_group_id="beta",
            turns=[
                TurnData(
                    expected_keywords=[["namespace", "openshift"]],
                    **shared,
                )
            ],
        ),
    ]


def _intervals(summary: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, stats in summary.by_metric.items():
        score = stats.score_statistics
        interval = None if score is None else score.confidence_interval
        found[name] = None if interval is None else interval.model_dump(mode="json")
    return found


def test_summary_leaves_intervals_off_by_default() -> None:
    """The library default is False, and a keyword run stores no interval."""
    parameter = inspect.signature(evaluate_with_summary).parameters
    assert parameter["compute_confidence_intervals"].default is False
    summary = evaluate_with_summary(_config(), _data())
    assert _intervals(summary) == {"custom:keywords_eval": None}
    scores = sorted(
        score
        for score in (result.score for result in summary.results)
        if score is not None
    )
    assert scores == [0.0, 1.0]


def test_seeded_intervals_match() -> None:
    """``np.random.seed(0)`` makes the bootstrap interval exact."""
    np.random.seed(0)
    summary = evaluate_with_summary(
        _config(), _data(), compute_confidence_intervals=True
    )
    payload = _intervals(summary)
    if os.environ.get("LS_GOLDEN_UPDATE") == "1":
        _EXPECTED.parent.mkdir(parents=True, exist_ok=True)
        _EXPECTED.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    assert payload == json.loads(_EXPECTED.read_text(encoding="utf-8"))
    assert (
        payload["custom:keywords_eval"]["low"]
        != payload["custom:keywords_eval"]["high"]
    )
