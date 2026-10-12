"""Judge-backed metric goldens. WP0c-1: the fake judge and the GEval logprob proof.

SPEC-01 §7 Q11: GEval must be scored from hand-written ``logprobs`` through the
wire fake, not DeepEval's text fallback. The cassette puts 0.75 on token "8" and
0.25 on "7", so the weighted score is (0.75*8 + 0.25*7)/10 = 0.775. Both
fallbacks give something else: a broken logprob payload makes ``g_eval.py``
return the text score 8 -> 0.8, and a failed logprob call makes it send a
``ReasonScore`` json_schema request (a replay miss here).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.golden
CASE = "geval_logprob_proof"
CASSETTE = Path(__file__).with_name("cassettes") / f"{CASE}.jsonl"


def _exchanges() -> list[dict]:
    return [
        json.loads(line) for line in CASSETTE.read_text().splitlines() if line.strip()
    ]


def _by_metric(rows: list[dict]) -> dict[str, dict]:
    return {r["metric_identifier"]: r for r in rows}


def test_geval_scored_from_logprobs(
    tmp_path: Path, run_case: Callable[..., Any]
) -> None:
    """Turn and conversation GEval both score exactly 0.775 via the logprob path."""
    code, rows, replay = run_case(CASE, tmp_path, CASSETTE)
    replay.assert_clean()  # 0 misses, 0 unused
    assert code == 0
    got = {
        m: (
            r["turn_id"],
            r["result"],
            r["score"],
            r["threshold"],
            r["reason"],
            r["judge_llm_input_tokens"],
            r["judge_llm_output_tokens"],
        )
        for m, r in _by_metric(rows).items()
    }
    assert got == {
        "geval:technical_accuracy": (
            "t1",
            "PASS",
            "0.775",
            "0.7",
            "The command is correct and matches the expected namespace.",
            "510",
            "70",
        ),
        "geval:conversation_coherence": (
            "",
            "PASS",
            "0.775",
            "0.6",
            "Turn two keeps the payments namespace from turn one.",
            "350",
            "21",
        ),
    }
    for row in rows:
        assert json.loads(row["judge_scores"])[0]["score"] == 0.775
    bodies = [json.loads(r["body"]) for r in replay.requests]
    scoring = [b for b in bodies if b.get("logprobs")]
    assert len(bodies) == 3 and len(scoring) == 2  # steps + 2 logprob scoring calls
    assert all(b["top_logprobs"] == 5 for b in scoring)
    assert not any(
        b.get("response_format", {}).get("json_schema", {}).get("name") == "ReasonScore"
        for b in bodies
    )


def test_pin_rejects_text_fallback(
    tmp_path: Path, run_case: Callable[..., Any]
) -> None:
    """Control: same requests, logprobs removed -> DeepEval falls back to 0.8, not 0.775."""
    entries = copy.deepcopy(_exchanges())
    for e in entries:
        e["response"]["body"]["choices"][0]["logprobs"] = None
    code, rows, replay = run_case(CASE, tmp_path, entries)
    replay.assert_clean()
    assert code == 0
    assert {m: r["score"] for m, r in _by_metric(rows).items()} == {
        "geval:technical_accuracy": "0.8",
        "geval:conversation_coherence": "0.8",
    }
