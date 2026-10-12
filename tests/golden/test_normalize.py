"""Normalizers for paths, CR suffixes, run ids, and SQLite rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.golden._normalize import (
    collapse_polls,
    normalize_addresses,
    normalize_obj,
    normalize_text,
)

pytestmark = pytest.mark.golden


def test_normalize_reason_paths() -> None:
    """The checkout becomes ``<REPO>`` and temp paths become ``<OUT>``."""
    work = str(Path.cwd() / "eval_output")
    scratch = "/tmp/golden-scratch/case"
    got = normalize_obj({"reason": f"read {work}/rows.csv and {scratch}/x.txt"})
    reason = got["reason"]
    assert isinstance(reason, str)
    assert work not in reason
    assert scratch not in reason
    assert (
        reason == "read <REPO>/eval_output/rows.csv and <OUT>/golden-scratch/case/x.txt"
    )


def test_normalize_eval_suffix() -> None:
    """The ``eval-<8 hex>`` suffix maps consistently and keeps the id prefix."""
    got = normalize_obj(
        {
            "first": "applied eval-myconv-abcd1234",
            "again": "applied eval-abcd1234",
            "other": "applied eval-ffff0000",
        }
    )
    assert got["first"] == "applied eval-myconv-<SFX1>"
    assert got["again"] == "applied eval-<SFX1>"
    assert got["other"] == "applied eval-<SFX2>"


def test_normalize_run_id_prefix() -> None:
    """Rewrite an 8-hex run id only inside MLflow and Langfuse run names."""
    raw = "12345678-aaaa-bbbb-cccc-ddddeeeeffff"
    prefix = raw[:8]
    got = normalize_obj(
        {
            "run_id": raw,
            "trace": f"eval_{prefix}",
            "dataset": f"evaluation__{prefix}",
            "reason": f"hash {prefix} stays",
            "longer": f"eval_{prefix}9abc",
        }
    )
    assert got["run_id"] == "<RUN1>"
    assert got["trace"] == "eval_<RUN1>"
    assert got["dataset"] == "evaluation__<RUN1>"
    assert got["reason"] == f"hash {prefix} stays"
    assert got["longer"] == f"eval_{prefix}9abc"


def test_normalize_sqlite_id_and_timestamp() -> None:
    """SQLite autoincrement ids and timestamp columns are placeholders."""
    got = normalize_obj(
        {
            "id": 4,
            "run_id": "12345678-aaaa-bbbb-cccc-ddddeeeeffff",
            "timestamp": "2024-05-06 07:08:09",
            "conversation_group_id": "c1",
            "score": 1.0,
            "execution_time": 1.25,
            "evaluation_latency": 0.0,
            "confidence_interval": {"low": 0.1, "high": 0.2},
        }
    )
    assert got["id"] == "<ID>"
    assert got["timestamp"] == "<TS>"
    assert got["run_id"] == "<RUN1>"
    assert got["score"] == 1.0
    assert got["execution_time"] == "<T>"
    assert got["evaluation_latency"] == 0.0
    assert got["confidence_interval"] == {"low": 0.1, "high": 0.2}


def test_normalize_progress_bar_is_tqdm_shaped() -> None:
    """Only a tqdm bar (percent, bar, count, timing) becomes ``<PROGRESS>``."""
    bars = "\n".join(
        (
            "  0%|          | 0/4 [00:00<?, ?it/s]",
            "100%|██████████| 4/4 [00:00<00:00, 10.00it/s]",
            "Conversations:  50%|█████     | 2/4 [00:01<00:01,  1.50it/s]",
        )
    )
    prose = "The score was 80%| of the labeled set"
    assert normalize_text(bars) == "<PROGRESS>\n<PROGRESS>\n<PROGRESS>"
    assert normalize_text(prose) == prose


def test_normalize_memory_address() -> None:
    """DeepEval RetryError addresses become ``<ADDR>``."""
    raw = "RetryError <bound method Chat.complete at 0x7f3a1b2c4d5e> then 0xabc"
    expected = "RetryError <bound method Chat.complete at <ADDR>> then <ADDR>"
    assert normalize_addresses(raw) == expected
    assert normalize_text(raw) == expected
    got = normalize_obj({"reason": raw})
    assert got["reason"] == expected


def test_keep_exact_preserves_named_timing() -> None:
    """Named fields such as ``agent_latency`` can stay exact."""
    raw = {"agent_latency": 1.25, "execution_time": 1.25, "reason": "at 0xabc"}
    got = normalize_obj(raw, keep_exact=("agent_latency",))
    assert got["agent_latency"] == 1.25
    assert got["execution_time"] == "<T>"
    assert got["reason"] == "at <ADDR>"
    untouched = normalize_obj(
        {"agent_latency": {"nested": "0xabc"}}, keep_exact=("agent_latency",)
    )
    assert untouched["agent_latency"] == {"nested": "0xabc"}


def test_collapse_polls_keeps_status_changes() -> None:
    """Identical consecutive polls collapse. A status change starts a new run."""
    pending = {"phase": "Pending"}
    done = {"phase": "Succeeded"}
    assert collapse_polls([pending, dict(pending), done, dict(done)]) == [pending, done]
    assert collapse_polls([{"n": 1}, {"n": 2}]) == [{"n": 1}, {"n": 2}]
    assert collapse_polls("Pending\nPending\nSucceeded\n") == "Pending\nSucceeded\n"
    got = normalize_obj({"polls": [pending, dict(pending), done]})
    assert got["polls"] == [pending, done]
    kept = normalize_obj({"polls": [pending, done]}, keep_exact=("polls",))
    assert kept["polls"] == [pending, done]


def test_root_conftest_registers_golden_for_harness(
    request: pytest.FixtureRequest,
) -> None:
    """``golden`` and ``slow`` are registered. Harness has no local conftest."""
    names = [
        str(line).split(":", 1)[0].strip() for line in request.config.getini("markers")
    ]
    assert "golden" in names
    assert "slow" in names
    harness = Path(__file__).resolve().parent / "harness"
    assert not (harness / "conftest.py").exists()
