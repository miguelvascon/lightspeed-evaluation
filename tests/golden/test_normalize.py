"""Normalizers for paths, CR suffixes, run ids, and SQLite rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.golden._normalize import normalize_obj

pytestmark = pytest.mark.golden


def test_normalize_reason_paths() -> None:
    """Workdir and tmp paths inside reason strings become ``<OUT>``."""
    work = str(Path.cwd() / "eval_output")
    scratch = "/tmp/golden-scratch/case"
    got = normalize_obj({"reason": f"read {work}/rows.csv and {scratch}/x.txt"})
    assert isinstance(got["reason"], str)
    assert work not in got["reason"]
    assert scratch not in got["reason"]
    assert "<OUT>" in got["reason"]
    assert got["reason"].startswith("read ")


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
    """A run id and the 8-hex prefix used in trace names share one token."""
    raw = "12345678-aaaa-bbbb-cccc-ddddeeeeffff"
    got = normalize_obj({"run_id": raw, "trace": f"eval_{raw[:8]}"})
    assert got["run_id"] == "<RUN1>"
    assert got["trace"] == "eval_<RUN1>"


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
