"""Fixtures for the metric goldens (lane E3).

The golden env, marker and sitecustomize (order pin, numpy seed, loopback
guard) come from WP0a's ``tests/golden/conftest.py``; this file only adds the
``run_case`` runner on top of the replay judge.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.golden._env import publish_openai_base_url
from tests.golden.harness import judge_server

FIXTURES = Path(__file__).with_name("fixtures")
CASSETTES = Path(__file__).with_name("cassettes")


def _run_case(
    case: str, tmp_path: Path, cassette: Path | list[dict]
) -> tuple[int, list[dict[str, str]], Any]:
    """Run the installed CLI on fixtures/<case> against the replay judge.

    Returns (exit code, CSV rows, replay state). The caller asserts replay.assert_clean().
    """
    fixture = FIXTURES / case
    cli = Path(sys.executable).with_name("lightspeed-eval")
    with judge_server.serve(cassette) as (base_url, replay):
        publish_openai_base_url(base_url)  # golden_env's Popen hook injects it
        proc = subprocess.run(
            [
                str(cli),
                "--system-config",
                str(fixture / "system.yaml"),
                "--eval-data",
                str(fixture / "eval_data.yaml"),
                "--output-dir",
                str(tmp_path),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    rows: list[dict[str, str]] = []
    for path in sorted(tmp_path.glob("*_detailed.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return proc.returncode, rows, replay


@pytest.fixture(name="run_case")
def run_case_fixture() -> Callable[..., tuple[int, list[dict[str, str]], Any]]:
    """``run_case(case, tmp_path, cassette) -> (exit code, CSV rows, replay)``."""
    return _run_case
