"""C5 offline: report files, transcripts, graphs, and the concurrency multiset."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from tests.golden._compare import passes, rate_fixture
from tests.golden.cli._cli import as_multiset, collect_output, run_cli

pytestmark = pytest.mark.golden

_ROOT = Path(__file__).parent
_EXPECTED = _ROOT / "expected"
_CSV_HEADER = (
    "conversation_group_id,turn_id,metric_identifier,score,threshold,"
    "result,reason,query,response"
)
_REPORTS = {
    "evaluation_<TS>_detailed.csv",
    "evaluation_<TS>_summary.json",
    "evaluation_<TS>_summary.txt",
    "evaluation_<TS>_quality_report.json",
    "graphs/evaluation_<TS>_pass_rates.png.plotted.json",
    "graphs/evaluation_<TS>_status_breakdown.png.plotted.json",
}
_GRAPHS = {
    "graphs/evaluation_<TS>_pass_rates.png",
    "graphs/evaluation_<TS>_status_breakdown.png",
}
# pytest's basetemp changes every run. The normalizer turns ``/tmp`` into
# ``<OUT>`` and leaves this suffix, so scrub the whole pytest directory.
_PYTEST_TMP = re.compile(r"/tmp/pytest-of-[^\s\"']+")


def _scrub(obj: Any) -> Any:
    if isinstance(obj, str):
        return _PYTEST_TMP.sub("<OUT>", obj)
    if isinstance(obj, dict):
        return {key: _scrub(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_scrub(item) for item in obj]
    return obj


def _run(fixture: str, output: Path) -> dict[str, Any]:
    base = _ROOT / "fixtures" / fixture
    completed = run_cli(
        "--system-config",
        str(base / "system.yaml"),
        "--eval-data",
        str(base / "eval_data.yaml"),
        "--output-dir",
        str(output),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return collect_output(output, completed)


def _assert_bundle(name: str, bundle: dict[str, Any], *, multiset: bool) -> None:
    actual = _scrub(as_multiset(bundle) if multiset else bundle)
    path = _EXPECTED / f"{name}.json"
    if os.environ.get("LS_GOLDEN_UPDATE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        return
    expected = _scrub(json.loads(path.read_text(encoding="utf-8")))
    if multiset:
        expected = as_multiset(expected)
    rating = rate_fixture(actual, expected)
    assert passes(rating, []), rating


def _assert_shape(bundle: dict[str, Any]) -> None:
    assert set(bundle["files"]) == _REPORTS
    assert set(bundle["png"]) == _GRAPHS
    csv_name = next(name for name in bundle["files"] if name.endswith(".csv"))
    header = str(bundle["files"][csv_name]).splitlines()[0]
    assert header == _CSV_HEADER


def test_offline_reports_match(tmp_path: Path) -> None:
    """File set, CSV header, transcript, JSON, TXT, quality report, and graphs."""
    bundle = _run("offline", tmp_path / "out")
    _assert_shape(bundle)
    _assert_bundle("offline", bundle, multiset=False)


def test_concurrency_reports_match_as_a_multiset(tmp_path: Path) -> None:
    """``max_threads: 4`` is compared with cross-conversation order sorted."""
    bundle = _run("concurrency", tmp_path / "out")
    _assert_shape(bundle)
    _assert_bundle("concurrency", bundle, multiset=True)
