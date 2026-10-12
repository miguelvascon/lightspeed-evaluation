"""HTTP-agent goldens for lsc-parity L09, L10, L18 and L19.

Expected artifacts live in ``expected/http/<case>/``. ``LS_GOLDEN_UPDATE=1``
rewrites them (spec section 5.3). Comparison uses ``rate_fixture``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.golden._compare import active_labels, passes, rate_fixture
from tests.golden._normalize import collapse_polls, normalize_obj, normalize_text
from tests.golden.agent.conftest import HERE, CaseRun, case_params

pytestmark = pytest.mark.golden
CASES = case_params("http")
_TIMING = frozenset(
    {
        "execution_time",
        "evaluation_latency",
        "agent_latency",
        "time_to_first_token",
        "streaming_duration",
        "tokens_per_second",
        "agent_latency_mean",
        "agent_latency_std",
        "agent_latency_mean_delta",
    }
)
_SUFFIXES = (".csv", ".json", ".yaml", ".yml")
_PORT = re.compile(r"127\.0\.0\.1:\d+")
_TO_STREAM = {"streaming_performance": "streaming"}
_FROM_STREAM = {"streaming": "streaming_performance"}


def _scrub(text: str) -> str:
    return _PORT.sub("127.0.0.1:<PORT>", text)


def _swap(obj: Any, table: dict[str, str]) -> Any:
    """Rename keys so ``normalize_obj`` blanks the streaming timing tree."""
    if isinstance(obj, list):
        return [_swap(item, table) for item in obj]
    if not isinstance(obj, dict):
        return obj
    return {table.get(key, key): _swap(value, table) for key, value in obj.items()}


def _stable(obj: Any, roots: tuple[str, ...], keep: tuple[str, ...]) -> Any:
    renamed = _swap(obj, _TO_STREAM)
    normalized = normalize_obj(renamed, roots=roots, keep_exact=keep)
    return _swap(normalized, _FROM_STREAM)


def _text(text: str, roots: tuple[str, ...], *, polls: bool) -> str:
    normalized = normalize_text(_scrub(text), roots=roots)
    if not polls:
        return normalized
    collapsed = collapse_polls(normalized)
    return collapsed if isinstance(collapsed, str) else normalized


def _csv(text: str, roots: tuple[str, ...], keep: tuple[str, ...]) -> str:
    reader = csv.DictReader(io.StringIO(text))
    fields = list(reader.fieldnames or [])
    rows: list[dict[str, Any]] = []
    for row in reader:
        built: dict[str, Any] = {}
        for key in fields:
            value: Any = row[key]
            if key in _TIMING and value not in (None, ""):
                try:
                    value = float(value)
                except ValueError:
                    pass
            built[key] = value
        rows.append(built)
    normalized = _stable(rows, roots, keep)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in normalized:
        writer.writerow(
            {key: "" if row.get(key) is None else row[key] for key in fields}
        )
    return buffer.getvalue()


def _render(path: Path, roots: tuple[str, ...], keep: tuple[str, ...]) -> str:
    text = _scrub(path.read_text(encoding="utf-8"))
    suffix = path.suffix.lower()
    if suffix == ".json":
        rendered = _stable(json.loads(text), roots, keep)
        return json.dumps(rendered, indent=2, ensure_ascii=False) + "\n"
    if suffix in {".yaml", ".yml"}:
        dumped = yaml.safe_dump(
            _stable(yaml.safe_load(text), roots, keep),
            sort_keys=False,
            allow_unicode=True,
        )
        return dumped if dumped.endswith("\n") else dumped + "\n"
    if suffix == ".csv":
        return _csv(text, roots, keep)
    return normalize_text(text, roots=roots)


def _tree(out: Path, roots: tuple[str, ...], keep: tuple[str, ...]) -> dict[str, str]:
    if not out.is_dir():
        return {}
    found = {}
    for path in sorted(item for item in out.rglob("*") if item.is_file()):
        rel = normalize_text(path.relative_to(out).as_posix(), roots=roots)
        found[rel] = _render(path, roots, keep)
    return found


def _reports(actual: dict[str, str], run: CaseRun, tree: dict[str, str]) -> None:
    """Add stdout, request logs, markers, and file modes."""
    wanted = set(run.case.get("assert", []))
    roots = (str(run.workdir), str(run.workdir.resolve()))
    polls = bool(run.case.get("oc_calls_collapse_polls"))
    if "lsc_requests" in wanted and run.lsc is not None:
        actual["lsc_requests.jsonl"] = _text(
            run.lsc.log_path.read_text(encoding="utf-8"), roots, polls=polls
        )
    if "stdout_summary" in wanted:
        tail = run.proc.stdout.split("🎉 Evaluation Complete!")[-1]
        actual["stdout_summary.txt"] = _text(tail, roots, polls=polls)
    if "stdout_tail" in wanted:
        tail = "\n".join(run.proc.stdout.rstrip().splitlines()[-3:]) + "\n"
        if run.noexec_target is not None:
            tail = tail.replace(str(run.noexec_target), "<NOEXEC_TARGET>")
        actual["stdout_tail.txt"] = _text(tail, roots, polls=polls)
    if "markers" in wanted:
        for rel, expect in sorted((run.case.get("markers") or {}).items()):
            path = run.workdir / str(rel)
            got = path.read_text(encoding="utf-8") if path.is_file() else "<missing>\n"
            assert got == expect, rel
            actual[f"markers/{rel}"] = _text(got, roots, polls=polls)
    if "file_modes" in run.case:
        modes = {
            rel: f"{stat.S_IMODE((run.workdir / str(rel)).stat().st_mode):04o}"
            for rel in sorted(run.case["file_modes"])
        }
        assert modes == run.case["file_modes"]
        actual["file_modes.json"] = json.dumps(modes, indent=1) + "\n"
    if "judge_requests" in wanted and run.judge is not None:
        body = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in run.judge.requests
        )
        actual["judge_requests.jsonl"] = _text(body, roots, polls=polls)
    if "flat_copy" in wanted:
        nested = {
            key.split("/run_1/", 1)[1]: text
            for key, text in tree.items()
            if "/run_1/" in key
        }
        flat = {key: text for key, text in tree.items() if "/" not in key}
        assert nested and nested == flat


def collect_actual(run: CaseRun) -> dict[str, str]:
    """Build the normalized artifact map ``case.yaml`` asked to freeze."""
    wanted = set(run.case.get("assert", []))
    roots = (str(run.workdir), str(run.workdir.resolve()))
    keep = ("agent_latency",) if run.case.get("agent_latency") == "exact" else ()
    tree = _tree(run.workdir / "eval_output", roots, keep)
    actual: dict[str, str] = {"exit_code": f"{run.proc.returncode}\n"}
    if "file_tree" in wanted:
        actual["file_tree.txt"] = "".join(f"{name}\n" for name in sorted(tree))
    if "results" in wanted:
        for rel, text in tree.items():
            if rel.endswith(_SUFFIXES):
                actual[f"out/{rel}"] = text
    _reports(actual, run, tree)
    return actual


def _write_expected(actual: dict[str, str], expected_dir: Path) -> None:
    if expected_dir.exists():
        shutil.rmtree(expected_dir)
    for name, text in actual.items():
        dest = expected_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")


def _assert_golden(actual: dict[str, str], expected_dir: Path) -> None:
    """Compare with ``rate_fixture``, or rewrite when ``LS_GOLDEN_UPDATE=1``."""
    ordered = {key: actual[key] for key in sorted(actual)}
    if os.environ.get("LS_GOLDEN_UPDATE") == "1":
        _write_expected(ordered, expected_dir)
        return
    expected = {
        path.relative_to(expected_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(expected_dir.rglob("*"))
        if path.is_file()
    }
    rating = rate_fixture(ordered, expected)
    if passes(rating, active_labels()):
        return
    shared = set(expected) & set(ordered)
    changed = [key for key in sorted(shared) if expected[key] != ordered[key]]
    raise AssertionError(f"golden {rating}: changed={changed[:8]}")


@pytest.mark.parametrize("fixture_dir", CASES)
def test_agent_http_case(fixture_dir: Path, run_agent_case: Any) -> None:
    """Run one HTTP fixture and freeze its normalized artifacts."""
    run: CaseRun = run_agent_case(fixture_dir)
    assert run.proc.returncode == run.case.get("exit_code", 0), run.proc.stderr[-2000:]
    if run.lsc is not None:
        assert not run.lsc.misses(), run.lsc.misses()
        assert not run.lsc.unused(), run.lsc.unused()
    if run.judge is not None:
        assert not run.judge.misses
        assert not run.judge.unused()
    _assert_golden(collect_actual(run), HERE / "expected" / "http" / fixture_dir.name)
