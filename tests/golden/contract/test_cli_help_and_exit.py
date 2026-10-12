"""C1: ``--help``, exit codes, cache warmup, and offline vs agent selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.golden.cli._cli import run_cli

pytestmark = pytest.mark.golden

_HELP = Path(__file__).parent / "expected" / "help"
_FIXTURE = Path(__file__).resolve().parents[1] / "cli" / "fixtures" / "offline"
_SYSTEM = _FIXTURE / "system.yaml"
_DATA = _FIXTURE / "eval_data.yaml"
_FAILED = "❌ Evaluation failed:"


def test_help_matches_the_python_minor() -> None:
    """``--help`` at ``COLUMNS=80`` matches the snapshot for this Python minor."""
    completed = run_cli("--help", extra_env={"COLUMNS": "80"})
    assert completed.returncode == 0, completed.stderr
    minor = f"py{sys.version_info.major}{sys.version_info.minor}"
    expected = (_HELP / f"{minor}.txt").read_text(encoding="utf-8")
    assert completed.stdout == expected


def test_missing_config_exits_1() -> None:
    """A missing system config is exit 1 and keeps the failure line."""
    completed = run_cli(
        "--system-config",
        "/no/such/system.yaml",
        "--eval-data",
        str(_DATA),
    )
    assert completed.returncode == 1
    assert _FAILED in completed.stdout
    assert "Configuration file not found" in completed.stdout


def test_empty_config_exits_1(tmp_path: Path) -> None:
    """An empty system config is exit 1."""
    empty = tmp_path / "system.yaml"
    empty.write_text("", encoding="utf-8")
    completed = run_cli("--system-config", str(empty), "--eval-data", str(_DATA))
    assert completed.returncode == 1
    assert _FAILED in completed.stdout
    assert "Empty or invalid YAML configuration file" in completed.stdout


def test_missing_fields_exits_1(tmp_path: Path) -> None:
    """A turn with no query and no OpenShift spec is exit 1."""
    data = tmp_path / "eval_data.yaml"
    data.write_text(
        "- conversation_group_id: missing\n  turns:\n  - turn_id: t1\n    response: hello\n",
        encoding="utf-8",
    )
    completed = run_cli("--system-config", str(_SYSTEM), "--eval-data", str(data))
    assert completed.returncode == 1
    assert _FAILED in completed.stdout
    assert "query is required" in completed.stdout


def test_normal_run_exits_0(tmp_path: Path) -> None:
    """A keyword fixture finishes offline and exits 0."""
    completed = run_cli(
        "--system-config",
        str(_SYSTEM),
        "--eval-data",
        str(_DATA),
        "--output-dir",
        str(tmp_path / "out"),
    )
    assert completed.returncode == 0, completed.stderr
    assert "🎉 Evaluation Complete!" in completed.stdout
    assert "Nothing to evaluate" not in completed.stdout
    names = {path.name for path in (tmp_path / "out").iterdir()}
    assert not any(name.startswith("eval_") for name in names)


def test_total_zero_exits_0() -> None:
    """A filter that matches nothing is exit 0, not a failure."""
    completed = run_cli(
        "--system-config",
        str(_SYSTEM),
        "--eval-data",
        str(_DATA),
        "--conv-ids",
        "not_a_conversation",
    )
    assert completed.returncode == 0, completed.stderr
    assert "Nothing to evaluate" in completed.stdout
    assert _FAILED not in completed.stdout


def test_cache_warmup_clears_a_tmp_cache(tmp_path: Path) -> None:
    """``--cache-warmup`` deletes the configured cache directory, then recreates it."""
    cache = tmp_path / "caches"
    sentinel = cache / "llm" / "stale.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("stale", encoding="utf-8")
    system = tmp_path / "system.yaml"
    text = _SYSTEM.read_text(encoding="utf-8")
    system.write_text(
        text.replace(
            "cache_enabled: false",
            f"cache_enabled: true\n  cache_base_dir: {cache}",
        ),
        encoding="utf-8",
    )
    completed = run_cli(
        "--system-config",
        str(system),
        "--eval-data",
        str(_DATA),
        "--output-dir",
        str(tmp_path / "out"),
        "--cache-warmup",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Cleared LLM Judge cache" in completed.stdout
    assert not sentinel.exists()
    assert (cache / "llm").is_dir()


def test_enabled_agents_write_the_agent_tree(tmp_path: Path) -> None:
    """An enabled agent writes ``eval_<ts>/http_api/run_1`` instead of a flat dir.

    The endpoint is closed. Those turns become ERROR rows and the process
    still exits 0. Offline runs never create the ``eval_`` directory.
    """
    system = tmp_path / "system.yaml"
    head, _, _tail = _SYSTEM.read_text(encoding="utf-8").partition("agents:\n")
    system.write_text(
        head
        + "agents:\n"
        + "  enabled: true\n"
        + "  default:\n"
        + "    agent: [http_api]\n"
        + "  http_api:\n"
        + "    type: http_api\n"
        + "    api_base: http://127.0.0.1:9\n"
        + "    timeout: 1\n"
        + "    num_retries: 0\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    completed = run_cli(
        "--system-config",
        str(system),
        "--eval-data",
        str(_DATA),
        "--output-dir",
        str(output),
    )
    assert completed.returncode == 0, completed.stderr
    assert "http_api/run_1" in completed.stdout
    runs = [path for path in output.glob("eval_*") if path.is_dir()]
    assert len(runs) == 1
    assert (runs[0] / "http_api" / "run_1").is_dir()
    amended = list(output.glob("eval_data_amended_*.yaml"))
    assert len(amended) == 1
