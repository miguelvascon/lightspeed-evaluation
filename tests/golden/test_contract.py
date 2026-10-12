"""HTTP routing, dependency pins, and the agents-block lint."""

from __future__ import annotations

import ast
import importlib.metadata
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from tests.golden.harness._http_server import Router, prefix, serve

pytestmark = pytest.mark.golden

_PINS = {
    "ragas": "0.4.3",
    "deepeval": "3.9.1",
    "litellm": "1.89.3",
    "instructor": "1.14.5",
    "openai": "2.48.0",
    "matplotlib": "3.10.6",
    "seaborn": "0.13.2",
    "numpy": "2.3.2",
}
_DEEP = ("lightspeed_evaluation.core", "lightspeed_evaluation.pipeline")


def _imported(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
            if node.module == "lightspeed_evaluation":
                found.extend(
                    f"lightspeed_evaluation.{alias.name}" for alias in node.names
                )
    return found


def test_golden_tree_does_not_import_deep_paths() -> None:
    """Fail if golden code imports ``core`` or ``pipeline`` outside ``contract/``."""
    root = Path(__file__).resolve().parent
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if "contract" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _imported(tree):
            if name.startswith(_DEEP):
                offenders.append(f"{path.relative_to(root)}:{name}")
    assert not offenders


def test_pinned_metric_dependencies() -> None:
    """§5.2 pins the judge and plotting stack the golden job installs."""
    for name, version in _PINS.items():
        assert importlib.metadata.version(name) == version


def test_serve_routes_v1_and_lsc() -> None:
    """One server dispatches ``/v1`` and ``/lsc`` and 404s anything else."""

    def judge(
        method: str, path: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        """Echo a judge-shaped response."""
        del method, headers, body
        assert path.startswith("/v1")
        return 200, {"Content-Type": "application/json"}, b'{"judge":true}'

    def lsc(
        method: str, path: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        """Echo the lightspeed-stack request body."""
        del method, headers
        assert path.startswith("/lsc")
        return 200, {"Content-Type": "application/octet-stream"}, body

    router = Router(prefix("/v1", judge), prefix("/lsc", lsc))
    with serve(router) as base:
        assert base.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(
            base + "/v1/chat/completions", data=b"{}"
        ) as response:
            assert response.status == 200
            assert b"judge" in response.read()
        request = urllib.request.Request(
            base + "/lsc/v1/query", data=b"query-body", method="POST"
        )
        with urllib.request.urlopen(request) as response:
            assert response.read() == b"query-body"
        try:
            with urllib.request.urlopen(base + "/v10/chat") as missed:
                missed.read()
            status = 200
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
        assert status == 404


def system_yamls_missing_agents(paths: Sequence[Path]) -> list[Path]:
    """Return system.yaml files that omit an explicit ``agents`` block.

    The default agent endpoint is localhost:8080. A golden fixture that omits
    ``agents`` would call a real lightspeed-stack.
    """
    missing: list[Path] = []
    for path in paths:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or "agents" not in loaded:
            missing.append(path)
    return missing


def test_checked_in_fixtures_set_agents() -> None:
    """Lint checked-in golden ``system.yaml`` files for an agents block."""
    root = Path(__file__).resolve().parent
    assert not system_yamls_missing_agents(list(root.rglob("system.yaml")))


def test_agents_lint_fails_on_bad_fixture(tmp_path: Path) -> None:
    """A fixture system.yaml without ``agents`` is reported as a lint failure."""
    bad = tmp_path / "system.yaml"
    bad.write_text("llm:\n  provider: openai\n", encoding="utf-8")
    assert system_yamls_missing_agents([bad]) == [bad]
    good = tmp_path / "system.yaml"
    good.write_text("agents:\n  enabled: false\n", encoding="utf-8")
    assert not system_yamls_missing_agents([good])
