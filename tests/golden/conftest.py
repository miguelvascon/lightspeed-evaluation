"""Golden pytest hooks (SPEC-01 §5.2).

``golden_env`` applies the determinism environment to the test process and to
every subprocess it starts. Other lanes should request this fixture and delete
local copies.

Per-test and per-directory additions (storage fakes, ``API_KEY``, PATH):

* Override the ``golden_env_extra`` fixture in a subdir ``conftest.py``. Return
  a ``dict[str, str]`` merged into the golden env for tests in that directory.
* Override ``golden_pythonpath_extra`` and return repo-relative or absolute
  paths to prepend. Storage tests prepend
  ``tests/golden/storage/_fake_modules`` (Engineer 2 owns that directory; this
  hook only adds it to ``PYTHONPATH``).
* Or mark a single test::

      @pytest.mark.golden_env(API_KEY="fake")
      @pytest.mark.golden_pythonpath("tests/golden/storage/_fake_modules")

  Marker kwargs win over the fixture override. Paths are prepended, in order,
  before the golden ``sitecustomize`` directory.

``pytest_configure`` registers the ``golden`` marker for this directory and
for ``tests/golden/harness/``. Harness tests do not need their own conftest
to satisfy ``--strict-markers``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from tests.golden._compare import (
    match_case_id,
    parse_pending_regen,
    pending_skip_reason,
    pending_summary,
    reject_unknown_cases,
)

_FIXED = {
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
    "LC_ALL": "C.UTF-8",
    "MPLBACKEND": "Agg",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "HF_HUB_OFFLINE": "1",
    "LS_GOLDEN_NP_SEED": "0",
    "OPENAI_API_KEY": "golden-fake",
}
_UNSET = frozenset(
    {
        "API_KEY",
        "KUBECONFIG",
        "SSL_CERT_FILE",
        "AZURE_API_BASE",
        "WATSONX_API_BASE",
        "WATSONX_PROJECT_ID",
        "HOSTED_VLLM_API_BASE",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OPENAI_API_BASE",
    }
)
_STATE: dict[str, str] = {}
_ORDER_ONLY: list[str] = []
_PENDING: dict[str, str] = {}
_PENDING_KEY = pytest.StashKey[dict[str, str]]()
_REPO = Path(__file__).resolve().parents[2]
_SPAWN = [subprocess.Popen]


def note_order_only(digest: str) -> None:
    """Record an ORDER-ONLY digest for the golden job summary."""
    _ORDER_ONLY.append(digest)


def publish_openai_base_url(url: str) -> None:
    """Point later golden subprocesses at a local fake judge."""
    _STATE["base_url"] = url
    os.environ["OPENAI_BASE_URL"] = url


def _secret(key: str) -> bool:
    return key.endswith(("_API_KEY", "_API_TOKEN")) or key in _UNSET


def _resolve(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _REPO / path
    return str(candidate)


def _pythonpath(env: dict[str, str]) -> None:
    """Prepend extra entries, then sitecustomize, keeping the rest."""
    ordered: list[str] = []
    incoming = _STATE.get("pythonpath", "")
    site = _STATE.get("site", "")
    prior = env.get("PYTHONPATH", "")
    for entry in (*incoming.split(os.pathsep), site, *prior.split(os.pathsep)):
        if entry and entry not in ordered:
            ordered.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(ordered)


def _child_env(caller: Mapping[str, str] | None) -> dict[str, str]:
    """Build the env passed to one golden subprocess.

    Inherited provider secrets are dropped. Keys from ``golden_env_extra`` and
    an explicit ``subprocess`` ``env`` mapping are applied afterwards, so a
    case can set a fake ``API_KEY``. ``OPENAI_API_KEY`` stays the placeholder.
    """
    extras = json.loads(_STATE.get("extra_env") or "{}")
    env = {key: value for key, value in os.environ.items() if not _secret(key)}
    env.update({str(key): str(value) for key, value in extras.items()})
    if caller is not None:
        env.update({str(key): str(value) for key, value in caller.items()})
    env.update(_FIXED)
    env["MPLCONFIGDIR"] = _STATE["mpl"]
    env["OPENAI_BASE_URL"] = _STATE["base_url"]
    _pythonpath(env)
    return env


def _popen(args: Any, *pargs: Any, **kwargs: Any) -> Any:
    kwargs["env"] = _child_env(kwargs.get("env"))
    return _SPAWN[0](args, *pargs, **kwargs)


@pytest.fixture(scope="session")
def _golden_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Session temp matplotlib config dir and a placeholder judge URL."""
    mpl = tmp_path_factory.mktemp("mplconfig")
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return {
        "mpl": str(mpl),
        "site": str(Path(__file__).parent / "_sitecustomize"),
        "base_url": f"http://127.0.0.1:{port}/v1",
    }


@pytest.fixture
def golden_env_extra() -> dict[str, str]:
    """Extra env vars. Override in a subdir conftest; default is empty."""
    return {}


@pytest.fixture
def golden_pythonpath_extra() -> list[str]:
    """Extra ``PYTHONPATH`` entries. Override in a subdir conftest."""
    return []


def _marker_env(request: pytest.FixtureRequest) -> dict[str, str]:
    marker = request.node.get_closest_marker("golden_env")
    if marker is None:
        return {}
    return {str(key): str(value) for key, value in marker.kwargs.items()}


def _marker_paths(request: pytest.FixtureRequest) -> list[str]:
    marker = request.node.get_closest_marker("golden_pythonpath")
    if marker is None:
        return []
    return [str(arg) for arg in marker.args]


@pytest.fixture(autouse=True)
def golden_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    _golden_dirs: dict[str, str],
) -> dict[str, str]:
    """Apply the §5.2 env and return the mapping subprocesses inherit.

    Does nothing for tests that are not marked ``golden``. Marker
    ``golden_env`` / ``golden_pythonpath`` and the ``golden_*_extra`` fixtures
    add entries for this test only.
    """
    if request.node.get_closest_marker("golden") is None:
        return {}
    extra_env = cast(dict[str, str], request.getfixturevalue("golden_env_extra"))
    extra_paths = cast(list[str], request.getfixturevalue("golden_pythonpath_extra"))
    _STATE.update(_golden_dirs)
    extras = dict(extra_env)
    extras.update(_marker_env(request))
    paths = [_resolve(path) for path in (*extra_paths, *_marker_paths(request))]
    _STATE["pythonpath"] = os.pathsep.join(paths)
    _STATE["extra_env"] = json.dumps(extras)
    env = _child_env(None)
    for key in list(os.environ):
        if key not in env:
            monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(subprocess, "Popen", _popen)
    return env


def pytest_configure(config: pytest.Config) -> None:
    """Register ``golden`` for this tree, including ``harness/``.

    ``pytest.ini`` lists the marker too. This hook is what a harness test
    sees when it is collected under ``tests/golden/`` without a local
    ``harness/conftest.py``.
    """
    known = {str(line).split(":", 1)[0].strip() for line in config.getini("markers")}
    if "golden" not in known:
        config.addinivalue_line(
            "markers",
            "golden: Characterization tests that require the golden "
            "determinism env (run by golden.yaml)",
        )
    if "golden_case" not in known:
        config.addinivalue_line(
            "markers",
            "golden_case(case_id): Golden case id skipped while listed in "
            "tests/golden/PENDING_REGEN",
        )


def case_id_from_item(item: Any) -> str | None:
    """Return the case id WP0c must expose, or ``None`` for a harness test.

    ``golden_case`` wins. Otherwise use a ``case_id`` parameter, or the
    directory name of a parameter path under ``fixtures/``. A
    ``fixtures/<id>/`` dir with ``system.yaml`` or ``eval_data.yaml`` is known
    even with no parameter. Harness tests use none of these and are not cases.
    """
    marker = item.get_closest_marker("golden_case")
    if marker is not None and marker.args:
        return match_case_id(marker.args[0])
    params = getattr(getattr(item, "callspec", None), "params", None)
    if not isinstance(params, Mapping):
        return None
    named = match_case_id(params.get("case_id"))
    if named is not None:
        return named
    for value in params.values():
        found = _fixture_dirname(value)
        if found is not None:
            return found
    return None


def _fixture_dirname(value: object) -> str | None:
    if isinstance(value, Path):
        raw = value
    elif isinstance(value, str):
        raw = Path(value)
    else:
        return None
    if "fixtures" not in raw.parts:
        return None
    return match_case_id(raw.name)


def known_case_ids(items: list[Any], root: Path) -> set[str]:
    """Return collected case ids plus fixture dirs that hold a case file."""
    found: set[str] = set()
    for item in items:
        case_id = case_id_from_item(item)
        if case_id is not None:
            found.add(case_id)
    for filename in ("system.yaml", "eval_data.yaml"):
        for path in root.rglob(filename):
            if "fixtures" not in path.parts:
                continue
            case_id = match_case_id(path.parent.name)
            if case_id is not None:
                found.add(case_id)
    return found


def skip_pending_items(items: list[Any], pending: Mapping[str, str]) -> None:
    """Skip a listed case. The id is derived from the test, not a marker."""
    for item in items:
        case_id = case_id_from_item(item)
        if case_id is None:
            continue
        reason = pending_skip_reason(case_id, pending)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Load ``PENDING_REGEN``, reject unknown ids, and skip listed cases."""
    root = Path(__file__).resolve().parent
    path = root / "PENDING_REGEN"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    pending = parse_pending_regen(text)
    golden = [item for item in items if item.get_closest_marker("golden") is not None]
    reject_unknown_cases(pending, known_case_ids(golden, root))
    _PENDING.clear()
    _PENDING.update(pending)
    config.stash[_PENDING_KEY] = dict(pending)
    skip_pending_items(golden, pending)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    """Print ORDER-ONLY and PENDING lines for the golden job summary."""
    digest = ",".join(_ORDER_ONLY) if _ORDER_ONLY else "none"
    terminalreporter.write_line(f"ORDER-ONLY digest: {digest}")
    pending = terminalreporter.config.stash.get(_PENDING_KEY, _PENDING)
    terminalreporter.write_line(pending_summary(pending))
