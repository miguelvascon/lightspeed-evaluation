"""Per-test env and PYTHONPATH hooks (see ``tests/golden/conftest.py``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import tests.golden.conftest as second_conftest
from tests.golden._env import RUNTIME, UNSET, publish_openai_base_url

pytestmark = pytest.mark.golden


def _python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def test_placeholder_key_reaches_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inherited provider secrets are stripped. The placeholder key stays."""
    monkeypatch.setenv("API_KEY", "leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leak")
    script = (
        "import json, os\n"
        "print(json.dumps({\n"
        " 'key': os.environ.get('OPENAI_API_KEY'),\n"
        " 'api': os.environ.get('API_KEY'),\n"
        " 'anthropic': os.environ.get('ANTHROPIC_API_KEY'),\n"
        " 'hash': os.environ.get('PYTHONHASHSEED'),\n"
        "}))\n"
    )
    completed = _python(script)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["key"] == "golden-fake"
    assert payload["api"] is None
    assert payload["anthropic"] is None
    assert payload["hash"] == "0"


@pytest.mark.golden_env(LS_GOLDEN_HOOK="1")
@pytest.mark.golden_pythonpath("tests/golden/harness")
def test_per_test_hooks_reach_subprocess() -> None:
    """Marker kwargs and pythonpath entries apply to that test's subprocesses."""
    script = (
        "import json, os\n"
        "print(json.dumps({'hook': os.environ.get('LS_GOLDEN_HOOK'),\n"
        " 'path': os.environ.get('PYTHONPATH', '')}))\n"
    )
    completed = _python(script)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["hook"] == "1"
    first = payload["path"].split(os.pathsep)[0]
    assert first.endswith(f"{os.sep}tests{os.sep}golden{os.sep}harness")


def test_conftest_import_updates_the_live_hook() -> None:
    """A second import of the conftest still points subprocesses at the judge."""
    assert second_conftest.publish_openai_base_url is publish_openai_base_url
    publish_openai_base_url("http://127.0.0.1:9/from-import/v1")
    assert RUNTIME["base_url"] == "http://127.0.0.1:9/from-import/v1"
    completed = _python("import os; print(os.environ['OPENAI_BASE_URL'])")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "http://127.0.0.1:9/from-import/v1"


@pytest.mark.golden_env(OPENAI_API_KEY=UNSET)
def test_unset_sentinel_drops_the_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """``UNSET`` removes the key. A host value is not copied in its place."""
    assert "OPENAI_API_KEY" not in os.environ
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret")
    completed = _python("import os; print(repr(os.environ.get('OPENAI_API_KEY')))")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "None"


@pytest.mark.golden_env(OPENAI_API_KEY="sk-not-the-dummy")
def test_other_key_values_stay_the_dummy() -> None:
    """Only ``UNSET`` changes the key. Anything else stays ``golden-fake``."""
    assert os.environ["OPENAI_API_KEY"] == "golden-fake"
