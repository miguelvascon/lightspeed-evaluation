"""Per-test env and PYTHONPATH hooks (see ``tests/golden/conftest.py``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

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
