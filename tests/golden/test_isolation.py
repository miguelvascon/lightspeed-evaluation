"""§5.2 subprocess env, plotted-data capture, egress, and fake oc/kubectl."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.golden


def _python(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_golden_env_reaches_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every golden subprocess sees the §5.2 env, the order pin, and the seed."""
    monkeypatch.setenv("API_KEY", "leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leak")
    monkeypatch.setenv("KUBECONFIG", "/tmp/leak")
    script = (
        "import concurrent.futures, json, os\n"
        "import numpy as np\n"
        "print(json.dumps({\n"
        " 'key': os.environ.get('OPENAI_API_KEY'),\n"
        " 'hash': os.environ.get('PYTHONHASHSEED'),\n"
        " 'tz': os.environ.get('TZ'),\n"
        " 'lc': os.environ.get('LC_ALL'),\n"
        " 'mpl': os.environ.get('MPLBACKEND'),\n"
        " 'mpldir': os.environ.get('MPLCONFIGDIR'),\n"
        " 'base': os.environ.get('OPENAI_BASE_URL'),\n"
        " 'litellm': os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP'),\n"
        " 'hf': os.environ.get('HF_HUB_OFFLINE'),\n"
        " 'seed': os.environ.get('LS_GOLDEN_NP_SEED'),\n"
        " 'path': os.environ.get('PYTHONPATH', ''),\n"
        " 'api': os.environ.get('API_KEY'),\n"
        " 'kube': os.environ.get('KUBECONFIG'),\n"
        " 'anthropic': os.environ.get('ANTHROPIC_API_KEY'),\n"
        " 'pin': bool(getattr(concurrent.futures.as_completed,"
        " 'ls_golden_order_pin', False)),\n"
        " 'draw': float(np.random.random()),\n"
        "}))\n"
    )
    completed = _python(script)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["key"] == "golden-fake"
    assert payload["hash"] == "0"
    assert payload["tz"] == "UTC"
    assert payload["lc"] == "C.UTF-8"
    assert payload["mpl"] == "Agg"
    assert payload["mpldir"]
    assert payload["base"].startswith("http://127.0.0.1:")
    assert payload["base"].endswith("/v1")
    assert payload["litellm"] == "True"
    assert payload["hf"] == "1"
    assert payload["seed"] == "0"
    assert payload["api"] is None
    assert payload["kube"] is None
    assert payload["anthropic"] is None
    assert payload["pin"] is True
    assert any(
        entry.endswith("_sitecustomize") for entry in payload["path"].split(os.pathsep)
    )
    state = np.random.get_state()
    np.random.seed(0)
    expected = float(np.random.random())
    np.random.set_state(state)
    assert payload["draw"] == expected


def test_savefig_captures_plotted_data(tmp_path: Path) -> None:
    """``Figure.savefig`` writes a plotted-data sidecar next to the PNG."""
    target = tmp_path / "fig.png"
    script = (
        "import matplotlib.pyplot as plt\n"
        "import sys\n"
        "fig, ax = plt.subplots()\n"
        "ax.bar(['a', 'b'], [1, 2])\n"
        "fig.savefig(sys.argv[1])\n"
    )
    completed = _python(script, str(target))
    assert completed.returncode == 0, completed.stderr
    sidecar = json.loads(
        Path(str(target) + ".plotted.json").read_text(encoding="utf-8")
    )
    assert sidecar["rcparams"]
    assert sidecar["axes"][0]["bars"]


def test_egress_blocked_and_loopback_allowed() -> None:
    """Non-loopback connects fail. Loopback is not treated as egress."""
    script = (
        "import socket\n"
        "probe = socket.socket()\n"
        "try:\n"
        "    probe.connect(('203.0.113.1', 80))\n"
        "    print('open')\n"
        "except OSError as exc:\n"
        "    print('blocked' if 'egress' in str(exc) else 'other')\n"
        "local = socket.socket()\n"
        "try:\n"
        "    local.connect(('127.0.0.1', 9))\n"
        "    print('local-open')\n"
        "except OSError as exc:\n"
        "    print('local-egress' if 'egress' in str(exc) else 'local-refused')\n"
    )
    completed = _python(script)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.split()
    assert lines[0] == "blocked"
    assert lines[1] == "local-refused"


def test_real_binary_rejected_and_allowlisted_fake_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``oc`` outside the allowlist fails. A script in the allowlist runs."""
    rejected = _python(
        "import subprocess\n"
        "try:\n"
        "    subprocess.run(['/usr/bin/oc'], check=False)\n"
        "    print('ran')\n"
        "except OSError as exc:\n"
        "    print('rejected' if 'allowlisted' in str(exc) else 'other')\n"
    )
    assert rejected.returncode == 0, rejected.stderr
    assert rejected.stdout.strip() == "rejected"
    fake_bin = tmp_path / "bins"
    fake_bin.mkdir()
    script = fake_bin / "oc"
    script.write_text("#!/bin/sh\necho fake-oc\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("LS_GOLDEN_FAKE_BIN", str(fake_bin))
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    allowed = _python(
        "import subprocess\n"
        "completed = subprocess.run(['oc'], check=True, capture_output=True, text=True)\n"
        "print(completed.stdout.strip())\n"
    )
    assert allowed.returncode == 0, allowed.stderr
    assert allowed.stdout.strip() == "fake-oc"
