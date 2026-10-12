"""Guards on the Miguel-only recorder: record mode can't start by accident.

Nothing here reaches the network or needs a real key: every route upstream
(``RecordingJudge.set_record``, ``RecordingJudge._forward``, ``urlopen``) is a tripwire
and the CLI estimate pass is stubbed. The key is a placeholder (R-AGENT-KEYS).
Run it inside a no-egress namespace too: ``unshare -rn``.
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from tests.golden.harness import record_cassettes as rc

REPO = Path(__file__).resolve().parents[3]
EST = {
    "per_metric": {},
    "sum_per_metric": 0,
    "full_run_chat": 0,
    "full_run_embeddings": 0,
    "full_run_keys": [],
    "upper_bound": 0,
    "capture_full": {},
}


class _Tripwire(Exception):
    """Raised if anything tries to go upstream."""


class _Reached(Exception):
    """Raised when main() gets past every gate to the record/dry-run pass."""


def _reached(*_a: Any, **_k: Any) -> None:
    raise _Reached("record pass reached")


def _trip(*_a: Any, **_k: Any) -> None:
    raise _Tripwire("upstream path reached")


class _Stdin(io.StringIO):
    def __init__(self, text: str, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(name="calls")
def guarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Tripwire every upstream route and stub the estimate; record what main() reached."""
    calls: list[str] = []
    monkeypatch.setenv("OPENAI_API_KEY", "verification-not-a-real-key")
    monkeypatch.setattr(urllib.request, "urlopen", _trip)
    monkeypatch.setattr(rc.cp.RecordingJudge, "set_record", _trip)
    monkeypatch.setattr(rc.cp.RecordingJudge, "_forward", _trip)
    real_init = rc.cp.RecordingJudge.__init__

    def _proxy_init(self: Any, *a: Any, **k: Any) -> None:  # loopback only
        calls.append("proxy")
        real_init(self, *a, **k)

    def _estimate(*_a: Any, **_k: Any) -> dict:
        calls.append("estimate")
        return dict(EST)

    monkeypatch.setattr(rc.cp.RecordingJudge, "__init__", _proxy_init)
    monkeypatch.setattr(rc, "estimate", _estimate)
    monkeypatch.setattr(rc.Runner, "run", _reached)
    monkeypatch.setattr(rc.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    return calls


def _main(monkeypatch: pytest.MonkeyPatch, argv: list[str], stdin: _Stdin) -> int:
    monkeypatch.setattr(
        sys, "argv", ["record_cassettes.py", "--repo", str(REPO), *argv]
    )
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr("builtins.input", lambda _p="": stdin.readline())
    return rc.main()


def test_refuses_record_mode_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    """Key present, no --record-with-my-key, no --dry-run: exit 2 before anything starts."""
    assert _main(monkeypatch, [], _Stdin("y\n", tty=True)) == 2
    assert not calls


def test_yes_flag_does_not_exist(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    """There is no way to skip the confirmation: argparse rejects --yes."""
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, ["--record-with-my-key", "--yes"], _Stdin("", tty=True))
    assert exc.value.code == 2 and not calls


def test_refuses_record_mode_without_a_tty(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    """Flag but non-interactive stdin: exit 1 after the local estimate, nothing sent."""
    assert _main(monkeypatch, ["--record-with-my-key"], _Stdin("y\n", tty=False)) == 1
    assert calls == ["proxy", "estimate"]


def test_aborts_when_the_answer_is_not_yes(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    """Flag plus TTY: the y/N prompt comes after the estimate; 'n' sends nothing."""
    assert _main(monkeypatch, ["--record-with-my-key"], _Stdin("n\n", tty=True)) == 1
    assert calls == ["proxy", "estimate"]


def test_tripwire_fires_if_record_path_is_reached(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    """Control: flag + TTY + 'y' reaches set_record, which the tripwire stops."""
    with pytest.raises(_Tripwire):
        _main(monkeypatch, ["--record-with-my-key"], _Stdin("y\n", tty=True))
    assert calls == ["proxy", "estimate"]


def test_dry_run_needs_no_flag_no_key_no_tty(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    """--dry-run passes every gate without the flag, a key or a TTY."""
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(_Reached):
        _main(monkeypatch, ["--dry-run"], _Stdin("", tty=False))
    assert calls == ["proxy", "estimate"]
