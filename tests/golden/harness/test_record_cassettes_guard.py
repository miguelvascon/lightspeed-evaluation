"""Guards on the Miguel-only recorder: record mode can't start by accident.

Nothing here reaches the network or needs a real key: every route upstream
(``RecordingJudge.set_record``, ``RecordingJudge._forward``, ``urlopen``) is a tripwire
and the CLI estimate pass is stubbed. The key is a placeholder (R-AGENT-KEYS).
Run it inside a no-egress namespace too: ``unshare -rn``.
"""

from __future__ import annotations

import io
import json
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


def test_max_requests_above_hard_cap_is_refused(
    monkeypatch: pytest.MonkeyPatch, calls: list[str]
) -> None:
    """--max-requests 5000 exits 2 before the proxy, the estimate, or any send."""
    assert (
        _main(
            monkeypatch,
            ["--record-with-my-key", "--max-requests", "5000"],
            _Stdin("y\n", tty=True),
        )
        == 2
    )
    assert calls == []


class _Resp:
    """In-memory urlopen body. Never opens a socket."""

    status = 200

    def read(self) -> bytes:
        """Return a tiny JSON object."""
        return b'{"ok":true}'

    def __enter__(self) -> "_Resp":
        """Context-manager entry."""
        return self

    def __exit__(self, *_a: object) -> None:
        """Context-manager exit."""
        return None


def _trap_upstream(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace urlopen with a counter. The real opener is never called."""
    calls: list[int] = []

    def _urlopen(*_a: Any, **_k: Any) -> _Resp:
        calls.append(1)
        return _Resp()

    monkeypatch.setattr(rc.cp.urllib.request, "urlopen", _urlopen)
    return calls


def _chat(judge: Any, n: int) -> tuple[int, dict]:
    """One tagged chat completion carrying a distinct body."""
    return judge.handle("/v1/chat/completions", json.dumps({"n": n}).encode())


def test_forwarding_stops_at_hard_cap_when_more_requests_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cap of 5000 is clamped; only HARD_CAP requests are forwarded."""
    calls = _trap_upstream(monkeypatch)
    judge = rc.cp.RecordingJudge(lambda _path, _body: (200, {}))
    judge.set_record("https://upstream.invalid/v1", "verification-not-a-real-key", 5000)
    assert judge.cap == rc.HARD_CAP
    for i in range(150):
        _chat(judge, i)
    assert len(calls) == rc.HARD_CAP
    assert judge.forwarded == rc.HARD_CAP


def test_request_100_is_not_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests 1..99 are forwarded; request 100 is refused and stays local."""
    calls = _trap_upstream(monkeypatch)
    judge = rc.cp.RecordingJudge(lambda _path, _body: (200, {}))
    judge.set_record(
        "https://upstream.invalid/v1",
        "verification-not-a-real-key",
        rc.HARD_CAP,
    )
    statuses = [_chat(judge, i)[0] for i in range(rc.HARD_CAP)]
    assert statuses == [200] * rc.HARD_CAP
    assert len(calls) == rc.HARD_CAP
    status, payload = _chat(judge, rc.HARD_CAP)
    assert status == 400
    assert "hard cap" in payload["error"]["message"]
    assert len(calls) == rc.HARD_CAP
    assert judge.forwarded == rc.HARD_CAP
