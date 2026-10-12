r"""OpenAI-compatible strict replay judge for the golden suite (SPEC-01 §5.3, §7 Q11).

The CLI under test reaches it through ``OPENAI_BASE_URL=<base>/v1``; nothing in
``src/`` is hooked. It serves ``POST /v1/chat/completions`` and
``POST /v1/embeddings`` from a cassette and never talks to a real judge.

Cassette format (one JSON object per line; the same format the Miguel-only
``record_cassettes.py`` writes, so a recorded cassette drops in unchanged)::

    {"key": sha256("POST <path>\n" + body), "ordinal": 0,
     "request": {"method": "POST", "path": "/v1/chat/completions", "body": "<raw JSON text>"},
     "response": {"status": 200, "body": {...}}, "source": "handwritten", "scrubbed": false}

Replay is strict: the n-th identical request gets the entry with the n-th
smallest ordinal for its key. An unknown request (or one past the last ordinal)
gets HTTP 599 and is recorded as a miss; ``assert_clean()`` then fails the test
with a unified diff against the nearest cassette request.
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import re
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

from tests.golden.harness._http_server import Router, prefix
from tests.golden.harness._http_server import serve as _serve

ROUTES = ("/v1/chat/completions", "/v1/embeddings")
MISS_STATUS = 599
# The recorder runs several CLIs at once under /t/<tag>/v1; the tag is not keyed.
_TAGGED = re.compile(r"^/t/[^/]+(/v1/.*)$")


def replay_key(method: str, path: str, body: bytes) -> str:
    """Replay key: sha256 of method, path and raw body (record_cassettes.py layout)."""
    return hashlib.sha256(f"{method} {path}\n".encode() + body).hexdigest()


def _untag(path: str) -> str:
    m = _TAGGED.match(path)
    return m.group(1) if m else path


def load_cassette(path: Path) -> list[dict]:
    """Read a cassette; blank lines are ignored."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pretty(body: str) -> list[str]:
    try:
        return json.dumps(json.loads(body), indent=1, sort_keys=True).splitlines()
    except ValueError:
        return body.splitlines()


class JudgeReplay:
    """Strict replay state machine; transport-free so any HTTP server can mount it."""

    def __init__(self, entries: list[dict]) -> None:
        """Index cassette entries by key, in ordinal order; reject inconsistent keys."""
        self._lock = threading.Lock()
        self._index: dict[str, list[dict]] = {}
        for entry in sorted(entries, key=lambda e: (e["key"], e["ordinal"])):
            req = entry["request"]
            expect = replay_key(req["method"], req["path"], req["body"].encode())
            if expect != entry["key"]:
                raise ValueError(
                    f"cassette key {entry['key'][:12]} != hash of its request {expect[:12]}"
                )
            self._index.setdefault(entry["key"], []).append(entry)
        self._used: dict[str, int] = {}
        self.requests: list[dict] = []  # every request seen, in arrival order
        self.misses: list[dict] = []

    @staticmethod
    def handles(path: str) -> bool:
        """True for the OpenAI routes this fake owns, bare or under ``/t/<tag>``."""
        return _untag(path) in ROUTES

    def handle(
        self, method: str, path: str, _headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        """Serve one request. Headers are ignored, so Authorization never matters."""
        path = _untag(path)
        key = replay_key(method, path, body)
        text = body.decode("utf-8", errors="replace")
        with self._lock:
            ordinal = self._used.get(key, 0)
            self.requests.append(
                {"key": key, "ordinal": ordinal, "path": path, "body": text}
            )
            cands = self._index.get(key, [])
            if method != "POST" or not self.handles(path) or ordinal >= len(cands):
                self.misses.append(
                    {"key": key, "ordinal": ordinal, "path": path, "body": text}
                )
                payload = {
                    "error": {
                        "message": f"golden replay miss {key[:12]}#{ordinal}",
                        "type": "golden_replay_miss",
                    }
                }
                return (
                    MISS_STATUS,
                    {"content-type": "application/json"},
                    json.dumps(payload).encode(),
                )
            self._used[key] = ordinal + 1
            resp = cands[ordinal]["response"]
        return (
            resp["status"],
            {"content-type": "application/json"},
            json.dumps(resp["body"]).encode(),
        )

    def unused(self) -> list[tuple[str, int]]:
        """(key, ordinal) of cassette entries that were never served."""
        return [
            (k, i)
            for k, v in self._index.items()
            for i in range(self._used.get(k, 0), len(v))
        ]

    def nearest_diff(self, miss: dict) -> str:
        """Unified diff of a missed body against the most similar cassette request."""
        bodies = [
            e["request"]["body"]
            for v in self._index.values()
            for e in v
            if e["request"]["path"] == miss["path"]
        ]
        if not bodies:
            return f"(no cassette entry for {miss['path']})\n" + "\n".join(
                _pretty(miss["body"])
            )
        best = max(
            bodies, key=lambda b: difflib.SequenceMatcher(None, b, miss["body"]).ratio()
        )
        return "\n".join(
            difflib.unified_diff(
                _pretty(best), _pretty(miss["body"]), "cassette", "request", lineterm=""
            )
        )

    def assert_clean(self) -> None:
        """Fail on any miss (with the nearest-match diff) or any unused entry."""
        # Clients retry a 599 two or three times; report each missed key once.
        first: dict[str, dict] = {}
        count: dict[str, int] = {}
        for m in self.misses:
            first.setdefault(m["key"], m)
            count[m["key"]] = count.get(m["key"], 0) + 1
        problems = [
            f"MISS {m['path']} {k[:12]}#{m['ordinal']} (x{count[k]}):\n"
            + self.nearest_diff(m)
            for k, m in first.items()
        ]
        problems += [f"UNUSED cassette entry {k[:12]}#{i}" for k, i in self.unused()]
        if problems:
            raise AssertionError("judge replay not clean:\n" + "\n\n".join(problems))


# ---------------------------------------------------------------- transport
@contextlib.contextmanager
def serve(cassette: Path | list[dict]) -> Iterator[tuple[str, JudgeReplay]]:
    """Run the judge on WP0a's loopback server; yields (OPENAI_BASE_URL, replay state).

    Mounted at ``/v1`` and at ``/t`` (the recorder's ``/t/<tag>/v1`` routes; the
    tag is stripped before keying). Responses are served as ``json.dumps`` of the
    stored body, not the original wire bytes: the cassette format stores the
    parsed body, clients only parse it, and the dry-run == shipped-cassette check
    compares written cassette lines, which never depend on served bytes.
    """
    entries = cassette if isinstance(cassette, list) else load_cassette(cassette)
    replay = JudgeReplay(entries)
    with _serve(
        Router(prefix("/v1", replay.handle), prefix("/t", replay.handle))
    ) as base:
        yield base + "/v1", replay
