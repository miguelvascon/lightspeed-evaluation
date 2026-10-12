"""Record-side judge proxy for ``record_cassettes.py`` (Miguel-only).

Replay is not here: the recorder replays through ``judge_server.JudgeReplay``.
This module only answers the recorder's estimate and record passes:

- ``fake``: every request is answered by the hand-written responder.
- ``record``: chat completions are forwarded upstream with the caller's key;
  embeddings are still answered locally and are never sent upstream.

It is a WP0a ``_http_server`` handler for the recorder's ``/t/<tag>/v1/...``
routes (the tag separates concurrent CLI runs). The cassette line format and
key are the ``judge_server`` contract: sha256("POST <path>\\n" + body) plus a
per-key ordinal; entries are written sorted by key and ordinal with
``sort_keys`` and ``ensure_ascii=False``, keeping ``source`` and ``scrubbed``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Optional

from tests.golden.harness.judge_server import replay_key

KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{8,}"),
]
API_KEY_FIELD = re.compile(r'("api_key"\s*:\s*)"[^"]*"')
TAGGED = re.compile(r"^/t/([^/]+)(/v1/.*)$")
JSON_HEADERS = {"content-type": "application/json"}

Responder = Callable[[str, dict], "tuple[int, dict]"]


def scrub_text(text: str, secret: Optional[str]) -> tuple[str, bool]:
    """Redact ``api_key`` fields, the caller's key and anything key-shaped."""
    out = API_KEY_FIELD.sub(r'\1"<REDACTED>"', text)
    if secret:
        out = out.replace(secret, "<REDACTED>")
    for pat in KEY_PATTERNS:
        out = pat.sub("<REDACTED>", out)
    return out, out != text


class RecordingJudge:  # pylint: disable=too-many-instance-attributes
    """The estimate/record server state, mounted under ``/t/<tag>/v1``."""

    def __init__(self, responder: Responder) -> None:
        """Start in fake mode with the hand-written responder."""
        self.responder = responder
        self.mode = "fake"
        self.upstream: Optional[str] = None
        self._secret: Optional[str] = None
        self.cap: Optional[int] = None
        self.forwarded = 0
        self.abort_reason: Optional[str] = None
        self.lock = threading.Lock()
        self.exchanges: list[dict] = []
        self.ordinals: dict[tuple[str, str], int] = {}

    def set_fake(self) -> None:
        """Answer everything locally (estimate pass and ``--dry-run``)."""
        self.mode, self.upstream, self._secret, self.cap = "fake", None, None, None

    def set_record(self, upstream: str, secret: str, cap: int) -> None:
        """Forward chat completions upstream (record mode only)."""
        self.mode, self.upstream, self._secret = "record", upstream.rstrip("/"), secret
        self.cap = cap

    def reset(self) -> None:
        """Forget exchanges, ordinals and the abort state."""
        with self.lock:
            self.exchanges, self.ordinals = [], {}
            self.forwarded, self.abort_reason = 0, None

    @staticmethod
    def handles(path: str) -> bool:
        """True for the recorder's tagged routes."""
        return TAGGED.match(path) is not None

    def __call__(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        """WP0a handler: answer one request and log the exchange."""
        del method, headers  # headers (incl. Authorization) are never read or kept
        status, payload = self.handle(path, body)
        return status, dict(JSON_HEADERS), json.dumps(payload).encode()

    def handle(self, raw_path: str, raw: bytes) -> tuple[int, dict]:
        """Answer one request: (status, JSON body)."""
        m = TAGGED.match(raw_path)
        tag, path = (m.group(1), m.group(2)) if m else ("default", raw_path)
        body_text = raw.decode("utf-8", errors="replace")
        scrubbed, changed = scrub_text(body_text, self._secret)
        key = replay_key("POST", path, scrubbed.encode())
        with self.lock:
            if self.abort_reason:
                return 400, {"error": {"message": "aborted"}}
            ordinal = self.ordinals.get((tag, key), 0)
            self.ordinals[(tag, key)] = ordinal + 1
        try:
            body = json.loads(body_text)
        except ValueError:
            body = {}
        is_embedding = path.endswith("/embeddings")
        if self.mode == "fake" or is_embedding:
            if body.get("stream"):
                with self.lock:
                    self.abort_reason = "streaming judge request is not supported"
                return 400, {"error": {"message": self.abort_reason}}
            status, resp = self.responder(path, body)
            source = "handwritten-embedding" if is_embedding else "handwritten"
        else:
            with self.lock:
                if self.cap is not None and self.forwarded >= self.cap:
                    self.abort_reason = f"hard cap of {self.cap} upstream requests reached"
                    return 400, {"error": {"message": self.abort_reason}}
                self.forwarded += 1
            status, resp = self._forward(path, raw)
            source = "recorded"
            if not 200 <= status < 300:
                with self.lock:
                    self.abort_reason = f"upstream returned HTTP {status}"
                return 400, {"error": {"message": "recording aborted (upstream error)"}}
        with self.lock:
            self.exchanges.append(
                {
                    "tag": tag,
                    "path": path,
                    "key": key,
                    "ordinal": ordinal,
                    "body": scrubbed,
                    "scrubbed": changed,
                    "response": {"status": status, "body": resp},
                    "source": source,
                }
            )
        return status, resp

    def _forward(self, path: str, raw: bytes) -> tuple[int, dict]:
        assert self.upstream and self._secret
        req = urllib.request.Request(
            self.upstream + path[len("/v1") :],
            data=raw,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._secret,
            },
        )
        try:
            with urllib.request.urlopen(  # nosec B310 - https upstream only
                req, timeout=300
            ) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode())
            except ValueError:
                return e.code, {"error": {"message": "non-JSON upstream error"}}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return 598, {"error": {"message": f"upstream unreachable: {type(e).__name__}"}}


# ------------------------------------------------------------------ cassettes
def to_entries(exchanges: list[dict], tag: str) -> list[dict]:
    """Cassette entries for one tag, sorted by key and ordinal."""
    return [
        {
            "key": e["key"],
            "ordinal": e["ordinal"],
            "request": {"method": "POST", "path": e["path"], "body": e["body"]},
            "response": e["response"],
            "source": e["source"],
            "scrubbed": e["scrubbed"],
        }
        for e in sorted(
            (x for x in exchanges if x["tag"] == tag),
            key=lambda x: (x["key"], x["ordinal"]),
        )
    ]


def per_key_conflicts(entries: list[dict]) -> list[str]:
    """Keys whose repeats carry different responses (replay needs them equal)."""
    seen: dict[str, str] = {}
    bad = []
    for e in entries:
        r = json.dumps(e["response"], sort_keys=True)
        if e["key"] in seen and seen[e["key"]] != r:
            bad.append(e["key"])
        seen.setdefault(e["key"], r)
    return sorted(set(bad))


def secret_scan(path: Path, secret: Optional[str]) -> list[str]:
    """Lines that contain the key, a key-shaped string, an api_key or Authorization."""
    hits = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if secret and secret in line:
            hits.append(f"line {i}: contains the OPENAI_API_KEY value")
        for pat in KEY_PATTERNS:
            if pat.search(line):
                hits.append(f"line {i}: matches {pat.pattern}")
        for m in API_KEY_FIELD.finditer(line):
            if '"<REDACTED>"' not in m.group(0) and not m.group(0).endswith('""'):
                hits.append(f"line {i}: unredacted api_key field")
        if re.search(r'(?i)"authorization"', line):
            hits.append(f"line {i}: contains an Authorization header")
    return hits


def write_cassette(entries: list[dict], dest: Path, secret: Optional[str]) -> list[str]:
    """Write atomically after a clean secret scan; return the scan hits."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
    hits = secret_scan(tmp, secret)
    if hits:
        tmp.unlink()
        return hits
    os.replace(tmp, dest)
    return []


__all__ = [
    "RecordingJudge",
    "per_key_conflicts",
    "scrub_text",
    "secret_scan",
    "to_entries",
    "write_cassette",
]
