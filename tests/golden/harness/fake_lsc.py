"""Scripted lightspeed-stack fake for the golden suite (WP0b, C9).

Owner: Engineer 2. Engineer 3's WP0d (#23, #27, #28b, #31b) uses this module
**read-only**; ask Engineer 2 for any change. The interface below is
``FAKE_LSC_API = 1``. It only grows in a backward-compatible way; a breaking
change bumps the number and is announced to E3 first.

Pure stdlib, no ``lightspeed_evaluation`` import (golden isolation rule).

Mounting
========
``FakeLsc`` is a route handler, not a server. Mount it on Engineer 1's
shared server (the same process as the judge fake) with ``prefix``, ``Router``
and ``serve``::

    prefix("/lsc", fake.http_handler())

``http_handler`` returns ``(status, headers, body)``, which is what ``serve``
writes. ``handle`` still returns ``Reply``. A fixture sets
``api_base: http://127.0.0.1:<port>/lsc`` so ``/lsc/v1/responses`` never
collides with the judge's ``/v1/...``. Standalone mode (spikes/debugging
only)::

    python fake_lsc.py <port> <lsc_script.json> <lsc_requests.jsonl>

Python API (stable)
===================
``FakeLsc.from_files(script_path, log_path)``  load + validate a script (a bad
    script raises ``ScriptError`` at load) and truncate the request log.
``FakeLsc.from_script(script: dict, log_path)`` same, from a dict.
``fake.handle(method, raw_path, headers, body) -> Reply(status, content_type, body)``
``fake.http_handler()``  callable for ``prefix("/lsc", ...)``. Same routing as
    ``handle``; the return is ``(status, {"Content-Type": ...}, body)``.
``fake.misses() -> list[dict]``   requests answered with 599 (+ ``why``)
``fake.unused() -> list[str]``    scripted responses never requested
``fake.requests() -> list[dict]`` every logged request (as in the log file)

Request log (``lsc_requests.jsonl``, C9), one line per request, keys sorted::

    {"seq": 1, "method": "POST", "path": "/v1/query",
     "headers": {<allow-list: authorization, mcp-headers, content-type, accept>},
     "body": "<raw request body text, byte-for-byte>"}

Script format (``lsc_script.json``)
===================================
::

    {"routes": [
      {"path": "/v1/streaming_query",     # path after /lsc
       "prompt": "hello",                 # optional; matches body query|question|input
       "match_conversation_id": "absent", # optional; "absent" = request has no
                                          # conversation_id key, "<id>" = equals
       "new_conversation_id": "conv-1",   # optional; see "@CONV@" below
       "responses": [ <response>, ... ]}  # served in order, one per request
    ]}

Routes are tried in file order; the first match wins. A request matching no
route, or a route whose ``responses`` are used up, gets **HTTP 599** with
``{"detail": "fake_lsc miss: ..."}`` and is recorded in ``misses()``.

A ``<response>`` is exactly one of:

``{"status": 200, "json": <any>}``          JSON body, verbatim.
``{"status": 200, "sse": [{"event": "start", "data": {...}}, ...]}``
                                            each item is one ``data:`` JSON line, then a blank line.
``{"status": 200, "raw": "<text>", "content_type": "text/event-stream"}``
                                            verbatim body (malformed SSE etc.).
``{"status": 200, "turn": <turn>}``         high-level answer, rendered for
                                            the route's endpoint (below).

``status`` defaults to 200 (use e.g. 503 with ``json`` for retry tests).

Conversation id echo (``@CONV@``)
---------------------------------
Any JSON string value exactly equal to ``"@CONV@"`` anywhere in a ``json``,
``sse`` or ``turn`` response is replaced, per request, by the request body's
``conversation_id`` if it has one, else by the route's ``new_conversation_id``.
If neither exists the request is a miss ("no conversation_id to echo").
``turn.conversation_id`` defaults to ``"@CONV@"``.

``<turn>`` (endpoints ``/v1/query`` and ``/v1/streaming_query`` only)
---------------------------------------------------------------------
::

    {"conversation_id": "@CONV@",     # default "@CONV@"
     "response": "final text",        # default ""
     "tool_calls": [ {...}, ... ],    # copied VERBATIM: omit "tool_name"/"name"
                                      # for a nameless call, omit "arguments"/
                                      # "args" (or set null/[]/"") for an
                                      # argument-less one
     "input_tokens": 0, "output_tokens": 0,
     "sse_tool_call_form": "token",   # streaming only: "token" -> data
                                      # {"token": call} (legacy form);
                                      # "flat" -> data = call (name/args form)
     "omit_events": []}               # streaming only: drop e.g. "start" or
                                      # "turn_complete" to provoke errors

``/v1/query`` renders ``{"conversation_id", "response", "tool_calls",
"input_tokens", "output_tokens"}``. ``/v1/streaming_query`` renders the SSE
events ``start{conversation_id}``, ``token{id: 0, token: response}``, one
``tool_call`` per call, ``turn_complete{token: response}``,
``end{input_tokens, output_tokens}``, minus ``omit_events``. Using ``turn``
on any other path is a ``ScriptError`` at load.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

FAKE_LSC_API = 1
PREFIX = "/lsc"
# Only headers the client controls and that are part of the C8/C9 contract.
HEADER_ALLOW_LIST = ("authorization", "mcp-headers", "content-type", "accept")
MISS_STATUS = 599
CONV = "@CONV@"
TURN_PATHS = ("/query", "/streaming_query")
SSE_EVENTS = ("start", "token", "tool_call", "turn_complete", "end")


class ScriptError(ValueError):
    """The lsc_script.json is malformed (raised at load, never mid-run)."""


class _NoConversationId(Exception):
    pass


@dataclass
class Reply:
    """One HTTP reply produced by the fake."""

    status: int
    content_type: str
    body: bytes

    def as_http(self) -> tuple[int, dict[str, str], bytes]:
        """Return the ``(status, headers, body)`` triple ``serve`` writes."""
        return self.status, {"Content-Type": self.content_type}, self.body


def _validate(script: dict[str, Any]) -> None:
    if not isinstance(script, dict) or not isinstance(script.get("routes"), list):
        raise ScriptError("script must be {'routes': [...]}")
    for i, route in enumerate(script["routes"]):
        if not isinstance(route, dict):
            raise ScriptError(f"route {i} must be a dict")
        where = f"route {i} ({route.get('path')!r})"
        if not str(route.get("path", "")).startswith("/"):
            raise ScriptError(f"{where}: 'path' must start with '/'")
        if not isinstance(route.get("responses"), list) or not route["responses"]:
            raise ScriptError(f"{where}: 'responses' must be a non-empty list")
        for j, resp in enumerate(route["responses"]):
            kinds = [k for k in ("json", "sse", "raw", "turn") if k in resp]
            if len(kinds) != 1:
                raise ScriptError(
                    f"{where} response {j}: exactly one of json/sse/raw/turn"
                )
            if kinds == ["turn"]:
                if not route["path"].endswith(TURN_PATHS):
                    raise ScriptError(
                        f"{where}: 'turn' only for /query and /streaming_query"
                    )
                bad = set(resp["turn"].get("omit_events", [])) - set(SSE_EVENTS)
                if bad:
                    raise ScriptError(
                        f"{where} response {j}: unknown omit_events {sorted(bad)}"
                    )
                if resp["turn"].get("sse_tool_call_form", "token") not in (
                    "token",
                    "flat",
                ):
                    raise ScriptError(
                        f"{where} response {j}: sse_tool_call_form token|flat"
                    )


def _substitute(obj: Any, conv: str | None) -> Any:
    if obj == CONV:
        if conv is None:
            raise _NoConversationId
        return conv
    if isinstance(obj, dict):
        return {k: _substitute(v, conv) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, conv) for v in obj]
    return obj


def _render_turn(path: str, turn: dict[str, Any], conv: str | None) -> Reply:
    t = _substitute({"conversation_id": CONV, **turn}, conv)
    calls = t.get("tool_calls", [])
    if path.endswith("/streaming_query"):
        omit = set(t.get("omit_events", []))
        flat = t.get("sse_tool_call_form", "token") == "flat"
        events = [
            ("start", {"conversation_id": t["conversation_id"]}),
            ("token", {"id": 0, "token": t.get("response", "")}),
        ]
        events += [("tool_call", c if flat else {"token": c}) for c in calls]
        events += [
            ("turn_complete", {"token": t.get("response", "")}),
            (
                "end",
                {
                    "input_tokens": t.get("input_tokens", 0),
                    "output_tokens": t.get("output_tokens", 0),
                },
            ),
        ]
        text = "".join(
            f"data: {json.dumps({'event': e, 'data': d})}\n\n"
            for e, d in events
            if e not in omit
        )
        return Reply(200, "text/event-stream", text.encode())
    body = {
        "conversation_id": t["conversation_id"],
        "response": t.get("response", ""),
        "tool_calls": calls,
        "input_tokens": t.get("input_tokens", 0),
        "output_tokens": t.get("output_tokens", 0),
    }
    return Reply(200, "application/json", json.dumps(body).encode())


@dataclass
class FakeLsc:
    """Route table + request recorder. Thread-safe (one lock)."""

    script: dict[str, Any]
    log_path: Path
    _served: dict[int, int] = field(default_factory=dict)
    _misses: list[dict[str, Any]] = field(default_factory=list)
    _requests: list[dict[str, Any]] = field(default_factory=list)
    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Validate the script before any request is served."""
        _validate(self.script)

    @classmethod
    def from_script(cls, script: dict[str, Any], log_path: Path) -> "FakeLsc":
        """Validate a script dict; truncate the request log."""
        fake = cls(script, log_path)
        log_path.write_text("", encoding="utf-8")
        return fake

    @classmethod
    def from_files(cls, script_path: Path, log_path: Path) -> "FakeLsc":
        """Load + validate a fixture script; truncate the request log."""
        return cls.from_script(
            json.loads(script_path.read_text(encoding="utf-8")), log_path
        )

    # -- matching -----------------------------------------------------------
    @staticmethod
    def _body_json(body: bytes) -> dict[str, Any]:
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _prompt(data: dict[str, Any]) -> str | None:
        for key in ("query", "question", "input"):
            if isinstance(data.get(key), str):
                return data[key]
        return None

    def _route_index(self, path: str, data: dict[str, Any]) -> int | None:
        prompt = self._prompt(data)
        for idx, route in enumerate(self.script["routes"]):
            if route["path"] != path or route.get("prompt") not in (None, prompt):
                continue
            want = route.get("match_conversation_id")
            if want == "absent" and "conversation_id" in data:
                continue
            if want not in (None, "absent") and data.get("conversation_id") != want:
                continue
            return idx
        return None

    # -- entry point --------------------------------------------------------
    def handle(
        self, method: str, raw_path: str, headers: Mapping[str, str], body: bytes
    ) -> Reply:
        """Serve one request. ``raw_path`` still carries the ``/lsc`` prefix."""
        path = raw_path[len(PREFIX) :] if raw_path.startswith(PREFIX) else raw_path
        lower = {k.lower(): v for k, v in headers.items()}
        data = self._body_json(body)
        with self._lock:
            self._seq += 1
            entry = {
                "seq": self._seq,
                "method": method,
                "path": path,
                "headers": {h: lower[h] for h in HEADER_ALLOW_LIST if h in lower},
                "body": body.decode("utf-8", errors="replace"),
            }
            self._requests.append(entry)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")

            idx = self._route_index(path, data)
            if idx is None:
                return self._miss(entry, "no route")
            route = self.script["routes"][idx]
            ordinal = self._served.get(idx, 0)
            if ordinal >= len(route["responses"]):
                return self._miss(entry, f"route {idx} exhausted")
            self._served[idx] = ordinal + 1
            conv = data.get("conversation_id")
            conv = conv if isinstance(conv, str) else route.get("new_conversation_id")
            try:
                return self._render(path, route["responses"][ordinal], conv)
            except _NoConversationId:
                return self._miss(entry, "no conversation_id to echo")

    def _miss(self, entry: dict[str, Any], why: str) -> Reply:
        self._misses.append({"why": why, **entry})
        body = json.dumps({"detail": f"fake_lsc miss: {why}"}).encode()
        return Reply(MISS_STATUS, "application/json", body)

    @staticmethod
    def _render(path: str, spec: dict[str, Any], conv: str | None) -> Reply:
        status = int(spec.get("status", 200))
        if "turn" in spec:
            reply = _render_turn(path, spec["turn"], conv)
            reply.status = status
            return reply
        if "sse" in spec:
            events = _substitute(spec["sse"], conv)
            text = "".join(f"data: {json.dumps(ev)}\n\n" for ev in events)
            return Reply(status, "text/event-stream", text.encode())
        if "raw" in spec:
            ctype = spec.get("content_type", "text/event-stream")
            return Reply(status, ctype, spec["raw"].encode())
        return Reply(
            status,
            "application/json",
            json.dumps(_substitute(spec["json"], conv)).encode(),
        )

    # -- assertion helpers --------------------------------------------------
    def misses(self) -> list[dict[str, Any]]:
        """Return requests that got HTTP 599."""
        with self._lock:
            return list(self._misses)

    def unused(self) -> list[str]:
        """Return scripted responses that were never requested."""
        out = []
        with self._lock:
            for idx, route in enumerate(self.script["routes"]):
                left = len(route["responses"]) - self._served.get(idx, 0)
                if left > 0:
                    out.append(
                        f"{route['path']} prompt={route.get('prompt')!r}: {left} unused"
                    )
        return out

    def requests(self) -> list[dict[str, Any]]:
        """Return every logged request, in arrival order."""
        with self._lock:
            return [dict(r) for r in self._requests]

    def http_handler(
        self,
    ) -> Callable[
        [str, str, Mapping[str, str], bytes], tuple[int, dict[str, str], bytes]
    ]:
        """Return the callable ``prefix("/lsc", ...)`` expects."""

        def respond(
            method: str, path: str, headers: Mapping[str, str], body: bytes
        ) -> tuple[int, dict[str, str], bytes]:
            return self.handle(method, path, headers, body).as_http()

        return respond


def _serve_standalone(port: int, script: Path, log: Path) -> None:
    """Bind 127.0.0.1:port. Tests mount ``http_handler`` via prefix and serve."""
    fake = FakeLsc.from_files(script, log)
    summary = log.parent / (log.stem + ".summary.json")
    summary.write_text(
        json.dumps({"misses": [], "unused": fake.unused()}, indent=1),
        encoding="utf-8",
    )

    class _Handler(BaseHTTPRequestHandler):
        """Spike server. Golden tests use ``prefix`` and ``serve`` instead."""

        protocol_version = "HTTP/1.1"

        def handle_one(self) -> None:
            """Read one request and write the fake's reply."""
            length = int(self.headers.get("content-length", 0))
            payload = self.rfile.read(length) if length else b""
            status, out_headers, body = fake.handle(
                self.command, self.path, dict(self.headers), payload
            ).as_http()
            summary.write_text(
                json.dumps(
                    {"misses": fake.misses(), "unused": fake.unused()}, indent=1
                ),
                encoding="utf-8",
            )
            self.send_response(status)
            for key, value in out_headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            """Drop the default stderr access log."""
            del code, size

    for verb in ("GET", "POST"):
        setattr(_Handler, f"do_{verb}", _Handler.handle_one)

    ThreadingHTTPServer(("127.0.0.1", port), _Handler).serve_forever()


if __name__ == "__main__":
    _serve_standalone(int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
