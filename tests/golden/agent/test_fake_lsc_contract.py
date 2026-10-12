"""Contract tests for harness/fake_lsc.py (FAKE_LSC_API = 1).

These pin the interface Engineer 3's WP0d relies on (read-only use):
conversation_id echo, scripted nameless / argument-less tool calls on both
/v1/query and /v1/streaming_query, strict 599 misses, the request log, and
mounting on the shared server with prefix, Router and serve.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.golden.harness._http_server import Router, prefix, serve
from tests.golden.harness.fake_lsc import FAKE_LSC_API, FakeLsc, ScriptError

pytestmark = pytest.mark.golden

ODD_CALLS = [
    {"tool_name": "no_args_tool"},  # no arguments key
    {"tool_name": "null_args_tool", "arguments": None},  # arguments null
    {"arguments": {"x": 1}},  # nameless
    {"tool_name": "get_pods", "arguments": {"namespace": "payments"}},
]


def _post(fake: FakeLsc, path: str, body: dict[str, Any]) -> tuple[int, bytes]:
    reply = fake.handle(
        "POST",
        "/lsc" + path,
        {"Content-Type": "application/json"},
        json.dumps(body).encode(),
    )
    return reply.status, reply.body


def _sse(body: bytes) -> list[dict[str, Any]]:
    return [
        json.loads(chunk[len("data: ") :])
        for chunk in body.decode().split("\n\n")
        if chunk
    ]


def _script(path: str, **turn: Any) -> dict[str, Any]:
    return {
        "routes": [
            {
                "path": path,
                "match_conversation_id": "absent",
                "new_conversation_id": "conv-ABC-123",
                "responses": [{"turn": turn}],
            },
            {
                "path": path,
                "match_conversation_id": "conv-ABC-123",
                "responses": [{"turn": turn}],
            },
        ]
    }


def _judge(
    method: str, path: str, headers: Mapping[str, str], body: bytes
) -> tuple[int, dict[str, str], bytes]:
    del method, path, headers, body
    return 200, {"Content-Type": "application/json"}, b'{"judge":true}'


def test_api_version() -> None:
    """Pin the read-only API number WP0d imports."""
    assert FAKE_LSC_API == 1


def test_query_echo_and_odd_tool_calls(tmp_path: Path) -> None:
    """Echo @CONV@ and copy nameless or argument-less calls verbatim."""
    fake = FakeLsc.from_script(
        _script("/v1/query", response="ok", tool_calls=ODD_CALLS),
        tmp_path / "log.jsonl",
    )
    status, body = _post(
        fake, "/v1/query", {"query": "q1", "provider": "p", "model": "m"}
    )
    assert status == 200
    first = json.loads(body)
    assert first["conversation_id"] == "conv-ABC-123"
    assert first["tool_calls"] == ODD_CALLS  # verbatim: keys stay absent / null
    status, body = _post(
        fake, "/v1/query", {"query": "q2", "conversation_id": "conv-ABC-123"}
    )
    assert status == 200 and json.loads(body)["conversation_id"] == "conv-ABC-123"
    assert not fake.misses()
    assert not fake.unused()


@pytest.mark.parametrize("form", ["token", "flat"])
def test_streaming_echo_and_odd_tool_calls(tmp_path: Path, form: str) -> None:
    """Render both SSE tool-call forms and thread the echoed conversation id."""
    fake = FakeLsc.from_script(
        _script(
            "/v1/streaming_query",
            response="ok",
            tool_calls=ODD_CALLS,
            sse_tool_call_form=form,
        ),
        tmp_path / "log.jsonl",
    )
    status, body = _post(fake, "/v1/streaming_query", {"query": "q1"})
    events = _sse(body)
    assert status == 200
    assert [e["event"] for e in events] == ["start", "token"] + ["tool_call"] * 4 + [
        "turn_complete",
        "end",
    ]
    assert events[0]["data"] == {"conversation_id": "conv-ABC-123"}
    calls = [
        e["data"] if form == "flat" else e["data"]["token"]
        for e in events
        if e["event"] == "tool_call"
    ]
    assert calls == ODD_CALLS
    status, _ = _post(
        fake, "/v1/streaming_query", {"query": "q2", "conversation_id": "conv-ABC-123"}
    )
    assert status == 200
    assert not fake.misses()
    assert not fake.unused()


def test_omit_events(tmp_path: Path) -> None:
    """Drop a named SSE event when the turn lists it in omit_events."""
    script = {
        "routes": [
            {
                "path": "/v1/streaming_query",
                "new_conversation_id": "c",
                "responses": [{"turn": {"omit_events": ["turn_complete"]}}],
            }
        ]
    }
    fake = FakeLsc.from_script(script, tmp_path / "log.jsonl")
    events = _sse(_post(fake, "/v1/streaming_query", {"query": "q"})[1])
    assert "turn_complete" not in [e["event"] for e in events]


def test_conv_placeholder_in_verbatim_json(tmp_path: Path) -> None:
    """Replace @CONV@ inside a verbatim json response from the request id."""
    script = {
        "routes": [
            {
                "path": "/v1/query",
                "responses": [{"json": {"conversation_id": "@CONV@", "response": "x"}}],
            }
        ]
    }
    fake = FakeLsc.from_script(script, tmp_path / "log.jsonl")
    _, body = _post(fake, "/v1/query", {"query": "q", "conversation_id": "given"})
    assert json.loads(body)["conversation_id"] == "given"


def test_strict_misses_and_log(tmp_path: Path) -> None:
    """Answer unmatched, unechoable and exhausted routes with HTTP 599."""
    script = {
        "routes": [
            {
                "path": "/v1/query",
                "prompt": "known",
                "responses": [{"json": {"conversation_id": "@CONV@"}}],
            }
        ]
    }
    fake = FakeLsc.from_script(script, tmp_path / "log.jsonl")
    assert _post(fake, "/v1/query", {"query": "unknown"})[0] == 599
    assert _post(fake, "/v1/query", {"query": "known"})[0] == 599  # nothing to echo
    assert _post(fake, "/v1/query", {"query": "known"})[0] == 599  # exhausted
    assert [m["why"] for m in fake.misses()] == [
        "no route",
        "no conversation_id to echo",
        "route 0 exhausted",
    ]
    logged = [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert logged == fake.requests() and [r["seq"] for r in logged] == [1, 2, 3]
    assert logged[0]["headers"] == {"content-type": "application/json"}


def test_mounted_on_shared_server(tmp_path: Path) -> None:
    """Serve /lsc beside /v1 on loopback through prefix, Router and serve."""
    fake = FakeLsc.from_script(
        _script("/v1/query", response="ok", tool_calls=ODD_CALLS),
        tmp_path / "log.jsonl",
    )
    router = Router(prefix("/v1", _judge), prefix("/lsc", fake.http_handler()))
    with serve(router) as base:
        assert base.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(base + "/v1/chat/completions", data=b"{}") as got:
            assert b"judge" in got.read()
        sent = json.dumps({"query": "q1"}).encode()
        posted = urllib.request.Request(
            base + "/lsc/v1/query", data=sent, method="POST"
        )
        with urllib.request.urlopen(posted) as got:
            payload = json.loads(got.read())
        assert payload["conversation_id"] == "conv-ABC-123"
        assert payload["tool_calls"] == ODD_CALLS
        missed = urllib.request.Request(base + "/lsc/nope", data=b"{}", method="POST")
        try:
            with urllib.request.urlopen(missed) as got:
                got.read()
            code = 200
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
        assert code == 599
    assert [row["path"] for row in fake.requests()] == ["/v1/query", "/nope"]


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"routes": [{"path": "v1/query", "responses": [{"json": {}}]}]},
        {"routes": [{"path": "/v1/query", "responses": []}]},
        {"routes": [{"path": "/v1/query", "responses": [{"json": {}, "sse": []}]}]},
        {"routes": [{"path": "/v1/responses", "responses": [{"turn": {}}]}]},
        {
            "routes": [
                {
                    "path": "/v1/streaming_query",
                    "responses": [{"turn": {"omit_events": ["nope"]}}],
                }
            ]
        },
    ],
)
def test_bad_scripts_fail_at_load(tmp_path: Path, bad: dict[str, Any]) -> None:
    """Reject a malformed script at load, before any request."""
    with pytest.raises(ScriptError):
        FakeLsc.from_script(bad, tmp_path / "log.jsonl")
