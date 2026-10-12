"""Self-tests for the strict replay judge (stdlib only; no CLI, no network beyond loopback).

Marked ``golden`` so golden.yaml runs it (EM ruling); it needs neither the golden
env nor ``nlp-metrics``.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request

import pytest

from tests.golden.harness import judge_server as js

pytestmark = pytest.mark.golden
PATH = "/v1/chat/completions"
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BODY = '{"messages":[{"content":"hi","role":"user"}],"model":"m","temperature":0.0}'


def _entry(body: str, ordinal: int, content: str) -> dict:
    reply = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return {
        "key": js.replay_key("POST", PATH, body.encode()),
        "ordinal": ordinal,
        "request": {"method": "POST", "path": PATH, "body": body},
        "response": {"status": 200, "body": reply},
        "source": "handwritten",
        "scrubbed": False,
    }


def _content(raw: bytes) -> str:
    return json.loads(raw)["choices"][0]["message"]["content"]


def test_key_is_ds_literal_format() -> None:
    """Key = sha256("POST <path>\\n" + raw body), as record_cassettes.py writes it."""
    want = hashlib.sha256(f"POST {PATH}\n{BODY}".encode()).hexdigest()
    assert js.replay_key("POST", PATH, BODY.encode()) == want


def test_ordinals_misses_unused_and_nearest_diff() -> None:
    """Identical requests are served in ordinal order; extras are 599 misses."""
    other = BODY.replace('"m"', '"other"')
    replay = js.JudgeReplay(
        [_entry(BODY, 1, "second"), _entry(BODY, 0, "first"), _entry(other, 0, "x")]
    )
    first, second = (replay.handle("POST", PATH, {}, BODY.encode()) for _ in "12")
    assert [first[0], _content(first[2])] == [200, "first"]
    assert [second[0], _content(second[2])] == [200, "second"]
    assert replay.handle("POST", PATH, {}, BODY.encode())[0] == 599  # past last ordinal
    altered = BODY.replace('"temperature":0.0', '"temperature":0.5')
    assert replay.handle("POST", PATH, {}, altered.encode())[0] == 599
    with pytest.raises(AssertionError) as err:
        replay.assert_clean()
    msg = str(err.value)
    assert msg.count("MISS ") == 2 and msg.count("UNUSED ") == 1
    assert '- "temperature": 0.0' in msg and '+ "temperature": 0.5' in msg


def test_rejects_entry_whose_key_does_not_match_body() -> None:
    """A hand edit to a request body without re-keying is caught at load."""
    bad = _entry(BODY, 0, "x")
    bad["request"]["body"] = BODY.replace("hi", "hello")
    with pytest.raises(ValueError, match="hash of its request"):
        js.JudgeReplay([bad])


def test_served_over_loopback_http() -> None:
    """Through the transport: a hit returns 200, an unknown body returns HTTP 599."""
    with js.serve([_entry(BODY, 0, "first")]) as (base_url, replay):
        assert base_url.startswith("http://127.0.0.1:") and base_url.endswith("/v1")

        def post(body: str) -> tuple[int, bytes]:
            req = urllib.request.Request(
                base_url + "/chat/completions",
                data=body.encode(),
                headers={"Authorization": "Bearer ignored"},
            )
            try:
                with _NO_PROXY.open(req, timeout=10) as resp:  # nosec B310
                    return resp.status, resp.read()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read()

        status, raw = post(BODY)
        assert (status, _content(raw)) == (200, "first")
        status, raw = post(BODY + " ")
        assert status == 599 and "golden replay miss" in raw.decode()
    assert len(replay.requests) == 2 and len(replay.misses) == 1 and not replay.unused()
