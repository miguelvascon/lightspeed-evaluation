"""Hand-written judge for the WP0c golden cassettes (DS-owned).

``respond(path, body)`` is a **pure function** of the request: the same request
bytes always get the same response. That gives the per-key identity the
concurrency rule needs (DeepEval ``async_mode`` and Ragas ``n``-splits send
identical bodies concurrently, and replay may serve them in any order).

It is used in two places only:
- ``author_cassettes.py`` runs it live, once, to write ``metrics/cassettes/*.jsonl``.
  The cassettes are the golden artifact. Tests replay them strictly and never
  call this module.
- ``record_cassettes.py --dry-run`` uses it as the local stand-in judge.

Behavior is driven by the fixture text, never by hidden state.
- Default: a schema-valid, *passing* answer for whatever the library asked.
- ``FAIL_PHRASES`` (happy-path fixture only; it must stay free of markers
  because Miguel records it against the real judge): a failing answer.
- Markers in the fixture text (error fixtures only; they're never recorded):

  | marker                  | effect |
  |-------------------------|--------|
  | ``[judge:fail]``        | failing verdicts / low scores |
  | ``[judge:badjson]``     | schema-invalid JSON on every attempt (instructor re-ask exhaustion; DeepEval ``None`` score) |
  | ``[judge:badjson-once]``| schema-invalid JSON first, valid once the re-ask carries the failed answer |
  | ``[judge:no-statements]``| Ragas statement generator returns ``[]`` |
  | ``[judge:http500]``     | HTTP 500 for every attempt |
  | ``[judge:http401]``     | HTTP 401 (bad key) |
  | ``[judge:panel-break]`` | only model ``gpt-4.1-mini`` answers garbage (mixed panel) |
  | ``[judge-reply:TEXT]``  | custom metrics: reply exactly TEXT (``\\n`` → newline) |

Embeddings are dyadic-rational 8-vectors (SPEC-01 §5.2), so the dot products are
exact. Every text gets the "on-topic" base direction plus one ±1/8 bump chosen by
a hash. A text carrying ``[judge:fail]`` gets an orthogonal direction instead.

GEval: a passing score is ``8`` with top_logprobs ``8``=ln 0.75 and ``7``=ln 0.25,
which gives **0.775** (the pinned value). A failing score is ``3``/``2``, which
gives **0.275**.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Optional

FAIL_PHRASES = ("virtual machine image",)
# Panel fixtures use three judge models; answer_correctness differs per model so
# max / average / majority_vote give different results. gpt-4o-mini keeps 0.9 / 0.2.
PANEL_SCORES = {"gpt-4.1-mini": ("0.7", "0.4"), "gpt-4.1-nano": ("0.8", "0.1")}
_REPLY = re.compile(r"\[judge-reply:(.*?)\]", re.S)


# ----------------------------------------------------------------- helpers
def _text(messages: list) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(parts)


def _final_input(text: str) -> dict:
    """Ragas prompts end with 'Now perform the same with the following input\\ninput: {...}\\nOutput: '."""
    i = text.rfind("input: {")
    if i < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i + len("input: "):])
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        return {}


def _system_schema_title(messages: list) -> Optional[str]:
    for m in messages:
        c = m.get("content")
        if m.get("role") == "system" and isinstance(c, str) and "json_schema" in c:
            start = c.find("{", c.find("json_schema"))
            try:
                obj, _ = json.JSONDecoder().raw_decode(c[start:])
                return obj.get("title")
            except ValueError:
                return None
    return None


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts or [text.strip()]


def _tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _chat(body: dict, content: str, logprobs: Optional[dict] = None) -> dict:
    prompt = _text(body.get("messages", []))
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    n = int(body.get("n") or 1)
    choices = [{"index": i, "finish_reason": "stop", "logprobs": logprobs,
                "message": {"role": "assistant", "content": content}} for i in range(n)]
    return {"id": f"chatcmpl-hw-{key}", "object": "chat.completion", "created": 1760000000,
            "model": body.get("model", "unknown"), "choices": choices,
            "usage": {"prompt_tokens": _tokens(prompt), "completion_tokens": _tokens(content) * n,
                      "total_tokens": _tokens(prompt) + _tokens(content) * n}}


def _error(status: int, message: str, etype: str, code: Optional[str] = None) -> tuple[int, dict]:
    return status, {"error": {"message": message, "type": etype, "param": None, "code": code}}


def _geval_logprobs(content: str, score: int) -> dict:
    """Tokens that concatenate to ``content``. The score token carries a 0.75/0.25 split."""
    s, alt = str(score), str(score - 1)
    head, tail = content.split(s, 1)

    def tok(t: str, lp: float, top: list) -> dict:
        return {"token": t, "logprob": lp, "bytes": list(t.encode()), "top_logprobs": top}

    def solo(t: str) -> list:
        return [{"token": t, "logprob": 0.0, "bytes": list(t.encode())}]

    return {"refusal": None, "content": [
        tok(head, 0.0, solo(head)),
        tok(s, math.log(0.75), [
            {"token": s, "logprob": math.log(0.75), "bytes": list(s.encode())},
            {"token": alt, "logprob": math.log(0.25), "bytes": list(alt.encode())},
        ]),
        tok(tail, 0.0, solo(tail)),
    ]}


# ------------------------------------------------------------- embeddings
def embedding_vector(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    if "[judge:fail]" in text:
        v = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
    else:
        v = [0.5] * 8
    v[h[0] % 8] += 0.125 if h[1] & 1 else -0.125
    return v


def respond_embedding(body: dict) -> tuple[int, dict]:
    inputs = body.get("input")
    if not isinstance(inputs, list):
        inputs = [inputs]
    data, toks = [], 0
    for i, t in enumerate(inputs):
        t = t if isinstance(t, str) else json.dumps(t)
        toks += _tokens(t)
        data.append({"object": "embedding", "index": i, "embedding": embedding_vector(t)})
    return 200, {"object": "list", "data": data, "model": body.get("model", "unknown"),
                 "usage": {"prompt_tokens": toks, "total_tokens": toks}}


# ------------------------------------------------------------------ judges
def _custom(body: dict, text: str, fail: bool) -> str:
    m = _REPLY.search(text)
    if m:
        return m.group(1).replace("\\n", "\n")
    if "automated remediation workflow" in text:
        if fail:
            return json.dumps({"reasoning": "The diagnosis misses the OOMKilled root cause and the "
                               "fix is unrelated.", "diagnosis": 0.2, "execution": 0.3,
                               "verification": 0.2, "average": 0.23})
        return json.dumps({"reasoning": "Correct OOMKilled diagnosis, the limit was raised as "
                           "expected and the pod is verified healthy.", "diagnosis": 0.9,
                           "execution": 0.8, "verification": 0.8, "average": 0.83})
    if "expected intent" in text:
        return ("Score: 0\nReason: The response does not serve the expected intent." if fail
                else "Score: 1\nReason: The response serves the expected intent.")
    good, bad = PANEL_SCORES.get(body.get("model", ""), ("0.9", "0.2"))
    return (f"Score: {bad}\nReason: The response contradicts the expected answer." if fail
            else f"Score: {good}\nReason: The response matches the expected answer and adds accurate detail.")


def _ragas(title: str, text: str, fail: bool, no_statements: bool) -> Any:
    inp = _final_input(text)
    if title == "StatementGeneratorOutput":
        if no_statements:
            return {"statements": []}
        sts = _sentences(str(inp.get("answer", "")))
        if fail:
            sts = [f"{s} [judge:fail]" for s in sts]
        return {"statements": sts}
    if title == "NLIStatementOutput":
        return {"statements": [
            {"statement": s, "verdict": 0 if fail or "[judge:fail]" in s else 1,
             "reason": ("The context does not support this statement." if fail or "[judge:fail]" in s
                        else "The context states this directly.")}
            for s in inp.get("statements", [])]}
    if title == "AnswerRelevanceOutput":
        resp = str(inp.get("response", ""))
        q = f"Which command answers this: {resp[:60]}?"
        if fail:
            q += " [judge:fail]"
        return {"question": q, "noncommittal": 0}
    if title == "ContextRecallOutput":
        return {"classifications": [
            {"statement": s, "attributed": 0 if fail else 1,
             "reason": "Not in the context." if fail else "The context contains this."}
            for s in _sentences(str(inp.get("answer", "")))]}
    if title == "ContextPrecisionOutput":
        return {"reason": "The context is not useful for the answer." if fail
                else "The context was useful in arriving at the answer.", "verdict": 0 if fail else 1}
    if title == "ContextRelevanceOutput":
        return {"rating": 0 if fail else 2}
    raise KeyError(f"unhandled Ragas schema {title!r}")


def _deepeval(name: str, text: str, fail: bool) -> Any:
    if name == "UserIntentions":
        return {"intentions": ["The user wants to inspect workloads in their namespace."]}
    if name in ("ConversationCompletenessVerdict", "TurnRelevancyVerdict"):
        if fail:
            return {"verdict": "no", "reason": "The assistant did not address the user's request. [judge:fail]"}
        return {"verdict": "yes", "reason": None}
    if name == "KnowledgeRetentionVerdict":
        if fail:
            return {"verdict": "yes", "reason": "The assistant forgot the namespace stated earlier. [judge:fail]"}
        return {"verdict": "no", "reason": None}
    if name == "Knowledge":
        return {"data": {"namespace": "payments"}}
    if name in ("ConversationCompletenessScoreReason", "TurnRelevancyScoreReason",
                "KnowledgeRetentionScoreReason"):
        return {"reason": "The score is low because the assistant missed the user's needs."
                if fail else "The score is high because every user need was met."}
    if name == "Steps":
        return {"steps": ["Check the commands are correct for the platform.",
                          "Check the response matches the expected output.",
                          "Penalize incorrect or unsafe instructions."]}
    raise KeyError(f"unhandled DeepEval schema {name!r}")


def respond(path: str, body: dict) -> tuple[int, dict]:
    """Return (HTTP status, JSON body) for one judge request. Pure."""
    if path.endswith("/embeddings"):
        return respond_embedding(body)
    messages = body.get("messages", [])
    text = _text(messages)
    if body.get("stream"):
        return _error(400, "streaming is not supported by the golden judge", "invalid_request_error")
    if "[judge:http500]" in text:
        return _error(500, "The server had an error while processing your request.", "server_error")
    if "[judge:http401]" in text:
        return _error(401, "Incorrect API key provided.", "invalid_request_error", "invalid_api_key")
    reasked = any(m.get("role") == "assistant" for m in messages)
    badjson = "[judge:badjson]" in text or ("[judge:badjson-once]" in text and not reasked)
    broken_judge = "[judge:panel-break]" in text and body.get("model") == "gpt-4.1-mini"
    fail = "[judge:fail]" in text or any(p in text for p in FAIL_PHRASES)

    rf = body.get("response_format")
    # GEval scoring (logprobs requested)
    if body.get("logprobs"):
        if badjson or broken_judge:
            return 200, _chat(body, "I cannot score this.")
        score = 3 if fail else 8
        content = json.dumps({"score": score, "reason": "The response contains incorrect commands."
                              if fail else "The response is technically accurate."})
        return 200, _chat(body, content, _geval_logprobs(content, score))
    # DeepEval structured outputs
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        if badjson or broken_judge:
            return 200, _chat(body, "Sorry, here is my answer: maybe")
        name = rf.get("json_schema", {}).get("name", "")
        return 200, _chat(body, json.dumps(_deepeval(name, text, fail)))
    # Ragas (instructor JSON mode, schema in the system prompt)
    title = _system_schema_title(messages)
    if title:
        if badjson or broken_judge:
            return 200, _chat(body, json.dumps({"unexpected_field": "not what the schema asks"}))
        return 200, _chat(body, json.dumps(_ragas(title, text, fail, "[judge:no-statements]" in text)))
    # custom metrics: free text
    if broken_judge:
        return 200, _chat(body, "I cannot grade this.")
    return 200, _chat(body, _custom(body, text, fail))
