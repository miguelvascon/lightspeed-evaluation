# Recording the happy-path judge cassette (Miguel only)

`tests/golden/harness/record_cassettes.py` replaces WP0c's hand-written
happy-path responses with real gpt-4o-mini responses. Only Miguel runs it, on
his machine, with his key. Agents and CI never run it in record mode.

## What it sends

The fixture is `tests/golden/metrics/fixtures/judge_happy_path/` (14 metric ids,
5 turns, 3 conversations). The dry run measured **30 chat requests**. The hard
cap defaults to **99**, which allows 3 attempts per request plus headroom for
extra DeepEval intentions. Embeddings are never sent upstream; they stay
hand-written (dyadic vectors). The expected cost is about **$0.007**. Even at
the cap, the worst case is under **$0.05** (gpt-4o-mini list prices; override
with `--price-in`/`--price-out`).

## Run it

```bash
uv sync --frozen --group dev --extra nlp-metrics
# 1. Optional: a dry run. It makes no network calls and needs no key.
uv run --no-sync python tests/golden/harness/record_cassettes.py --dry-run
# 2. Record. Run it in an interactive terminal: it prints the estimate and asks y/N
#    before sending anything. Without --record-with-my-key it refuses.
OPENAI_API_KEY=sk-... uv run --no-sync python tests/golden/harness/record_cassettes.py --record-with-my-key
```

Flags: `--record-with-my-key` is required for record mode (it makes real, paid
calls). The y/N prompt cannot be skipped, and a non-interactive run refuses.
`--max-requests N` sets the cap. `--upstream URL`
must be https. `--keep-work` keeps the scratch directory.

## Guarantees

- The key is read only from your environment, sent only in the proxy's
  `Authorization` header to the upstream, and never logged or written. The
  `lightspeed-eval` subprocess gets a placeholder key, the local proxy URL, and
  a startup guard that blocks every non-loopback connection.
- Request bodies are scrubbed (`api_key` fields, `sk-…`, `Bearer …`) before
  hashing and writing. A secret scan runs before the file is moved into place.
  If the scan finds a hit, the file is deleted and the script exits with code 4.
- An upstream non-2xx response, a streaming request, or reaching the cap aborts
  the run. Nothing is written in that case.
- After writing, the script replays the cassette in strict mode with no
  upstream. It requires 0 misses, 0 unused entries, and identical metric rows.

## Output

- `tests/golden/metrics/cassettes/judge_happy_path.jsonl` has one exchange per
  line: `key` (sha256 of `POST <path>\n` + body), `ordinal`, `request`,
  `response`, `source` (`recorded` | `handwritten-embedding`), and `scrubbed`.
- `judge_happy_path.record-summary.json` records the counts and the replay
  result.

Commit both through a `golden-regen` PR. That PR has no `src/` changes, is
recorded at the G0 baseline commit, and needs DS sign-off and Miguel's OK.
Then run `LS_GOLDEN_UPDATE=1 uv run pytest tests/golden/metrics` to refresh
`expected/`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Not confirmed |
| 2 | Bad setup (no key or fixture) |
| 3 | Aborted, or the counts don't match the estimate |
| 4 | The secret scan found a hit |
| 5 | The replay check failed |
