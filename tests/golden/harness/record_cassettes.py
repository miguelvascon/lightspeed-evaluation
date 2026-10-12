#!/usr/bin/env python3
"""Record (or dry-run) the WP0c happy-path judge cassette.

Miguel runs this himself, when and where he chooses. No agent runs it in
record mode, and no agent holds or uses an OpenAI key.

What it does
------------
1. Estimate pass (always, fully local). Starts the hand-written judge
   (``judge_responses.py``, the same one WP0c's cassettes come from) on 127.0.0.1, and runs the real
   ``lightspeed-eval`` CLI against it once per metric (``--metrics <id>``)
   and once for the whole fixture. It counts judge requests per metric and
   in total.
2. Prints the request count per metric, the total, the hard cap and an
   approximate cost at gpt-4o-mini prices. Then it asks ``Proceed? [y/N]``
   (always; there is no way to skip it, and a non-TTY stdin refuses).
3. Record pass (skipped with ``--dry-run``). The same local server becomes a
   recording proxy:
   - Chat completions are forwarded to the upstream (default
     https://api.openai.com/v1) with ``Authorization: Bearer $OPENAI_API_KEY``.
     The key is read from this process's environment only. It is never given
     to the CLI (the CLI gets a placeholder), never logged and never written.
   - Embeddings are never forwarded. They are synthesized locally as dyadic
     vectors (SPEC-01 §8.c.2), at no cost.
   - Any upstream non-2xx answer, or any request beyond the hard cap, aborts
     the run before anything is written.
   With ``--dry-run`` this pass uses the hand-written judge instead, writes the
   cassette to a scratch directory, and makes no network call at all.
4. Writes ``tests/golden/metrics/cassettes/judge_happy_path.jsonl`` (one
   exchange per line: key, ordinal, request, response, source). In record mode
   it replaces the hand-written happy-path cassette. Request bodies are
   scrubbed (``api_key`` fields and anything key-shaped). Then it runs a
   secret scan; on a hit the file is deleted and the script exits 4.
5. Replay check: re-runs the CLI against the written cassette in strict replay
   mode, with no upstream. It requires 0 misses, 0 unused entries, and
   identical metric results to the recording run.

Run it from the repo root with the repo's locked environment:
    uv sync --frozen
    uv run --no-sync python tests/golden/harness/record_cassettes.py --dry-run
    OPENAI_API_KEY=... uv run --no-sync python \
        tests/golden/harness/record_cassettes.py --record-with-my-key
"""

# pylint: disable=line-too-long,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-return-statements,too-many-branches,too-many-statements
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root: run as a script
HARD_CAP = 99  # absolute ceiling on upstream forwards; --max-requests can only lower it
from tests.golden._normalize import normalize_addresses, normalize_obj  # noqa: E402
from tests.golden.harness import cassette_proxy as cp  # noqa: E402
from tests.golden.harness import judge_responses, judge_server  # noqa: E402
from tests.golden.harness._http_server import serve  # noqa: E402

SITE_DIR = HERE.parent / "_sitecustomize"  # WP0a: order pin, numpy seed, socket guard
ENV_DROP_PREFIXES = (
    "OPENAI_",
    "AZURE_",
    "ANTHROPIC_",
    "GEMINI_",
    "GOOGLE_",
    "WATSONX_",
    "HOSTED_VLLM_",
    "LITELLM_",
    "LANGFUSE_",
    "MLFLOW_",
)
ENV_DROP_EXACT = {
    "API_KEY",
    "OLLAMA_HOST",
    "KUBECONFIG",
    "DEEPEVAL_API_KEY",
    "SSL_CERT_FILE",
}
CLI_PLACEHOLDER_KEY = "golden-fake"
CSV_TIMING = {
    "execution_time",
    "evaluation_latency",
    "agent_latency",
    "time_to_first_token",
    "streaming_duration",
    "tokens_per_second",
}

FIXTURE = "judge_happy_path"
DEFAULT_UPSTREAM = "https://api.openai.com/v1"
DEFAULT_PRICE_IN = 0.15  # gpt-4o-mini USD per 1M input tokens
DEFAULT_PRICE_OUT = 0.60  # gpt-4o-mini USD per 1M output tokens
TYPICAL_OUT_TOKENS = 200
RETRY_FACTOR = 3  # instructor re-asks, DeepEval tenacity, LiteLLM num_retries
INTENTION_HEADROOM = 3  # DeepEval completeness: one verdict per extracted intention


def log(msg: str) -> None:
    """Print one progress line, flushed."""
    print(msg, flush=True)


def fixture_metrics(fixture_dir: Path) -> list[str]:
    """Metric ids the fixture uses, in first-seen order."""
    import yaml

    seen: list[str] = []
    for conv in yaml.safe_load((fixture_dir / "eval_data.yaml").read_text()):
        seen += conv.get("conversation_metrics") or []
        for t in conv.get("turns", []):
            seen += t.get("turn_metrics") or []
    return list(dict.fromkeys(seen))


def cli_env(base_url: str, mpl_dir: str) -> dict:
    """The CLI's env: placeholder key, §5.2 pins, WP0a sitecustomize; no provider vars."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(ENV_DROP_PREFIXES) and k not in ENV_DROP_EXACT
    }
    env.update(
        {
            "OPENAI_API_KEY": CLI_PLACEHOLDER_KEY,
            "OPENAI_BASE_URL": base_url,
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
            "LC_ALL": "C.UTF-8",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": mpl_dir,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONPATH": str(SITE_DIR)
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "LS_GOLDEN_NP_SEED": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    for k in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(k, None)
    return env


def run_cli(
    exe: list[str],
    cwd: Path,
    cfg: tuple[Path, Path],
    out_dir: Path,
    env: dict,
    proxy: cp.RecordingJudge,
    extra_args: tuple[str, ...] = (),
    timeout: int = 1800,
) -> int:
    """Run the CLI; kill it if the proxy aborts (cap, upstream error) or it times out."""
    cmd = exe + [
        "--system-config",
        str(cfg[0]),
        "--eval-data",
        str(cfg[1]),
        "--output-dir",
        str(out_dir),
        *extra_args,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(
        out_dir.parent / f"{out_dir.name}.cli.log", "w", encoding="utf-8"
    ) as logf:
        with subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as proc:
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if proxy.abort_reason or time.monotonic() > deadline:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    return -9
                time.sleep(0.2)
    return proc.returncode


def capture(out: Path, rc: int, roots: list[str]) -> dict:
    """CSV rows and summary JSON, normalized with WP0a's ``_normalize``."""
    rows: list[dict] = []
    for path in sorted(out.glob("*_detailed.csv"))[:1]:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [
                {k: (float(v) if k in CSV_TIMING and v else v) for k, v in row.items()}
                for row in csv.DictReader(f)
            ]  # CSV timings as floats so _normalize masks them
    sums = sorted(out.glob("*_summary.json"))
    summary = json.loads(sums[0].read_text()) if sums else None
    doc: Any = normalize_obj({"csv_rows": rows, "summary": summary}, roots=roots)
    doc = json.loads(normalize_addresses(json.dumps(doc)))
    return {"exit_code": rc, **doc}


class Runner:  # pylint: disable=too-many-instance-attributes
    """Runs the CLI against one loopback server whose handler can be swapped."""

    def __init__(
        self, repo: Path, fixture_dir: Path, work: Path, proxy: cp.RecordingJudge
    ) -> None:
        """Start the server with the recording judge as the active handler."""
        self.repo, self.fx, self.work, self.proxy = repo, fixture_dir, work, proxy
        self.exe = ["uv", "run", "--no-sync", "lightspeed-eval"]
        self.active: Any = proxy
        self._stack = contextlib.ExitStack()
        self.base = self._stack.enter_context(serve(self._dispatch))

    def _dispatch(self, *args: Any) -> Any:
        return self.active(*args)

    def close(self) -> None:
        """Stop the loopback server."""
        self._stack.close()

    def run(
        self, label: str, tag: str, metric: Optional[str] = None
    ) -> tuple[int, Path]:
        """Run the CLI once under ``/t/<tag>/v1``; returns (exit code, output dir)."""
        out = self.work / label
        env = cli_env(f"{self.base}/t/{tag}/v1", str(self.work / f"mpl_{label}"))
        args = ("--metrics", metric) if metric else ()
        cfg = (self.fx / "system.yaml", self.fx / "eval_data.yaml")
        return run_cli(self.exe, self.repo, cfg, out, env, self.proxy, args), out

    def capture(self, out: Path, rc: int) -> dict:
        """Normalized results of one run."""
        return capture(out, rc, [str(out), str(self.fx)])


def estimate(r: Runner, jobs: int) -> dict:
    """Count judge requests per metric and in total with the hand-written judge."""
    r.proxy.set_fake()
    r.proxy.reset()
    metrics = fixture_metrics(r.fx)
    log(
        f"[estimate] hand-written judge, {len(metrics)} metrics + 1 full run (no network)"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        codes = dict(
            ex.map(
                lambda im: (im[1], r.run(f"est_{im[0]}", f"m{im[0]}", im[1])[0]),
                enumerate(metrics),
            )
        )
    rc_full, out_full = r.run("est_full", "full")
    bad = {m: c for m, c in codes.items() if c != 0}
    if bad or rc_full != 0:
        raise SystemExit(
            f"[estimate] CLI failed: {bad or ''} full={rc_full}. See {r.work}/*.cli.log"
        )
    per: dict[str, dict] = {}
    for i, m in enumerate(metrics):
        chat = [
            e
            for e in r.proxy.exchanges
            if e["tag"] == f"m{i}" and e["path"].endswith("/chat/completions")
        ]
        in_tok = sum(max(1, len(e["body"]) // 4) for e in chat)
        max_out = sum(
            int(
                json.loads(e["body"]).get("max_tokens")
                or json.loads(e["body"]).get("max_completion_tokens")
                or 512
            )
            for e in chat
        )
        # Printed retry estimate only. The enforced upstream ceiling is HARD_CAP.
        bound = min(
            HARD_CAP,
            len(chat) * RETRY_FACTOR
            + (
                INTENTION_HEADROOM * RETRY_FACTOR
                if m == "deepeval:conversation_completeness"
                else 0
            ),
        )
        per[m] = {
            "chat": len(chat),
            "in_tokens": in_tok,
            "max_out_tokens": max_out,
            "upper_bound": bound,
        }
    full_chat = [
        e
        for e in r.proxy.exchanges
        if e["tag"] == "full" and e["path"].endswith("/chat/completions")
    ]
    full_emb = [
        e
        for e in r.proxy.exchanges
        if e["tag"] == "full" and e["path"].endswith("/embeddings")
    ]
    return {
        "per_metric": per,
        "sum_per_metric": sum(p["chat"] for p in per.values()),
        "full_run_chat": len(full_chat),
        "full_run_embeddings": len(full_emb),
        "full_run_keys": sorted((e["key"], e["ordinal"]) for e in full_chat),
        "upper_bound": sum(p["upper_bound"] for p in per.values()),
        "capture_full": r.capture(out_full, rc_full),
    }


def print_estimate(est: dict, cap: int, price_in: float, price_out: float) -> None:
    """Print requests, cap and approximate cost before the y/N prompt."""
    per = est["per_metric"]
    log("")
    log(f"{'metric':52} {'requests':>8} {'bound':>6} {'~in tok':>8}")
    for m, p in per.items():
        log(f"{m:52} {p['chat']:>8} {p['upper_bound']:>6} {p['in_tokens']:>8}")
    tin = sum(p["in_tokens"] for p in per.values())
    tmax = sum(p["max_out_tokens"] for p in per.values())
    n = est["full_run_chat"]
    exp_cost = tin * price_in / 1e6 + n * TYPICAL_OUT_TOKENS * price_out / 1e6
    worst = (
        RETRY_FACTOR * (tin * price_in / 1e6 + tmax * price_out / 1e6)
        + INTENTION_HEADROOM * RETRY_FACTOR * 1024 * price_out / 1e6
    )
    log(f"{'TOTAL (full fixture run)':52} {n:>8} {est['upper_bound']:>6} {tin:>8}")
    log(
        f"sum of per-metric runs: {est['sum_per_metric']} (must equal the full run: "
        f"{'OK' if est['sum_per_metric'] == n else 'MISMATCH'})"
    )
    log(
        f"embeddings: {est['full_run_embeddings']} requests, answered locally, never sent upstream"
    )
    log(
        f"expected upstream requests: {n} (exact if the judge returns schema-valid JSON first time)"
    )
    log(f"hard cap: {cap} upstream requests (abort beyond this; nothing is written)")
    log(
        f"approx cost at ${price_in}/1M in, ${price_out}/1M out: expected ~${exp_cost:.4f}, "
        f"worst case at the cap <= ${worst:.4f}"
    )
    log("")


def main() -> int:
    """Estimate, confirm, record (or dry-run), scan and replay; return the exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=".", help="repo root (default: cwd)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="hand-written judge only, no network, no key; cassette goes to a scratch dir",
    )
    ap.add_argument(
        "--record-with-my-key",
        action="store_true",
        help="required for record mode: confirms you intend real, paid calls with your own OPENAI_API_KEY",
    )
    ap.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help=(
            f"lower the upstream cap (1..{HARD_CAP}, default {HARD_CAP}); "
            "a value above the hard cap is refused"
        ),
    )
    ap.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN)
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None, help="cassette path override")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()
    if args.max_requests is not None and not 1 <= args.max_requests <= HARD_CAP:
        log(
            f"--max-requests must be from 1 to {HARD_CAP} "
            f"(got {args.max_requests}); refusing before anything is sent"
        )
        return 2

    repo = Path(args.repo).resolve()
    fixture_dir = repo / "tests/golden/metrics/fixtures" / FIXTURE
    if not (fixture_dir / "system.yaml").exists():
        log(f"fixture not found: {fixture_dir}")
        return 2
    secret: Optional[str] = None
    if not args.dry_run:
        if not args.record_with_my_key:
            log(
                "record mode makes real, paid API calls; pass --record-with-my-key to proceed (or use --dry-run)"
            )
            return 2
        secret = os.environ.get("OPENAI_API_KEY") or None
        if not secret:
            log(
                "record mode needs OPENAI_API_KEY in your environment (or use --dry-run)"
            )
            return 2
        if not args.upstream.startswith("https://"):
            log("refusing a non-https upstream in record mode")
            return 2

    work = Path(tempfile.mkdtemp(prefix="golden-record-"))
    proxy = cp.RecordingJudge(judge_responses.respond)
    r = Runner(repo, fixture_dir, work, proxy)
    try:
        est = estimate(r, args.jobs)
        cap = HARD_CAP if args.max_requests is None else args.max_requests
        print_estimate(est, cap, args.price_in, args.price_out)
        if est["sum_per_metric"] != est["full_run_chat"]:
            log("per-metric and full-run counts differ; refusing to continue")
            return 3
        if args.dry_run:
            log("[dry-run] no confirmation needed; nothing leaves this machine")
        else:
            if not sys.stdin.isatty():
                log(
                    "record mode needs an interactive terminal for the y/N prompt: aborting"
                )
                return 1
            if input(
                f"Send ~{est['full_run_chat']} requests (cap {cap}) to {args.upstream}? [y/N] "
            ).strip().lower() not in ("y", "yes"):
                log("aborted, nothing sent")
                return 1

        proxy.reset()
        if args.dry_run:
            proxy.set_fake()
        else:
            assert secret is not None  # checked above for record mode
            proxy.set_record(args.upstream, secret, cap)
        rc, rec_out = r.run("record_run", "rec")
        if proxy.abort_reason or rc != 0:
            log(
                f"[record] ABORTED ({proxy.abort_reason or f'CLI exit {rc}'}); nothing written. "
                f"Upstream requests sent: {proxy.forwarded}"
            )
            return 3
        rec_chat = [
            e
            for e in proxy.exchanges
            if e["tag"] == "rec" and e["path"].endswith("/chat/completions")
        ]
        entries = cp.to_entries(proxy.exchanges, "rec")
        dest = (
            Path(args.out)
            if args.out
            else (
                work / f"{FIXTURE}.dryrun.jsonl"
                if args.dry_run
                else repo / "tests/golden/metrics/cassettes" / f"{FIXTURE}.jsonl"
            )
        )
        hits = cp.write_cassette(entries, dest, secret)
        if hits:
            for h in hits[:20]:
                log(f"[secret-scan] {h}")
            return 4
        rec_cap = r.capture(rec_out, rc)

        proxy.reset()
        replay = judge_server.JudgeReplay(judge_server.load_cassette(dest))
        r.active = replay.handle  # strict replay through judge_server (no upstream)
        rc2, rep_out = r.run("replay_run", "rec")
        rep_cap = r.capture(rep_out, rc2)
        unused = replay.unused()
        same = (
            rep_cap["csv_rows"] == rec_cap["csv_rows"]
            and rep_cap["summary"] == rec_cap["summary"]
        )
        replay_ok = rc2 == 0 and not replay.misses and not unused and same
        by_source: dict[str, int] = {}
        for e in entries:
            by_source[e["source"]] = by_source.get(e["source"], 0) + 1
        summary = {
            "fixture": FIXTURE,
            "mode": "dry-run" if args.dry_run else "record",
            "cassette": str(dest),
            "entries": len(entries),
            "entries_by_source": by_source,
            "chat_requests": len(rec_chat),
            "estimate_chat_requests": est["full_run_chat"],
            "matches_estimate": len(rec_chat) == est["full_run_chat"],
            "upstream_requests_sent": proxy.forwarded if not args.dry_run else 0,
            "scrubbed_bodies": sum(1 for e in entries if e["scrubbed"]),
            "per_key_conflicts": len(cp.per_key_conflicts(entries)),
            "secret_scan": "clean",
            "replay": {
                "cli_exit": rc2,
                "misses": len(replay.misses),
                "unused_entries": len(unused),
                "results_identical": same,
                "results_rows": len(rep_cap["csv_rows"]),
            },
        }
        if args.dry_run:
            summary["dry_run_matches_estimate_keys"] = (
                sorted((e["key"], e["ordinal"]) for e in rec_chat)
                == est["full_run_keys"]
            )
            summary["results_identical_to_estimate_run"] = (
                rec_cap["csv_rows"] == est["capture_full"]["csv_rows"]
            )
        log(json.dumps(summary, indent=2))
        if not args.dry_run:
            (dest.parent / f"{FIXTURE}.record-summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            log(
                "Next: review `git diff tests/golden/metrics/cassettes`, then regenerate "
                "metrics/expected/judge_happy_path in a golden-regen PR (HANDOFF.md)."
            )
        if not replay_ok:
            log(f"[replay] FAILED: misses={replay.misses[:3]} unused={len(unused)}")
            return 5
        log("[replay] OK: 0 misses, 0 unused, identical results")
        return 0
    finally:
        r.close()
        if args.keep_work or args.dry_run:
            log(f"scratch dir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
