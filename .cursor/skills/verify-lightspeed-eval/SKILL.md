---
name: verify-lightspeed-eval
description: Drive the lightspeed-eval CLI for offline keyword evaluation, conversation filters, and invalid-input rejection, and capture the terminal transcript plus report files. Use when changing evaluation runs, CLI flags, report output, or config validation, and before claiming that CLI behavior works.
---

# Verify lightspeed-eval

Primary surface is the `lightspeed-eval` CLI (`lightspeed-evaluation` package, entry point `lightspeed_evaluation.runner.evaluation:main`). A user runs it from the repo root with a system config and an evaluation-data file. It exits when the run finishes. It does not listen on a port.

Other surfaces, not driven by this skill:

- Python library: `from lightspeed_evaluation import evaluate`. See `examples/01_getting_started/library_mode/`.
- Dashboard proof of concept: `dashboard/`, Vite on port 5173 (`make -C dashboard dev`). It needs Node and, to run evaluations from the browser, a live API plus `OPENAI_API_KEY`.
- `generate_answers` is a second console script and is moving out of this repo.
- `lsc_agent_eval/` is deprecated.

This skill drives the offline keywords example. That metric scores pre-written responses in the eval file and does not call an agent. Startup still checks the default judge provider. The keywords yaml leaves `llm` unset, so the default provider is `openai`, and the process requires `OPENAI_API_KEY` to be non-empty before it scores anything. A placeholder value is enough: keyword scoring does not send that key to a provider. Judge-LLM metrics (`custom:answer_correctness` and the Ragas/DeepEval metrics) and live agent runs (`config/system.yaml`, `agents.enabled: true`, default API `http://localhost:8080`) are outside this harness. Do not treat a keywords run as proof of those paths.

## Launch

From the repo root, install the locked runtime once:

```bash
uv sync
```

There is no server to keep running. The CLI is ready when this exits 0 and prints `Lightspeed Evaluation Framework / Tool`:

```bash
uv run --no-sync lightspeed-eval --help
```

Each drive is a new process. Start it with `scripts/drive.sh` (below), which allocates a PTY. Do not run a bare `uv run lightspeed-eval`: the default files are `config/system.yaml` and `config/evaluation_data.yaml`, and that system config turns agents on.

Teardown is `scripts/cleanup.sh` on the scratch directory for that drive. There is nothing else to stop.

## Doctor

Run this before driving whenever a command fails or the environment looks stale:

```bash
.cursor/skills/verify-lightspeed-eval/scripts/doctor.sh
```

It is read-only. It does not sync packages and it does not start an evaluation. Ready output is:

```text
doctor: ready
repo: <absolute repo root>
package: <version from pyproject.toml, currently 0.8.0>
surface: lightspeed-eval CLI (no listening port; each drive is a short-lived process)
```

The installed distribution version must match `pyproject.toml`. There is no port to check. If doctor says the package is not installed, run Launch and then doctor again.

## Drive

Harness scripts (all executable):

- `.cursor/skills/verify-lightspeed-eval/scripts/drive.sh <feature-id> <scratch-output-dir> -- <lightspeed-eval args>`
- `.cursor/skills/verify-lightspeed-eval/scripts/pty_run.py --pid-file PATH --transcript PATH --cwd DIR -- CMD [ARGS...]`

`drive.sh` is the command to run. It cds to the repo root, forces `MPLBACKEND=Agg` when that variable is unset (so matplotlib writes PNG files and does not open a window), and runs:

```text
uv run --no-sync lightspeed-eval --output-dir <scratch-output-dir> <args>
```

inside a new PTY via `pty_run.py`. `pty_run.py` writes the child pid, tees the combined stdout/stderr into the transcript, strips carriage returns, and exits with the child status. The PTY wait is 300 seconds; a timeout exits 124 and kills that process group.

`<scratch-output-dir>` must be an absolute path under `${TMPDIR}` or `/tmp`, and must not be inside the repo. `drive.sh` rejects a second `--output-dir` in the extra args. Two drives can run at the same time only with different scratch directories. Do not write verification output into `examples/02_metrics/keywords_evaluation/eval_output/` or any other example `eval_output/` directory.

The keywords system config sets `core.cache_enabled: false`. Do not point a verification drive at a config that enables the shared `.caches/` directory, and do not pass `--cache-warmup` unless the feature you are proving is cache warmup. `--cache-warmup` deletes cache directories named by the config.

Stable CLI handles are the flags `--system-config`, `--eval-data`, `--output-dir`, `--tags`, `--conv-ids`, `--metrics`, and `--cache-warmup`. Conversation identity in the keywords file is the `conversation_group_id` (`single_keyword_set`, `alternative_keywords`, `case_insensitive`, `missing_keyword`). The metric id is `custom:keywords_eval`.

Read `features/README.md` and the matching feature file before choosing a drive. A proof of one entry point does not cover the others listed there.

## Evidence

`drive.sh` copies proof to:

```text
.cursor/skills/verify-lightspeed-eval/evidence/<feature-id>/<UTC-timestamp>-<pid>/
```

That directory contains:

- `command.txt` — working directory, `MPLBACKEND`, and the argv
- `transcript.txt` — PTY transcript of the run
- `exit_code.txt` — process exit code
- `reports/` — copy of the scratch output, excluding `.verify-pid`

Proof standards:

- Drive `lightspeed-eval` itself. Do not call `evaluate()` or a unit test and call that the CLI.
- Keep the transcript (the action) and the report files (the resulting state). A transcript alone is not enough when the feature writes reports.
- For the keywords run, also show the side effect on disk: `evaluation_*_detailed.csv`, `evaluation_*_summary.json`, `evaluation_*_summary.txt`, and `graphs/evaluation_*_pass_rates.png` plus `graphs/evaluation_*_status_breakdown.png`.
- Record the feature id. `drive.sh` does that by using it as the evidence directory name.
- The keywords path is the real offline evaluation, not a dry-run flag. Set `OPENAI_API_KEY=verification-not-a-real-key` and unset `API_KEY` (see the feature file). That placeholder only satisfies the startup check for the default `openai` provider. A completed run whose CSV reasons describe keyword substring matches, with that placeholder key, shows the scorer did not call OpenAI: a real completion with that key would fail authentication. Unsetting `OPENAI_API_KEY` is not a valid proof; the process raises `LLMError: OPENAI_API_KEY environment variable is required for OpenAI provider` and exits 1 before scoring.
- Do not mock the metric. The responses and `expected_keywords` in `examples/02_metrics/keywords_evaluation/eval_data.yaml` are the fixture.

On failure, keep the transcript. Say which feature and entry point failed, and quote the `❌ Evaluation failed:` line or the exit code. Do not mark a skipped entry point as verified through a different one.

## Cleanup

```bash
.cursor/skills/verify-lightspeed-eval/scripts/cleanup.sh <scratch-output-dir>
```

This reads `<scratch-output-dir>/.verify-pid`. It sends `TERM`, then `KILL` if needed, only when that pid's command line contains `lightspeed-eval` or `lightspeed_evaluation`. It never kills by process name. It then deletes the scratch directory. It refuses paths inside the repo or outside the temp roots.

Evidence under `.cursor/skills/verify-lightspeed-eval/evidence/` stays. After cleanup, the evidence directory from the drive must still exist.

Run cleanup after a failed drive too, so a timed-out process and its scratch directory are not left behind.

## Helpers

| Script | Invocation |
| --- | --- |
| Doctor | `.cursor/skills/verify-lightspeed-eval/scripts/doctor.sh` |
| Drive | `.cursor/skills/verify-lightspeed-eval/scripts/drive.sh <feature-id> <scratch-output-dir> -- <lightspeed-eval args>` |
| PTY | `.cursor/skills/verify-lightspeed-eval/scripts/pty_run.py --pid-file PATH --transcript PATH --cwd DIR -- CMD [ARGS...]` (called by `drive.sh`) |
| Cleanup | `.cursor/skills/verify-lightspeed-eval/scripts/cleanup.sh <scratch-output-dir>` |
