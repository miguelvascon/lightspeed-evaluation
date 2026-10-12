# lightspeed-eval verification map

This directory is the maintained source for verifying the user-facing behavior of the `lightspeed-eval` CLI. Read this index before driving the app, then use the matching feature file as the recipe.

## Baseline preconditions

- Repo root is the working directory the harness uses. `drive.sh` cds there.
- Dependencies are installed with `uv sync`. `scripts/doctor.sh` prints `doctor: ready` and a package version that matches `pyproject.toml`.
- Each drive gets its own absolute scratch directory under `${TMPDIR}` or `/tmp`, outside the repo. Pass that directory as the second argument of `drive.sh`. The harness sets `--output-dir` to it.
- Offline drives use `examples/02_metrics/keywords_evaluation/system.yaml` and `examples/02_metrics/keywords_evaluation/eval_data.yaml`. That config has `agents.enabled: false` and `core.cache_enabled: false`.
- Set `OPENAI_API_KEY=verification-not-a-real-key` and unset `API_KEY`. The CLI checks that the default `openai` provider has a non-empty key before scoring. Keyword scoring does not call OpenAI.
- Do not start from `config/system.yaml`. That file enables agents against `http://localhost:8080`.
- Never drive an output directory that another verification run, or the example's own `eval_output/`, is already using.

## Driving conventions

- Start every recipe from the baseline unless its preconditions say otherwise.
- Treat every command as literal. Keep flags and conversation ids unchanged.
- Run the CLI only through `scripts/drive.sh`. The PTY transcript and the exit code are the command proof.
- Restore nothing in the repo. Scratch dirs are disposable. Do not delete `.cursor/skills/verify-lightspeed-eval/evidence/` during cleanup.

## Proof and skip reporting

- Capture the command transcript and the files the command wrote.
- CLI proof is `transcript.txt`, `exit_code.txt`, and `command.txt` in the evidence directory `drive.sh` prints.
- Report proof is the copy under that evidence directory's `reports/`.
- Record the feature id by passing it as the first argument to `drive.sh`.
- Report an unreachable path with the attempted command and the unmet precondition.
- Do not report a skipped entry point as verified through a different path.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with drive.sh` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

## Features

- [Run an offline evaluation](./run-offline-evaluation.md) scores the keywords example and writes CSV, JSON, text, and graph reports.
- [Filter the run](./filter-the-run.md) limits a run with `--tags`, `--conv-ids`, and `--metrics`, including a filter that matches nothing.
- [Reject invalid input](./reject-invalid-input.md) exits non-zero for a missing config, an empty config, and eval data that lacks required fields.
