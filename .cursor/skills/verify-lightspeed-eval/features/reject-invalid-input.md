# Reject invalid input

Invalid input stops the CLI before it scores conversations. The process exits 1, prints `❌ Evaluation failed:`, and does not write evaluation reports.

## Sub-features

- `reject-missing-config` fails when `--system-config` points at a file that is not there.
- `reject-empty-config` fails when the system config file is empty.
- `reject-missing-fields` fails when a turn names `custom:keywords_eval` and omits `expected_keywords`.

## How to get to it (user POV)

- Run `lightspeed-eval --system-config` with a path that does not exist.
- Run `lightspeed-eval --system-config` with an empty YAML file.
- Run `lightspeed-eval --eval-data` with a YAML file whose turn sets `turn_metrics: [custom:keywords_eval]` and has no `expected_keywords`.

## Driving it with drive.sh

Preconditions:

- `scripts/doctor.sh` prints `doctor: ready`.
- Use a new scratch directory for each bullet.
- Write the bad files into that scratch directory, not into the repo.
- Set `OPENAI_API_KEY=verification-not-a-real-key` and unset `API_KEY`. The missing-config and empty-config failures happen before the judge check, but use the same environment so the recipes stay comparable.

- **Missing system config.** Run:

```bash
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/lseval-verify.XXXXXX")"
env -u API_KEY OPENAI_API_KEY=verification-not-a-real-key \
  .cursor/skills/verify-lightspeed-eval/scripts/drive.sh \
  reject-invalid-input "$SCRATCH" -- \
  --system-config "$SCRATCH/missing.yaml" \
  --eval-data examples/02_metrics/keywords_evaluation/eval_data.yaml
```

Exit code `1`. The transcript contains `❌ Evaluation failed: Configuration file not found:` and the missing path. `reports/` has no `evaluation_*_detailed.csv`.

- **Empty system config.** Use a new scratch directory. Create the empty file, then drive it:

```bash
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/lseval-verify.XXXXXX")"
: > "$SCRATCH/empty.yaml"
env -u API_KEY OPENAI_API_KEY=verification-not-a-real-key \
  .cursor/skills/verify-lightspeed-eval/scripts/drive.sh \
  reject-invalid-input "$SCRATCH" -- \
  --system-config "$SCRATCH/empty.yaml" \
  --eval-data examples/02_metrics/keywords_evaluation/eval_data.yaml
```

Exit code `1`. The transcript contains `❌ Evaluation failed: Empty or invalid YAML configuration file`. `reports/` has no `evaluation_*_detailed.csv`.

- **Missing required field.** Use a new scratch directory. Write this file to `$SCRATCH/broken.yaml`:

```yaml
- conversation_group_id: broken
  turns:
  - turn_id: t1
    query: What is a namespace?
    response: A namespace provides isolation.
    turn_metrics:
    - custom:keywords_eval
```

Then run:

```bash
env -u API_KEY OPENAI_API_KEY=verification-not-a-real-key \
  .cursor/skills/verify-lightspeed-eval/scripts/drive.sh \
  reject-invalid-input "$SCRATCH" -- \
  --system-config examples/02_metrics/keywords_evaluation/system.yaml \
  --eval-data "$SCRATCH/broken.yaml"
```

Exit code `1`. The transcript contains `❌ Validation Errors:`, `TurnData t1: Metric 'custom:keywords_eval' requires 'response' and 'expected_keywords' fields`, and `❌ Evaluation failed: Evaluation data validation failed`. `reports/` has no `evaluation_*_detailed.csv`. The same scratch directory now also contains `broken.yaml` because it was created there; that file is input, not a report.

- **Proof.** Each bullet's evidence directory has exit code `1` in `exit_code.txt`, the failure line in `transcript.txt`, and no detailed CSV under `reports/`.

## Gotchas

- A transcript that includes a Python traceback is expected. The stable line is `❌ Evaluation failed:`.
- Leaving `turn_metrics` off the broken turn does not reject it up front. Inherited defaults are checked later, and a missing `expected_keywords` then becomes an `ERROR` result inside a run that can still exit 0. The rejection proof must set `turn_metrics` to `custom:keywords_eval`.
- `fail_on_invalid_data` defaults to true. A config that sets it false is a different behavior and is not this feature.
- Exit code 1 with no CSV is the proof. Do not accept a run that prints the error and also writes `evaluation_*_detailed.csv`.
