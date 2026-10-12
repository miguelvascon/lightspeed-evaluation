# Run an offline evaluation

Running an offline evaluation scores responses that are already in the eval file and writes a pass/fail summary plus report files. The keywords example checks that required words appear in each response. One of the four conversations is written to fail.

## Sub-features

- `offline-run` runs all four keyword conversations and prints the summary counts.
- `offline-pass-fail` records three passes and one fail for `missing_keyword`.
- `offline-reports` writes the CSV, JSON, text summary, and both graphs under the output directory.
- `offline-placeholder-key` completes when `OPENAI_API_KEY` is a placeholder and `API_KEY` is unset.

## How to get to it (user POV)

- Run `lightspeed-eval` with `--system-config examples/02_metrics/keywords_evaluation/system.yaml` and `--eval-data examples/02_metrics/keywords_evaluation/eval_data.yaml`.
- Pass `--output-dir` to choose where the reports go. The harness supplies this from the scratch directory.

## Driving it with drive.sh

Preconditions:

- `scripts/doctor.sh` prints `doctor: ready`.
- `OPENAI_API_KEY` is the placeholder `verification-not-a-real-key`. `API_KEY` is unset.
- The scratch directory is empty and is not used by another drive.

- **Run the example.** From the repo root, create a scratch directory and drive the keywords configs. Run:

```bash
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/lseval-verify.XXXXXX")"
env -u API_KEY OPENAI_API_KEY=verification-not-a-real-key \
  .cursor/skills/verify-lightspeed-eval/scripts/drive.sh \
  run-offline-evaluation "$SCRATCH" -- \
  --system-config examples/02_metrics/keywords_evaluation/system.yaml \
  --eval-data examples/02_metrics/keywords_evaluation/eval_data.yaml
```

Exit code `0`. The transcript contains `🚀 Lightspeed Evaluation Framework`, `✅ All data validation passed`, `📋 Evaluation data loaded: 4 conversations`, `🎉 Evaluation Complete!`, `📊 4 evaluations completed`, and `✅ Pass: 3, ❌ Fail: 1, ⚠️ Error: 0, ⏭️ Skipped: 0`. It also contains `📁 Reports generated in:` followed by the absolute scratch path.

- **Confirm the fail on disk.** Open `reports/evaluation_*_detailed.csv` in the evidence directory. It has four data rows, all with metric `custom:keywords_eval`. `single_keyword_set` / `namespace_keywords`, `alternative_keywords` / `deployment_alternatives`, and `case_insensitive` / `case_test` have result `PASS` and score `1.0`. `missing_keyword` / `incomplete_match` has result `FAIL`, score `0.0`, and a reason that names unmatched `'openshift'`. Reasons that say `Keywords eval successful` or `Keywords eval failed` are the local scorer. A reason that mentions an API or authentication error means this proof failed.
- **Confirm the summaries.** `reports/evaluation_*_summary.json` and `reports/evaluation_*_summary.txt` exist beside the CSV. The text summary repeats Pass `3` and Fail `1`. Under `Token Usage (Judge LLM)` and `Token Usage (API Calls)` the input and output token counts are `0`.
- **Confirm the graphs.** `reports/graphs/` contains `evaluation_*_pass_rates.png` and `evaluation_*_status_breakdown.png`.
- **Proof.** The evidence directory printed as `evidence:` contains `command.txt`, `transcript.txt`, `exit_code.txt`, and `reports/` with those files. `command.txt` shows `MPLBACKEND=Agg` and does not point `--output-dir` at the repo.

## Gotchas

- The transcript line `✅ System config: openai/gpt-4o-mini` is the default judge identity. The keywords yaml does not set `llm` or `llm_pool`, so startup also prints `DEPRECATION: 'llm' configuration is deprecated`. That warning is expected. The placeholder `OPENAI_API_KEY` plus keyword reasons in the CSV is the check that scoring stayed local.
- Unsetting `OPENAI_API_KEY` aborts before scoring with `LLMError: OPENAI_API_KEY environment variable is required for OpenAI provider`. That traceback does not print `❌ Evaluation failed:`.
- A log line `Pipeline storage: file backend (output_dir=examples/02_metrics/keywords_evaluation/eval_output)` names the directory in the yaml. The directory that received files is the one in `📁 Reports generated in:`. Confirm the CSV is under the evidence `reports/` copy, and that `examples/02_metrics/keywords_evaluation/eval_output/` was not created by this drive.
- Omitting the harness `--output-dir` (by running `lightspeed-eval` yourself without one) writes into `examples/02_metrics/keywords_evaluation/eval_output/`, which is shared with anyone using the example.
- A bare `lightspeed-eval` uses `config/system.yaml`, which enables agents. That is a different feature and needs a live API.
- `alternative_keywords` passes because `namespace` is a substring of `namespaces` in that response. Assert the result code, not a whole-word match.
- Keyword scores are `1.0` or `0.0`. The pass threshold used when the yaml omits one is `0.5`, so the fail row is the zero score, not a mid-range score.
- Reports are timestamped (`evaluation_YYYYMMDD_HHMMSS_...`). Match the glob. Do not require a fixed filename.
- No `*_amended_*.yaml` file is written. Amended data is saved only when agents are enabled.
