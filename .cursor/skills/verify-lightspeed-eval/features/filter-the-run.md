# Filter the run

Filters choose which conversations or metrics an offline run scores. Tag and conversation-id filters combine with OR. A filter that matches nothing exits 0 and writes no report files.

## Sub-features

- `filter-tag` keeps conversations whose tag is listed in `--tags`.
- `filter-conv-id` keeps conversations whose `conversation_group_id` is listed in `--conv-ids`.
- `filter-or` unions `--tags` and `--conv-ids` when both are set.
- `filter-empty` exits 0 with a warning when nothing matches, and writes no reports.
- `filter-metric` with `custom:keywords_eval` still scores the keyword rows.
- `filter-metric-miss` with a name that is not a default metric scores nothing and writes no reports.

## How to get to it (user POV)

- Run `lightspeed-eval` with the keywords system config and eval data, and add `--tags keywords`.
- Add `--conv-ids missing_keyword` instead of a tag.
- Pass both `--tags keywords` and `--conv-ids missing_keyword`.
- Pass `--tags not-a-tag`.
- Pass `--metrics custom:keywords_eval`.
- Pass `--metrics custom:not_a_metric`.

## Driving it with drive.sh

Preconditions:

- `scripts/doctor.sh` prints `doctor: ready`.
- Use a new scratch directory for each bullet. Do not reuse a directory that already holds reports.
- Set `OPENAI_API_KEY=verification-not-a-real-key` and unset `API_KEY`, same as the offline run.

- **Tag match.** Run:

```bash
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/lseval-verify.XXXXXX")"
env -u API_KEY OPENAI_API_KEY=verification-not-a-real-key \
  .cursor/skills/verify-lightspeed-eval/scripts/drive.sh \
  filter-the-run "$SCRATCH" -- \
  --system-config examples/02_metrics/keywords_evaluation/system.yaml \
  --eval-data examples/02_metrics/keywords_evaluation/eval_data.yaml \
  --tags keywords
```

Exit code `0`. The transcript contains `📋 Evaluation data: 4 of 4 conversations (filtered)` and `📊 4 evaluations completed`. The CSV has four data rows.

- **One conversation.** Use a new scratch directory and replace `--tags keywords` with `--conv-ids missing_keyword`. Exit code `0`. The transcript contains `📋 Evaluation data: 1 of 4 conversations (filtered)`, `📊 1 evaluations completed`, and `✅ Pass: 0, ❌ Fail: 1`. The CSV has one data row: `missing_keyword`, result `FAIL`, score `0.0`.

- **OR combination.** Use a new scratch directory and pass both `--tags keywords` and `--conv-ids missing_keyword`. Exit code `0`. The transcript contains `📋 Evaluation data: 4 of 4 conversations (filtered)` and `📊 4 evaluations completed`. The tag already covers every row, including `missing_keyword`.

- **Nothing matched.** Use a new scratch directory and pass `--tags not-a-tag`. Exit code `0`. The transcript contains `📋 Evaluation data: 0 of 4 conversations (filtered)`, `⚠️ No conversation groups matched the filter criteria`, and `Nothing to evaluate - returning empty results`. It does not contain `🎉 Evaluation Complete!`. `reports/` has no `evaluation_*_detailed.csv`.

- **Known metric.** Use a new scratch directory and pass `--metrics custom:keywords_eval` with no tag filter. Exit code `0`. The transcript contains `📊 4 evaluations completed`. Every CSV row has metric `custom:keywords_eval`.

- **Unknown metric name.** Use a new scratch directory and pass `--metrics custom:not_a_metric`. Exit code `0`. The transcript contains `🎉 Evaluation Complete!` and `📊 0 evaluations completed`. `reports/` has no `evaluation_*_detailed.csv`. The CLI does not say the metric name is unknown when conversations inherit defaults.

- **Proof.** Keep a separate evidence directory per bullet (each `drive.sh` call prints its own `evidence:` path). For a bullet that writes reports, the CSV row count must match the transcript's evaluation count. For a bullet that matches nothing, the transcript and the empty `reports/` directory are both required.

## Gotchas

- `--tags` and `--conv-ids` together are a union. They do not intersect.
- Every conversation in the keywords file has tag `keywords`. `--tags keywords` is the full set, not a smaller sample.
- `--metrics` filters inherited defaults down to the names you listed. A name that is not configured becomes zero metrics and exit 0, not a validation error. An unknown metric is rejected only when a turn lists it explicitly under `turn_metrics`; that path is covered in [Reject invalid input](./reject-invalid-input.md).
- Reusing one scratch directory mixes timestamps from two runs. Assertions then see extra CSV files.
- An empty match still exits 0. Assert the warning text and the absence of a detailed CSV, not a non-zero exit.
