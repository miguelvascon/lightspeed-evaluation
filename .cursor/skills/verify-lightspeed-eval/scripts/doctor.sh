#!/usr/bin/env bash
# Read-only check: is this checkout worth driving with lightspeed-eval?
#
# Usage (from anywhere):
#   .cursor/skills/verify-lightspeed-eval/scripts/doctor.sh
#
# Does not start an evaluation, does not write reports, and does not sync
# dependencies. Exit 0 prints "doctor: ready".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT"

fail() {
  echo "doctor: $*" >&2
  exit 1
}

command -v uv >/dev/null 2>&1 || fail "uv is not on PATH. Install uv, then run: uv sync"
command -v python3 >/dev/null 2>&1 || fail "python3 is not on PATH (needed for the PTY harness)"

if ! uv run --no-sync python -c 'import lightspeed_evaluation' >/dev/null 2>&1; then
  fail "lightspeed-evaluation is not installed in this checkout. From the repo root run: uv sync"
fi

installed="$(uv run --no-sync python -c 'import importlib.metadata as m; print(m.version("lightspeed-evaluation"))')"
expected="$(uv run --no-sync python -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
if [[ "$installed" != "$expected" ]]; then
  fail "installed package ${installed} does not match pyproject.toml ${expected}. From the repo root run: uv sync"
fi

help_text="$(uv run --no-sync lightspeed-eval --help)"
if ! grep -q "Lightspeed Evaluation Framework / Tool" <<<"$help_text"; then
  fail "lightspeed-eval --help did not identify the evaluation CLI"
fi
for flag in --system-config --eval-data --output-dir --tags --conv-ids --metrics --cache-warmup; do
  if ! grep -q -- "$flag" <<<"$help_text"; then
    fail "lightspeed-eval --help is missing ${flag}"
  fi
done

echo "doctor: ready"
echo "repo: $ROOT"
echo "package: $installed"
echo "surface: lightspeed-eval CLI (no listening port; each drive is a short-lived process)"
