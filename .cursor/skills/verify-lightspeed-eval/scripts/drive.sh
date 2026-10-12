#!/usr/bin/env bash
# Drive one lightspeed-eval invocation in a private PTY.
#
# Usage (from anywhere; the script cds to the repo root):
#   .cursor/skills/verify-lightspeed-eval/scripts/drive.sh \
#     <feature-id> <scratch-output-dir> -- <lightspeed-eval args>
#
# <scratch-output-dir> must be an absolute path under ${TMPDIR} or /tmp,
# outside this repo. The script injects --output-dir itself. Do not pass
# --output-dir in the extra args.
#
# Evidence is copied to:
#   .cursor/skills/verify-lightspeed-eval/evidence/<feature-id>/<run-id>/
# and is not deleted by cleanup.sh.
set -euo pipefail

if [[ $# -lt 3 || "$3" != "--" ]]; then
  echo "usage: drive.sh <feature-id> <scratch-output-dir> -- <lightspeed-eval args>" >&2
  exit 2
fi

FEATURE="$1"
SCRATCH="$2"
shift 3

for arg in "$@"; do
  if [[ "$arg" == "--output-dir" ]]; then
    echo "drive.sh: do not pass --output-dir; the scratch directory is the output dir" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"

python3 - "$SCRATCH" "$ROOT" <<'PY'
import os
import sys
import tempfile

scratch, root = sys.argv[1], sys.argv[2]
if not os.path.isabs(scratch):
    sys.exit("drive.sh: scratch output dir must be absolute")
scratch_real = os.path.realpath(scratch)
root_real = os.path.realpath(root)
tmp_roots = {
    os.path.realpath(tempfile.gettempdir()),
    os.path.realpath("/tmp"),
    os.path.realpath("/private/tmp"),
}
allowed = any(
    scratch_real == tmp or scratch_real.startswith(tmp + os.sep) for tmp in tmp_roots
)
if not allowed:
    sys.exit(
        "drive.sh: scratch output dir must live under TMPDIR or /tmp, got "
        + scratch_real
    )
if scratch_real == root_real or scratch_real.startswith(root_real + os.sep):
    sys.exit("drive.sh: scratch output dir must not be inside the repo")
PY

mkdir -p "$SCRATCH"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
EVIDENCE="$SKILL_DIR/evidence/$FEATURE/$RUN_ID"
mkdir -p "$EVIDENCE/reports"

CMD=(uv run --no-sync lightspeed-eval --output-dir "$SCRATCH" "$@")
{
  printf 'cwd=%s\n' "$ROOT"
  printf 'MPLBACKEND=%s\n' "${MPLBACKEND:-Agg}"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} >"$EVIDENCE/command.txt"

export MPLBACKEND="${MPLBACKEND:-Agg}"

set +e
python3 "$SCRIPT_DIR/pty_run.py" \
  --pid-file "$SCRATCH/.verify-pid" \
  --transcript "$EVIDENCE/transcript.txt" \
  --cwd "$ROOT" \
  -- "${CMD[@]}"
rc=$?
set -e

printf '%s\n' "$rc" >"$EVIDENCE/exit_code.txt"

if [[ -d "$SCRATCH" ]]; then
  while IFS= read -r -d '' entry; do
    base="$(basename "$entry")"
    if [[ "$base" == ".verify-pid" ]]; then
      continue
    fi
    rel="${entry#"$SCRATCH"/}"
    dest="$EVIDENCE/reports/$rel"
    mkdir -p "$(dirname "$dest")"
    if [[ -d "$entry" ]]; then
      mkdir -p "$dest"
    else
      cp -p "$entry" "$dest"
    fi
  done < <(find "$SCRATCH" -mindepth 1 -print0)
fi

echo "evidence: $EVIDENCE"
echo "exit: $rc"
exit "$rc"
