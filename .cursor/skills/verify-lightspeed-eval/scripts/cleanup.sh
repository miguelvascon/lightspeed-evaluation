#!/usr/bin/env bash
# Remove one verification scratch directory and the process it recorded.
#
# Usage:
#   .cursor/skills/verify-lightspeed-eval/scripts/cleanup.sh <scratch-output-dir>
#
# Kills only the pid stored in <scratch-output-dir>/.verify-pid, and only when
# that process command line contains lightspeed-eval or lightspeed_evaluation.
# Deletes the scratch directory. Does not delete
# .cursor/skills/verify-lightspeed-eval/evidence/.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: cleanup.sh <scratch-output-dir>" >&2
  exit 2
fi

SCRATCH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

python3 - "$SCRATCH" "$ROOT" <<'PY'
import os
import sys
import tempfile

scratch, root = sys.argv[1], sys.argv[2]
if not os.path.isabs(scratch):
    sys.exit("cleanup.sh: scratch output dir must be absolute")
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
        "cleanup.sh: refusing to delete a path outside TMPDIR or /tmp: "
        + scratch_real
    )
if scratch_real == root_real or scratch_real.startswith(root_real + os.sep):
    sys.exit("cleanup.sh: refusing to delete a path inside the repo")
PY

if [[ ! -d "$SCRATCH" ]]; then
  echo "cleanup: scratch already gone: $SCRATCH"
  exit 0
fi

pidfile="$SCRATCH/.verify-pid"
if [[ -f "$pidfile" ]]; then
  pid="$(tr -d '[:space:]' <"$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$cmd" == *lightspeed-eval* || "$cmd" == *lightspeed_evaluation* ]]; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        if ! ps -p "$pid" >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
      if ps -p "$pid" >/dev/null 2>&1; then
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      fi
      echo "cleanup: stopped pid $pid"
    elif [[ -n "$cmd" ]]; then
      echo "cleanup: pid $pid is not lightspeed-eval ($cmd); not killing" >&2
    fi
  fi
fi

rm -rf "$SCRATCH"
echo "cleanup: removed $SCRATCH"
