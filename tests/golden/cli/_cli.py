"""Run ``lightspeed-eval`` and collect one output directory."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_STAMP = re.compile(r"\d{8}_\d{6}")
_MAPS = frozenset({"by_conversation", "by_metric", "by_tag"})


def run_cli(
    *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the console script. The golden hook supplies the child env."""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    binary = Path(sys.executable).with_name("lightspeed-eval")
    return subprocess.run(
        [str(binary), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def collect_output(
    root: Path, completed: subprocess.CompletedProcess[str]
) -> dict[str, Any]:
    """Bundle transcripts and report files. Timestamps in names become ``<TS>``."""
    files: dict[str, Any] = {}
    png: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = _STAMP.sub("<TS>", path.relative_to(root).as_posix())
        if path.suffix == ".png":
            png[rel] = path.read_bytes().hex()
            continue
        text = path.read_text(encoding="utf-8")
        files[rel] = json.loads(text) if path.suffix == ".json" else text
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "files": files,
        "png": png,
    }


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("conversation_group_id", "")),
        str(row.get("turn_id") or ""),
        str(row.get("metric_identifier") or ""),
    )


def _sort_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        built = {key: _sort_tree(value) for key, value in obj.items()}
        for key in _MAPS:
            child = built.get(key)
            if isinstance(child, dict):
                built[key] = {name: child[name] for name in sorted(child)}
        return built
    if isinstance(obj, list):
        items = [_sort_tree(item) for item in obj]
        if items and all(
            isinstance(item, dict) and "conversation_group_id" in item for item in items
        ):
            return sorted(items, key=_row_key)
        return items
    return obj


def as_multiset(bundle: dict[str, Any]) -> dict[str, Any]:
    """Sort cross-conversation rows. Within a conversation, order stays."""
    files: dict[str, Any] = {}
    for name, payload in bundle["files"].items():
        if isinstance(payload, str) and name.endswith(".csv"):
            rows = payload.splitlines()
            body = "\n".join(sorted(rows[1:]))
            newline = "\n" if payload.endswith("\n") else ""
            files[name] = f"{rows[0]}\n{body}{newline}" if rows else payload
        else:
            files[name] = _sort_tree(payload)
    return {**bundle, "files": files}
