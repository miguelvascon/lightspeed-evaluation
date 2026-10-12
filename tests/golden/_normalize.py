"""Normalize golden artifacts (SPEC-01 §5.1).

Wall-clock values, random ids, and paths become placeholders. Scores,
thresholds, statuses, reasons (aside from embedded paths), token counts, and
confidence intervals stay as written. Patterns cite baseline ``1433c32``.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# generator.py:113,162 data_persistence.py:40 orchestrator.py:75,118
_STAMP = re.compile(r"\d{8}_\d{6}")
# generator.py:458
_GENERATED = re.compile(r"Generated: .*")
# constants.py:147
_ASCTIME = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}")
# summary.py:86 generator.py:316,381 behavioral/models.py:131
# protocol.py:28 sql_storage.py:360
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)")
# pipeline.py:250 tqdm
_PROGRESS = re.compile(r"^.*\d+%\|.*$", re.M)
# runner tracebacks and warnings file:line (F2)
_FRAME = re.compile(r'File "[^"]+", line \d+')
_WARN = re.compile(r"(?<![\w/])[\w./\\-]+\.py:\d+")
# driver.py:172 eval-<id>-<8 hex> or eval-<8 hex>
_CR = re.compile(r"\beval-((?:[a-z0-9]+-)*)([0-9a-f]{8})\b")
_TMP = re.compile(r"(?<![\w])/(?:tmp|private/tmp|var/folders)(?:/[^\s\"']*)?")
_TIMING = frozenset(
    {
        "execution_time",
        "evaluation_latency",
        "agent_latency",
        "time_to_first_token",
        "streaming_duration",
        "tokens_per_second",
        "agent_latency_mean",
        "agent_latency_std",
        "agent_latency_mean_delta",
    }
)
_TIMING_TREES = frozenset({"agent_latency_stats", "streaming"})
_SKIP = frozenset({"requests", "png", "confidence_interval"})
_TS_KEYS = frozenset({"timestamp", "started_at"})
_TEXT_SUBS = (
    (_GENERATED, "Generated: <TS>"),
    (_STAMP, "<TS>"),
    (_ASCTIME, "<TS>"),
    (_ISO, "<TS>"),
    (_PROGRESS, "<PROGRESS>"),
    (_FRAME, 'File "<OUT>", line <LINE>'),
    (_WARN, "<OUT>:<LINE>"),
)


def _token(table: dict[str, str], raw: str, kind: str) -> str:
    if raw not in table:
        table[raw] = f"<{kind}{len(table) + 1}>"
    return table[raw]


def _roots(extra: Sequence[str]) -> list[str]:
    found = [str(Path.cwd()), tempfile.gettempdir(), *extra]
    unique = {root for root in found if root and root not in {"/", "."}}
    return sorted(unique, key=len, reverse=True)


def _collect_run_ids(obj: Any, found: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "run_id" and isinstance(value, str):
                found.append(value)
            elif key not in _SKIP:
                _collect_run_ids(value, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect_run_ids(item, found)


def _timing_tree(value: Any) -> Any:
    """Replace nonzero floats inside streaming and agent-latency stat blocks."""
    if isinstance(value, float) and not isinstance(value, bool):
        return "<T>" if value != 0.0 else value
    if isinstance(value, dict):
        return {
            name: item if name == "count" else _timing_tree(item)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_timing_tree(item) for item in value]
    return value


def _text(text: str, state: dict[str, Any]) -> str:
    for root in state["roots"]:
        text = text.replace(root, "<OUT>")
    text = _TMP.sub("<OUT>", text)
    runs: dict[str, str] = state["runs"]
    for raw in state["ids"]:
        token = _token(runs, raw, "RUN")
        text = text.replace(raw, token)
        prefix = raw[:8]
        text = re.sub(
            rf"(?<![0-9a-fA-F]){re.escape(prefix)}(?![0-9a-fA-F])",
            token,
            text,
        )

    def _suffix(match: re.Match[str]) -> str:
        token = _token(state["suffixes"], match.group(2), "SFX")
        return f"eval-{match.group(1)}{token}"

    text = _CR.sub(_suffix, text)
    for pattern, repl in _TEXT_SUBS:
        text = pattern.sub(repl, text)
    return text


def _scalar(key: str, value: Any, parent: dict[str, Any]) -> str | None:
    if (
        key == "id"
        and isinstance(value, int)
        and not isinstance(value, bool)
        and ("run_id" in parent or "conversation_group_id" in parent)
    ):
        return "<ID>"
    if key in _TS_KEYS and not isinstance(value, (dict, list)):
        return "<TS>"
    if key in _TIMING and isinstance(value, float) and value != 0.0:
        return "<T>"
    return None


def _item(key: str, value: Any, parent: dict[str, Any], state: dict[str, Any]) -> Any:
    if key in _SKIP:
        return value
    if key == "by_latency" and isinstance(value, list):
        return ["<TIMING_RANK>", *sorted(str(item) for item in value)]
    if key == "run_id" and isinstance(value, str):
        return _token(state["runs"], value, "RUN")
    if key in _TIMING_TREES:
        return _timing_tree(value)
    replaced = _scalar(key, value, parent)
    if replaced is not None:
        return replaced
    return _apply(value, state)


def _apply(obj: Any, state: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {key: _item(key, value, obj, state) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_apply(item, state) for item in obj]
    if isinstance(obj, str):
        return _text(obj, state)
    return obj


def normalize_obj(obj: Any, *, roots: Sequence[str] = ()) -> Any:
    """Return a normalized copy of a JSON-like golden document."""
    found: list[str] = []
    _collect_run_ids(obj, found)
    state: dict[str, Any] = {
        "runs": {},
        "suffixes": {},
        "ids": found,
        "roots": _roots(roots),
    }
    return _apply(obj, state)


def normalize_text(text: str, *, roots: Sequence[str] = ()) -> str:
    """Normalize one text blob (logs, TXT reports, reason strings)."""
    return str(normalize_obj(text, roots=roots))
