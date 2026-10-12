"""C4: eval-data roots, query, tool calls, tags, filters, and schemas."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from lightspeed_evaluation.core.models.data import EvaluationData, TurnData
from lightspeed_evaluation.core.system import ConfigLoader, DataValidator
from lightspeed_evaluation.core.system.exceptions import DataValidationError

pytestmark = pytest.mark.golden

_FIX = Path(__file__).parent / "fixtures" / "evaldata"
_EXPECTED = Path(__file__).parent / "expected" / "evaldata"
_SKIP = frozenset({"metrics.yaml", "scoped.yaml"})


def _cases() -> list[Path]:
    return sorted(path for path in _FIX.glob("*.yaml") if path.name not in _SKIP)


def _stable(obj: Any) -> Any:
    """Sort tag sets so the snapshot does not depend on hash order."""
    if isinstance(obj, dict):
        built = {key: _stable(value) for key, value in obj.items()}
        tag = built.get("tag")
        if isinstance(tag, list):
            built["tag"] = sorted(str(item) for item in tag)
        return built
    if isinstance(obj, list):
        return [_stable(item) for item in obj]
    return obj


def _check(name: str, payload: Any) -> None:
    path = _EXPECTED / f"{name}.json"
    if os.environ.get("LS_GOLDEN_UPDATE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    assert payload == json.loads(path.read_text(encoding="utf-8"))


def _system() -> Any:
    return ConfigLoader().load_system_config(str(_FIX / "metrics.yaml"))


def _load(path: Path, **kwargs: Any) -> dict[str, Any]:
    validator = DataValidator(system_config=_system())
    try:
        rows = validator.load_evaluation_data(str(path), **kwargs)
    except DataValidationError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "validation_errors": list(validator.validation_errors),
        }
    metadata = validator.dataset_metadata
    return {
        "ok": True,
        "error": "",
        "metadata": (
            None if metadata is None else _stable(metadata.model_dump(mode="json"))
        ),
        "data": _stable([row.model_dump(mode="json") for row in rows]),
        "validation_errors": list(validator.validation_errors),
    }


def test_model_json_schema() -> None:
    """``TurnData`` and ``EvaluationData`` schemas stay byte-for-byte."""
    _check("turn_schema", TurnData.model_json_schema())
    _check("evaluation_schema", EvaluationData.model_json_schema())


@pytest.mark.parametrize("path", _cases(), ids=lambda item: item.stem)
def test_eval_data_case(path: Path) -> None:
    """List and dict roots, query, tool-call shape, aliases, and error text."""
    _check(path.stem, _load(path))


def test_filters_keep_matching_conversations() -> None:
    """Tags and conv ids are OR'd. A metric filter materializes defaults."""
    path = _FIX / "scoped.yaml"
    payload = {
        "all": _load(path),
        "tag": _load(path, tags=["keywords"]),
        "conv": _load(path, conv_ids=["beta"]),
        "either": _load(path, tags=["other"], conv_ids=["alpha"]),
        "keywords_only": _load(path, metrics=["custom:keywords_eval"]),
        "correctness_only": _load(path, metrics=["custom:answer_correctness"]),
    }
    _check("filters", payload)
