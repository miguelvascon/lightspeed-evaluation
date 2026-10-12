"""C3: system-config accept/reject matrix, migrations, and ``model_dump``."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

pytestmark = pytest.mark.golden

_FIX = Path(__file__).parent / "fixtures" / "config"
_EXPECTED = Path(__file__).parent / "expected" / "config"


def _paths(kind: str) -> list[Path]:
    return sorted((_FIX / kind).glob("*.yaml"))


def _check(name: str, payload: dict[str, Any]) -> None:
    path = _EXPECTED / f"{name}.json"
    if os.environ.get("LS_GOLDEN_UPDATE") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert payload == expected


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return sorted(
        f"{record.name}: {record.getMessage()}"
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


@pytest.mark.parametrize("path", _paths("accept"), ids=lambda item: item.stem)
def test_accepted_config_matches_dump(
    path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Accepted configs freeze warnings and ``SystemConfig.model_dump()``."""
    caplog.set_level(logging.WARNING)
    config = ConfigLoader().load_system_config(str(path))
    _check(
        path.stem,
        {"warnings": _warnings(caplog), "dump": config.model_dump(mode="json")},
    )


@pytest.mark.parametrize("path", _paths("reject"), ids=lambda item: item.stem)
def test_rejected_config_matches_error(path: Path) -> None:
    """Rejected configs freeze the exception type and message."""
    with pytest.raises((ConfigurationError, ValidationError)) as caught:
        ConfigLoader().load_system_config(str(path))
    error = caught.value
    _check(path.stem, {"type": type(error).__name__, "message": str(error)})
