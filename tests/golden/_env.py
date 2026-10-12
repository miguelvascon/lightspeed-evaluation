"""Shared golden-env state.

pytest loads ``tests/golden/conftest.py`` as ``golden.conftest``. Importing
that path again builds a second module. Judge URL updates and the
``OPENAI_API_KEY`` opt-out live here so every copy talks to one dict.
"""

from __future__ import annotations

import os

# Drop the placeholder key. Any other value is ignored and the dummy stays.
UNSET = "UNSET"
RUNTIME: dict[str, str] = {}


def publish_openai_base_url(url: str) -> None:
    """Point later golden subprocesses at a local fake judge."""
    RUNTIME["base_url"] = url
    os.environ["OPENAI_BASE_URL"] = url
