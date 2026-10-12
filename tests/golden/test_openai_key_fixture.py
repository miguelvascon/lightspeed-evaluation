"""``golden_env_extra`` can drop ``OPENAI_API_KEY`` for one directory."""

from __future__ import annotations

import os

import pytest

from tests.golden._env import UNSET

pytestmark = pytest.mark.golden


@pytest.fixture
def golden_env_extra() -> dict[str, str]:
    """Opt this module out of the placeholder key."""
    return {"OPENAI_API_KEY": UNSET}


def test_fixture_override_drops_the_key() -> None:
    """A directory fixture returning ``UNSET`` leaves the key absent."""
    assert "OPENAI_API_KEY" not in os.environ
