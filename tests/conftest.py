"""Shared fixtures. Everything is offline: synthetic history plus saved HTML pages."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Never pick up a developer's real .env (and Telegram token) while testing.
os.environ["HUATBOT_NO_DOTENV"] = "1"

from huatbot.models import PrizeRules, Settings
from huatbot.synth import synth_fourd, synth_next_fourd, synth_next_toto, synth_toto

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def toto_df():
    """600 synthetic TOTO draws with a planted crowd preference (see synth.DEFAULT_POPULAR)."""
    return synth_toto(n_draws=600)


@pytest.fixture(scope="session")
def fourd_df():
    return synth_fourd(n_draws=500)


@pytest.fixture(scope="session")
def next_toto(toto_df):
    return synth_next_toto(toto_df)


@pytest.fixture(scope="session")
def next_fourd(fourd_df):
    return synth_next_fourd(fourd_df)


@pytest.fixture
def rules():
    return PrizeRules()


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def fixture_html():
    def _read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")
    return _read
