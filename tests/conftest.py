from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fakes import FakeModel  # noqa: E402
from harness.api import Harness, HarnessConfig  # noqa: E402
from harness.tools.builtin import builtin_tools  # noqa: E402


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def config(fake_model) -> HarnessConfig:
    # A large budget keeps unit tests fast and deterministic.
    return HarnessConfig(model_client=fake_model, max_context_tokens=1_000_000)


@pytest.fixture
def tools(tmp_path):
    return builtin_tools(tmp_path / "tooldata", flaky=False)


@pytest.fixture
def harness(config, tools, tmp_path):
    h = Harness(config, tools, tmp_path / "state")
    yield h
    h.close()
