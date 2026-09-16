"""Tests gegen die echten Provider-APIs.

Diese Tests laufen nur, wenn Zugangsdaten vorhanden sind, und sind separat
ausgewiesen (Markierung `live`). Standardlauf:

    pytest                      # nur Fakes, kein Netzwerk
    pytest -m live              # zusätzlich echte API-Aufrufe (kostenpflichtig)
"""

from __future__ import annotations

import os

import pytest

from app.config import ModelConfig, Settings
from app.providers import ProviderRegistry

pytestmark = pytest.mark.live

OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY nicht gesetzt"
)
ANTHROPIC = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY nicht gesetzt"
)

FRAGE = "Antworte mit genau einem kurzen Satz: Wofür steht die Abkürzung API?"


def _settings(model: ModelConfig) -> Settings:
    return Settings(
        env="production",
        models=[model],
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        request_timeout_s=60,
    )


@OPENAI
@pytest.mark.asyncio
async def test_openai_antwortet_echt():
    model = ModelConfig(
        id="openai", label="OpenAI", provider="openai",
        model=os.environ.get("XT_OPENAI_MODEL", "gpt-4.1"),
    )
    registry = ProviderRegistry(_settings(model))
    try:
        response = await registry.get("openai").complete(
            prompt=FRAGE, model=model.model, timeout_s=60
        )
        assert response.text.strip()
    finally:
        await registry.aclose()


@ANTHROPIC
@pytest.mark.asyncio
async def test_anthropic_antwortet_echt():
    model = ModelConfig(
        id="anthropic", label="Anthropic", provider="anthropic",
        model=os.environ.get("XT_ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    )
    registry = ProviderRegistry(_settings(model))
    try:
        response = await registry.get("anthropic").complete(
            prompt=FRAGE, model=model.model, timeout_s=60
        )
        assert response.text.strip()
    finally:
        await registry.aclose()
