"""Konfiguration, Healthcheck und die Trennung von Test-Fakes und Produktion."""

from __future__ import annotations

import pytest

from app.config import ModelConfig, Settings, load_settings
from conftest import FAKE_MODELS


def test_fakes_sind_im_produktionsmodus_verboten():
    settings = Settings(env="production", allow_fake_providers=True)
    with pytest.raises(RuntimeError, match="Produktionsmodus"):
        settings.validate()


def test_fake_modell_ohne_freischaltung_verboten():
    settings = Settings(env="development", allow_fake_providers=False, models=list(FAKE_MODELS))
    with pytest.raises(RuntimeError, match="nicht freigeschaltet|fake"):
        settings.validate()


def test_load_settings_produktion_hat_keine_fakes(monkeypatch):
    for key in ("XT_ENV", "XT_ALLOW_FAKE_PROVIDERS", "XT_USE_FAKE_MODELS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XT_ENV", "production")
    settings = load_settings()
    assert settings.allow_fake_providers is False
    assert all(m.provider != "fake" for m in settings.models)


def test_load_settings_fake_modelle_brauchen_freischaltung(monkeypatch):
    monkeypatch.setenv("XT_ENV", "development")
    monkeypatch.setenv("XT_USE_FAKE_MODELS", "1")
    monkeypatch.delenv("XT_ALLOW_FAKE_PROVIDERS", raising=False)
    with pytest.raises(RuntimeError, match="XT_ALLOW_FAKE_PROVIDERS"):
        load_settings()


def test_fehlende_zugangsdaten_werden_benannt(monkeypatch):
    from app.provider_table import ProviderRecord, ProviderTable

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tabelle = ProviderTable(Settings(env="production"))
    tabelle._records = [
        ProviderRecord(id="openai", label="OpenAI", kind="openai", base_url=None,
                       model="gpt-4.1", stored_key="", enabled=True, is_preset=True),
    ]
    assert tabelle.missing_keys() == ["OPENAI_API_KEY"]
    assert tabelle.ready() == []
    assert "Kein Schlüssel hinterlegt" in tabelle.all()[0].reason


@pytest.mark.asyncio
async def test_healthcheck(client):
    body = (await client.get("/api/health")).json()
    assert body["database"] == "ok"
    assert body["status"] == "ok"
    assert body["providers"]["fake-a"]["ready"] is True
    assert body["providers"]["fake-b"]["ready"] is True
    assert body["fake_providers_enabled"] is True


@pytest.mark.asyncio
async def test_config_enthaelt_keine_zugangsdaten(client):
    raw = (await client.get("/api/config")).text
    body = (await client.get("/api/config")).json()
    assert {m["id"] for m in body["models"]} == {"fake-a", "fake-b"}
    for verboten in ("api_key", "API_KEY", "secret", "sk-"):
        assert verboten not in raw
