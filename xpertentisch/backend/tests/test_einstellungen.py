"""Einstellungsseite: Zugangsschutz, Speicherung, Wirksamkeit, Verschwiegenheit."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from app.config import ModelConfig, Settings
from app.main import create_app

TOKEN = "ein-langes-zugangswort-zum-testen"
ECHTE_MODELLE = [
    ModelConfig(id="openai-gpt", label="OpenAI GPT", provider="openai", model="gpt-4.1"),
    ModelConfig(id="anthropic-claude", label="Anthropic Claude",
                provider="anthropic", model="claude-sonnet-4-5"),
]


def echte_settings(tmp_path: Path, **over) -> Settings:
    s = Settings(env="test", db_path=tmp_path / "e.sqlite3", models=list(ECHTE_MODELLE), **over)
    s.validate()
    return s


@pytest_asyncio.fixture
async def admin_client(tmp_path: Path):
    app = create_app(echte_settings(tmp_path, admin_token=TOKEN))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=20
        ) as c:
            c.headers["X-Admin-Token"] = TOKEN
            yield c


@pytest.mark.asyncio
async def test_ohne_zugangswort_bleibt_die_seite_gesperrt(tmp_path):
    app = create_app(echte_settings(tmp_path))  # kein admin_token
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            antwort = await c.get("/api/admin/settings")
            konfig = (await c.get("/api/config")).json()
    assert antwort.status_code == 503
    assert "XT_ADMIN_TOKEN" in antwort.json()["detail"]
    assert konfig["settings_available"] is False


@pytest.mark.asyncio
async def test_falsches_zugangswort_wird_abgewiesen(admin_client):
    antwort = await admin_client.get(
        "/api/admin/settings", headers={"X-Admin-Token": "falsch"}
    )
    assert antwort.status_code == 401
    ohne = await admin_client.get("/api/admin/settings", headers={"X-Admin-Token": ""})
    assert ohne.status_code == 401


@pytest.mark.asyncio
async def test_uebersicht_zeigt_zustand_ohne_schluessel(admin_client):
    body = (await admin_client.get("/api/admin/settings")).json()
    anbieter = {p["provider"]: p for p in body["providers"]}
    assert set(anbieter) == {"openai", "anthropic"}
    assert anbieter["openai"]["key_source"] == "fehlt"
    assert anbieter["openai"]["ready"] is False
    assert anbieter["openai"]["key_hint"] == ""
    assert (await admin_client.get("/api/config")).json()["settings_available"] is True


@pytest.mark.asyncio
async def test_schluessel_eintragen_macht_den_provider_bereit(admin_client):
    vorher = (await admin_client.get("/api/health")).json()
    assert "OPENAI_API_KEY" in vorher["missing_credentials"]

    antwort = await admin_client.post(
        "/api/admin/settings", json={"openai_api_key": "sk-test-0123456789abcdef"}
    )
    assert antwort.status_code == 200
    assert antwort.json()["changed"] == ["openai_api_key"]

    anbieter = {p["provider"]: p for p in antwort.json()["providers"]}
    assert anbieter["openai"]["key_source"] == "einstellungen"
    assert anbieter["openai"]["key_hint"] == "…cdef"
    assert anbieter["openai"]["ready"] is True

    nachher = (await admin_client.get("/api/health")).json()
    assert "OPENAI_API_KEY" not in nachher["missing_credentials"]
    assert nachher["providers"]["openai"]["ready"] is True


@pytest.mark.asyncio
async def test_schluessel_taucht_in_keiner_antwort_auf(admin_client):
    schluessel = "sk-geheim-9876543210zyxwvu"
    await admin_client.post("/api/admin/settings", json={"openai_api_key": schluessel})
    for pfad in ("/api/config", "/api/health", "/api/admin/settings"):
        text = (await admin_client.get(pfad)).text
        assert schluessel not in text
        assert "geheim" not in text


@pytest.mark.asyncio
async def test_leerer_wert_loescht_und_gibt_der_umgebung_wieder_vorrang(tmp_path):
    settings = echte_settings(tmp_path, admin_token=TOKEN, openai_api_key="aus-der-umgebung-1234")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            c.headers["X-Admin-Token"] = TOKEN
            body = (await c.get("/api/admin/settings")).json()
            assert body["providers"][0]["key_source"] == "umgebung"

            body = (await c.post("/api/admin/settings",
                                 json={"openai_api_key": "sk-neu-abcdefgh"})).json()
            assert body["providers"][0]["key_source"] == "einstellungen"

            body = (await c.post("/api/admin/settings", json={"openai_api_key": ""})).json()
            assert body["providers"][0]["key_source"] == "umgebung"
            assert body["providers"][0]["key_hint"] == "…1234"


@pytest.mark.asyncio
async def test_modellname_und_zeitgrenze_wirken_sofort(admin_client):
    await admin_client.post("/api/admin/settings",
                            json={"openai_model": "gpt-4.1-mini", "request_timeout_s": 42})
    konfig = (await admin_client.get("/api/config")).json()
    modelle = {m["provider"]: m["model"] for m in konfig["models"]}
    assert modelle["openai"] == "gpt-4.1-mini"

    body = (await admin_client.get("/api/admin/settings")).json()
    assert body["request_timeout_s"] == 42
    assert body["timeout_source"] == "einstellungen"


@pytest.mark.asyncio
async def test_unsinnige_zeitgrenze_wird_abgewiesen(admin_client):
    assert (await admin_client.post("/api/admin/settings",
                                    json={"request_timeout_s": 99999})).status_code == 422
    assert (await admin_client.post("/api/admin/settings",
                                    json={"request_timeout_s": 1})).status_code == 422


@pytest.mark.asyncio
async def test_einstellungen_ueberleben_den_neustart(tmp_path):
    for durchgang in (1, 2):
        app = create_app(echte_settings(tmp_path, admin_token=TOKEN))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as c:
                c.headers["X-Admin-Token"] = TOKEN
                if durchgang == 1:
                    await c.post("/api/admin/settings",
                                 json={"anthropic_api_key": "sk-ant-bleibt-erhalten-42"})
                body = (await c.get("/api/admin/settings")).json()
    anbieter = {p["provider"]: p for p in body["providers"]}
    assert anbieter["anthropic"]["key_source"] == "einstellungen"
    assert anbieter["anthropic"]["key_hint"] == "…n-42"
    assert anbieter["anthropic"]["ready"] is True


@pytest.mark.asyncio
async def test_pruefknopf_meldet_fehlenden_schluessel_statt_zu_werfen(admin_client):
    antwort = await admin_client.post("/api/admin/test", json={"provider": "anthropic"})
    assert antwort.status_code == 200
    assert antwort.json()["ok"] is False
    assert "Schlüssel" in antwort.json()["detail"]


@pytest.mark.asyncio
async def test_pruefknopf_kennt_nur_eingerichtete_provider(admin_client):
    assert (await admin_client.post("/api/admin/test",
                                    json={"provider": "erfunden"})).status_code == 404


@pytest.mark.asyncio
async def test_einstellungen_brauchen_das_zugangswort_auch_beim_schreiben(admin_client):
    antwort = await admin_client.post(
        "/api/admin/settings",
        json={"openai_api_key": "sk-darf-nicht-durch"},
        headers={"X-Admin-Token": "falsch"},
    )
    assert antwort.status_code == 401
    body = (await admin_client.get("/api/admin/settings")).json()
    assert body["providers"][0]["key_source"] == "fehlt"
