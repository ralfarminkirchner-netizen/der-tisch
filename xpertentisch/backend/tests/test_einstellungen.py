"""Anbieterverwaltung: Zugangsschutz, Speicherung, Wirksamkeit, Verschwiegenheit."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from app.config import Settings
from app.main import create_app

TOKEN = "ein-langes-zugangswort-zum-testen"


def leere_settings(tmp_path: Path, **over) -> Settings:
    """Ein Tisch, der aus der Datenbank kommt (kein fester Testtisch)."""
    s = Settings(env="test", db_path=tmp_path / "e.sqlite3", models=[], **over)
    s.validate()
    return s


@pytest_asyncio.fixture
async def admin_client(tmp_path: Path, monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                 "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(leere_settings(tmp_path, admin_token=TOKEN))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=20
        ) as c:
            c.headers["X-Admin-Token"] = TOKEN
            yield c


async def anbieter(client) -> dict[str, dict]:
    body = (await client.get("/api/admin/providers")).json()
    return {p["id"]: p for p in body["providers"]}


# ------------------------------------------------------------------- Zugang

@pytest.mark.asyncio
async def test_ohne_zugangswort_bleibt_alles_gesperrt(tmp_path):
    app = create_app(leere_settings(tmp_path))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            lesen = await c.get("/api/admin/providers")
            schreiben = await c.post("/api/admin/providers/openai", json={"enabled": True})
            konfig = (await c.get("/api/config")).json()
    assert lesen.status_code == 503 and "XT_ADMIN_TOKEN" in lesen.json()["detail"]
    assert schreiben.status_code == 503
    assert konfig["settings_available"] is False


@pytest.mark.asyncio
async def test_falsches_zugangswort_wird_abgewiesen(admin_client):
    for anfrage in (
        admin_client.get("/api/admin/providers", headers={"X-Admin-Token": "falsch"}),
        admin_client.post("/api/admin/providers/openai", json={"enabled": True},
                          headers={"X-Admin-Token": "falsch"}),
        admin_client.delete("/api/admin/providers/openai",
                            headers={"X-Admin-Token": "falsch"}),
    ):
        assert (await anfrage).status_code == 401


# --------------------------------------------------------- Mitgelieferte Liste

@pytest.mark.asyncio
async def test_alle_mitgelieferten_anbieter_stehen_bereit_zur_auswahl(admin_client):
    liste = await anbieter(admin_client)
    assert {"openai", "anthropic", "google", "deepseek", "mistral", "xai"} <= set(liste)
    # Ohne Schlüssel startet niemand eingeschaltet — der Tisch beginnt nicht mit Fehlern.
    assert all(not p["enabled"] for p in liste.values())
    assert (await admin_client.get("/api/config")).json()["models"] == []
    assert liste["deepseek"]["kind"] == "openai"
    assert liste["deepseek"]["base_url"] == "https://api.deepseek.com/v1"
    assert liste["google"]["kind"] == "google"
    assert liste["openai"]["key_url"].startswith("https://")


@pytest.mark.asyncio
async def test_umgebungsschluessel_schaltet_einen_anbieter_von_allein_ein(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-aus-der-umgebung-1234")
    app = create_app(leere_settings(tmp_path, admin_token=TOKEN))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            c.headers["X-Admin-Token"] = TOKEN
            liste = await anbieter(c)
            konfig = (await c.get("/api/config")).json()
    assert liste["deepseek"]["enabled"] is True
    assert liste["deepseek"]["ready"] is True
    assert liste["deepseek"]["key_source"] == "umgebung"
    assert [m["id"] for m in konfig["models"]] == ["deepseek"]


# ------------------------------------------------------------------ Ändern

@pytest.mark.asyncio
async def test_schluessel_eintragen_macht_einen_anbieter_bereit(admin_client):
    antwort = await admin_client.post(
        "/api/admin/providers/google",
        json={"api_key": "AIza-test-0123456789abcdef", "enabled": True},
    )
    assert antwort.status_code == 200
    assert set(antwort.json()["changed"]) == {"api_key", "enabled"}

    liste = {p["id"]: p for p in antwort.json()["providers"]}
    assert liste["google"]["ready"] is True
    assert liste["google"]["key_source"] == "einstellungen"
    assert liste["google"]["key_hint"] == "…cdef"

    gesundheit = (await admin_client.get("/api/health")).json()
    assert gesundheit["providers"]["google"]["ready"] is True
    assert "GOOGLE_API_KEY" not in gesundheit["missing_credentials"]
    assert [m["id"] for m in (await admin_client.get("/api/config")).json()["models"]] == ["google"]


@pytest.mark.asyncio
async def test_abschalten_nimmt_einen_anbieter_vom_tisch(admin_client):
    await admin_client.post("/api/admin/providers/mistral",
                            json={"api_key": "sk-mistral-abcdefgh", "enabled": True})
    assert [m["id"] for m in (await admin_client.get("/api/config")).json()["models"]] == ["mistral"]

    await admin_client.post("/api/admin/providers/mistral", json={"enabled": False})
    assert (await admin_client.get("/api/config")).json()["models"] == []


@pytest.mark.asyncio
async def test_leerer_schluessel_gibt_der_umgebung_wieder_vorrang(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-umgebung-9999")
    app = create_app(leere_settings(tmp_path, admin_token=TOKEN))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            c.headers["X-Admin-Token"] = TOKEN
            assert (await anbieter(c))["xai"]["key_source"] == "umgebung"

            await c.post("/api/admin/providers/xai", json={"api_key": "sk-eigener-1111"})
            assert (await anbieter(c))["xai"]["key_source"] == "einstellungen"

            await c.post("/api/admin/providers/xai", json={"api_key": ""})
            zeile = (await anbieter(c))["xai"]
    assert zeile["key_source"] == "umgebung"
    assert zeile["key_hint"] == "…9999"


@pytest.mark.asyncio
async def test_modellname_laesst_sich_aendern(admin_client):
    await admin_client.post("/api/admin/providers/openai",
                            json={"model": "gpt-4.1-mini", "api_key": "sk-x-abcdefgh",
                                  "enabled": True})
    modelle = {m["id"]: m["model"] for m in (await admin_client.get("/api/config")).json()["models"]}
    assert modelle["openai"] == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_aenderung_ohne_inhalt_wird_abgewiesen(admin_client):
    assert (await admin_client.post("/api/admin/providers/openai", json={})).status_code == 422
    assert (await admin_client.post("/api/admin/providers/gibtsnicht",
                                    json={"enabled": True})).status_code == 404


# ------------------------------------------------------------ Eigene Anbieter

@pytest.mark.asyncio
async def test_eigenen_anbieter_anlegen_und_wieder_loeschen(admin_client):
    antwort = await admin_client.post("/api/admin/providers", json={
        "label": "Groq", "kind": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "api_key": "gsk-test-0123456789",
    })
    assert antwort.status_code == 201
    neue_id = antwort.json()["created"]
    assert neue_id == "groq"

    liste = {p["id"]: p for p in antwort.json()["providers"]}
    assert liste["groq"]["is_preset"] is False
    assert liste["groq"]["ready"] is True
    assert [m["id"] for m in (await admin_client.get("/api/config")).json()["models"]] == ["groq"]

    weg = await admin_client.delete(f"/api/admin/providers/{neue_id}")
    assert weg.status_code == 200
    assert "groq" not in {p["id"] for p in weg.json()["providers"]}


@pytest.mark.asyncio
async def test_gleicher_name_bekommt_eine_eigene_kennung(admin_client):
    rumpf = {"label": "Mein Server", "kind": "openai",
             "base_url": "http://127.0.0.1:8000/v1", "model": "lokal"}
    erste = (await admin_client.post("/api/admin/providers", json=rumpf)).json()["created"]
    zweite = (await admin_client.post("/api/admin/providers", json=rumpf)).json()["created"]
    assert erste == "mein-server"
    assert zweite == "mein-server-2"


@pytest.mark.asyncio
async def test_openai_kompatibler_anbieter_braucht_eine_adresse(admin_client):
    antwort = await admin_client.post("/api/admin/providers", json={
        "label": "Ohne Adresse", "kind": "openai", "model": "irgendwas",
    })
    assert antwort.status_code == 422
    assert "Basis-Adresse" in antwort.json()["detail"]


@pytest.mark.asyncio
async def test_unbekannte_art_wird_abgewiesen(admin_client):
    antwort = await admin_client.post("/api/admin/providers", json={
        "label": "Fantasie", "kind": "telepathie", "model": "x", "base_url": "http://x",
    })
    assert antwort.status_code == 422


@pytest.mark.asyncio
async def test_mitgelieferte_anbieter_lassen_sich_nicht_loeschen(admin_client):
    antwort = await admin_client.delete("/api/admin/providers/openai")
    assert antwort.status_code == 409
    assert "abschalten" in antwort.json()["detail"]
    assert "openai" in await anbieter(admin_client)


# -------------------------------------------------------------- Verschwiegen

@pytest.mark.asyncio
async def test_schluessel_taucht_in_keiner_antwort_auf(admin_client):
    schluessel = "sk-streng-geheim-9876543210"
    await admin_client.post("/api/admin/providers/deepseek",
                            json={"api_key": schluessel, "enabled": True})
    for pfad in ("/api/config", "/api/health", "/api/admin/providers", "/api/admin/settings"):
        text = (await admin_client.get(pfad)).text
        assert schluessel not in text
        assert "geheim" not in text


# ----------------------------------------------------------------- Dauerhaft

@pytest.mark.asyncio
async def test_anbieter_ueberleben_den_neustart(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for durchgang in (1, 2):
        app = create_app(leere_settings(tmp_path, admin_token=TOKEN))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as c:
                c.headers["X-Admin-Token"] = TOKEN
                if durchgang == 1:
                    await c.post("/api/admin/providers", json={
                        "label": "Eigener", "kind": "openai",
                        "base_url": "https://eigen.example/v1", "model": "m-1",
                        "api_key": "sk-bleibt-erhalten-42",
                    })
                    await c.post("/api/admin/providers/google",
                                 json={"model": "gemini-2.5-pro", "enabled": True})
                liste = await anbieter(c)
    assert liste["eigener"]["ready"] is True
    assert liste["eigener"]["key_hint"] == "…n-42"
    assert liste["google"]["model"] == "gemini-2.5-pro"
    assert liste["google"]["enabled"] is True


# --------------------------------------------------------------------- Prüfen

@pytest.mark.asyncio
async def test_pruefknopf_meldet_fehlenden_schluessel_statt_zu_werfen(admin_client):
    antwort = await admin_client.post("/api/admin/providers/anthropic/test")
    assert antwort.status_code == 200
    assert antwort.json()["ok"] is False
    assert "Schlüssel" in antwort.json()["detail"]


@pytest.mark.asyncio
async def test_pruefknopf_kennt_nur_vorhandene_anbieter(admin_client):
    assert (await admin_client.post("/api/admin/providers/erfunden/test")).status_code == 404


# ----------------------------------------------------------------- Zeitgrenze

@pytest.mark.asyncio
async def test_zeitgrenze_wirkt_und_wird_geprueft(admin_client):
    body = (await admin_client.post("/api/admin/settings",
                                    json={"request_timeout_s": 42})).json()
    assert body["request_timeout_s"] == 42
    assert body["timeout_source"] == "einstellungen"
    for unsinn in (99999, 1):
        assert (await admin_client.post("/api/admin/settings",
                                        json={"request_timeout_s": unsinn})).status_code == 422
