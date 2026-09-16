"""Warteschlangen, Abbruch, Streaming, Kosten, Ping-Pong und Kuratierung.

Deckt die Abnahmefälle B (ein schneller Anbieter wartet nicht auf den
langsamen) und I (das Ping-Pong erreicht sein Limit und stoppt) ab.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from app.config import ModelConfig, Settings
from app.main import create_app
from app.providers.fake import FakeProvider, FakeScenario
from conftest import scenario, wait_for_jobs

TOKEN = "ein-langes-zugangswort-zum-testen"


def stueckweise(model: str, text: str, chunks: int, delay: float = 0.05) -> FakeScenario:
    return FakeProvider.configure(
        model, FakeScenario(text=text, chunks=chunks, chunk_delay_s=delay)
    )


# --------------------------------------------------------- Warteschlangen

@pytest.mark.asyncio
async def test_schneller_anbieter_wartet_nicht_auf_den_langsamen(client, session_id):
    """Abnahmefall B: xAI ist fertig und nimmt den nächsten Auftrag an,
    während OpenAI noch am vorherigen arbeitet."""
    scenario("fake-a-model", text="Langsam, aber sicher.", delay_s=1.2)
    scenario("fake-b-model", text="Sofort da.", delay_s=0.0)

    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Erster Gedanke.", "client_request_id": "req-1"})
    await asyncio.sleep(0.3)
    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Zweiter Gedanke.", "client_request_id": "req-2"})
    await asyncio.sleep(0.5)

    stand = (await client.get(f"/api/sessions/{session_id}")).json()
    zustaende = {
        (e["spark"]["seq"], j["model_id"]): j["status"]
        for e in stand["sparks"] for j in e["jobs"]
    }
    # Der schnelle Anbieter hat beide Gedanken schon bearbeitet …
    assert zustaende[(1, "fake-b")] == "done"
    assert zustaende[(2, "fake-b")] == "done"
    # … während der langsame noch am ersten sitzt und der zweite wartet.
    assert zustaende[(1, "fake-a")] in ("running", "streaming")
    assert zustaende[(2, "fake-a")] == "queued"

    await wait_for_jobs(client, session_id, timeout=20)


@pytest.mark.asyncio
async def test_ein_anbieter_arbeitet_seine_auftraege_der_reihe_nach_ab(client, session_id):
    scenario("fake-a-model", text="A.", delay_s=0.4)
    scenario("fake-b-model", text="B.", delay_s=0.0)
    for n in (1, 2, 3):
        await client.post(f"/api/sessions/{session_id}/sparks",
                          json={"prompt": f"Gedanke {n}.", "client_request_id": f"req-{n}"})
    await asyncio.sleep(0.25)
    stand = (await client.get(f"/api/sessions/{session_id}")).json()
    laufend = [
        j for e in stand["sparks"] for j in e["jobs"]
        if j["model_id"] == "fake-a" and j["status"] in ("running", "streaming")
    ]
    assert len(laufend) == 1, "der Anbieter soll nur einen Auftrag gleichzeitig haben"
    await wait_for_jobs(client, session_id, timeout=20)


# ------------------------------------------------------------------ Abbruch

@pytest.mark.asyncio
async def test_laufenden_auftrag_abbrechen(client, session_id):
    scenario("fake-a-model", text="Wird nie fertig.", delay_s=5)
    scenario("fake-b-model", text="Ich bin schnell durch.", delay_s=0.0)
    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Abbrechen bitte.", "client_request_id": "req-ab"})
    await asyncio.sleep(0.3)

    stand = (await client.get(f"/api/sessions/{session_id}")).json()
    langsam = next(j for j in stand["sparks"][0]["jobs"] if j["model_id"] == "fake-a")
    antwort = await client.post(f"/api/jobs/{langsam['id']}/cancel")
    assert antwort.status_code == 200

    bundle = await wait_for_jobs(client, session_id, timeout=15)
    jobs = {j["model_id"]: j for j in bundle["sparks"][0]["jobs"]}
    assert jobs["fake-a"]["status"] == "cancelled"
    assert "abgebrochen" in jobs["fake-a"]["error"]
    # Der andere Auftrag ist davon unberührt.
    assert jobs["fake-b"]["status"] == "done"
    assert jobs["fake-b"]["text"] == "Ich bin schnell durch."


@pytest.mark.asyncio
async def test_wartenden_auftrag_abbrechen(client, session_id):
    scenario("fake-a-model", text="Erster.", delay_s=1.0)
    scenario("fake-b-model", text="Egal.", delay_s=0.0)
    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Eins.", "client_request_id": "req-1"})
    await asyncio.sleep(0.2)
    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Zwei.", "client_request_id": "req-2"})
    await asyncio.sleep(0.2)

    stand = (await client.get(f"/api/sessions/{session_id}")).json()
    wartend = next(
        j for e in stand["sparks"] if e["spark"]["seq"] == 2
        for j in e["jobs"] if j["model_id"] == "fake-a"
    )
    assert wartend["status"] == "queued"
    assert (await client.post(f"/api/jobs/{wartend['id']}/cancel")).status_code == 200

    await asyncio.sleep(1.5)
    stand = (await client.get(f"/api/sessions/{session_id}")).json()
    danach = next(
        j for e in stand["sparks"] if e["spark"]["seq"] == 2
        for j in e["jobs"] if j["model_id"] == "fake-a"
    )
    assert danach["status"] == "cancelled"
    assert FakeProvider.call_count("fake-a-model") == 1, "der abgebrochene Auftrag lief nie"


@pytest.mark.asyncio
async def test_fertigen_auftrag_kann_man_nicht_abbrechen(client, session_id):
    scenario("fake-a-model", text="Schon fertig.")
    scenario("fake-b-model", text="Auch fertig.")
    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Fertig.", "client_request_id": "req-f"})
    bundle = await wait_for_jobs(client, session_id)
    job = bundle["sparks"][0]["jobs"][0]
    antwort = await client.post(f"/api/jobs/{job['id']}/cancel")
    assert antwort.status_code == 409
    assert "läuft nicht mehr" in antwort.json()["detail"]


# ---------------------------------------------------------------- Streaming

@pytest.mark.asyncio
async def test_teilantwort_wird_waehrend_des_entstehens_festgehalten(client, session_id):
    langer_text = "Erst dies. Dann das. Und schließlich jenes. " * 6
    stueckweise("fake-a-model", langer_text, chunks=12, delay=0.14)
    scenario("fake-b-model", text="Kurz.", delay_s=0.0)

    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Langsam schreiben.", "client_request_id": "req-stream"})
    await asyncio.sleep(0.75)

    stand = (await client.get(f"/api/sessions/{session_id}")).json()
    laufend = next(j for j in stand["sparks"][0]["jobs"] if j["model_id"] == "fake-a")
    # Der Zwischenstand steht schon in der Datenbank — ein Neuladen zeigt ihn.
    assert laufend["status"] == "streaming"
    assert 0 < len(laufend["text"]) < len(langer_text)

    bundle = await wait_for_jobs(client, session_id, timeout=20)
    fertig = next(j for j in bundle["sparks"][0]["jobs"] if j["model_id"] == "fake-a")
    assert fertig["status"] == "done"
    assert fertig["text"] == langer_text


# ------------------------------------------------------------------- Kosten

@pytest_asyncio.fixture
async def preis_client(tmp_path, monkeypatch):
    """Ein Tisch mit echten Anbieterzeilen, damit Preise hinterlegt werden können."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                 "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(env="test", db_path=tmp_path / "p.sqlite3", models=[],
                        admin_token=TOKEN)
    settings.validate()
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=20
        ) as c:
            c.headers["X-Admin-Token"] = TOKEN
            yield c


@pytest.mark.asyncio
async def test_ohne_hinterlegten_preis_bleiben_kosten_unbekannt(client, session_id):
    FakeProvider.configure("fake-a-model",
                           FakeScenario(text="Mit Verbrauch.", tokens_in=1000, tokens_out=500))
    scenario("fake-b-model", text="Ohne alles.")
    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Was kostet das?", "client_request_id": "req-kosten"})
    bundle = await wait_for_jobs(client, session_id)
    job = next(j for j in bundle["sparks"][0]["jobs"] if j["model_id"] == "fake-a")

    assert job["tokens_in"] == 1000
    assert job["tokens_out"] == 500
    assert job["cost_micro"] is None
    assert job["cost_source"] == "unbekannt"


@pytest.mark.asyncio
async def test_preise_lassen_sich_hinterlegen(preis_client):
    antwort = await preis_client.post("/api/admin/providers/deepseek",
                                      json={"price_in": 0.27, "price_out": 1.1})
    assert antwort.status_code == 200
    zeile = {p["id"]: p for p in antwort.json()["providers"]}["deepseek"]
    assert zeile["price_in"] == 0.27
    assert zeile["price_out"] == 1.1

    # Ein leerer Preis löscht ihn wieder.
    antwort = await preis_client.post("/api/admin/providers/deepseek", json={"price_in": 0})
    zeile = {p["id"]: p for p in antwort.json()["providers"]}["deepseek"]
    assert zeile["price_in"] is None


# ----------------------------------------------------------------- Ping-Pong

@pytest.mark.asyncio
async def test_pingpong_haelt_seine_obergrenze_ein(client, session_id):
    scenario("fake-a-model", text="Ich sage A.", delay_s=0.05)
    scenario("fake-b-model", text="Ich sage B.", delay_s=0.05)

    antwort = await client.post(f"/api/sessions/{session_id}/pingpong", json={
        "prompt": "Diskutiert das bitte.",
        "participants": ["fake-a", "fake-b"],
        "max_turns": 4,
    })
    assert antwort.status_code == 201
    lauf = antwort.json()
    assert lauf["max_turns"] == 4
    assert lauf["labels"] == ["Modell A", "Modell B"]

    for _ in range(80):
        stand = (await client.get(f"/api/sessions/{session_id}/pingpong")).json()["runs"][0]
        if stand["status"] != "laeuft":
            break
        await asyncio.sleep(0.15)

    # Abnahmefall I.
    assert stand["status"] == "beendet"
    assert stand["turn"] == 4
    assert "Obergrenze" in stand["stopped_reason"]

    bundle = (await client.get(f"/api/sessions/{session_id}")).json()
    pingpong = [e for e in bundle["sparks"] if e["spark"]["kind"] == "pingpong"]
    assert len(pingpong) == 4
    # Abwechselnd, und jede Runde bezieht sich auf die vorherige Antwort.
    sprecher = [
        next(j["model_id"] for j in e["jobs"] if j["status"] == "done") for e in pingpong
    ]
    assert sprecher == ["fake-a", "fake-b", "fake-a", "fake-b"]
    assert pingpong[1]["spark"]["refs"], "die zweite Runde braucht einen Bezug"


@pytest.mark.asyncio
async def test_pingpong_laesst_sich_stoppen(client, session_id):
    scenario("fake-a-model", text="A.", delay_s=0.3)
    scenario("fake-b-model", text="B.", delay_s=0.3)
    lauf = (await client.post(f"/api/sessions/{session_id}/pingpong", json={
        "prompt": "Lang und breit.", "participants": ["fake-a", "fake-b"], "max_turns": 12,
    })).json()

    await asyncio.sleep(0.5)
    gestoppt = await client.post(f"/api/sessions/{session_id}/pingpong/{lauf['id']}/stop")
    assert gestoppt.status_code == 200

    for _ in range(40):
        stand = (await client.get(f"/api/sessions/{session_id}/pingpong")).json()["runs"][0]
        if stand["status"] != "laeuft":
            break
        await asyncio.sleep(0.15)
    assert stand["status"] in ("gestoppt", "beendet")
    assert stand["turn"] < 12


@pytest.mark.asyncio
async def test_pingpong_braucht_zwei_einsatzbereite_anbieter(client, session_id):
    antwort = await client.post(f"/api/sessions/{session_id}/pingpong", json={
        "prompt": "Allein?", "participants": ["fake-a", "gibtsnicht"], "max_turns": 2,
    })
    assert antwort.status_code == 422
    assert "mindestens zwei" in antwort.json()["detail"]


# --------------------------------------------------------------- Kuratierung

@pytest.mark.asyncio
async def test_ohne_eingestellten_kurator_laeuft_die_runde_trotzdem(client, session_id):
    scenario("fake-a-model", text="Antwort A.")
    scenario("fake-b-model", text="Antwort B.")
    antwort = await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Mit Kuratierung bitte.", "client_request_id": "req-kur", "curate": True,
    })
    assert antwort.status_code == 201
    assert antwort.json()["curation"]["gestartet"] is False
    assert "Kein Kurator" in antwort.json()["curation"]["grund"]

    bundle = await wait_for_jobs(client, session_id)
    assert len(bundle["sparks"]) == 1, "keine Kuratierung, aber die Runde lief"


@pytest.mark.asyncio
async def test_kuratierung_ist_ein_eigener_beitrag_und_ersetzt_nichts(client, session_id):
    scenario("fake-a-model", text="Die Kuratorin ordnet ein.")
    scenario("fake-b-model", text="Die eigentliche Antwort.")
    # Modell A als Kurator einsetzen.
    store = client.app.state.store
    await store.set_setting("curator", "fake-a")
    client.app.state.settings.overrides = await store.all_settings()

    original = "Mein ganz eigener Wortlaut, unverändert."
    antwort = await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": original, "client_request_id": "req-kur2", "curate": True,
        "model_ids": ["fake-b"],
    })
    assert antwort.json()["curation"]["gestartet"] is True
    assert antwort.json()["curation"]["label"] == "Modell A"

    await asyncio.sleep(1.2)
    bundle = (await client.get(f"/api/sessions/{session_id}")).json()
    arten = {e["spark"]["kind"] for e in bundle["sparks"]}
    assert "kuratierung" in arten

    # Der Originalfunke steht unverändert da.
    original_funke = next(e for e in bundle["sparks"] if e["spark"]["kind"] == "funke")
    assert original_funke["spark"]["prompt"] == original
    # Und die Kuratierung verweist auf ihn, statt ihn zu ersetzen.
    kuratierung = next(e for e in bundle["sparks"] if e["spark"]["kind"] == "kuratierung")
    assert original_funke["spark"]["id"] in kuratierung["spark"]["refs"]
