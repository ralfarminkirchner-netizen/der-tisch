"""Aufträge: Ausfallisolation, Verzögerungen, Teilantworten, Doppelübertragung."""

from __future__ import annotations

import asyncio

import pytest

from app.providers.fake import FakeProvider
from conftest import scenario, wait_for_jobs


@pytest.mark.asyncio
async def test_ausfall_eines_providers_blockiert_den_anderen_nicht(client, session_id):
    scenario("fake-a-model", error="Provider A ist ausgefallen (503).", delay_s=0.05)
    scenario("fake-b-model", text="Antwort B steht trotzdem.", delay_s=0.05)

    resp = await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Wie belastbar ist das System?", "client_request_id": "req-1"},
    )
    assert resp.status_code == 201

    bundle = await wait_for_jobs(client, session_id)
    jobs = {j["model_id"]: j for j in bundle["sparks"][0]["jobs"]}

    assert jobs["fake-a"]["status"] == "error"
    assert "ausgefallen" in jobs["fake-a"]["error"]
    assert jobs["fake-b"]["status"] == "done"
    assert jobs["fake-b"]["text"] == "Antwort B steht trotzdem."


@pytest.mark.asyncio
async def test_langsamer_provider_haelt_den_schnellen_nicht_auf(client, session_id):
    scenario("fake-a-model", delay_s=0.6, text="Späte Antwort A.")
    scenario("fake-b-model", delay_s=0.0, text="Sofortige Antwort B.")

    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Dauert das lange?", "client_request_id": "req-delay"},
    )

    # Kurz nach dem Start ist B bereits fertig, A noch unterwegs.
    await asyncio.sleep(0.25)
    bundle = (await client.get(f"/api/sessions/{session_id}")).json()
    jobs = {j["model_id"]: j for j in bundle["sparks"][0]["jobs"]}
    assert jobs["fake-b"]["status"] == "done"
    assert jobs["fake-a"]["status"] in ("queued", "running")

    bundle = await wait_for_jobs(client, session_id)
    jobs = {j["model_id"]: j for j in bundle["sparks"][0]["jobs"]}
    assert jobs["fake-a"]["status"] == "done"


@pytest.mark.asyncio
async def test_teilantwort_wird_als_solche_gekennzeichnet(client, session_id):
    scenario("fake-a-model", text="Angefangene Antwort, die abbricht", partial=True)
    scenario("fake-b-model", error="Abbruch nach Teilübertragung", partial_text="Halber Satz")

    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Was bleibt bei Abbruch?", "client_request_id": "req-partial"},
    )
    bundle = await wait_for_jobs(client, session_id)
    jobs = {j["model_id"]: j for j in bundle["sparks"][0]["jobs"]}

    assert jobs["fake-a"]["status"] == "done" and jobs["fake-a"]["partial"] is True
    assert jobs["fake-b"]["status"] == "error"
    assert jobs["fake-b"]["text"] == "Halber Satz"
    assert jobs["fake-b"]["partial"] is True


@pytest.mark.asyncio
async def test_doppelte_uebertragung_erzeugt_keine_doppelaufrufe(client, session_id):
    scenario("fake-a-model", text="Einmalige Antwort A.", delay_s=0.1)
    scenario("fake-b-model", text="Einmalige Antwort B.", delay_s=0.1)

    payload = {"prompt": "Nur einmal bitte.", "client_request_id": "req-idem"}
    first = await client.post(f"/api/sessions/{session_id}/sparks", json=payload)
    second = await client.post(f"/api/sessions/{session_id}/sparks", json=payload)

    assert first.status_code == 201 and first.json()["duplicate"] is False
    assert second.status_code == 200 and second.json()["duplicate"] is True
    assert first.json()["spark"]["id"] == second.json()["spark"]["id"]

    bundle = await wait_for_jobs(client, session_id)
    assert len(bundle["sparks"]) == 1
    assert len(bundle["sparks"][0]["jobs"]) == 2
    assert FakeProvider.call_count("fake-a-model") == 1
    assert FakeProvider.call_count("fake-b-model") == 1


@pytest.mark.asyncio
async def test_gleichzeitige_doppelte_uebertragung(client, session_id):
    scenario("fake-a-model", text="A", delay_s=0.05)
    scenario("fake-b-model", text="B", delay_s=0.05)

    payload = {"prompt": "Parallel doppelt gesendet.", "client_request_id": "req-race"}
    await asyncio.gather(
        *[client.post(f"/api/sessions/{session_id}/sparks", json=payload) for _ in range(4)]
    )
    bundle = await wait_for_jobs(client, session_id)
    assert len(bundle["sparks"]) == 1
    assert len(bundle["sparks"][0]["jobs"]) == 2
    assert FakeProvider.call_count("fake-a-model") == 1


@pytest.mark.asyncio
async def test_abgeschlossene_sitzung_nimmt_keine_funken_mehr(client, session_id):
    scenario("fake-a-model", text="A")
    scenario("fake-b-model", text="B")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Letzte Frage.", "client_request_id": "req-last"},
    )
    await wait_for_jobs(client, session_id)

    closed = await client.post(
        f"/api/sessions/{session_id}/close", json={"note": "Erledigt."}
    )
    assert closed.json()["status"] == "abgeschlossen"
    assert closed.json()["closing_note"] == "Erledigt."

    blocked = await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Doch noch etwas?", "client_request_id": "req-after"},
    )
    assert blocked.status_code == 409
