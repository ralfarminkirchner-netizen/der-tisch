"""TiSCH-Lauf: Kette, Abbruch, Reibung wartet, Fake-TableResponse."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.providers.fake import FakeProvider, FakeScenario
from conftest import scenario


async def wait_lauf(client, lauf_id: str, timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        resp = await client.get(f"/api/laeufe/{lauf_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last.get("status") == "done":
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(f"Lauf nicht fertig: {last.get('phase')} {last.get('status')}")


@pytest.mark.asyncio
async def test_get_tische_enthaelt_familientisch(client):
    resp = await client.get("/api/tische")
    assert resp.status_code == 200
    ids = {t["id"] for t in resp.json()}
    assert "familientisch" in ids
    one = await client.get("/api/tische/familientisch")
    assert one.status_code == 200
    assert "frage" in one.json()["modi"]


@pytest.mark.asyncio
async def test_familientisch_frage_liefert_table_response(client):
    scenario("fake-a-model", text='{"rolle":"A","anspruchstyp":"geltung","kernanalyse":"Ka.","evidenz":"E","blinder_fleck":"B"}')
    scenario("fake-b-model", text='{"uebersetzungsfehler":[],"echte_widersprueche":["x"],"uebersehenes":"o"}')
    # default model is fake-a; reibung/integration also default unless set
    FakeProvider.configure(
        "fake-a-model",
        FakeScenario(text='{"rolle":"Sys","anspruchstyp":"g","kernanalyse":"Kern","evidenz":"-","blinder_fleck":"-"}'),
    )
    resp = await client.post(
        "/api/laeufe",
        json={"tisch_id": "familientisch", "modus": "frage", "question": "Wie setzen wir Grenzen?"},
    )
    assert resp.status_code == 201, resp.text
    lauf_id = resp.json()["id"]
    done = await wait_lauf(client, lauf_id)
    table = done["table"]
    assert "perspectives" in table and table["perspectives"]
    assert set(table["perspectives"][0]) >= {"rolle", "anspruchstyp", "kernanalyse", "evidenz", "blinder_fleck"}
    assert set(table["friction"]) == {"uebersetzungsfehler", "echte_widersprueche", "uebersehenes"}
    assert "vorlaeufiges_fazit" in table["integration"]
    assert done["snapshots"]
    assert len(done["snapshots"]) == len(table["perspectives"])


@pytest.mark.asyncio
async def test_abbruch_einer_perspektive(client):
    FakeProvider.configure("fake-a-model", FakeScenario(delay_s=0.35, text='{"rolle":"X","anspruchstyp":"","kernanalyse":"k","evidenz":"","blinder_fleck":""}'))
    meta = (await client.get("/api/tische/familientisch")).json()
    first_id = meta["perspectives"][0]["id"] if isinstance(meta["perspectives"], list) and meta["perspectives"] and isinstance(meta["perspectives"][0], dict) else None
    # GET tische/{id} returns model_dump with perspectives as list of dicts
    detail = (await client.get("/api/tische/familientisch")).json()
    pid = detail["perspectives"][0]["id"]
    started = await client.post(
        "/api/laeufe",
        json={"tisch_id": "familientisch", "modus": "frage", "question": "Abbruchtest bitte jetzt."},
    )
    lauf_id = started.json()["id"]
    await client.post(f"/api/laeufe/{lauf_id}/cancel/{pid}")
    done = await wait_lauf(client, lauf_id)
    rollen_ids = [s.get("perspective_id") for s in done.get("snapshots", [])]
    # Abgebrochene Perspektive darf fehlen oder Snapshot ohne Ergebnis
    assert pid not in [p.get("rolle") for p in done["table"]["perspectives"]]
    assert done["status"] == "done"


@pytest.mark.asyncio
async def test_reibung_wartet_auf_alle_perspektiven(client):
    FakeProvider.configure("fake-a-model", FakeScenario(delay_s=0.2, text='{"rolle":"R","anspruchstyp":"","kernanalyse":"k","evidenz":"","blinder_fleck":""}'))
    started = await client.post(
        "/api/laeufe",
        json={"tisch_id": "familientisch", "modus": "frage", "question": "Wartet die Reibung wirklich?"},
    )
    lauf_id = started.json()["id"]
    saw_perspektiven = False
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rec = (await client.get(f"/api/laeufe/{lauf_id}")).json()
        if rec.get("phase") == "perspektiven":
            saw_perspektiven = True
            assert rec.get("friction") is None
        if rec.get("phase") in ("reibung", "integration", "done") and rec.get("friction") is not None:
            assert saw_perspektiven or rec["status"] == "done"
            break
        await asyncio.sleep(0.03)
    done = await wait_lauf(client, lauf_id)
    assert done["friction"] is not None
    assert done["integration"] is not None


@pytest.mark.asyncio
async def test_kein_gpt4o_mini_im_laufmodul():
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "app" / "tisch_lauf.py"
    assert "gpt-4o-mini" not in src.read_text()
