"""Ereignisstrom, Wiederverbindung und Serverneustart.

Die SSE-Tests laufen gegen einen echten uvicorn-Server, weil httpx'
ASGITransport Antworten vollständig puffert und einen laufenden Strom
deshalb nicht abbilden kann.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.main import create_app
from app.providers.fake import FakeProvider
from conftest import make_settings, scenario


async def read_events(
    base_url: str,
    session_id: str,
    *,
    last_event_id: int = 0,
    header: str | None = None,
    expect: int = 1,
    timeout: float = 10.0,
) -> list[dict]:
    """Liest eine begrenzte Zahl von SSE-Ereignissen und trennt danach."""
    events: list[dict] = []
    current: dict = {}
    headers = {"Last-Event-ID": header} if header else {}
    params = {} if header else {"last_event_id": last_event_id}

    async def run():
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as c:
            async with c.stream(
                "GET", f"/api/sessions/{session_id}/events", params=params, headers=headers
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                async for line in response.aiter_lines():
                    line = line.rstrip("\r\n")
                    if line.startswith(":") or line.startswith("retry:"):
                        continue
                    if line == "":
                        if current.get("type"):
                            events.append(dict(current))
                            current.clear()
                            if len(events) >= expect:
                                return
                        continue
                    key, _, value = line.partition(":")
                    value = value.lstrip()
                    if key == "id":
                        current["id"] = int(value)
                    elif key == "event":
                        current["type"] = value
                    elif key == "data":
                        current["data"] = json.loads(value)

    await asyncio.wait_for(run(), timeout=timeout)
    return events


async def wait_done(base_url: str, session_id: str, timeout: float = 10.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as c:
        while time.monotonic() < deadline:
            bundle = (await c.get(f"/api/sessions/{session_id}")).json()
            jobs = [j for e in bundle["sparks"] for j in e["jobs"]]
            if jobs and all(j["status"] not in ("queued", "running") for j in jobs):
                if all(e.get("summary") for e in bundle["sparks"]):
                    return bundle
            await asyncio.sleep(0.05)
    raise AssertionError("Aufträge wurden nicht rechtzeitig fertig")


@pytest.mark.asyncio
async def test_sse_liefert_laufende_ereignisse(live_server):
    scenario("fake-a-model", text="Laufende Antwort A.", delay_s=0.3)
    scenario("fake-b-model", text="Laufende Antwort B.", delay_s=0.3)

    async with httpx.AsyncClient(base_url=live_server, timeout=10) as c:
        sid = (await c.post("/api/sessions", json={"title": "Live"})).json()["id"]
        reader = asyncio.create_task(read_events(live_server, sid, expect=6))
        await asyncio.sleep(0.2)
        await c.post(
            f"/api/sessions/{sid}/sparks",
            json={"prompt": "Live mitlesen.", "client_request_id": "req-sse"},
        )
        events = await reader

    typen = [e["type"] for e in events]
    assert "funke.angelegt" in typen
    assert typen.count("auftrag.laeuft") == 2
    assert typen.count("auftrag.fertig") == 2


@pytest.mark.asyncio
async def test_reconnect_liefert_verpasste_ereignisse_ohne_neue_auftraege(live_server):
    scenario("fake-a-model", text="Antwort A mit Inhalt.")
    scenario("fake-b-model", text="Antwort B mit Inhalt.")

    async with httpx.AsyncClient(base_url=live_server, timeout=10) as c:
        sid = (await c.post("/api/sessions", json={"title": "Reconnect"})).json()["id"]
        await c.post(
            f"/api/sessions/{sid}/sparks",
            json={"prompt": "Was war?", "client_request_id": "req-replay"},
        )
    bundle = await wait_done(live_server, sid)
    aufrufe_vorher = FakeProvider.call_count("fake-a-model")
    assert aufrufe_vorher == 1

    alle = await read_events(live_server, sid, expect=6)
    assert [e["id"] for e in alle] == sorted(e["id"] for e in alle)
    assert "sitzung.angelegt" in [e["type"] for e in alle]

    # Wiederverbindung ab der Mitte: nur der Rest kommt nach.
    cursor = alle[2]["id"]
    rest = await read_events(live_server, sid, last_event_id=cursor, expect=2)
    assert all(e["id"] > cursor for e in rest)

    # Weder Nachlieferung noch Neuladen lösen neue Modellaufrufe aus.
    async with httpx.AsyncClient(base_url=live_server, timeout=10) as c:
        neu = (await c.get(f"/api/sessions/{sid}")).json()
    assert FakeProvider.call_count("fake-a-model") == aufrufe_vorher
    assert FakeProvider.call_count("fake-b-model") == 1
    assert neu["sparks"][0]["summary"] is not None
    assert neu["sparks"][0]["jobs"] == bundle["sparks"][0]["jobs"]


@pytest.mark.asyncio
async def test_last_event_id_header_wird_beachtet(live_server):
    scenario("fake-a-model", text="Antwort A mit Inhalt.")
    scenario("fake-b-model", text="Antwort B mit Inhalt.")

    async with httpx.AsyncClient(base_url=live_server, timeout=10) as c:
        sid = (await c.post("/api/sessions", json={"title": "Header"})).json()["id"]
        await c.post(
            f"/api/sessions/{sid}/sparks",
            json={"prompt": "Kopfzeile prüfen.", "client_request_id": "req-header"},
        )
    await wait_done(live_server, sid)

    alle = await read_events(live_server, sid, expect=3)
    cursor = alle[0]["id"]
    rest = await read_events(live_server, sid, header=str(cursor), expect=1)
    assert rest[0]["id"] > cursor


@pytest.mark.asyncio
async def test_neustart_erhaelt_sitzung_und_kennzeichnet_unterbrechung(tmp_path):
    settings = make_settings(tmp_path)
    scenario("fake-a-model", text="Kommt nie an.", delay_s=30)
    scenario("fake-b-model", text="Kommt auch nicht an.", delay_s=30)

    app1 = create_app(settings)
    async with app1.router.lifespan_context(app1):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app1), base_url="http://testserver"
        ) as c:
            session = (await c.post("/api/sessions", json={"title": "Absturzsitzung"})).json()
            await c.post(
                f"/api/sessions/{session['id']}/sparks",
                json={"prompt": "Frage vor dem Absturz.", "client_request_id": "req-crash"},
            )
            await asyncio.sleep(0.2)
    # Hier endet der Serverprozess mitten im Lauf.

    settings2 = make_settings(tmp_path)
    app2 = create_app(settings2)
    async with app2.router.lifespan_context(app2):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app2), base_url="http://testserver"
        ) as c:
            bundle = (await c.get(f"/api/sessions/{session['id']}")).json()

    assert bundle["session"]["title"] == "Absturzsitzung"
    assert bundle["sparks"][0]["spark"]["prompt"] == "Frage vor dem Absturz."
    jobs = bundle["sparks"][0]["jobs"]
    assert len(jobs) == 2
    assert all(j["status"] == "interrupted" for j in jobs)
    assert all("Serverneustart" in (j["error"] or "") for j in jobs)
    # Die Unterbrechung ist auch im Ereignisstrom vermerkt.
    typen = [e["type"] for e in bundle["sparks"][0].get("markers", [])]
    assert isinstance(typen, list)


@pytest.mark.asyncio
async def test_neustart_meldet_unterbrechung_im_ereignisstrom(tmp_path):
    settings = make_settings(tmp_path)
    scenario("fake-a-model", text="Hängt.", delay_s=30)
    scenario("fake-b-model", text="Hängt auch.", delay_s=30)

    app1 = create_app(settings)
    async with app1.router.lifespan_context(app1):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app1), base_url="http://testserver"
        ) as c:
            session = (await c.post("/api/sessions", json={"title": "Strom"})).json()
            await c.post(
                f"/api/sessions/{session['id']}/sparks",
                json={"prompt": "Frage vor dem Neustart.", "client_request_id": "req-evt"},
            )
            await asyncio.sleep(0.2)

    app2 = create_app(make_settings(tmp_path))
    async with app2.router.lifespan_context(app2):
        store = app2.state.store
        events = await store.events_since(session["id"], 0)

    typen = [e["type"] for e in events]
    assert typen.count("auftrag.unterbrochen") == 2
    assert "einschaetzung.fertig" in typen
