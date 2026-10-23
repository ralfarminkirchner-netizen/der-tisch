"""Berichtsexporte: unverändert, offline, druckbar, gegen Injection geschützt."""

from __future__ import annotations

import re

import pytest

from conftest import scenario, wait_for_jobs

BOESE_ANTWORT = (
    '<script>alert("xss")</script> Eine Antwort mit <b>Markup</b> & Sonderzeichen, '
    "dazu ein Zitat: \"gefährlich\" und ein Bild <img src=x onerror=alert(1)>."
)
BOESE_FRAGE = "Was passiert bei </textarea><script>alert('frage')</script>?"


@pytest.mark.asyncio
async def test_html_export_ist_injektionssicher_und_offline(client, session_id):
    scenario("fake-a-model", text=BOESE_ANTWORT)
    scenario("fake-b-model", text="Eine harmlose zweite Antwort mit Inhalt und Aussage.")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": BOESE_FRAGE, "client_request_id": "req-html"},
    )
    await wait_for_jobs(client, session_id)
    await client.post(f"/api/sessions/{session_id}/close", json={"note": "<i>fertig</i>"})

    resp = await client.get(f"/api/sessions/{session_id}/report.html")
    assert resp.status_code == 200
    html = resp.text

    # Injection: keine ausführbaren Skripte, alles escaped.
    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html
    assert "&lt;i&gt;fertig&lt;/i&gt;" in html

    # Offline: keine externen Ressourcen, kein Netzwerkzugriff nötig.
    for muster in ("http://", "https://", "//cdn", "@import"):
        assert muster not in html.replace("http-equiv", "")
    assert not re.search(r"<(link|iframe|object|embed)\b", html, re.I)
    assert "<style>" in html

    # Druckbar
    assert "@media print" in html

    # Vollständigkeit
    assert "XPERTENTiSCH" in html
    assert "Modell A" in html and "Modell B" in html


@pytest.mark.asyncio
async def test_originalantwort_bleibt_in_ansicht_und_export_unveraendert(client, session_id):
    original = (
        "Zeile eins mit Aussage.\n\n  Zeile zwei mit Einrückung — und Gedankenstrich.\n"
        "Drittens: 100 % unverändert, inklusive \"Anführungszeichen\"."
    )
    scenario("fake-a-model", text=original)
    scenario("fake-b-model", text="Zweite Antwort, damit ein Vergleich entsteht.")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Unverändert?", "client_request_id": "req-orig"},
    )
    bundle = await wait_for_jobs(client, session_id)

    job = next(j for j in bundle["sparks"][0]["jobs"] if j["model_id"] == "fake-a")
    assert job["text"] == original  # Ansicht (API-Zustand)

    html = (await client.get(f"/api/sessions/{session_id}/report.html")).text
    import html as html_mod

    assert html_mod.escape(original, quote=True) in html  # Export

    md = (await client.get(f"/api/sessions/{session_id}/report.md")).text
    assert original in md


@pytest.mark.asyncio
async def test_markdown_export(client, session_id):
    scenario("fake-a-model", text="Antwort A mit ausreichendem Inhalt für einen Satz.")
    scenario("fake-b-model", text="Antwort B mit ausreichendem Inhalt für einen Satz.")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Zwei Berichte?", "client_request_id": "req-md"},
    )
    await wait_for_jobs(client, session_id)

    resp = await client.get(f"/api/sessions/{session_id}/report.md")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    md = resp.text
    assert md.startswith("# XPERTENTiSCH")
    assert "| Modell | Provider |" in md
    assert "Zwei Berichte?" in md


@pytest.mark.asyncio
async def test_bericht_zeigt_fehler_und_unterbrechungen(client, session_id):
    scenario("fake-a-model", error="Provider antwortet nicht (504).")
    scenario("fake-b-model", text="Nur ein Modell hat geantwortet, das reicht für den Bericht.")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Was bei Ausfall?", "client_request_id": "req-err"},
    )
    await wait_for_jobs(client, session_id)

    html = (await client.get(f"/api/sessions/{session_id}/report.html")).text
    assert "Provider antwortet nicht (504)." in html
    assert "Fehler" in html


@pytest.mark.asyncio
async def test_bericht_zeigt_berliner_zeit(client, session_id):
    scenario("fake-a-model", text="Antwort A mit ausreichendem Inhalt.")
    scenario("fake-b-model", text="Antwort B mit ausreichendem Inhalt.")
    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Wann war das?", "client_request_id": "req-zeit"})
    await wait_for_jobs(client, session_id)

    md = (await client.get(f"/api/sessions/{session_id}/report.md")).text
    assert "UTC" not in md
    assert "CET" in md or "CEST" in md


@pytest.mark.asyncio
async def test_bericht_weist_nicht_angefragte_stimmen_aus(client, session_id):
    scenario("fake-a-model", text="Nur A wurde gefragt und antwortet ausführlich.")
    scenario("fake-b-model", text="B wird nie gefragt.")
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Nur einer bitte.", "client_request_id": "req-einer",
        "model_ids": ["fake-a"],
    })
    await wait_for_jobs(client, session_id)

    html = (await client.get(f"/api/sessions/{session_id}/report.html")).text
    assert "nicht angefragt" in html
    assert "Für diesen Funken nicht angefragt." in html
    md = (await client.get(f"/api/sessions/{session_id}/report.md")).text
    assert "nicht angefragt" in md


@pytest.mark.asyncio
async def test_bericht_nennt_bezuege_mit_herkunft_und_stand(client, session_id):
    scenario("fake-a-model",
             text="Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll.")
    scenario("fake-b-model",
             text="Eine tägliche Sicherung der Datenbank ist nicht notwendig und nicht sinnvoll.")
    await client.post(f"/api/sessions/{session_id}/sparks",
                      json={"prompt": "Täglich?", "client_request_id": "req-bez"})
    await wait_for_jobs(client, session_id)

    html = (await client.get(f"/api/sessions/{session_id}/report.html")).text
    assert "Bezüge" in html
    assert "maschineller Vorschlag" in html
    assert "unbestätigt" in html
    assert "keine von dir getroffenen" in html


def test_bericht_unterscheidet_acht_zustaende_ohne_neunten():
    from app.reports import _status_label

    erwartet = {
        "not_requested": "nicht angefragt",
        "queued": "wartet",
        "running": "denkt nach",
        "streaming": "schreibt",
        "done": "fertig",
        "error": "Fehler",
        "interrupted": "unterbrochen",
        "cancelled": "abgebrochen",
    }
    for status, label in erwartet.items():
        assert _status_label({"status": status}) == label
    assert _status_label({"status": "spark.created"}) == "spark.created"
