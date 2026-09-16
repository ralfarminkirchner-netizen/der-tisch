"""Gesprächskontext, Bezugnahmen, Kontext-Schnappschüsse und Beziehungen.

Deckt die Abnahmefälle C (Bezug nachweislich im Kontext), D (ein neuer
Einwurf ändert keinen laufenden Schnappschuss) und J (maschinelle Beziehung
erscheint nicht als bestätigte Tatsache) des Bauauftrags ab.
"""

from __future__ import annotations

import asyncio

import pytest

from app.context import build, collect
from app.providers.fake import FakeProvider
from conftest import scenario, wait_for_jobs


def funke(fid: str, seq: int, prompt: str, refs=None) -> dict:
    return {"id": fid, "seq": seq, "prompt": prompt, "refs": refs or []}


def auftrag(jid: str, spark_id: str, label: str, text: str) -> dict:
    return {"id": jid, "spark_id": spark_id, "label": label, "model": "m-1", "text": text}


# ------------------------------------------------------------- Kontextaufbau

def test_gewaehlter_bezug_steht_wortwoertlich_im_kontext():
    sparks = [funke("f1", 1, "Was hältst du von täglichen Sicherungen?"),
              funke("f2", 2, "Prüfe das bitte.", refs=["j1"])]
    jobs = [auftrag("j1", "f1", "DeepSeek", "Tägliche Sicherungen sind unverzichtbar.")]
    text, eintraege, gekuerzt = build(spark=sparks[1], sparks=sparks, jobs=jobs)

    assert "Tägliche Sicherungen sind unverzichtbar." in text
    assert "Prüfe das bitte." in text
    assert gekuerzt is False
    gewaehlt = [e for e in eintraege if e.reason == "ausdrücklich gewählt"]
    assert [e.id for e in gewaehlt] == ["j1"]


def test_fremde_modellbeitraege_sind_als_zitat_gekennzeichnet():
    sparks = [funke("f1", 1, "Erste Frage."), funke("f2", 2, "Zweite.", refs=["j1"])]
    jobs = [auftrag("j1", "f1", "Mistral", "Eine fremde Modellantwort.")]
    text, _, _ = build(spark=sparks[1], sparks=sparks, jobs=jobs)

    assert "keine Anweisung" in text
    assert "keine Aussage des Menschen" in text
    assert "AUFTRAG:" in text
    # Der Auftrag des Menschen steht nach dem Zitatblock.
    assert text.index("Eine fremde Modellantwort.") < text.index("AUFTRAG:")


def test_gewaehlte_bezuege_verschwinden_auch_bei_engem_platz_nicht():
    langer_verlauf = "A" * 500
    sparks = [funke("f1", 1, langer_verlauf), funke("f2", 2, "Jetzt das.", refs=["j1"])]
    jobs = [auftrag("j1", "f1", "Google Gemini", "Der ausdrücklich gewählte Bezug.")]
    text, eintraege, gekuerzt = build(
        spark=sparks[1], sparks=sparks, jobs=jobs, budget=60
    )

    assert "Der ausdrücklich gewählte Bezug." in text
    assert gekuerzt is True
    assert "gekürzt" in text
    assert langer_verlauf not in text
    assert [e.id for e in eintraege] == ["j1"]


def test_ohne_bezug_und_ohne_verlauf_bleibt_der_funke_pur():
    sparks = [funke("f1", 1, "Ganz allein.")]
    text, eintraege, _ = build(spark=sparks[0], sparks=sparks, jobs=[])
    assert text == "Ganz allein."
    assert eintraege == []


def test_leere_antworten_kommen_nicht_in_den_verlauf():
    sparks = [funke("f1", 1, "Frage."), funke("f2", 2, "Weiter.")]
    jobs = [auftrag("j1", "f1", "Leer", ""), auftrag("j2", "f1", "Voll", "Inhalt.")]
    eintraege, _ = collect(spark=sparks[1], sparks=sparks, jobs=jobs)
    assert [e.id for e in eintraege] == ["f1", "j2"]


# ------------------------------------------------------- Zusammenspiel am Tisch

@pytest.mark.asyncio
async def test_weitergabe_an_ein_anderes_modell_traegt_den_bezug_mit(client, session_id):
    scenario("fake-a-model", text="Sicherungen täglich, verschlüsselt, geprobt.")
    scenario("fake-b-model", text="Ich sehe das anders.")
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Wie sichert man eine Datenbank?", "client_request_id": "req-1",
    })
    bundle = await wait_for_jobs(client, session_id)
    antwort_a = next(j for j in bundle["sparks"][0]["jobs"] if j["model_id"] == "fake-a")

    # Diese Antwort an das andere Modell zur Prüfung geben.
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Prüfe diese Aussage kritisch.",
        "client_request_id": "req-2",
        "refs": [antwort_a["id"]],
        "kind": "weitergabe",
        "model_ids": ["fake-b"],
    })
    bundle = await wait_for_jobs(client, session_id)
    zweiter = bundle["sparks"][1]
    geprueft = next(j for j in zweiter["jobs"] if j["model_id"] == "fake-b")

    # Abnahmefall C: der Bezugsbeitrag ist nachweislich im Kontext.
    kontext = (await client.get(f"/api/jobs/{geprueft['id']}/context")).json()
    assert "Sicherungen täglich, verschlüsselt, geprobt." in kontext["rendered"]
    assert "Prüfe diese Aussage kritisch." in kontext["rendered"]
    gewaehlt = [e for e in kontext["entries"] if e["reason"] == "ausdrücklich gewählt"]
    assert [e["id"] for e in gewaehlt] == [antwort_a["id"]]

    # Das nicht angefragte Modell steht als solches da — nicht als Fehler.
    nicht_gefragt = next(j for j in zweiter["jobs"] if j["model_id"] == "fake-a")
    assert nicht_gefragt["status"] == "not_requested"
    assert nicht_gefragt["error"] is None
    # Und es wurde tatsächlich nicht angerufen: ein Aufruf aus Funke 1, sonst keiner.
    assert FakeProvider.call_count("fake-a-model") == 1


@pytest.mark.asyncio
async def test_spaeterer_einwurf_aendert_keinen_laufenden_schnappschuss(client, session_id):
    scenario("fake-a-model", text="Langsame Antwort.", delay_s=0.9)
    scenario("fake-b-model", text="Schnelle Antwort.", delay_s=0.0)

    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Erster Gedanke.", "client_request_id": "req-lang",
    })
    await asyncio.sleep(0.25)  # der langsame Auftrag läuft bereits
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "GANZ NEUER EINWURF, der nicht mehr dazugehört.",
        "client_request_id": "req-neu",
    })
    bundle = await wait_for_jobs(client, session_id, timeout=15)

    langsam = next(j for j in bundle["sparks"][0]["jobs"] if j["model_id"] == "fake-a")
    kontext = (await client.get(f"/api/jobs/{langsam['id']}/context")).json()

    # Abnahmefall D.
    assert "GANZ NEUER EINWURF" not in kontext["rendered"]
    assert "Erster Gedanke." in kontext["rendered"]


@pytest.mark.asyncio
async def test_der_verlauf_landet_im_kontext_des_naechsten_funkens(client, session_id):
    scenario("fake-a-model", text="Antwort auf die erste Frage.")
    scenario("fake-b-model", text="Zweite Stimme zur ersten Frage.")
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Erste Frage.", "client_request_id": "req-1",
    })
    await wait_for_jobs(client, session_id)

    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Und was folgt daraus?", "client_request_id": "req-2",
    })
    bundle = await wait_for_jobs(client, session_id)
    job = bundle["sparks"][1]["jobs"][0]
    kontext = (await client.get(f"/api/jobs/{job['id']}/context")).json()

    assert "Erste Frage." in kontext["rendered"]
    assert "Antwort auf die erste Frage." in kontext["rendered"]
    assert kontext["rule"]


@pytest.mark.asyncio
async def test_ohne_schnappschuss_gibt_es_eine_erklaerung_statt_einer_luege(client, session_id):
    antwort = await client.get("/api/jobs/auf_gibtsnicht/context")
    assert antwort.status_code == 404


# ------------------------------------------------------------------ Beziehungen

@pytest.mark.asyncio
async def test_menschlicher_bezug_gilt_als_bestaetigt(client, session_id):
    scenario("fake-a-model", text="Erste Antwort mit Inhalt.")
    scenario("fake-b-model", text="Zweite Antwort mit Inhalt.")
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Los.", "client_request_id": "req-1",
    })
    bundle = await wait_for_jobs(client, session_id)
    ziel = bundle["sparks"][0]["jobs"][0]["id"]

    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Dazu eine Rückfrage.", "client_request_id": "req-2",
        "refs": [ziel], "kind": "antwort",
    })
    bundle = await wait_for_jobs(client, session_id)

    menschlich = [r for r in bundle["relations"] if r["origin"] == "mensch"]
    assert len(menschlich) == 1
    assert menschlich[0]["to_id"] == ziel
    assert menschlich[0]["type"] == "antwortet_auf"
    assert menschlich[0]["status"] == "bestaetigt"


@pytest.mark.asyncio
async def test_maschinelle_beziehung_ist_ein_vorschlag_und_bleibt_es(client, session_id):
    scenario("fake-a-model",
             text="Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll.")
    scenario("fake-b-model",
             text="Eine tägliche Sicherung der Datenbank ist nicht notwendig und nicht sinnvoll.")
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Täglich sichern?", "client_request_id": "req-1",
    })
    bundle = await wait_for_jobs(client, session_id)

    # Abnahmefall J.
    maschinell = [r for r in bundle["relations"] if r["origin"] == "maschine"]
    assert maschinell, "die Auswertung sollte einen Bezug gefunden haben"
    assert all(r["status"] == "vorschlag" for r in maschinell)
    assert any(r["type"] == "widerspricht" for r in maschinell)

    # Erst der Mensch macht daraus einen Befund.
    vorschlag = maschinell[0]
    antwort = await client.post(
        f"/api/sessions/{session_id}/relations/{vorschlag['id']}",
        json={"status": "bestaetigt"},
    )
    assert antwort.status_code == 200
    danach = {r["id"]: r for r in antwort.json()["relations"]}
    assert danach[vorschlag["id"]]["status"] == "bestaetigt"

    # Und er kann ihn auch verwerfen.
    await client.post(
        f"/api/sessions/{session_id}/relations/{vorschlag['id']}",
        json={"status": "abgelehnt"},
    )
    bundle = (await client.get(f"/api/sessions/{session_id}")).json()
    assert {r["id"]: r for r in bundle["relations"]}[vorschlag["id"]]["status"] == "abgelehnt"


@pytest.mark.asyncio
async def test_unbekannte_beziehung_wird_abgewiesen(client, session_id):
    antwort = await client.post(
        f"/api/sessions/{session_id}/relations/bez_gibtsnicht", json={"status": "bestaetigt"}
    )
    assert antwort.status_code == 404


# ---------------------------------------------------------------- JSON-Export

@pytest.mark.asyncio
async def test_json_export_traegt_wortlaut_bezuege_und_zustaende(client, session_id):
    scenario("fake-a-model", text="Originalwortlaut der ersten Stimme.")
    scenario("fake-b-model", error="Anbieter nicht erreichbar.")
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Wortlaut prüfen.", "client_request_id": "req-json",
    })
    await wait_for_jobs(client, session_id)

    antwort = await client.get(f"/api/sessions/{session_id}/report.json")
    assert antwort.status_code == 200
    daten = antwort.json()
    text = antwort.text

    assert "Originalwortlaut der ersten Stimme." in text
    assert "Wortlaut prüfen." in text
    assert "Anbieter nicht erreichbar." in text
    zustaende = {j["status"] for e in daten["sparks"] for j in e["jobs"]}
    assert {"done", "error"} <= zustaende
    assert "relations" in daten
    assert daten["context_snapshots"], "die Schnappschüsse gehören in den Export"
    assert "vorschlag" in text or not [r for r in daten["relations"] if r["origin"] == "maschine"]


@pytest.mark.asyncio
async def test_bundle_meldet_ob_noch_etwas_laeuft(client, session_id):
    scenario("fake-a-model", text="Kommt gleich.", delay_s=0.8)
    scenario("fake-b-model", text="Auch gleich.", delay_s=0.8)
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Läuft noch?", "client_request_id": "req-pending",
    })
    await asyncio.sleep(0.2)
    laufend = (await client.get(f"/api/sessions/{session_id}")).json()
    assert laufend["pending"] is True

    await wait_for_jobs(client, session_id, timeout=15)
    fertig = (await client.get(f"/api/sessions/{session_id}")).json()
    assert fertig["pending"] is False


def test_bezuege_kommen_immer_als_liste_zurueck(tmp_path):
    """Regression: als JSON-Text durchgereichte Bezüge wurden zeichenweise gelesen."""
    import asyncio as _asyncio

    from app.db import Store

    async def lauf():
        store = Store(tmp_path / "refs.sqlite3")
        await store.connect()
        session = await store.create_session("T")
        spark, neu = await store.insert_spark(
            session["id"], "Frage.", "req-1", kind="antwort", refs=["auf_abc"]
        )
        gelesen = await store.get_spark(spark["id"])
        await store.close()
        return spark, gelesen

    frisch, gelesen = _asyncio.run(lauf())
    assert frisch["refs"] == ["auf_abc"]
    assert gelesen["refs"] == ["auf_abc"]
    assert gelesen["kind"] == "antwort"


@pytest.mark.asyncio
async def test_gleichzeitiges_ende_erzeugt_keine_doppelten_bezuege(client, session_id):
    """Regression: enden zwei Aufträge gleichzeitig, lief der Abschlusslauf doppelt."""
    scenario("fake-a-model",
             text="Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll.",
             delay_s=0.15)
    scenario("fake-b-model",
             text="Eine tägliche Sicherung der Datenbank ist nicht notwendig und nicht sinnvoll.",
             delay_s=0.15)
    await client.post(f"/api/sessions/{session_id}/sparks", json={
        "prompt": "Gleichzeitig fertig?", "client_request_id": "req-race",
    })
    bundle = await wait_for_jobs(client, session_id)
    await asyncio.sleep(0.3)
    bundle = (await client.get(f"/api/sessions/{session_id}")).json()

    maschinell = [r for r in bundle["relations"] if r["origin"] == "maschine"]
    schluessel = [(r["from_id"], r["to_id"], r["type"]) for r in maschinell]
    assert len(schluessel) == len(set(schluessel)), f"doppelte Bezüge: {schluessel}"
