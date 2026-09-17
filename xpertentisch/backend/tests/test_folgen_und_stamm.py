"""Stammformen, Konsequenzkarte und die Linsen im Bericht.

Geprüft wird nicht nur, dass etwas herauskommt, sondern dass es auch über
leeren, einstimmigen und widersprüchlichen Daten das Richtige ist. Eine
Zusicherung, die über leeren Daten ebenso besteht, sichert nichts zu.
"""

from __future__ import annotations

import pytest

from app.analysis import (
    KIND_CONTRADICTION,
    analyse,
    folgen,
    szenario_block,
    themen_aus_markern,
)
from app.db import JOB_DONE
from app.reports import _verlauf_svg, _zweige, render_html, render_markdown
from app.stemmer import stamm
from conftest import scenario, wait_for_jobs


def make_job(job_id: str, text: str, label: str | None = None, status: str = JOB_DONE) -> dict:
    return {
        "id": job_id,
        "model_id": job_id,
        "label": label or job_id.upper(),
        "provider": "fake",
        "model": "m",
        "status": status,
        "text": text,
        "error": None,
        "partial": 0,
        "latency_ms": 10,
    }


# ------------------------------------------------------------------ Stamm

def test_beugung_faellt_zusammen_bedeutung_nicht():
    # Der Zweck der Übung: Ein- und Mehrzahl sind ein Begriff.
    assert stamm("Sicherung") == stamm("Sicherungen")
    assert stamm("Ausfall") == stamm("Ausfälle")
    assert stamm("Datenbank") == stamm("Datenbanken")
    assert stamm("Ergebnis") == stamm("Ergebnisse")
    # Ableitungen bleiben getrennt: „Sicherung" ist nicht „sicher".
    assert stamm("Sicherung") != stamm("sicher")
    # Verschiedene Wörter bleiben verschieden.
    assert stamm("Sicherung") != stamm("Wiederherstellung")


def test_stamm_ist_idempotent_und_greift_kurze_woerter_nicht_an():
    for wort in ["Sicherungen", "Ausfälle", "Servern", "Teams", "das", "wer", "ab"]:
        assert stamm(stamm(wort)) == stamm(wort), wort
    # Zu kurz zum Kürzen: es bleibt etwas übrig, das noch ein Wort ist.
    assert stamm("ab") == "ab"
    assert stamm("") == ""


def test_stemmer_hinterlaesst_keine_leeren_staemme():
    for wort in ["ernen", "eier", "esse", "aaa", "Ehe", "Reise", "Sees"]:
        assert len(stamm(wort)) >= 2, wort


def test_themen_tragen_den_wortlaut_nicht_den_stamm():
    """Der Stamm ist Schlüssel, nie Beschriftung."""
    jobs = [
        make_job("j1", "Regelmäßige Sicherungen der Datenbank schützen vor Verlust."),
        make_job("j2", "Regelmäßige Sicherung der Datenbank schützt vor Verlust."),
    ]
    markers, _ = analyse("s1", "f1", jobs)
    begriffe = {b for m in markers for b in m["topics"]}
    assert begriffe, "ohne Begriffe wäre die Themen-Linse leer"
    # Ein- und Mehrzahl landen im selben Begriff …
    assert not ({"sicherung", "sicherungen"} <= begriffe)
    # … und jeder gezeigte Begriff steht so auch in einer der Antworten.
    volltext = " ".join(j["text"] for j in jobs).lower()
    for begriff in begriffe:
        assert begriff in volltext, begriff


# ------------------------------------------------------------------ Folgen

def test_folgen_zaehlen_stimmen_und_erfinden_nichts():
    jobs = [
        make_job("j1", "Die Wiederherstellung dauert dann deutlich länger als geplant.",
                 label="Alto"),
        make_job("j2", "Die Wiederherstellung dauert dann deutlich länger als erwartet.",
                 label="Basso"),
        make_job("j3", "Der Betrieb bleibt in diesem Fall vollständig unberührt davon.",
                 label="Cantus"),
    ]
    karte = folgen(jobs)
    assert karte["folgen"], "drei auswertbare Antworten müssen Folgen ergeben"
    for eintrag in karte["folgen"]:
        # Jede Folge steht wörtlich in einer Antwort — nichts wird umformuliert.
        assert any(eintrag["text"] in j["text"] for j in jobs), eintrag["text"]
        assert eintrag["von"] == 3
        assert 1 <= eintrag["anzahl"] <= 3
        assert eintrag["anzahl"] == len({n["job_id"] for n in eintrag["nennungen"]})

    geteilt = [e for e in karte["folgen"] if e["anzahl"] >= 2]
    assert geteilt, "die gleichlautende Folge zweier Stimmen muss zusammenfallen"
    assert {n["label"] for n in geteilt[0]["nennungen"]} == {"Alto", "Basso"}


def test_folgen_belegen_sich_selbst_im_originaltext():
    jobs = [
        make_job("j1", "Die Kosten für den Betrieb steigen im ersten Jahr spürbar an. "
                       "Zusätzlich wächst der Aufwand für die Prüfung der Sicherungen."),
        make_job("j2", "Die Kosten für den Betrieb steigen im ersten Jahr merklich an."),
    ]
    texte = {j["id"]: j["text"] for j in jobs}
    karte = folgen(jobs)
    assert karte["folgen"]
    for eintrag in karte["folgen"]:
        for nennung in eintrag["nennungen"]:
            text = texte[nennung["job_id"]]
            assert text[nennung["start_offset"]:nennung["end_offset"]] == nennung["quote"]


def test_folgen_stellen_gegensaetze_gegenueber_aber_nur_bei_gleichem_thema():
    jobs = [
        make_job("j1", "Der Aufwand für den Betrieb steigt dadurch deutlich an."),
        make_job("j2", "Der Aufwand für den Betrieb sinkt dadurch deutlich."),
        make_job("j3", "Ein Zwiebelkuchen gelingt besonders gut mit frischem Federweißer."),
    ]
    karte = folgen(jobs)
    mit_gegensatz = [e for e in karte["folgen"] if e["gegensatz"]]
    assert mit_gegensatz, "steigt/sinkt beim selben Thema ist ein Gegensatz"
    # Der Gegensatz zeigt wechselseitig aufeinander.
    for eintrag in mit_gegensatz:
        for ziel in eintrag["gegensatz"]:
            partner = next(e for e in karte["folgen"] if e["id"] == ziel)
            assert eintrag["id"] in partner["gegensatz"]
    # Das themenfremde Gebäck steht für sich und widerspricht niemandem.
    kuchen = next(e for e in karte["folgen"] if "Zwiebelkuchen" in e["text"])
    assert kuchen["gegensatz"] == []
    assert kuchen["anzahl"] == 1


def test_folgen_bei_leeren_daten_und_bei_einer_einzigen_stimme():
    # Leer: keine Folgen, keine Stimmen, und vor allem keine erfundene Karte.
    leer = folgen([])
    assert leer["folgen"] == []
    assert leer["stimmen"] == []
    assert leer["uebergangen"] == 0

    # Nur Fehler und leerer Text: ebenfalls nichts — „von 0 Stimmen" darf nirgends stehen.
    nur_fehler = folgen([
        {**make_job("j1", ""), "status": "error", "error": "Ausfall"},
        make_job("j2", ""),
    ])
    assert nur_fehler["folgen"] == []

    # Eine einzige Stimme: ihre Folgen sind Folgen, gezählt als 1 von 1.
    eine = folgen([make_job("j1", "Der Betrieb steht dann für mehrere Stunden still.")])
    assert len(eine["folgen"]) == 1
    assert eine["folgen"][0]["anzahl"] == 1
    assert eine["folgen"][0]["von"] == 1
    assert eine["folgen"][0]["gegensatz"] == []


def test_folgen_sind_wiederholbar():
    jobs = [
        make_job("j1", "Die Wiederherstellung dauert dann deutlich länger als geplant."),
        make_job("j2", "Die Wiederherstellung dauert dann deutlich länger als erwartet."),
        make_job("j3", "Die Kosten für den Betrieb steigen im ersten Jahr spürbar an."),
    ]
    erste = folgen(jobs)
    zweite = folgen(jobs)
    assert [e["text"] for e in erste["folgen"]] == [e["text"] for e in zweite["folgen"]]
    assert [e["anzahl"] for e in erste["folgen"]] == [e["anzahl"] for e in zweite["folgen"]]


def test_szenario_block_nennt_die_ausgangsaussage():
    spark = {"id": "f2", "seq": 2, "prompt": "Welche Folgen hätte das?", "kind": "szenario",
             "refs": ["j-quelle"]}
    quelle = make_job("j-quelle", "Eine tägliche Sicherung ist notwendig.", label="Alto")
    block = szenario_block(spark, [make_job("j1", "Der Aufwand steigt dadurch deutlich an.")],
                           ausgang=quelle)
    assert block["ausgang"]["label"] == "Alto"
    assert block["ausgang"]["auszug"] == "Eine tägliche Sicherung ist notwendig."
    assert block["ausgang"]["gekuerzt"] is False
    assert block["seq"] == 2

    # Ohne gesetzten Bezug wird keiner erfunden.
    ohne = szenario_block(spark, [make_job("j1", "Der Aufwand steigt dadurch deutlich an.")])
    assert ohne["ausgang"] is None


def test_langer_ausgang_wird_gekuerzt_und_sagt_es():
    spark = {"id": "f2", "seq": 2, "prompt": "Folgen?", "kind": "szenario", "refs": []}
    lang = make_job("j-quelle", "Satz. " * 200)
    block = szenario_block(spark, [], ausgang=lang)
    assert block["ausgang"]["gekuerzt"] is True
    assert block["ausgang"]["auszug"].endswith("…")
    assert len(block["ausgang"]["auszug"]) <= 240


# ------------------------------------------------------ Themen und Verlauf

def test_themen_aus_markern_zaehlt_stimmen_nicht_fundstellen():
    markers = [
        {"job_id": "j1", "kind": "uebereinstimmung", "topics": ["sicherung"]},
        {"job_id": "j1", "kind": "uebereinstimmung", "topics": ["sicherung"]},
        {"job_id": "j2", "kind": "uebereinstimmung", "topics": ["sicherung"]},
        {"job_id": "j1", "kind": "widerspruch", "topics": ["rhythmus"]},
    ]
    zeilen = {z["begriff"]: z for z in themen_aus_markern(markers)}
    assert zeilen["sicherung"]["einig"] == 2  # zwei Stimmen, drei Fundstellen
    assert zeilen["sicherung"]["stimmen"] == 2
    assert zeilen["rhythmus"]["gegen"] == 1
    assert themen_aus_markern([]) == []


def test_verlaufsbaum_im_bericht_haengt_beitraege_aneinander():
    bundle = {
        "sparks": [
            {"spark": {"id": "f1", "seq": 1, "kind": "funke", "prompt": "Erste Frage",
                       "refs": []},
             "jobs": [make_job("f1-j", "x")], "markers": []},
            {"spark": {"id": "f2", "seq": 2, "kind": "szenario", "prompt": "Folgen?",
                       "refs": ["f1-j"]},
             "jobs": [make_job("f2-j", "y")], "markers": []},
        ]
    }
    zweige = {z["spark"]["id"]: z for z in _zweige(bundle)}
    assert zweige["f2"]["eltern_id"] == "f1"
    assert zweige["f2"]["tiefe"] == 1
    assert zweige["f1"]["tiefe"] == 0

    svg = _verlauf_svg(bundle)
    assert svg.count('class="zweig"') == 2
    assert 'class="ast"' in svg
    # Kein Namensraum im Quelltext: sonst stünde eine Adresse im Bericht.
    assert "http" not in svg


def test_verlaufsbaum_haengt_sich_an_einem_ring_nicht_auf():
    bundle = {
        "sparks": [
            {"spark": {"id": "f1", "seq": 1, "kind": "funke", "prompt": "A",
                       "refs": ["f2-j"]},
             "jobs": [make_job("f1-j", "x")], "markers": []},
            {"spark": {"id": "f2", "seq": 2, "kind": "antwort", "prompt": "B",
                       "refs": ["f1-j"]},
             "jobs": [make_job("f2-j", "y")], "markers": []},
        ]
    }
    zweige = _zweige(bundle)
    assert len(zweige) == 2
    assert all(isinstance(z["tiefe"], int) for z in zweige)


def test_leerer_verlauf_zeichnet_nichts_statt_eines_leeren_rahmens():
    assert _verlauf_svg({"sparks": []}) == ""


# ------------------------------------------------------------ Im Zusammenspiel

@pytest.mark.asyncio
async def test_szenario_kommt_als_konsequenzkarte_in_der_sitzung_an(client, session_id):
    scenario("fake-a-model",
             text="Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll.")
    scenario("fake-b-model",
             text="Eine tägliche Sicherung der Datenbank ist notwendig und geboten.")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Täglich sichern?", "client_request_id": "req-1"},
    )
    bundle = await wait_for_jobs(client, session_id)
    quelle = bundle["sparks"][0]["jobs"][0]["id"]

    scenario("fake-a-model", text="Der Aufwand für den Betrieb steigt dadurch deutlich an.")
    scenario("fake-b-model", text="Der Aufwand für den Betrieb steigt dadurch merklich an.")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Welche Folgen hätte das?", "client_request_id": "req-2",
              "kind": "szenario", "refs": [quelle]},
    )
    bundle = await wait_for_jobs(client, session_id)

    assert len(bundle["szenarien"]) == 1
    karte = bundle["szenarien"][0]
    assert karte["ausgang"]["job_id"] == quelle
    assert karte["folgen"]
    geteilt = [f for f in karte["folgen"] if f["anzahl"] == 2]
    assert geteilt, "zwei Stimmen nennen dieselbe Folge"
    assert geteilt[0]["von"] == 2

    # Der Bericht kennt die Konsequenzkarte und beschriftet sie als Häufigkeit.
    html = (await client.get(f"/api/sessions/{session_id}/report.html")).text
    assert "von 2 von 2 Stimmen genannt" in html
    assert "keine Wahrscheinlichkeit" in html
    markdown = (await client.get(f"/api/sessions/{session_id}/report.md")).text
    assert "von 2 von 2 Stimmen genannt" in markdown


@pytest.mark.asyncio
async def test_bericht_traegt_themen_und_verlauf(client, session_id):
    scenario("fake-a-model",
             text="Regelmäßige Sicherungen der Datenbank schützen wirksam vor Verlust.")
    scenario("fake-b-model",
             text="Regelmäßige Sicherung der Datenbank schützt wirksam vor Verlust.")
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Sichern?", "client_request_id": "req-t"},
    )
    await wait_for_jobs(client, session_id)

    html = (await client.get(f"/api/sessions/{session_id}/report.html")).text
    assert "<h3>Themen</h3>" in html
    assert "<h2>Verlauf</h2>" in html
    assert 'class="verlauf"' in html
    # Der Bericht bleibt offline und ohne Skript, auch mit dem neuen SVG.
    assert "<script" not in html.lower()
    assert "http://" not in html and "https://" not in html

    markdown = (await client.get(f"/api/sessions/{session_id}/report.md")).text
    assert "**Themen**" in markdown
    assert "## Verlauf" in markdown


def test_bericht_ohne_szenario_zeigt_keinen_leeren_folgenabschnitt():
    bundle = {
        "session": {"id": "s1", "title": "T", "status": "offen", "created_at": 1.0,
                    "closed_at": None, "closing_note": None},
        "sparks": [], "relations": [], "szenarien": [], "pending": False,
        "exported_at": 2.0,
    }
    html = render_html(bundle)
    assert "<h2>Folgen</h2>" not in html
    assert "<h2>Verlauf</h2>" not in html
    markdown = render_markdown(bundle)
    assert "## Folgen" not in markdown
    assert "## Verlauf" not in markdown


def test_widerspruch_bleibt_widerspruch_auch_mit_stammformen():
    """Die Stammformen dürfen die Kernregel nicht aushebeln."""
    jobs = [
        make_job("j1", "Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll."),
        make_job("j2", "Eine tägliche Sicherung der Datenbank ist nicht notwendig "
                       "und nicht sinnvoll."),
    ]
    _, summary = analyse("s1", "f1", jobs)
    assert summary["counts"][KIND_CONTRADICTION] >= 1
