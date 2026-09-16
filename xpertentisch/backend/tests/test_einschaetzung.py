"""Automatische Einschätzungen: gültige Bezüge, echte Textbelege, keine Scheinwidersprüche."""

from __future__ import annotations

import pytest

from app.analysis import (
    KIND_AGREEMENT,
    KIND_CONTRADICTION,
    KIND_UNIQUE,
    analyse,
    split_sentences,
    validate_markers,
)
from app.db import JOB_DONE
from conftest import scenario, wait_for_jobs


def make_job(job_id: str, text: str, model_id: str | None = None) -> dict:
    return {
        "id": job_id,
        "model_id": model_id or job_id,
        "label": job_id.upper(),
        "provider": "fake",
        "model": "m",
        "status": JOB_DONE,
        "text": text,
        "error": None,
        "partial": 0,
        "latency_ms": 10,
    }


def test_satzpositionen_zeigen_exakt_auf_den_text():
    text = "  Erster Satz mit Inhalt.  Zweiter Satz mit weiterem Inhalt!  "
    for sentence in split_sentences("j1", text):
        assert text[sentence.start:sentence.end] == sentence.text


def test_uebereinstimmung_wird_erkannt():
    jobs = [
        make_job("j1", "Regelmäßige Sicherungskopien schützen die Daten wirksam vor Verlust."),
        make_job("j2", "Regelmäßige Sicherungskopien schützen Daten wirksam vor Verlust."),
    ]
    markers, summary = analyse("s1", "f1", jobs)
    kinds = {m["kind"] for m in markers}
    assert KIND_AGREEMENT in kinds
    assert KIND_CONTRADICTION not in kinds
    assert summary["pairs"][0]["agreements"] >= 1
    assert not validate_markers(markers, jobs)


def test_unterschiedlichkeit_ist_kein_widerspruch():
    jobs = [
        make_job("j1", "Regelmäßige Sicherungskopien schützen die Daten vor Verlust."),
        make_job("j2", "Ein Zwiebelkuchen gelingt besonders gut mit frischem Federweißer."),
    ]
    markers, summary = analyse("s1", "f1", jobs)
    assert summary["counts"][KIND_CONTRADICTION] == 0
    assert all(m["kind"] != KIND_CONTRADICTION for m in markers)
    # Themenfremde Aussagen gelten als einzigartig, nicht als widersprüchlich.
    assert summary["counts"][KIND_UNIQUE] >= 2
    assert not validate_markers(markers, jobs)


def test_andere_formulierung_gleicher_aussage_ist_kein_widerspruch():
    jobs = [
        make_job("j1", "Die Datenbank sollte täglich gesichert werden, am besten nachts."),
        make_job("j2", "Die Datenbank sollte täglich gesichert werden, vorzugsweise nachts."),
    ]
    markers, summary = analyse("s1", "f1", jobs)
    assert summary["counts"][KIND_CONTRADICTION] == 0


def test_widerspruch_nur_bei_gegensaetzlicher_polaritaet():
    jobs = [
        make_job("j1", "Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll."),
        make_job("j2", "Eine tägliche Sicherung der Datenbank ist nicht notwendig und nicht sinnvoll."),
    ]
    markers, summary = analyse("s1", "f1", jobs)
    assert summary["counts"][KIND_CONTRADICTION] >= 1
    widersprueche = [m for m in markers if m["kind"] == KIND_CONTRADICTION]
    # Ein Widerspruch wird immer auf beiden Seiten markiert und verweist aufeinander.
    assert {m["job_id"] for m in widersprueche} == {"j1", "j2"}
    assert all(m["related_job_id"] in ("j1", "j2") for m in widersprueche)
    assert not validate_markers(markers, jobs)


def test_gegensatzpaar_erzeugt_widerspruch():
    jobs = [
        make_job("j1", "Dieses Verfahren gilt bei korrekter Anwendung als sicher und erprobt."),
        make_job("j2", "Dieses Verfahren gilt bei korrekter Anwendung als unsicher und erprobt."),
    ]
    _, summary = analyse("s1", "f1", jobs)
    assert summary["counts"][KIND_CONTRADICTION] >= 1


def test_marker_belegen_sich_selbst():
    jobs = [
        make_job("j1", "Sicherungskopien schützen Daten zuverlässig. Zusätzlich hilft eine Prüfung."),
        make_job("j2", "Sicherungskopien schützen Daten zuverlässig. Ein Notfallplan gehört dazu."),
    ]
    markers, _ = analyse("s1", "f1", jobs)
    assert markers
    texte = {j["id"]: j["text"] for j in jobs}
    for m in markers:
        assert texte[m["job_id"]][m["start_offset"]:m["end_offset"]] == m["quote"]
        assert m["quote"] in texte[m["job_id"]]
    assert validate_markers(markers, jobs) == []


def test_validate_markers_erkennt_falsche_belege():
    jobs = [make_job("j1", "Ein Satz, der wirklich so im Text steht.")]
    markers = [
        {"id": "m1", "job_id": "j1", "related_job_id": None, "kind": KIND_UNIQUE,
         "start_offset": 0, "end_offset": 5, "quote": "FALSCH"},
        {"id": "m2", "job_id": "unbekannt", "related_job_id": None, "kind": KIND_UNIQUE,
         "start_offset": 0, "end_offset": 3, "quote": "Ein"},
    ]
    probleme = validate_markers(markers, jobs)
    assert len(probleme) == 2


def test_fehlerhafte_antworten_werden_nicht_ausgewertet():
    jobs = [
        make_job("j1", "Eine vollständige Antwort mit Inhalt und Aussage."),
        {**make_job("j2", ""), "status": "error", "error": "Ausfall"},
    ]
    markers, summary = analyse("s1", "f1", jobs)
    assert summary["analysed_jobs"] == ["j1"]
    assert all(m["job_id"] == "j1" for m in markers)
    assert len(summary["models"]) == 2


@pytest.mark.asyncio
async def test_einschaetzung_landet_in_der_sitzung(client, session_id):
    scenario(
        "fake-a-model",
        text="Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll.",
    )
    scenario(
        "fake-b-model",
        text="Eine tägliche Sicherung der Datenbank ist nicht notwendig und nicht sinnvoll.",
    )
    await client.post(
        f"/api/sessions/{session_id}/sparks",
        json={"prompt": "Täglich sichern?", "client_request_id": "req-analyse"},
    )
    bundle = await wait_for_jobs(client, session_id)
    entry = bundle["sparks"][0]

    assert entry["summary"]["counts"][KIND_CONTRADICTION] >= 1
    assert entry["markers"]
    assert validate_markers(entry["markers"], entry["jobs"]) == []
    # Graph-Kanten für die Oberfläche
    assert entry["summary"]["pairs"][0]["a_model_id"] == "fake-a"
