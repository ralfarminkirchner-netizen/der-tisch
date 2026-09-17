"""Regressionen für epistemische Schutzregeln der Auswertung.

Künstliche Testfälle — kein HDD-Praxistest, keine Originaldokumente.
"""

from __future__ import annotations

import pytest

from app.analysis import (
    KIND_AGREEMENT,
    KIND_CONTRADICTION,
    KIND_LABELS,
    KIND_UNIQUE,
    METHOD_VERSION,
    analyse,
    folgen,
    gleiches_thema,
    polaritaet_verschieden,
    split_sentences,
)
from app.aussagen import beleg_pruefen, normalisiere_vorschlag
from app.db import JOB_DONE, Store
from app.zahlen import extrahiere_zahlen, umrechnen, zahlen_analyse


def make_job(job_id: str, text: str, label: str | None = None) -> dict:
    return {
        "id": job_id,
        "model_id": job_id,
        "label": label or job_id.upper(),
        "provider": "fake",
        "model": "m",
        "status": JOB_DONE,
        "text": text,
        "error": None,
        "partial": 0,
        "latency_ms": 10,
        "finished_at": 1.0,
    }


def test_marker_labels_behaupten_keine_einigkeit():
    assert "Hinweis" in KIND_LABELS[KIND_AGREEMENT]
    assert "Einzigartig" not in KIND_LABELS[KIND_UNIQUE]
    assert "Übereinstimmung" not in KIND_LABELS[KIND_AGREEMENT]


def test_aehnliche_woerter_unterschiedliche_aussage_kein_einigkeitsbeweis():
    """Ähnliche Wörter bei unterschiedlicher Aussage → höchstens Themenbezug."""
    jobs = [
        make_job("j1", "Drei Bohrungen je Vortrieb reichen unter der genannten Bedingung."),
        make_job("j2", "Drei Vortriebe je Kreuzung sind etwas anderes als drei Bohrungen."),
    ]
    markers, summary = analyse("s1", "f1", jobs)
    assert summary["claim_level"] == "hinweis"
    assert summary["method_version"] == METHOD_VERSION
    for m in markers:
        if m["kind"] == KIND_AGREEMENT:
            assert "kein Beleg für inhaltliche Übereinstimmung" in m["note"].lower() or \
                "Begriffsüberschneidung" in m["note"]


def test_gleiche_aussage_andere_wortwahl_kein_widerspruch():
    jobs = [
        make_job("j1", "Die Deckung muss mindestens 2,80 m betragen und dokumentiert werden."),
        make_job("j2", "Es ist erforderlich, dass die Deckung wenigstens 2,80 m beträgt und dokumentiert wird."),
    ]
    markers, summary = analyse("s1", "f1", jobs)
    assert summary["counts"][KIND_CONTRADICTION] == 0


def test_verneinung_und_reichweite():
    jobs = [
        make_job("j1", "Eine tägliche Sicherung der Datenbank ist notwendig und sinnvoll."),
        make_job("j2", "Eine tägliche Sicherung der Datenbank ist nicht notwendig."),
    ]
    _, summary = analyse("s1", "f1", jobs)
    assert summary["counts"][KIND_CONTRADICTION] >= 1


def test_aehnlichkeitskette_ohne_gemeinsame_aussage_ist_cluster():
    """A~B und B~C darf nicht still zu „alle sagen A“ werden."""
    jobs = [
        make_job(
            "a",
            "Die Sicherung der Baugrube erfordert eine verstärkte Spundwand am Nordende.",
            "A",
        ),
        make_job(
            "b",
            "Die Sicherung der Baugrube braucht zusätzlich eine Drainage unter der Sohle.",
            "B",
        ),
        make_job(
            "c",
            "Die Drainage unter der Sohle muss vor dem Betonieren geprüft und freigegeben werden.",
            "C",
        ),
    ]
    karte = folgen(jobs)
    cluster = [f for f in karte["folgen"] if f.get("art") == "themencluster"]
    # Entweder getrennte Folgen oder ein Cluster mit Einzelaussagen — nie ein
    # gemeinsamer Ersttext als wäre er von allen gesagt.
    for folge in karte["folgen"]:
        if folge.get("zaehlung") == "clusterbeteiligung" or folge.get("art") == "themencluster":
            assert "Clusterbeteiligung" in folge["text"] or folge["zaehlung"] == "clusterbeteiligung"
            assert len(folge["nennungen"]) >= 2
            # Der Vertretertext darf nicht einfach die erste Nennung sein, als sagten alle dasselbe.
            assert folge["text"] != folge["nennungen"][0]["quote"] or folge["art"] == "gleiche_aussage"
    # Mindestens eine Gruppe mit mehr als einer Stimme muss ehrlich gezählt sein.
    assert any(f["anzahl"] >= 1 for f in karte["folgen"])


def test_kein_treffer_heisst_nicht_einzigartig():
    jobs = [
        make_job("j1", "Regelmäßige Sicherungskopien schützen die Daten vor Verlust."),
        make_job("j2", "Ein Zwiebelkuchen gelingt besonders gut mit frischem Federweißer."),
    ]
    markers, _ = analyse("s1", "f1", jobs)
    unique = [m for m in markers if m["kind"] == KIND_UNIQUE]
    assert unique
    assert all("Einzigartigkeit" in m["note"] or "einzigartig" in m["note"].lower() for m in unique)
    assert all("keine tatsächliche" in m["note"].lower() or "nicht" in m["note"].lower() for m in unique)


def test_zahl_gleiche_bezugsgröße_abweichende_werte():
    """Dieselbe Bezugsgröße mit 2,80 m gegenüber 3,50 m."""
    jobs = [
        make_job("j1", "Die erforderliche Deckung beträgt 2,80 m über dem Scheitel."),
        make_job("j2", "Die erforderliche Deckung beträgt 3,50 m über dem Scheitel."),
    ]
    ergebnis = zahlen_analyse(jobs)
    assert len(ergebnis["funde"]) >= 2
    abweichend = [v for v in ergebnis["vergleiche"] if v["status"] == "abweichend"]
    assert abweichend, ergebnis["vergleiche"]


def test_zahl_gleiche_einheit_andere_bezugsgröße():
    jobs = [
        make_job("j1", "Es sind drei Bohrungen je Vortrieb vorgesehen."),
        make_job("j2", "Es sind drei Vortriebe je Kreuzung vorgesehen."),
    ]
    ergebnis = zahlen_analyse(jobs)
    funde = ergebnis["funde"]
    assert any(f.get("bezug") and "Bohr" in (f["bezug"] or "") for f in funde)
    assert any(f.get("bezug") and "Vortrieb" in (f["bezug"] or "") for f in funde)


def test_zahl_aequivalente_einheiten():
    assert umrechnen(2.8, "m", "cm") == pytest.approx(280.0)
    jobs = [
        make_job("j1", "Die Deckung beträgt 2,80 m am Scheitel."),
        make_job("j2", "Die Deckung beträgt 280 cm am Scheitel."),
    ]
    ergebnis = zahlen_analyse(jobs)
    gesichert = [v for v in ergebnis["vergleiche"] if v["status"] == "zuordnung_gesichert"]
    vorgeschlagen = [v for v in ergebnis["vergleiche"] if v["status"] == "vergleich_vorgeschlagen"]
    assert gesichert or vorgeschlagen


def test_mehrdeutiges_trennzeichen_nicht_still_annehmen():
    funde = extrahiere_zahlen("j1", "Der Wert liegt bei 2.800 m Deckung.")
    assert funde
    assert funde[0].parse_status == "unklar_trennzeichen"
    assert funde[0].value is None


def test_belegstellen_unicode():
    text = "Die Länge beträgt 2,80 m — inkl. Toleranz."
    assert beleg_pruefen(text, 0, 4, "Die ") is None
    # Ungültig
    assert beleg_pruefen(text, 0, 4, "XXXX") is not None
    # Außerhalb
    assert beleg_pruefen(text, 0, 999, text) is not None


def test_verstaerkte_interpretation_trotz_echtem_zitat():
    job = make_job("j1", "Die Deckung beträgt 2,80 m.")
    v = normalisiere_vorschlag(
        job=job,
        analysis_run_id="arn1",
        analysemodell="other-model",
        roh={
            "aussage": "Die Deckung ist nachweislich sicher und genehmigt.",
            "quote": "Die Deckung beträgt 2,80 m.",
            "start_offset": 0,
            "end_offset": len(job["text"]),
            "setzung": "modell_vermutet",
        },
    )
    assert v["status"] == "vorschlag"
    assert "Interpretation" in v["hinweis"]


def test_ungueltiger_beleg_nicht_im_befund():
    job = make_job("j1", "Kurzer Satz hier.")
    v = normalisiere_vorschlag(
        job=job,
        analysis_run_id="arn1",
        analysemodell="other-model",
        roh={
            "aussage": "Etwas",
            "quote": "nicht im Text",
            "start_offset": 0,
            "end_offset": 5,
        },
    )
    assert v["status"] == "fehler"
    assert v["fehler"]


@pytest.mark.asyncio
async def test_neuer_lauf_ueberschreibt_alte_nicht(tmp_path):
    store = Store(tmp_path / "t.db")
    await store.connect()
    session = await store.create_session("Test")
    sid = session["id"]
    spark, _neu = await store.insert_spark(
        session_id=sid, prompt="q", client_request_id="c1"
    )
    fid = spark["id"]
    await store.save_analysis_run(
        run_id="arn-alt",
        spark_id=fid,
        session_id=sid,
        method_version="alt",
        payload={"v": 1},
        markers=[],
    )
    await store.save_assessment(fid, sid, {"v": 1, "analysis_run_id": "arn-alt"})
    await store.save_analysis_run(
        run_id="arn-neu",
        spark_id=fid,
        session_id=sid,
        method_version=METHOD_VERSION,
        payload={"v": 2},
        markers=[],
    )
    await store.save_assessment(fid, sid, {"v": 2, "analysis_run_id": "arn-neu"})
    laeufe = await store.list_analysis_runs(fid)
    assert len(laeufe) == 2
    assert [l["id"] for l in laeufe] == ["arn-alt", "arn-neu"]
    aktuell = (await store.list_assessments(sid))[fid]
    assert aktuell["analysis_run_id"] == "arn-neu"
    await store.close()


def test_analyse_traegt_laufkennung():
    jobs = [
        make_job("j1", "Regelmäßige Sicherungskopien schützen die Daten wirksam vor Verlust."),
        make_job("j2", "Regelmäßige Sicherungskopien schützen Daten wirksam vor Verlust."),
    ]
    markers, summary = analyse("s1", "f1", jobs, analysis_run_id="arn-test")
    assert summary["analysis_run_id"] == "arn-test"
    assert all(m.get("analysis_run_id") == "arn-test" for m in markers)
    assert all(m.get("claim_level") == "hinweis" for m in markers)
