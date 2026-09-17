"""Automatische Einschätzungen über die Antworten eines Funkens.

Die Auswertung ist rein rechnerisch (keine LLM-Aufrufe) und damit
deterministisch, reproduzierbar und kostenlos wiederholbar. Sie verändert
die Originalantworten nicht: Marker verweisen nur über Zeichenpositionen
auf den unveränderten Text.

Epistemische Leitregeln (Schutz vor Fehlschlüssen):

- Begriffsüberschneidung ist ein **Hinweis** auf möglichen Text- oder
  Themenbezug. Sie ist kein hinreichender Beleg für inhaltliche Übereinstimmung.
- Kein erkannter Gegensatz bedeutet: mit diesem Verfahren kein Gegensatz
  erkannt — nicht: Übereinstimmung bewiesen.
- Keine passende Fundstelle bedeutet: im ausgewerteten Bestand mit diesem
  Verfahren keine passende Fundstelle gefunden — nicht: die Aussage ist
  tatsächlich einzigartig.
- A ähnlich B und B ähnlich C ergibt einen Themencluster mit einzelnen
  Aussagen (Clusterbeteiligung), nicht still „alle sagen A“.

Leitregel für Widerspruchshinweise: Unterschiedlichkeit allein ist keiner.
Ein Gegensatzhinweis entsteht nur, wenn zwei Aussagen dasselbe Thema
betreffen (hohe Begriffsüberschneidung) UND entgegengesetzte Polarität
haben (Verneinung oder bekanntes Gegensatzpaar).

Verglichen wird über Wortstämme (siehe :mod:`app.stemmer`), damit „Sicherung"
und „Sicherungen" ein Begriff sind und nicht zwei halb so schwere. Der Stamm
ist nur der Schlüssel; angezeigt und zitiert wird immer die Form, die
tatsächlich im Text steht.

Die Marker-``kind``-Schlüssel ``uebereinstimmung`` / ``widerspruch`` /
``einzigartig`` bleiben aus Kompatibilität zu Farbe und Speicherung; ihre
Bedeutung in Hinweis-Läufen ist die der Labels in :data:`KIND_LABELS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .db import JOB_DONE, new_id, now
from .stemmer import stamm

#: Verfahrensversion — jeder gespeicherte Lauf trägt sie mit.
METHOD_VERSION = "auswertung-hinweis-v1"

KIND_AGREEMENT = "uebereinstimmung"
KIND_CONTRADICTION = "widerspruch"
KIND_UNIQUE = "einzigartig"

#: Anzeigenamen — bewusst keine Einigkeits- oder Einzigartigkeitsbehauptung.
KIND_LABELS = {
    KIND_AGREEMENT: "Themenbezug (Hinweis)",
    KIND_CONTRADICTION: "Gegensatzhinweis (Verfahren)",
    KIND_UNIQUE: "Kein Treffer in diesem Verfahren",
}

#: Ab dieser Begriffsüberschneidung gelten zwei Sätze als themengleich.
TOPIC_THRESHOLD = 0.34
#: Mindestzahl gemeinsamer Inhaltswörter für einen Themenbezug.
MIN_SHARED_TERMS = 2
#: Unterhalb dieser Ähnlichkeit zu allen anderen Antworten: kein Treffer hier.
UNIQUE_THRESHOLD = 0.10
#: Ab dieser Satzähnlichkeit gelten zwei Nennungen als dieselbe Aussage.
SAME_STATEMENT_THRESHOLD = 0.82
MIN_TERMS_PER_SENTENCE = 3
#: Obergrenze je Art und Antwortpaar, damit die Oberfläche lesbar bleibt.
MAX_MARKERS_PER_PAIR = 6
MAX_UNIQUE_PER_JOB = 5

STOPWORDS = {
    # Deutsch
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an",
    "auch", "auf", "aus", "bei", "beim", "bin", "bis", "bist", "da", "damit", "dann",
    "das", "dass", "dem", "den", "der", "des", "deshalb", "die", "dies", "diese",
    "diesem", "diesen", "dieser", "dieses", "doch", "dort", "durch", "ein", "eine",
    "einem", "einen", "einer", "eines", "er", "es", "etwa", "für", "gegen", "hat",
    "hatte", "hier", "ich", "ihr", "ihre", "im", "in", "ist", "ja", "jede", "jeder",
    "kann", "können", "mehr", "mit", "muss", "nach", "noch", "nur", "ob", "oder",
    "ohne", "schon", "sehr", "sein", "seine", "sich", "sie", "sind", "so", "solche",
    "soll", "sollte", "über", "um", "und", "uns", "unter", "vom", "von", "vor",
    "war", "waren", "was", "weil", "welche", "wenn", "werden", "wie", "wir", "wird",
    "wurde", "zu", "zum", "zur", "zwar", "zwischen", "man", "eher", "etwas", "dabei",
    # Füll- und Funktionswörter, die sonst als vermeintliches „Thema" oben
    # landen: sie werden von allen benutzt und sagen über den Inhalt nichts.
    "allein", "außerdem", "darf", "dürfen", "müssen", "sollen", "sollten",
    "wollen", "können", "konnte", "könnte", "würde", "wäre", "sei", "seien",
    "bereits", "immer", "jedoch", "dennoch", "sobald", "sofern", "falls",
    "meist", "meistens", "oft", "häufig", "selten", "jeweils", "ebenfalls",
    "insbesondere", "beispielsweise", "grundsätzlich", "generell", "gerade",
    "genau", "darum", "deswegen", "somit", "folglich", "zudem", "ferner",
    "weiter", "weitere", "weiteren", "weiterhin", "viel", "viele", "vielen",
    "wenig", "wenige", "groß", "große", "klein", "kleine", "gut", "gute",
    "besser", "schlecht", "neu", "neue", "alt", "alte", "erst", "erste",
    "letzte", "beim", "ihrem", "ihren", "einfach", "möglich", "nötig",
    # Englisch
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "are", "was",
    "were", "but", "not", "you", "your", "its", "it's", "they", "their", "can",
    "will", "would", "should", "could", "there", "here", "into", "about", "than",
    "then", "them", "these", "those", "which", "when", "what", "who", "how",
}

NEGATIONS = {
    "nicht", "kein", "keine", "keinen", "keiner", "keinem", "keines", "nie",
    "niemals", "nichts", "weder", "ohne", "unmöglich", "no", "not", "never",
    "none", "cannot", "can't", "don't", "doesn't", "isn't", "without",
}

#: Gegensatzpaare, die auch ohne Verneinung eine Polaritätsumkehr anzeigen.
ANTONYMS = [
    ("sicher", "unsicher"),
    ("möglich", "unmöglich"),
    ("geeignet", "ungeeignet"),
    ("wirksam", "unwirksam"),
    ("zulässig", "unzulässig"),
    ("nötig", "unnötig"),
    ("steigt", "sinkt"),
    ("erhöht", "senkt"),
    ("mehr", "weniger"),
    ("vorteil", "nachteil"),
    ("empfohlen", "abgeraten"),
    ("safe", "unsafe"),
    ("possible", "impossible"),
    ("increase", "decrease"),
    ("recommended", "discouraged"),
]

WORD_RE = re.compile(r"[\wäöüßÄÖÜ'’-]+", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*", re.UNICODE)


@dataclass
class Sentence:
    job_id: str
    start: int
    end: int
    text: str
    #: Die Wortstämme des Satzes — der Schlüssel, über den verglichen wird.
    terms: frozenset[str]
    negated: bool
    raw_terms: frozenset[str]
    #: Stamm -> Wortform, die im Satz wirklich steht. Nur diese wird gezeigt.
    forms: dict[str, str]


def _inhaltswort(w: str) -> bool:
    """Trägt dieses Wort Inhalt? Geprüft wird die Form, die dasteht.

    Die Stoppwortliste wird ausdrücklich **vor** dem Stemmen angewandt. Ein
    Stamm kann mit dem Stamm eines Funktionsworts zusammenfallen („sicher" und
    „sich" ergeben beide „sich"); würde nach dem Stemmen gefiltert, verschwände
    ein tragender Begriff still aus der Auswertung.
    """
    return len(w) >= 3 and w not in STOPWORDS and w not in NEGATIONS and not w.isdigit()


def split_sentences(job_id: str, text: str) -> list[Sentence]:
    """Zerlegt einen Text in Sätze und behält die Zeichenpositionen bei."""
    sentences: list[Sentence] = []
    for match in SENTENCE_RE.finditer(text):
        raw = match.group(0)
        stripped = raw.strip()
        if len(stripped) < 15:
            continue
        # Positionen auf den getrimmten Satz nachziehen, damit das Zitat exakt passt.
        lead = len(raw) - len(raw.lstrip())
        start = match.start() + lead
        end = start + len(stripped)
        words = [w.lower() for w in WORD_RE.findall(stripped)]
        raw_terms = frozenset(words)
        forms: dict[str, str] = {}
        for w in words:
            if not _inhaltswort(w):
                continue
            schluessel = stamm(w)
            if len(schluessel) < 2:
                continue
            # Die kürzeste Form gewinnt, bei Gleichstand das Alphabet. So hängt
            # die Beschriftung nicht daran, in welcher Reihenfolge gelesen wurde.
            vorhanden = forms.get(schluessel)
            if vorhanden is None or (len(w), w) < (len(vorhanden), vorhanden):
                forms[schluessel] = w
        sentences.append(
            Sentence(
                job_id=job_id,
                start=start,
                end=end,
                text=stripped,
                terms=frozenset(forms),
                negated=bool(raw_terms & NEGATIONS),
                raw_terms=raw_terms,
                forms=forms,
            )
        )
    return sentences


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def anzeigeform(schluessel: str, *saetze: Sentence) -> str:
    """Die Wortform, die für einen Stamm gezeigt wird.

    Genommen wird, was in den beteiligten Sätzen wirklich steht — nie der
    Stamm selbst. Bei mehreren Formen entscheidet Länge, dann Alphabet, damit
    dieselbe Fundstelle immer dieselbe Beschriftung trägt.
    """
    formen = sorted(
        {s.forms[schluessel] for s in saetze if schluessel in s.forms},
        key=lambda w: (len(w), w),
    )
    return formen[0] if formen else schluessel


def _antonym_conflict(a: Sentence, b: Sentence) -> str | None:
    for left, right in ANTONYMS:
        if (left in a.raw_terms and right in b.raw_terms) or (
            right in a.raw_terms and left in b.raw_terms
        ):
            return f"{left} ↔ {right}"
    return None


def gleiches_thema(a: Sentence, b: Sentence) -> tuple[bool, float, frozenset[str]]:
    """Hinweis auf möglichen Text-/Themenbezug — keine inhaltliche Einigkeit.

    Die eine Stelle, an der diese Frage beantwortet wird. Alles, was Aussagen
    zusammenlegt — Marker, Themen-Linse, Folgen im Szenario —, ruft sie auf; es
    gibt dafür bewusst keine zweite Heuristik.

    Ein Treffer bedeutet nur: die Begriffsüberschneidung liegt über dem
    Schwellenwert. Er beweist keine inhaltliche Übereinstimmung.
    """
    score = similarity(a.terms, b.terms)
    shared = a.terms & b.terms
    return (score >= TOPIC_THRESHOLD and len(shared) >= MIN_SHARED_TERMS), score, shared


def polaritaet_verschieden(a: Sentence, b: Sentence) -> str | None:
    """Entgegengesetzte Polarität mit diesem Verfahren? Sonst None.

    Erst zusammen mit :func:`gleiches_thema` ergibt das einen Gegensatzhinweis.
    Unterschiedlichkeit allein ist keiner. ``None`` bedeutet: kein Gegensatz
    erkannt — nicht: Übereinstimmung bewiesen.
    """
    antonym = _antonym_conflict(a, b)
    if antonym is not None:
        return f"Gegensatz: {antonym}"
    if a.negated != b.negated:
        return "gegensätzliche Verneinung"
    return None


def gleiche_aussage(a: Sentence, b: Sentence) -> bool:
    """Sehr ähnliche Formulierung derselben Aussage — strenger als Themenbezug."""
    if not gleiches_thema(a, b)[0]:
        return False
    if polaritaet_verschieden(a, b) is not None:
        return False
    return similarity(a.terms, b.terms) >= SAME_STATEMENT_THRESHOLD


def _marker(
    *,
    session_id: str,
    spark_id: str,
    sentence: Sentence,
    kind: str,
    related_job_id: str | None,
    note: str,
    topics: list[str] | None = None,
    analysis_run_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": new_id("mrk"),
        "session_id": session_id,
        "spark_id": spark_id,
        "job_id": sentence.job_id,
        "related_job_id": related_job_id,
        "kind": kind,
        "start_offset": sentence.start,
        "end_offset": sentence.end,
        "quote": sentence.text,
        "note": note,
        # Woran diese Fundstelle hängt — Grundlage der Themen-Linse.
        "topics": list(topics or []),
        "claim_level": "hinweis",
        "analysis_run_id": analysis_run_id,
        "created_at": now(),
    }


def analyse(
    session_id: str,
    spark_id: str,
    jobs: Iterable[dict[str, Any]],
    *,
    analysis_run_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Berechnet Hinweis-Marker und Zusammenfassung für einen Funken."""
    run_id = analysis_run_id or new_id("arn")
    job_list = list(jobs)
    usable = [j for j in job_list if j["status"] == JOB_DONE and (j.get("text") or "").strip()]
    job_versions = [
        {
            "job_id": j["id"],
            "status": j["status"],
            "text_len": len(j.get("text") or ""),
            "finished_at": j.get("finished_at"),
            "model_id": j.get("model_id"),
            "provider": j.get("provider"),
        }
        for j in usable
    ]

    by_job: dict[str, list[Sentence]] = {
        j["id"]: split_sentences(j["id"], j["text"]) for j in usable
    }

    markers: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for i, job_a in enumerate(usable):
        for job_b in usable[i + 1:]:
            agreements = 0
            contradictions = 0
            topics: list[str] = []
            sents_a = by_job[job_a["id"]]
            sents_b = by_job[job_b["id"]]

            for sa in sents_a:
                best: tuple[float, Sentence] | None = None
                for sb in sents_b:
                    score = similarity(sa.terms, sb.terms)
                    if best is None or score > best[0]:
                        best = (score, sb)
                if best is None:
                    continue
                score, sb = best
                topical, score, shared = gleiches_thema(sa, sb)
                if not topical:
                    # Unterschiedliche Themen sind kein Gegensatzhinweis.
                    continue

                grund = polaritaet_verschieden(sa, sb)
                kind = KIND_CONTRADICTION if grund else KIND_AGREEMENT
                if kind == KIND_CONTRADICTION and contradictions >= MAX_MARKERS_PER_PAIR:
                    continue
                if kind == KIND_AGREEMENT and agreements >= MAX_MARKERS_PER_PAIR:
                    continue

                # Beschriftet wird mit dem Wortlaut aus den Antworten, nicht mit
                # dem Stamm: „Sicherungen" zählt wie „Sicherung", steht aber so
                # da, wie eine der beiden Stimmen es geschrieben hat.
                begriffe = [anzeigeform(k, sa, sb) for k in sorted(shared)[:4]]
                topic = ", ".join(begriffe)
                if grund:
                    note = (
                        f"{KIND_LABELS[KIND_CONTRADICTION]}: {topic} "
                        f"(Überschneidung {score:.0%}); {grund}. "
                        "Kein hinreichender Beleg für inhaltliche Einigkeit oder Wahrheit."
                    )
                else:
                    note = (
                        f"{KIND_LABELS[KIND_AGREEMENT]}: {topic} "
                        f"(Überschneidung {score:.0%}). "
                        "Begriffsüberschneidung ist kein Beleg für inhaltliche Übereinstimmung; "
                        "kein Gegensatz mit diesem Verfahren erkannt bedeutet nicht Einigkeit."
                    )

                markers.append(
                    _marker(
                        session_id=session_id, spark_id=spark_id, sentence=sa,
                        kind=kind, related_job_id=sb.job_id, note=note,
                        topics=begriffe, analysis_run_id=run_id,
                    )
                )
                markers.append(
                    _marker(
                        session_id=session_id, spark_id=spark_id, sentence=sb,
                        kind=kind, related_job_id=sa.job_id, note=note,
                        topics=begriffe, analysis_run_id=run_id,
                    )
                )
                if kind == KIND_CONTRADICTION:
                    contradictions += 1
                else:
                    agreements += 1
                if topic and topic not in topics:
                    topics.append(topic)

            pairs.append(
                {
                    "a_job_id": job_a["id"],
                    "b_job_id": job_b["id"],
                    "a_model_id": job_a["model_id"],
                    "b_model_id": job_b["model_id"],
                    "a_label": job_a["label"],
                    "b_label": job_b["label"],
                    # Zählungen bleiben unter den alten Schlüsseln (API-Kompatibilität),
                    # bedeuten aber Hinweis-Treffer, keine bestätigte Einigkeit.
                    "agreements": agreements,
                    "contradictions": contradictions,
                    "themenbezuege": agreements,
                    "gegensatzhinweise": contradictions,
                    "topics": topics[:6],
                }
            )

    # Kein Treffer in diesem Verfahren — nicht: tatsächlich einzigartig.
    unique_counts: dict[str, int] = {}
    for job in usable:
        others = [s for jid, sents in by_job.items() if jid != job["id"] for s in sents]
        count = 0
        for sa in by_job[job["id"]]:
            if len(sa.terms) < MIN_TERMS_PER_SENTENCE:
                continue
            if others and max(similarity(sa.terms, sb.terms) for sb in others) >= UNIQUE_THRESHOLD:
                continue
            if count >= MAX_UNIQUE_PER_JOB:
                break
            markers.append(
                _marker(
                    session_id=session_id, spark_id=spark_id, sentence=sa,
                    kind=KIND_UNIQUE, related_job_id=None,
                    note=(
                        f"{KIND_LABELS[KIND_UNIQUE]}: im ausgewerteten Bestand "
                        "keine passende Fundstelle mit diesem Verfahren. "
                        "Das beweist keine tatsächliche Einzigartigkeit."
                    ),
                    topics=[anzeigeform(k, sa) for k in sorted(sa.terms)[:4]],
                    analysis_run_id=run_id,
                )
            )
            count += 1
        unique_counts[job["id"]] = count

    counts = {
        KIND_AGREEMENT: sum(1 for m in markers if m["kind"] == KIND_AGREEMENT),
        KIND_CONTRADICTION: sum(1 for m in markers if m["kind"] == KIND_CONTRADICTION),
        KIND_UNIQUE: sum(1 for m in markers if m["kind"] == KIND_UNIQUE),
    }

    summary = {
        "spark_id": spark_id,
        "analysis_run_id": run_id,
        "method_version": METHOD_VERSION,
        "computed_at": now(),
        "claim_level": "hinweis",
        "labels": dict(KIND_LABELS),
        "epistemik": (
            "Begriffsüberschneidung = Hinweis auf möglichen Textbezug. "
            "Kein Gegensatz erkannt ≠ Übereinstimmung. "
            "Kein Treffer ≠ tatsächliche Einzigartigkeit."
        ),
        "job_versions": job_versions,
        "models": [
            {
                "job_id": j["id"],
                "model_id": j["model_id"],
                "label": j["label"],
                "provider": j["provider"],
                "status": j["status"],
                "chars": len(j.get("text") or ""),
                "sentences": len(by_job.get(j["id"], [])),
                "latency_ms": j.get("latency_ms"),
                "partial": bool(j.get("partial")),
                "error": j.get("error"),
                "unique": unique_counts.get(j["id"], 0),
                "agreements": sum(
                    p["agreements"] for p in pairs
                    if j["id"] in (p["a_job_id"], p["b_job_id"])
                ),
                "contradictions": sum(
                    p["contradictions"] for p in pairs
                    if j["id"] in (p["a_job_id"], p["b_job_id"])
                ),
            }
            for j in job_list
        ],
        "pairs": pairs,
        "counts": counts,
        "analysed_jobs": [j["id"] for j in usable],
        "method": (
            f"{METHOD_VERSION}: regelbasierter Vergleich auf Satzebene. "
            "Themenbezug über Begriffsüberschneidung (Hinweis, keine Einigkeit); "
            "Gegensatzhinweis nur bei gegensätzlicher Polarität; "
            "kein Treffer bedeutet fehlende Fundstelle in diesem Verfahren."
        ),
    }
    return markers, summary


def validate_markers(markers: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> list[str]:
    """Prüft Bezüge und Textbelege. Gibt eine Liste von Beanstandungen zurück."""
    texts = {j["id"]: (j.get("text") or "") for j in jobs}
    problems: list[str] = []
    for m in markers:
        if m["job_id"] not in texts:
            problems.append(f"Marker {m['id']}: unbekannter Auftrag {m['job_id']}")
            continue
        if m["related_job_id"] is not None and m["related_job_id"] not in texts:
            problems.append(f"Marker {m['id']}: unbekannter Bezug {m['related_job_id']}")
        text = texts[m["job_id"]]
        start, end = m["start_offset"], m["end_offset"]
        if not (0 <= start < end <= len(text)):
            problems.append(f"Marker {m['id']}: Position {start}-{end} außerhalb des Textes")
            continue
        if text[start:end] != m["quote"]:
            problems.append(f"Marker {m['id']}: Textbeleg stimmt nicht mit der Antwort überein")
    return problems


# ---------------------------------------------------------------- Szenarien

#: Höchstens so viele Folgen werden gezeigt. Was darüber hinausgeht, wird
#: gezählt und benannt — es verschwindet nicht stillschweigend.
MAX_FOLGEN = 12
#: So lang darf der Auszug der Ausgangsaussage sein.
AUSGANG_ZEICHEN = 240


def _kuerze(text: str, zeichen: int) -> tuple[str, bool]:
    sauber = " ".join(text.split())
    if len(sauber) <= zeichen:
        return sauber, False
    return sauber[: zeichen - 1].rstrip() + "…", True


def _cluster_ist_gleiche_aussage(gruppe: list[tuple[dict[str, Any], Sentence]]) -> bool:
    """True nur wenn jede Nennung zur ersten als dieselbe Aussage gilt."""
    if len(gruppe) <= 1:
        return True
    anker = gruppe[0][1]
    return all(gleiche_aussage(anker, satz) for _, satz in gruppe[1:])


def folgen(jobs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Die Konsequenzkarte einer Szenario-Runde.

    Die Antworten auf einen Szenario-Beitrag werden mit derselben
    Satzzerlegung zerlegt und über Themenbezug gruppiert. A~B und B~C
    erzeugen einen **Themencluster** mit einzelnen Aussagen — nicht still
    „alle sagen A“. Nur wenn die Nennungen dieselbe Aussage tragen
    (:func:`gleiche_aussage`), gilt die Zählung als gemeinsame Nennung.

    Gezählt wird Clusterbeteiligung bzw. gemeinsame Nennung — beides eine
    Häufigkeit, keine Wahrscheinlichkeit und keine Zustimmung.
    """
    job_list = list(jobs)
    usable = [j for j in job_list if j["status"] == JOB_DONE and (j.get("text") or "").strip()]
    stimmen = [{"job_id": j["id"], "label": j["label"]} for j in usable]

    # Sätze in fester Reihenfolge: Tischreihenfolge, darin Lesereihenfolge.
    # Nur davon hängt die Gruppierung ab — sie ist damit wiederholbar.
    saetze: list[tuple[dict[str, Any], Sentence]] = []
    for job in usable:
        for satz in split_sentences(job["id"], job["text"]):
            if len(satz.terms) < MIN_TERMS_PER_SENTENCE:
                continue
            saetze.append((job, satz))

    gruppen: list[list[tuple[dict[str, Any], Sentence]]] = []
    for job, satz in saetze:
        ziel: list[tuple[dict[str, Any], Sentence]] | None = None
        for gruppe in gruppen:
            for _, anderer in gruppe:
                gleich, _score, _shared = gleiches_thema(satz, anderer)
                if gleich and polaritaet_verschieden(satz, anderer) is None:
                    ziel = gruppe
                    break
            if ziel is not None:
                break
        if ziel is None:
            gruppen.append([(job, satz)])
        else:
            ziel.append((job, satz))

    eintraege: list[dict[str, Any]] = []
    for index, gruppe in enumerate(gruppen, start=1):
        nennungen: list[dict[str, Any]] = []
        gesehen: set[str] = set()
        for job, satz in gruppe:
            nennungen.append(
                {
                    "job_id": job["id"],
                    "label": job["label"],
                    "quote": satz.text,
                    "start_offset": satz.start,
                    "end_offset": satz.end,
                }
            )
            gesehen.add(job["id"])
        erster = gruppe[0][1]
        schluessel = sorted(erster.terms)[:4]
        gleiche = _cluster_ist_gleiche_aussage(gruppe)
        art = "gleiche_aussage" if gleiche else "themencluster"
        zaehlung = "gemeinsame_nenung" if gleiche else "clusterbeteiligung"
        # Vertretertext nur bei gleicher Aussage; sonst Cluster-Beschriftung,
        # damit nicht „alle sagen A“ aus A~B~C entsteht.
        if gleiche:
            text = erster.text
        else:
            themen = [anzeigeform(k, *[s for _, s in gruppe]) for k in schluessel]
            text = (
                f"Themencluster ({', '.join(themen) or 'Thema'}): "
                f"{len(nennungen)} einzelne Aussagen, {len(gesehen)} Stimmen — "
                "Clusterbeteiligung, keine Zustimmung."
            )
        eintraege.append(
            {
                "id": f"flg{index}",
                "art": art,
                "zaehlung": zaehlung,
                "text": text,
                "themen": [anzeigeform(k, *[s for _, s in gruppe]) for k in schluessel],
                "nennungen": nennungen,
                "anzahl": len(gesehen),
                "von": len(usable),
                "gegensatz": [],
            }
        )

    # Gegensatzhinweis zwischen zwei Folgen: gleiches Thema + Polarität.
    for i, gruppe_a in enumerate(gruppen):
        for j in range(i + 1, len(gruppen)):
            gruppe_b = gruppen[j]
            treffer = False
            for _, sa in gruppe_a:
                for _, sb in gruppe_b:
                    gleich, _score, _shared = gleiches_thema(sa, sb)
                    if gleich and polaritaet_verschieden(sa, sb) is not None:
                        treffer = True
                        break
                if treffer:
                    break
            if treffer:
                eintraege[i]["gegensatz"].append(eintraege[j]["id"])
                eintraege[j]["gegensatz"].append(eintraege[i]["id"])

    # Häufigkeit nach vorn: Clusterbeteiligung/Nennung, keine Rangfolge der Stimmen.
    geordnet = sorted(
        enumerate(eintraege), key=lambda paar: (-paar[1]["anzahl"], paar[0])
    )
    gezeigt = [eintrag for _, eintrag in geordnet[:MAX_FOLGEN]]
    sichtbare_ids = {e["id"] for e in gezeigt}
    for eintrag in gezeigt:
        eintrag["gegensatz"] = [g for g in eintrag["gegensatz"] if g in sichtbare_ids]

    return {
        "stimmen": stimmen,
        "folgen": gezeigt,
        "uebergangen": max(0, len(eintraege) - len(gezeigt)),
        "methode": (
            f"{METHOD_VERSION}: Sätze über Themenbezug gruppiert. "
            "Ähnlichkeitsketten ohne gemeinsame Aussage = Themencluster mit "
            "einzelnen Aussagen (Clusterbeteiligung, keine Zustimmung). "
            "Die Zahl ist eine Häufigkeit, keine Wahrscheinlichkeit."
        ),
    }


def szenario_block(
    spark: dict[str, Any],
    jobs: Iterable[dict[str, Any]],
    ausgang: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ein Szenario mit seiner Ausgangsaussage und den genannten Folgen.

    ``ausgang`` ist der Auftrag, aus dessen Antwort das Szenario hervorging —
    ermittelt aus den ausdrücklich gesetzten Bezügen des Beitrags, nicht
    geraten. Fehlt er, steht die Eingabe selbst als Ausgang da.
    """
    karte = folgen(jobs)
    herkunft: dict[str, Any] | None = None
    if ausgang is not None:
        auszug, gekuerzt = _kuerze(ausgang.get("text") or "", AUSGANG_ZEICHEN)
        herkunft = {
            "job_id": ausgang["id"],
            "label": ausgang["label"],
            "auszug": auszug,
            "gekuerzt": gekuerzt,
        }
    return {
        "spark_id": spark["id"],
        "seq": spark["seq"],
        "prompt": spark["prompt"],
        "ausgang": herkunft,
        **karte,
    }


def themen_aus_markern(markers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sammelt die Begriffe der Fundstellen mit den Stimmen, die daran hängen.

    Reine Projektion vorhandener Marker — es wird nichts gerechnet und nichts
    hinzugefügt. Dieselbe Ordnung wie in der Themen-Linse der Oberfläche:
    schwerste zuerst, bei Gleichstand das Alphabet.
    """
    nach: dict[str, dict[str, set[str]]] = {}
    for marker in markers:
        for begriff in marker.get("topics") or []:
            if not begriff:
                continue
            knoten = nach.setdefault(
                begriff, {"einig": set(), "gegen": set(), "einzeln": set()}
            )
            if marker["kind"] == KIND_AGREEMENT:
                knoten["einig"].add(marker["job_id"])
            elif marker["kind"] == KIND_CONTRADICTION:
                knoten["gegen"].add(marker["job_id"])
            else:
                knoten["einzeln"].add(marker["job_id"])

    zeilen = [
        {
            "begriff": begriff,
            "einig": len(mengen["einig"]),
            "gegen": len(mengen["gegen"]),
            "einzeln": len(mengen["einzeln"]),
            "stimmen": len(mengen["einig"] | mengen["gegen"] | mengen["einzeln"]),
        }
        for begriff, mengen in nach.items()
    ]
    zeilen.sort(key=lambda z: (-(z["einig"] + z["gegen"] + z["einzeln"]), z["begriff"]))
    return zeilen
