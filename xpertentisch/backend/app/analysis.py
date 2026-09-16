"""Automatische Einschätzungen über die Antworten eines Funkens.

Die Auswertung ist rein rechnerisch (keine LLM-Aufrufe) und damit
deterministisch, reproduzierbar und kostenlos wiederholbar. Sie verändert
die Originalantworten nicht: Marker verweisen nur über Zeichenpositionen
auf den unveränderten Text.

Leitregel für Widersprüche: Unterschiedlichkeit allein ist kein Widerspruch.
Ein Widerspruch wird nur dann markiert, wenn zwei Aussagen dasselbe Thema
betreffen (hohe Begriffsüberschneidung) UND entgegengesetzte Polarität
haben (Verneinung oder bekanntes Gegensatzpaar).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .db import JOB_DONE, new_id, now

KIND_AGREEMENT = "uebereinstimmung"
KIND_CONTRADICTION = "widerspruch"
KIND_UNIQUE = "einzigartig"

#: Ab dieser Begriffsüberschneidung gelten zwei Sätze als themengleich.
TOPIC_THRESHOLD = 0.34
#: Mindestzahl gemeinsamer Inhaltswörter für einen Themenbezug.
MIN_SHARED_TERMS = 2
#: Unterhalb dieser Ähnlichkeit zu allen anderen Antworten gilt ein Satz als einzigartig.
UNIQUE_THRESHOLD = 0.10
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
    terms: frozenset[str]
    negated: bool
    raw_terms: frozenset[str]


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
        terms = frozenset(
            w for w in words
            if len(w) >= 3 and w not in STOPWORDS and w not in NEGATIONS and not w.isdigit()
        )
        sentences.append(
            Sentence(
                job_id=job_id,
                start=start,
                end=end,
                text=stripped,
                terms=terms,
                negated=bool(raw_terms & NEGATIONS),
                raw_terms=raw_terms,
            )
        )
    return sentences


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _antonym_conflict(a: Sentence, b: Sentence) -> str | None:
    for left, right in ANTONYMS:
        if (left in a.raw_terms and right in b.raw_terms) or (
            right in a.raw_terms and left in b.raw_terms
        ):
            return f"{left} ↔ {right}"
    return None


def _marker(
    *,
    session_id: str,
    spark_id: str,
    sentence: Sentence,
    kind: str,
    related_job_id: str | None,
    note: str,
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
        "created_at": now(),
    }


def analyse(
    session_id: str, spark_id: str, jobs: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Berechnet Marker und Zusammenfassung für einen Funken."""
    job_list = list(jobs)
    usable = [j for j in job_list if j["status"] == JOB_DONE and (j.get("text") or "").strip()]

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
                shared = sa.terms & sb.terms
                antonym = _antonym_conflict(sa, sb)
                topical = score >= TOPIC_THRESHOLD and len(shared) >= MIN_SHARED_TERMS
                if not topical:
                    # Unterschiedliche Themen sind kein Widerspruch.
                    continue

                polarity_differs = sa.negated != sb.negated or antonym is not None
                kind = KIND_CONTRADICTION if polarity_differs else KIND_AGREEMENT
                if kind == KIND_CONTRADICTION and contradictions >= MAX_MARKERS_PER_PAIR:
                    continue
                if kind == KIND_AGREEMENT and agreements >= MAX_MARKERS_PER_PAIR:
                    continue

                topic = ", ".join(sorted(shared)[:4])
                note_base = f"Themenbezug: {topic} (Überschneidung {score:.0%})"
                note = (
                    f"{note_base}; Gegensatz: {antonym}"
                    if antonym
                    else (f"{note_base}; gegensätzliche Verneinung" if polarity_differs else note_base)
                )

                markers.append(
                    _marker(
                        session_id=session_id, spark_id=spark_id, sentence=sa,
                        kind=kind, related_job_id=sb.job_id, note=note,
                    )
                )
                markers.append(
                    _marker(
                        session_id=session_id, spark_id=spark_id, sentence=sb,
                        kind=kind, related_job_id=sa.job_id, note=note,
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
                    "agreements": agreements,
                    "contradictions": contradictions,
                    "topics": topics[:6],
                }
            )

    # Einzigartige Aussagen: kein hinreichend ähnlicher Satz in einer anderen Antwort.
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
                    note="Kein vergleichbarer Satz in den anderen Antworten.",
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
        "computed_at": now(),
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
            "Regelbasierter Vergleich auf Satzebene: Themenbezug über "
            "Begriffsüberschneidung, Widerspruch nur bei gegensätzlicher Polarität."
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
