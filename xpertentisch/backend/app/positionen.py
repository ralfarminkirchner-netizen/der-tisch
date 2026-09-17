"""Positionswechsel und Szenarien — zusätzliche Verfahren, keine Wertung.

Erhält parallele Antworten. Optionaler Durchgang mit gegenseitiger Lektüre
hält fest, welche fremden Antworten mitgegeben wurden. Keine automatische
Wertung von Wechsel oder Beibehaltung. Spätere Übernahme fremder Argumente
ist keine unabhängige Erstübereinstimmung.

Stimmenhäufigkeit bleibt beschreibende Zählung — weder Wahrscheinlichkeit
noch Begründungsstärke.
"""

from __future__ import annotations

from typing import Any

from .db import new_id, now


def positionswechsel_protokollieren(
    *,
    aussage_id: str,
    job_id: str,
    vorher: str | None,
    nachher: str | None,
    belege_vorher: list[dict[str, Any]] | None = None,
    belege_nachher: list[dict[str, Any]] | None = None,
    begruendung: str | None = None,
    mitgegebene_antworten: list[str] | None = None,
) -> dict[str, Any]:
    """Dokumentiert Wechsel oder Beibehaltung ohne Wertung."""
    if vorher is None and nachher is None:
        art = "unbekannt"
    elif vorher == nachher:
        art = "beibehaltung"
    elif vorher is None:
        art = "erstposition"
    else:
        art = "wechsel"
    return {
        "id": new_id("pos"),
        "aussage_id": aussage_id,
        "job_id": job_id,
        "art": art,
        "vorher": vorher,
        "nachher": nachher,
        "belege_vorher": list(belege_vorher or []),
        "belege_nachher": list(belege_nachher or []),
        "begruendung": begruendung or "",
        "mitgegebene_antworten": list(mitgegebene_antworten or []),
        "wertung": None,
        "hinweis": (
            "Keine automatische Wertung. Übernahme fremder Argumente nach "
            "gegenseitiger Lektüre ist keine unabhängige Erstübereinstimmung."
        ),
        "created_at": now(),
    }


def szenario_voraussetzungen(
    *,
    folge: str,
    vorschlaege: list[dict[str, Any]],
) -> dict[str, Any]:
    """Von betrachteter Folge zu möglichen Voraussetzungen — nur soweit begründet."""
    eintraege = []
    for v in vorschlaege:
        stufe = v.get("stufe") or "moeglich"
        if stufe not in ("notwendig", "hinreichend", "moeglich"):
            stufe = "moeglich"
        eintraege.append(
            {
                "text": v.get("text") or "",
                "stufe": stufe,
                "begruendung": v.get("begruendung") or "",
                "fehlende_information": bool(v.get("fehlende_information")),
                "pruefbedarf": bool(v.get("pruefbedarf", True)),
                "hypothese": True,
            }
        )
    return {
        "richtung": "folge_zu_voraussetzung",
        "folge": folge,
        "voraussetzungen": eintraege,
        "hinweis": (
            "Notwendig/hinreichend/möglich nur, soweit die Begründung das trägt. "
            "Stimmenhäufigkeit ≠ Wahrscheinlichkeit."
        ),
        "created_at": now(),
    }


def szenario_folgen_aus_annahmen(
    *,
    annahmen: list[str],
    moegliche_folgen: list[dict[str, Any]],
) -> dict[str, Any]:
    """Von benannten Annahmen zu möglichen Folgen."""
    return {
        "richtung": "annahme_zu_folge",
        "annahmen": list(annahmen),
        "folgen": [
            {
                "text": f.get("text") or "",
                "alternative": bool(f.get("alternative")),
                "fehlende_information": bool(f.get("fehlende_information")),
                "pruefbedarf": bool(f.get("pruefbedarf", True)),
                "hypothese": True,
            }
            for f in moegliche_folgen
        ],
        "hinweis": "Hypothesen und Alternativen — keine automatische Wertung.",
        "created_at": now(),
    }
