"""KI-gestützte Aussagenzerlegung — Vorschläge, keine Wahrheitsprüfung.

Wird ausdrücklich ausgelöst. Nutzt ein anderes Modell als die zerlegte Antwort.
Fehlt ein solches Modell, bleibt der Lauf als „nicht ausgeführt“ sichtbar —
kein stiller Ersatz.

Jede Aussage braucht wörtliche Belegstelle mit Zeichenpositionen. Ungültige
Belege gelangen nicht in den gültigen Befundbestand; der Fehler bleibt
nachvollziehbar. Ein korrektes Zitat beweist keine korrekte Interpretation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Awaitable

from .db import new_id, now

STATUS_VORSCHLAG = "vorschlag"
STATUS_UEBERNOMMEN = "uebernommen"
STATUS_VERWORFEN = "verworfen"
STATUS_FEHLER = "fehler"
STATUS_NICHT_AUSGEFUEHRT = "nicht_ausgefuehrt"

SETZUNG_GENANNT = "ausdruecklich_genannt"
SETZUNG_VERMUTET = "modell_vermutet"
SETZUNG_FEHLEND = "fehlende_information"


def beleg_pruefen(text: str, start: int, end: int, quote: str) -> str | None:
    """Gibt None bei gültigem Beleg, sonst einen Fehlergrund."""
    if not isinstance(start, int) or not isinstance(end, int):
        return "Zeichenpositionen müssen ganze Zahlen sein"
    if start < 0 or end > len(text) or start >= end:
        return f"Position {start}-{end} außerhalb des Textes (Länge {len(text)})"
    ausschnitt = text[start:end]
    if ausschnitt != quote:
        # Häufiger Unicode-/Whitespace-Fall: nachvollziehbarer Fehler, kein Eintrag.
        if ausschnitt.replace("\u00a0", " ") == quote.replace("\u00a0", " "):
            return "Beleg weicht nur in Leerzeichen/NBSP ab — nicht übernommen"
        return "Belegstelle stimmt nicht mit dem Antworttext überein"
    return None


def normalisiere_vorschlag(
    *,
    job: dict[str, Any],
    analysis_run_id: str,
    analysemodell: str,
    roh: dict[str, Any],
) -> dict[str, Any]:
    """Prüft einen Roh-Vorschlag und setzt Status."""
    text = job.get("text") or ""
    start = roh.get("start_offset")
    end = roh.get("end_offset")
    quote = roh.get("quote") or ""
    fehler = None
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        start_i, end_i = -1, -1
        fehler = "Zeichenpositionen ungültig"
    if fehler is None:
        fehler = beleg_pruefen(text, start_i, end_i, quote)

    setzung = roh.get("setzung") or SETZUNG_VERMUTET
    if setzung not in (SETZUNG_GENANNT, SETZUNG_VERMUTET, SETZUNG_FEHLEND):
        setzung = SETZUNG_VERMUTET

    return {
        "id": new_id("aus"),
        "job_id": job["id"],
        "job_label": job.get("label"),
        "analysis_run_id": analysis_run_id,
        "analysemodell": analysemodell,
        "aussage": (roh.get("aussage") or "").strip(),
        "quote": quote,
        "start_offset": start_i,
        "end_offset": end_i,
        "bedingungen": list(roh.get("bedingungen") or []),
        "unsicherheiten": list(roh.get("unsicherheiten") or []),
        "setzung": setzung,
        "status": STATUS_FEHLER if fehler else STATUS_VORSCHLAG,
        "fehler": fehler,
        "hinweis": (
            "Korrektes Zitat beweist keine korrekte Interpretation. "
            "Übernahme ist menschliche Entscheidung, kein Wahrheitsnachweis."
        ),
        "created_at": now(),
    }


async def aussagen_zerlegen(
    *,
    job: dict[str, Any],
    analysis_run_id: str,
    analysemodell: str | None,
    antwort_modell: str | None,
    llm_call: Callable[[str, str], Awaitable[str]] | None,
) -> dict[str, Any]:
    """Führt die Zerlegung aus oder meldet „nicht ausgeführt“."""
    if not analysemodell:
        return {
            "status": STATUS_NICHT_AUSGEFUEHRT,
            "grund": "Kein anderes Analysemodell verfügbar — kein stiller Ersatz.",
            "aussagen": [],
            "analysis_run_id": analysis_run_id,
        }
    if antwort_modell and analysemodell == antwort_modell:
        return {
            "status": STATUS_NICHT_AUSGEFUEHRT,
            "grund": (
                "Analysemodell muss von dem der zerlegten Antwort verschieden sein. "
                "Kein stiller Ersatz."
            ),
            "aussagen": [],
            "analysis_run_id": analysis_run_id,
        }
    if llm_call is None:
        return {
            "status": STATUS_NICHT_AUSGEFUEHRT,
            "grund": "Kein LLM-Aufruf angebunden — Zerlegung nicht ausgeführt.",
            "aussagen": [],
            "analysis_run_id": analysis_run_id,
        }

    prompt = (
        "Zerlege die Antwort in einzelne überprüfbare Aussagen. "
        "Antworte nur mit JSON-Array. Jedes Element: "
        '{"aussage":"...","quote":"wörtlicher Ausschnitt","start_offset":0,'
        '"end_offset":10,"bedingungen":[],"unsicherheiten":[],'
        f'"setzung":"{SETZUNG_GENANNT}|{SETZUNG_VERMUTET}|{SETZUNG_FEHLEND}"}}. '
        "Erfinde keine Annahmen nur weil etwas fehlt. "
        "start_offset/end_offset müssen exakt auf quote im Text zeigen.\n\n"
        f"TEXT:\n{job.get('text') or ''}"
    )
    rohtext = await llm_call(analysemodell, prompt)
    try:
        # Ersten JSON-Array-Block suchen.
        m = re.search(r"\[.*\]", rohtext, re.DOTALL)
        daten = json.loads(m.group(0) if m else rohtext)
        if not isinstance(daten, list):
            raise ValueError("kein Array")
    except Exception as exc:  # noqa: BLE001 — Fehler bleibt nachvollziehbar
        return {
            "status": STATUS_FEHLER,
            "grund": f"Antwort des Analysemodells nicht parsebar: {exc}",
            "aussagen": [],
            "analysis_run_id": analysis_run_id,
            "analysemodell": analysemodell,
        }

    aussagen = [
        normalisiere_vorschlag(
            job=job,
            analysis_run_id=analysis_run_id,
            analysemodell=analysemodell,
            roh=item if isinstance(item, dict) else {},
        )
        for item in daten
    ]
    gueltig = [a for a in aussagen if a["status"] == STATUS_VORSCHLAG]
    fehlerhaft = [a for a in aussagen if a["status"] == STATUS_FEHLER]
    return {
        "status": "ausgefuehrt",
        "analysemodell": analysemodell,
        "analysis_run_id": analysis_run_id,
        "aussagen": gueltig,
        "verworfene_vorschlaege": fehlerhaft,
        "hinweis": (
            "Anderes Modell ≠ unabhängiger Wahrheitsprüfer. "
            "Nur gültige Belegstellen im Befundbestand."
        ),
    }
