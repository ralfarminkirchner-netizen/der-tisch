"""Zahlen-Linse: Extraktion mit Originalschreibweise und vorsichtigem Vergleich.

Kein stilles Umrechnen, keine erfundenen Bezugsgrößen. Gleiche Einheit allein
begründet keine gleiche Bezugsgröße. Statusstufen:

- gefunden
- vergleich_vorgeschlagen
- zuordnung_gesichert
- abweichend
- unklar
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .db import JOB_DONE, new_id, now

#: Explizite, getestete Umrechnungen — nur diese, keine stillen Annahmen.
UMRECHNUNGEN: dict[tuple[str, str], float] = {
    ("m", "cm"): 100.0,
    ("cm", "m"): 0.01,
    ("m", "mm"): 1000.0,
    ("mm", "m"): 0.001,
    ("cm", "mm"): 10.0,
    ("mm", "cm"): 0.1,
}

EINHEIT_ALIASE = {
    "meter": "m",
    "metre": "m",
    "metres": "m",
    "meters": "m",
    "zentimeter": "cm",
    "centimeter": "cm",
    "centimetre": "cm",
    "millimeter": "mm",
    "millimetre": "mm",
}

WORTZAHLEN = {
    "null": 0,
    "eins": 1,
    "eine": 1,
    "einem": 1,
    "einen": 1,
    "einer": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
    "elf": 11,
    "zwölf": 12,
    "dreizehn": 13,
    "vierzehn": 14,
    "fünfzehn": 15,
    "achtzehn": 18,
    "zwanzig": 20,
    "dreißig": 30,
    "dreissig": 30,
    "zweiundvierzig": 42,
}

# Komma oder Punkt als Dezimaltrenner nur, wenn eindeutig (eine Trennung).
ZAHL_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<raw>\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
    r"(?:\s*(?P<unit>mm|cm|m|km|kg|g|%|€|EUR|USD|\$))?"
    r"(?![\w.])",
    re.IGNORECASE,
)

WORTZAHL_RE = re.compile(
    r"(?<!\w)(?P<raw>" + "|".join(sorted(WORTZAHLEN.keys(), key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)

BEZUG_RE = re.compile(
    r"(?P<bezug>Bohrungen?|Vortrieben?|Rohre?|DN|Durchmesser|Länge|Tiefe|Radius|"
    r"Toleranz|Deckung|Freigaberadius|Sondierradius)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ZahlFund:
    job_id: str
    raw: str
    value: float | None
    unit: str | None
    bezug: str | None
    start: int
    end: int
    context: str
    parse_status: str  # ok | unklar_trennzeichen | unklar


def _norm_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.strip().lower()
    return EINHEIT_ALIASE.get(u, u)


def _parse_number(raw: str) -> tuple[float | None, str]:
    """Parst ohne stille Tausender-/Dezimalannahmen bei Mehrdeutigkeit."""
    s = raw.strip().replace(" ", "")
    if not s:
        return None, "unklar"
    # Beide Trenner vorhanden → Rollen müssen klar sein.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # 1.234,56
            try:
                return float(s.replace(".", "").replace(",", ".")), "ok"
            except ValueError:
                return None, "unklar_trennzeichen"
        # 1,234.56
        try:
            return float(s.replace(",", "")), "ok"
        except ValueError:
            return None, "unklar_trennzeichen"
    if "," in s:
        links, _, rechts = s.partition(",")
        if len(rechts) == 3 and links.isdigit() and rechts.isdigit():
            # Mehrdeutig: 2,800 könnte 2.8 oder 2800 sein → unklar
            return None, "unklar_trennzeichen"
        try:
            return float(s.replace(",", ".")), "ok"
        except ValueError:
            return None, "unklar"
    if "." in s:
        links, _, rechts = s.partition(".")
        if len(rechts) == 3 and links.isdigit() and rechts.isdigit() and s.count(".") == 1:
            # Mehrdeutig: 2.800
            return None, "unklar_trennzeichen"
        try:
            return float(s), "ok"
        except ValueError:
            return None, "unklar"
    try:
        return float(s), "ok"
    except ValueError:
        return None, "unklar"


def _kontext(text: str, start: int, end: int, radius: int = 40) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b].replace("\n", " ").strip()


def _bezug_nahe(text: str, start: int, end: int) -> str | None:
    fenster = text[max(0, start - 48) : min(len(text), end + 48)]
    m = BEZUG_RE.search(fenster)
    return m.group("bezug") if m else None


def extrahiere_zahlen(job_id: str, text: str) -> list[ZahlFund]:
    funde: list[ZahlFund] = []
    text = text or ""
    for m in ZAHL_RE.finditer(text):
        raw = m.group("raw")
        value, status = _parse_number(raw)
        unit = _norm_unit(m.group("unit"))
        funde.append(
            ZahlFund(
                job_id=job_id,
                raw=raw,
                value=value,
                unit=unit,
                bezug=_bezug_nahe(text, m.start(), m.end()),
                start=m.start(),
                end=m.end(),
                context=_kontext(text, m.start(), m.end()),
                parse_status=status,
            )
        )
    # Wortzahlen („drei Bohrungen“) — unterscheidbar von „drei Vortrieben“.
    belegt = {(f.start, f.end) for f in funde}
    for m in WORTZAHL_RE.finditer(text):
        span = (m.start(), m.end())
        if any(not (span[1] <= a or span[0] >= b) for a, b in belegt):
            continue
        raw = m.group("raw")
        value = float(WORTZAHLEN[raw.lower()])
        funde.append(
            ZahlFund(
                job_id=job_id,
                raw=raw,
                value=value,
                unit=None,
                bezug=_bezug_nahe(text, m.start(), m.end()),
                start=m.start(),
                end=m.end(),
                context=_kontext(text, m.start(), m.end()),
                parse_status="ok",
            )
        )
    funde.sort(key=lambda f: f.start)
    return funde


def umrechnen(value: float, von: str, nach: str) -> float | None:
    von_n, nach_n = _norm_unit(von), _norm_unit(nach)
    if von_n is None or nach_n is None:
        return None
    if von_n == nach_n:
        return value
    faktor = UMRECHNUNGEN.get((von_n, nach_n))
    if faktor is None:
        return None
    return value * faktor


def _werte_aequivalenz(a: ZahlFund, b: ZahlFund) -> str:
    """Status eines Paarvergleichs — ohne erfundene Bezugsgröße."""
    if a.parse_status != "ok" or b.parse_status != "ok" or a.value is None or b.value is None:
        return "unklar"
    if (a.bezug or "").lower() != (b.bezug or "").lower():
        # Gleiche Einheit allein reicht nicht.
        if a.unit and b.unit and _norm_unit(a.unit) == _norm_unit(b.unit) and not a.bezug and not b.bezug:
            return "vergleich_vorgeschlagen"
        if a.bezug and b.bezug and (a.bezug or "").lower() != (b.bezug or "").lower():
            return "unklar"
        return "vergleich_vorgeschlagen"

    if a.unit and b.unit:
        ua, ub = _norm_unit(a.unit), _norm_unit(b.unit)
        if ua == ub:
            if abs(a.value - b.value) < 1e-9:
                return "zuordnung_gesichert" if a.bezug else "vergleich_vorgeschlagen"
            return "abweichend" if a.bezug else "vergleich_vorgeschlagen"
        um = umrechnen(a.value, ua or "", ub or "")
        if um is None:
            return "unklar"
        if abs(um - b.value) < 1e-6:
            return "zuordnung_gesichert" if a.bezug else "vergleich_vorgeschlagen"
        return "abweichend" if a.bezug else "vergleich_vorgeschlagen"

    if a.unit is None and b.unit is None:
        if abs(a.value - b.value) < 1e-9:
            return "vergleich_vorgeschlagen"
        return "vergleich_vorgeschlagen"
    return "unklar"


def zahlen_analyse(jobs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Extrahiert Zahlen und schlägt vorsichtige Vergleiche vor."""
    usable = [j for j in jobs if j.get("status") == JOB_DONE and (j.get("text") or "").strip()]
    funde: list[dict[str, Any]] = []
    for job in usable:
        for f in extrahiere_zahlen(job["id"], job["text"]):
            funde.append(
                {
                    "id": new_id("zal"),
                    "job_id": f.job_id,
                    "label": job.get("label"),
                    "raw": f.raw,
                    "value": f.value,
                    "unit": f.unit,
                    "bezug": f.bezug,
                    "start_offset": f.start,
                    "end_offset": f.end,
                    "context": f.context,
                    "parse_status": f.parse_status,
                    "status": "gefunden",
                }
            )

    vergleiche: list[dict[str, Any]] = []
    for i, a in enumerate(funde):
        for b in funde[i + 1 :]:
            if a["job_id"] == b["job_id"]:
                continue
            fa = ZahlFund(
                job_id=a["job_id"],
                raw=a["raw"],
                value=a["value"],
                unit=a["unit"],
                bezug=a["bezug"],
                start=a["start_offset"],
                end=a["end_offset"],
                context=a["context"],
                parse_status=a["parse_status"],
            )
            fb = ZahlFund(
                job_id=b["job_id"],
                raw=b["raw"],
                value=b["value"],
                unit=b["unit"],
                bezug=b["bezug"],
                start=b["start_offset"],
                end=b["end_offset"],
                context=b["context"],
                parse_status=b["parse_status"],
            )
            status = _werte_aequivalenz(fa, fb)
            if status == "unklar" and (fa.unit != fb.unit or fa.bezug != fb.bezug):
                # Nur sinnvolle Paare melden.
                if not (fa.unit and fb.unit) and not (fa.bezug and fb.bezug):
                    continue
            vergleiche.append(
                {
                    "a_id": a["id"],
                    "b_id": b["id"],
                    "a_job_id": a["job_id"],
                    "b_job_id": b["job_id"],
                    "status": status,
                    "hinweis": (
                        "Gleiche Einheit allein begründet keine gleiche Bezugsgröße."
                        if (fa.unit and fb.unit and fa.bezug != fb.bezug)
                        else ""
                    ),
                }
            )

    return {
        "computed_at": now(),
        "funde": funde,
        "vergleiche": vergleiche,
        "methode": (
            "Zahlenextraktion mit Originalschreibweise; Umrechnung nur über "
            "explizite Regeln; keine stillen Trennzeichen-Annahmen."
        ),
        "claim_level": "hinweis",
    }
