"""
tisch_shared_core/kintegrity_synthesis.py — KiNTEGRiTY-Synthese (gewickelt).

Phase-2-Auftrag: Wenn KiNTEGRiTY im Repo existiert, importieren und WICKELN —
nicht neu schreiben. KiNTEGRiTY existiert als Top-Level-Modul
`der-tisch-backend/kintegrity.py` (Survey, genau eine Stelle).

Robustheit: `kintegrity.py` importiert `anthropic` auf Modulebene und baut
beim Import einen `anthropic.Anthropic`-Client. `anthropic` ist NICHT in
requirements.txt. Deshalb wird KiNTEGRiTY hier DEFENSIV importiert
(try/except über jede Import-Exception) und bei fehlender Verfügbarkeit ODER
fehlgeschlagenem Aufruf auf eine lokale, dependency-freie TL;DR-Verdichtung
zurückgefallen. So bleibt der TiSCH Shared Core auch ohne `anthropic` /
`ANTHROPIC_API_KEY` lauffähig.

Output: ein `MemoryCandidate` mit
  source_role  = kintegrity_synthesis
  memory_layer = reusable_context
  curation_state = synthesized
— passend als Eingang für den Curator (curator.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from typing import Optional

from . import store
from .models import (
    CurationState,
    MemoryCandidate,
    MemoryLayer,
    ProvenanceStep,
    SourceRole,
    new_urn,
)

# --- Defensiver Import von KiNTEGRiTY -------------------------------------
KINTEGRITY_AVAILABLE = False
_KINTEGRITY_IMPORT_ERROR: Optional[str] = None
try:  # kintegrity.py ist ein Top-Level-Modul neben api_server.py
    from kintegrity import (  # type: ignore
        InputBlock,
        KintegrityRequest,
        kintegrity_synthesize,
    )
    KINTEGRITY_AVAILABLE = True
except Exception as exc:  # ImportError, fehlendes anthropic, Client-Init ...
    _KINTEGRITY_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Lokale TL;DR-Verdichtung — Fallback ohne externe Abhängigkeit
# ---------------------------------------------------------------------------

def local_tldr(text: str, max_sentences: int = 3, max_chars: int = 600) -> str:
    """Einfache, deterministische Verdichtung: erste N Sätze, hart gekappt."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:max_sentences]).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip() + " …"
    return summary or text[:max_chars]


def _fallback_result(text: str, reason: str) -> dict:
    return {
        "synthesis": local_tldr(text),
        "engine": "local_tldr_fallback",
        "note": f"Lokale TL;DR-Verdichtung — KiNTEGRiTY nicht verfügbar ({reason}).",
        "meta": {
            "aber_section": "",
            "questionable": "",
            "redundancies_removed": [],
            "confidence": 0.3,
            "must_keep_honored": True,
            "provenance": [],
            "fallback_reason": reason,
        },
    }


# ---------------------------------------------------------------------------
# Synthese-Aufruf (gewickeltes KiNTEGRiTY mit Fallback)
# ---------------------------------------------------------------------------

async def run_synthesis(
    *,
    text: str,
    question: str = "",
    profile: str = "default_profile",
    integrity_field: str = "",
    source_id: str = "src",
) -> dict:
    """KiNTEGRiTY aufrufen; bei Nichtverfügbarkeit/Fehler: lokaler Fallback.

    Rückgabe-dict: {synthesis, engine, note, meta}.
    """
    if not KINTEGRITY_AVAILABLE:
        return _fallback_result(text, reason=_KINTEGRITY_IMPORT_ERROR or "unavailable")
    try:
        req = KintegrityRequest(
            inputs=[InputBlock(id=source_id, content=text, is_user_authored=False)],
            integrity_field=integrity_field,
            profile=profile,
            lang="de",
            question=question,
        )
        resp = await kintegrity_synthesize(req)
        return {
            "synthesis": resp.synthesis or local_tldr(text),
            "engine": "kintegrity",
            "note": "KiNTEGRiTY-Synthese (gewickeltes Modul kintegrity.py).",
            "meta": {
                "aber_section": resp.aber_section,
                "questionable": resp.questionable,
                "redundancies_removed": list(resp.redundancies_removed),
                "confidence": float(resp.confidence),
                "must_keep_honored": bool(resp.must_keep_honored),
                "provenance": [p.model_dump() for p in resp.provenance],
            },
        }
    except Exception as exc:
        # API-Key fehlt, Netzfehler, JSON-Fehler ... -> nie hart fehlschlagen.
        return _fallback_result(text, reason=f"call failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Synthese eines MemoryCandidate -> synthetisierter MemoryCandidate
# ---------------------------------------------------------------------------

async def synthesize_candidate(
    candidate: MemoryCandidate,
    *,
    profile: str = "default_profile",
    integrity_field: str = "",
    persist: bool = True,
) -> MemoryCandidate:
    """Aus einem Candidate einen synthetisierten Candidate erzeugen.

    Der Quell-Candidate bleibt unverändert (Provenance-Erhalt); das Ergebnis
    ist ein neuer Record mit curation_state=synthesized und einer um den
    Schritt `kintegrity_synthesis` verlängerten provenance_chain.
    """
    result = await run_synthesis(
        text=candidate.content,
        question=candidate.title,
        profile=profile,
        integrity_field=integrity_field,
        source_id=candidate.short_id,
    )

    chain = list(candidate.provenance_chain) + [
        ProvenanceStep(
            step="kintegrity_synthesis",
            origin=result["engine"],
            summary=result["note"],
        )
    ]

    synthesized = MemoryCandidate(
        id=new_urn(candidate.title, prefix="syn"),
        family_line=candidate.family_line,
        source_role=SourceRole.KINTEGRITY_SYNTHESIS,
        memory_layer=MemoryLayer.REUSABLE_CONTEXT,
        curation_state=CurationState.SYNTHESIZED,
        visibility=candidate.visibility,
        moonfingers_use=candidate.moonfingers_use,
        provenance_chain=chain,
        title=f"Synthese — {candidate.title}",
        content=result["synthesis"],
        source_app=candidate.source_app,
        raw_payload={
            "source_candidate_id": candidate.id,
            "synthesis_engine": result["engine"],
            "synthesis": result["meta"],
        },
        notes=result["note"],
    )
    if persist:
        await store.upsert(store.CANDIDATES, synthesized.model_dump(mode="json"))
    return synthesized


# ---------------------------------------------------------------------------
# Smoke-Test / Demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    from .capture import capture_candidate
    from .models import FamilyLine

    print(f"[demo] KINTEGRITY_AVAILABLE = {KINTEGRITY_AVAILABLE}")
    if not KINTEGRITY_AVAILABLE:
        print(f"[demo] (Grund: {_KINTEGRITY_IMPORT_ERROR}) -> lokaler TL;DR-Fallback")

    candidate = await capture_candidate(
        title="Synthese-Demo — TiSCH-Durchlauf",
        content=(
            "Der TiSCH ist eine Agenten-Orchestrierungs-Engine. Acht "
            "Methoden-Perspektiven antworten auf eine Frage. Danach folgt eine "
            "Reibungsanalyse, die Widersprüche sichtbar macht. Zum Schluss "
            "integriert eine Synthese die Perspektiven, ohne Widersprüche "
            "vorschnell zu glätten. Das Ergebnis ist editierbares "
            "Arbeitsmaterial, kein Endurteil."
        ),
        source_role=SourceRole.TISCH_RUN_RESULT,
        family_line=[FamilyLine.DER_TISCH],
        source_app="der-tisch",
        origin="synthesis_demo",
    )
    print(f"[demo] Quell-Candidate: {candidate.id} ({candidate.curation_state})")

    synthesized = await synthesize_candidate(candidate)
    assert synthesized.source_role == SourceRole.KINTEGRITY_SYNTHESIS
    assert synthesized.memory_layer == MemoryLayer.REUSABLE_CONTEXT
    assert synthesized.curation_state == CurationState.SYNTHESIZED
    assert synthesized.raw_payload.get("source_candidate_id") == candidate.id
    print(f"[demo] Synthese-Candidate: {synthesized.id} ({synthesized.curation_state})")
    print(f"[demo]   engine = {synthesized.raw_payload.get('synthesis_engine')}")
    print("[demo] Synthese-Inhalt:")
    print("   " + synthesized.content)
    print("[demo] OK — Synthese erzeugt source_role/memory_layer/state korrekt.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tisch_shared_core.kintegrity_synthesis",
        description="TiSCH Shared Core — KiNTEGRiTY-Synthese (gewickelt)",
    )
    parser.add_argument("--demo", action="store_true", help="Smoke-Test")
    args = parser.parse_args()
    if args.demo:
        asyncio.run(_demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
