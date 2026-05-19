"""
tisch_shared_core/capture.py — Raw Capture.

Nimmt rohe TiSCH-Run-Outputs / Chat-Excerpts entgegen und erzeugt daraus
`MemoryCandidate`s mit curation_state `raw` oder `candidate`. Persistierung
über store.py (SQLite via aiosqlite).

Smoke-Test:  python -m tisch_shared_core.capture --demo
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Dict, List, Optional, Union

from . import store
from .models import (
    CurationState,
    FamilyLine,
    MemoryCandidate,
    MemoryLayer,
    MoonfingersUse,
    ProvenanceStep,
    SourceRole,
    Visibility,
    content_fingerprint,
    new_urn,
)

# ---------------------------------------------------------------------------
# Default-Mapping je source_role — abgeleitet aus dem Vertrag
# ("Mapping bestehender Quellen" / Implementation Defaults Punkt 4).
# ---------------------------------------------------------------------------
_CAPTURE_DEFAULTS: Dict[SourceRole, dict] = {
    SourceRole.TISCH_RUN_RESULT: {
        "memory_layer": MemoryLayer.PROJECT_MEMORY,
        "curation_state": CurationState.CANDIDATE,
        "moonfingers_use": [MoonfingersUse.PERSONAL_OVERLAY, MoonfingersUse.CONTEXT_PACK],
    },
    SourceRole.CHAT_EXCERPT: {
        "memory_layer": MemoryLayer.PERSONAL_MEMORY,
        "curation_state": CurationState.RAW,
        "moonfingers_use": [MoonfingersUse.PERSONAL_OVERLAY, MoonfingersUse.CONTEXT_PACK],
    },
    SourceRole.PROMPT: {
        "memory_layer": MemoryLayer.REUSABLE_CONTEXT,
        "curation_state": CurationState.RAW,
        "moonfingers_use": [MoonfingersUse.CONTEXT_PACK],
    },
    SourceRole.USER_AUTHORED_NOTE: {
        "memory_layer": MemoryLayer.PERSONAL_MEMORY,
        "curation_state": CurationState.CANDIDATE,
        "moonfingers_use": [MoonfingersUse.PERSONAL_OVERLAY],
    },
}

_FALLBACK_DEFAULTS = {
    "memory_layer": MemoryLayer.PERSONAL_MEMORY,
    "curation_state": CurationState.RAW,
    "moonfingers_use": [MoonfingersUse.PERSONAL_OVERLAY, MoonfingersUse.CONTEXT_PACK],
}


def capture_defaults_for(source_role: SourceRole) -> dict:
    """Sinnvolle Provenance-Defaults für eine source_role (überschreibbar)."""
    return dict(_CAPTURE_DEFAULTS.get(source_role, _FALLBACK_DEFAULTS))


def build_candidate(
    *,
    title: str,
    content: str,
    source_role: Union[SourceRole, str] = SourceRole.CHAT_EXCERPT,
    family_line: Optional[List[Union[FamilyLine, str]]] = None,
    source_app: str = "",
    visibility: Union[Visibility, str] = Visibility.PRIVATE,
    memory_layer: Optional[Union[MemoryLayer, str]] = None,
    curation_state: Optional[Union[CurationState, str]] = None,
    moonfingers_use: Optional[List[Union[MoonfingersUse, str]]] = None,
    raw_payload: Optional[dict] = None,
    notes: str = "",
    origin: str = "",
) -> MemoryCandidate:
    """Einen `MemoryCandidate` aus rohen Eingaben bauen (ohne Persistierung).

    Nicht gesetzte Provenance-Felder werden aus `capture_defaults_for` gefüllt.
    """
    if isinstance(source_role, str):
        source_role = SourceRole(source_role)
    defaults = capture_defaults_for(source_role)

    if memory_layer is None:
        memory_layer = defaults["memory_layer"]
    if curation_state is None:
        curation_state = defaults["curation_state"]
    if moonfingers_use is None:
        moonfingers_use = list(defaults["moonfingers_use"])
    if not family_line:
        family_line = [FamilyLine.DER_TISCH]

    chain = [
        ProvenanceStep(
            step="raw_capture",
            origin=origin or source_app or "der-tisch",
            summary=f"Captured as {source_role.value} ({curation_state})",
        )
    ]

    return MemoryCandidate(
        id=new_urn(title),
        family_line=family_line,
        source_role=source_role,
        memory_layer=memory_layer,
        curation_state=curation_state,
        visibility=visibility,
        moonfingers_use=moonfingers_use,
        provenance_chain=chain,
        title=title,
        content=content,
        source_app=source_app,
        raw_payload=raw_payload or {},
        notes=notes,
        content_hash=content_fingerprint(content),
    )


async def find_candidate_by_hash(content_hash: str) -> Optional[MemoryCandidate]:
    """Bestehenden Candidate mit gleichem content_hash finden (Dedupe)."""
    matches = await store.find(
        store.CANDIDATES, lambda r: r.get("content_hash") == content_hash
    )
    return MemoryCandidate(**matches[0]) if matches else None


async def capture_candidate(
    *, dedupe_on_hash: bool = True, **kwargs
) -> MemoryCandidate:
    """Einen `MemoryCandidate` bauen UND persistieren.

    Bei `dedupe_on_hash` wird ein bereits vorhandener inhaltsgleicher Candidate
    zurückgegeben statt ein Duplikat zu schreiben (idempotentes Capture).
    """
    candidate = build_candidate(**kwargs)
    if dedupe_on_hash:
        existing = await find_candidate_by_hash(candidate.content_hash)
        if existing is not None:
            return existing
    await store.upsert(store.CANDIDATES, candidate.model_dump(mode="json"))
    return candidate


async def get_candidate(candidate_id: str) -> Optional[MemoryCandidate]:
    """Einen Candidate per URN-id zurücklesen."""
    rec = await store.get(store.CANDIDATES, candidate_id)
    return MemoryCandidate(**rec) if rec else None


async def list_candidates(
    *, curation_state: Optional[CurationState] = None, limit: int = 200
) -> List[MemoryCandidate]:
    """Candidates auflisten, optional nach curation_state gefiltert."""
    records = await store.all_records(store.CANDIDATES)
    out: List[MemoryCandidate] = []
    for rec in records:
        cand = MemoryCandidate(**rec)
        if curation_state is not None and cand.curation_state != curation_state:
            continue
        out.append(cand)
    return out[:limit]


# ---------------------------------------------------------------------------
# Smoke-Test / Demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    print("[demo] capture: erzeuge Demo-MemoryCandidate ...")
    candidate = await capture_candidate(
        title="Demo — Dual-Core Capture",
        content=(
            "MOONFiNGERS ist die poetisch-visuelle Dual-Core-Oberfläche über "
            "zwei separate, föderierte Shared Cores. Der TiSCH-Core hält "
            "persönliches Arbeits- und Antwortgedächtnis."
        ),
        source_role=SourceRole.CHAT_EXCERPT,
        family_line=[FamilyLine.DER_TISCH, FamilyLine.MOONFINGERS],
        source_app="der-tisch",
        origin="capture_demo",
    )
    print(f"[demo] erzeugt: {candidate.id}")
    print(f"[demo]   curation_state = {candidate.curation_state}")
    print(f"[demo]   content_hash   = {candidate.content_hash[:16]}...")

    read_back = await get_candidate(candidate.id)
    assert read_back is not None, "FEHLER: read-back ergab None"
    assert read_back.id == candidate.id, "FEHLER: id stimmt nicht"
    assert read_back.content_hash == candidate.content_hash, "FEHLER: hash stimmt nicht"
    assert read_back.core_id == "tisch_shared_core", "FEHLER: core_id falsch"
    print("[demo] read-back OK — Modell + Provenance konsistent.")

    again = await capture_candidate(
        title="Demo — Dual-Core Capture",
        content=(
            "MOONFiNGERS ist die poetisch-visuelle Dual-Core-Oberfläche über "
            "zwei separate, föderierte Shared Cores. Der TiSCH-Core hält "
            "persönliches Arbeits- und Antwortgedächtnis."
        ),
        source_role=SourceRole.CHAT_EXCERPT,
        source_app="der-tisch",
    )
    assert again.id == candidate.id, "FEHLER: Dedupe schrieb ein Duplikat"
    print("[demo] dedupe OK — inhaltsgleiches Capture ist idempotent.")

    print("[demo] --- MemoryCandidate (JSON) ---")
    print(json.dumps(read_back.model_dump(mode="json"), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tisch_shared_core.capture",
        description="TiSCH Shared Core — Raw Capture",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Demo-Candidate erzeugen und zurücklesen (Smoke-Test)",
    )
    args = parser.parse_args()
    if args.demo:
        asyncio.run(_demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
