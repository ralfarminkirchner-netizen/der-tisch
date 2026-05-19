"""
tisch_shared_core/api.py — FastAPI-Router des TiSCH Shared Core.

Endpunkte (Vertrag „API-Gruppen A"):
  POST /api/tisch-memory/candidates       Candidate anlegen
  GET  /api/tisch-memory/search           stabile Antworten / Cards suchen
  POST /api/tisch-memory/context-pack     Context Pack bauen
  GET  /api/tisch-memory/cards/{card_id}  eine MemoryCard lesen
  POST /api/tisch-memory/curate           kuratieren / canonical-Promotion
  GET  /api/tisch-memory/obsidian/export  Cards als Obsidian-Markdown

Einbindung: api_server.py bindet `router` über genau eine include_router-Zeile
ein. Der Router setzt seine vollen Pfade selbst (prefix="").

Standalone-Smoke-Test (nur dieser Router, ohne die übrige TiSCH-API):
    python -m tisch_shared_core.api --port 8000
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

from . import capture, store
from .context_packs import build_context_pack, get_context_pack
from .curator import (
    CanonicalGuardError,
    curate_candidate,
    get_card,
    promote_to_canonical,
)
from .models import (
    URN_PREFIX,
    ContextPack,
    CurationState,
    MemoryCandidate,
    MemoryCard,
)
from .obsidian_export import export_all_cards, export_card, vault_path
from .stable_answers import StableAnswerHit, find_stable_answers

router = APIRouter(tags=["tisch-shared-core"])


def _as_urn(record_id: str) -> str:
    """Kurz-ID zu URN-Form ergänzen (Innenform -> Außenform)."""
    record_id = (record_id or "").strip()
    if record_id and not record_id.startswith(URN_PREFIX):
        return URN_PREFIX + record_id
    return record_id


# ===========================================================================
# Request-Modelle
# ===========================================================================

class CandidateCreateRequest(BaseModel):
    title: str
    content: str
    source_role: str = "chat_excerpt"
    family_line: List[str] = Field(default_factory=lambda: ["der-tisch"])
    source_app: str = ""
    visibility: str = "private"
    memory_layer: Optional[str] = None
    curation_state: Optional[str] = None
    moonfingers_use: Optional[List[str]] = None
    raw_payload: dict = Field(default_factory=dict)
    notes: str = ""
    origin: str = ""


class ContextPackRequest(BaseModel):
    task_or_question: str
    top_k: int = 8
    max_tokens: int = 2000
    title: str = ""


class CurateRequest(BaseModel):
    target_state: str                       # reviewed | curated_draft | canonical
    candidate_id: Optional[str] = None       # für reviewed / curated_draft
    card_id: Optional[str] = None            # für canonical
    approved_by: Optional[str] = None        # Pflicht bei canonical
    approved_for_reuse: bool = True


# ===========================================================================
# Endpunkte
# ===========================================================================

@router.post(
    "/api/tisch-memory/candidates",
    status_code=201,
    response_model=MemoryCandidate,
    summary="Roh-Capture: einen MemoryCandidate anlegen",
)
async def create_candidate(req: CandidateCreateRequest) -> MemoryCandidate:
    """Erzeugt einen `MemoryCandidate` (curation_state raw|candidate)."""
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="title und content sind Pflicht.")
    try:
        return await capture.capture_candidate(
            title=req.title,
            content=req.content,
            source_role=req.source_role,
            family_line=req.family_line,
            source_app=req.source_app,
            visibility=req.visibility,
            memory_layer=req.memory_layer,
            curation_state=req.curation_state,
            moonfingers_use=req.moonfingers_use,
            raw_payload=req.raw_payload,
            notes=req.notes,
            origin=req.origin or "api",
        )
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Ungültige Capture-Eingabe: {exc}")


@router.get(
    "/api/tisch-memory/search",
    response_model=List[StableAnswerHit],
    summary="Stabile Antworten / Cards suchen",
)
async def search(
    q: str = Query(..., description="Suchanfrage"),
    top_k: int = Query(5, ge=1, le=50),
    include_all: bool = Query(
        False, description="True = über ALLE Cards suchen, nicht nur stabile"
    ),
) -> List[StableAnswerHit]:
    """Liefert nach Relevanz sortierte Treffer.

    Default: nur stabile Cards (approved_for_reuse oder canonical).
    """
    return await find_stable_answers(q, top_k=top_k, include_all=include_all)


@router.post(
    "/api/tisch-memory/context-pack",
    response_model=ContextPack,
    summary="Context Pack aus stabilen Cards bauen",
)
async def context_pack(req: ContextPackRequest) -> ContextPack:
    """Baut einen kompakten, provenance-erhaltenden Context Pack."""
    if not req.task_or_question.strip():
        raise HTTPException(status_code=400, detail="task_or_question ist Pflicht.")
    return await build_context_pack(
        req.task_or_question,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
        title=req.title,
    )


@router.get(
    "/api/tisch-memory/cards/{card_id}",
    response_model=MemoryCard,
    summary="Eine MemoryCard per id lesen",
)
async def read_card(card_id: str) -> MemoryCard:
    """Liefert eine `MemoryCard`. Akzeptiert URN-Form oder Kurz-ID."""
    card = await get_card(_as_urn(card_id))
    if card is None:
        raise HTTPException(status_code=404, detail=f"MemoryCard nicht gefunden: {card_id}")
    return card


@router.post(
    "/api/tisch-memory/curate",
    response_model=MemoryCard,
    summary="Candidate kuratieren oder Card zu canonical befördern",
)
async def curate(req: CurateRequest) -> MemoryCard:
    """Kurations-Endpunkt.

    - target_state `reviewed`/`curated_draft`: braucht `candidate_id` —
      autonome Kuration zu einer MemoryCard.
    - target_state `canonical`: braucht `card_id` + `approved_by` —
      schließt das Approval-Tor. Autonom ist `canonical` verboten (403).
    """
    try:
        target = CurationState(req.target_state)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Unbekannter target_state: {req.target_state!r}"
        )

    if target == CurationState.CANONICAL:
        if not req.card_id:
            raise HTTPException(
                status_code=400, detail="card_id ist Pflicht für target_state=canonical."
            )
        card = await get_card(_as_urn(req.card_id))
        if card is None:
            raise HTTPException(
                status_code=404, detail=f"MemoryCard nicht gefunden: {req.card_id}"
            )
        try:
            return await promote_to_canonical(
                _as_urn(req.card_id), approved_by=req.approved_by or ""
            )
        except ValueError as exc:
            # leeres approved_by -> Approval-Tor nicht erfüllt
            raise HTTPException(status_code=400, detail=str(exc))

    if target in (CurationState.REVIEWED, CurationState.CURATED_DRAFT):
        if not req.candidate_id:
            raise HTTPException(
                status_code=400,
                detail="candidate_id ist Pflicht für target_state reviewed/curated_draft.",
            )
        candidate = await capture.get_candidate(_as_urn(req.candidate_id))
        if candidate is None:
            raise HTTPException(
                status_code=404, detail=f"Candidate nicht gefunden: {req.candidate_id}"
            )
        try:
            return await curate_candidate(
                candidate,
                target_state=target,
                approved_for_reuse=req.approved_for_reuse,
            )
        except CanonicalGuardError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    raise HTTPException(
        status_code=400,
        detail=(
            "target_state muss reviewed, curated_draft oder canonical sein "
            f"(erhalten: {req.target_state!r})."
        ),
    )


@router.get(
    "/api/tisch-memory/obsidian/export",
    summary="MemoryCards als Obsidian-Markdown exportieren",
)
async def obsidian_export(
    card_id: Optional[str] = Query(None, description="Nur diese Card; sonst alle"),
    curation_state: Optional[str] = Query(None, description="Filter (nur ohne card_id)"),
    write: bool = Query(False, description="True = zusätzlich in den Vault schreiben"),
) -> dict:
    """Rendert Cards als Markdown + Frontmatter.

    Schreibt nur bei `write=true` UND gesetztem `OBSIDIAN_VAULT_PATH` —
    sonst No-op (Railway-Sicherheitsregel). Das gerenderte Markdown wird
    immer zurückgegeben.
    """
    if card_id:
        card = await get_card(_as_urn(card_id))
        if card is None:
            raise HTTPException(status_code=404, detail=f"MemoryCard nicht gefunden: {card_id}")
        exports = [export_card(card, write=write)]
    else:
        state_filter = None
        if curation_state:
            try:
                state_filter = CurationState(curation_state)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Unbekannter curation_state: {curation_state!r}"
                )
        exports = await export_all_cards(curation_state=state_filter, write=write)

    return {
        "vault_configured": vault_path() is not None,
        "written": bool(write and vault_path() is not None),
        "count": len(exports),
        "exports": [e.model_dump(mode="json") for e in exports],
    }


# ===========================================================================
# Standalone-Smoke-Server (nur dieser Router)
# ===========================================================================

def build_standalone_app():
    """Minimale FastAPI-App nur mit diesem Router — für den Smoke-Test."""
    from fastapi import FastAPI

    app = FastAPI(title="TiSCH Shared Core — standalone")
    app.include_router(router, prefix="")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tisch_shared_core.api",
        description="TiSCH Shared Core — Standalone-Router-Server",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_standalone_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
