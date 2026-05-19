"""
tisch_memory/schemas.py — Pydantic-Schemas für den TiSCH Shared Core.

Verbindliche Pflichtfelder laut TISCH_SHARED_CORE_AUFTRAG_PATCH_2026-05-18.md.
Alle Cross-Core-IDs als URN: tisch_shared_core:<slug>
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
import uuid

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums — nur kanonische Werte, keine Erfindungen
# ---------------------------------------------------------------------------

class SourceRole(str, Enum):
    tisch_run_result       = "tisch_run_result"
    chat_excerpt           = "chat_excerpt"
    kintegrity_synthesis   = "kintegrity_synthesis"
    user_authored_note     = "user_authored_note"
    curator_decision       = "curator_decision"


class MemoryLayer(str, Enum):
    personal_memory    = "personal_memory"
    project_memory     = "project_memory"
    reusable_context   = "reusable_context"
    verified_knowledge = "verified_knowledge"   # nur aus knowledge_shared_core
    poetic_surface     = "poetic_surface"        # nur aus knowledge_shared_core


class CurationState(str, Enum):
    raw                = "raw"
    model_generated    = "model_generated"
    candidate          = "candidate"
    synthesized        = "synthesized"
    reviewed           = "reviewed"
    curated            = "curated"
    approved_for_reuse = "approved_for_reuse"
    canonical          = "canonical"

# Welcher curation_state ist für Export/Context-Pack freigegeben?
REUSE_ELIGIBLE_STATES = {
    CurationState.reviewed,
    CurationState.curated,
    CurationState.approved_for_reuse,
    CurationState.canonical,
}


# ---------------------------------------------------------------------------
# Hilfs-Modelle
# ---------------------------------------------------------------------------

class ProvenanceLink(BaseModel):
    parent_id: str       # URN
    relation: Literal["derived_from", "synthesized_from", "approved_from", "imported_from"]
    at: datetime


class ReuseState(BaseModel):
    used_in_context_packs: int = 0
    last_used_at: Optional[datetime] = None
    moonfingers_use: list[Literal[
        "personal_overlay",
        "context_pack",
        "entity_dossier",
        "globe_node",
        "story_step",
        "fog_hint",
        "relation",
        "image",
    ]] = []


# ---------------------------------------------------------------------------
# MemoryCandidate — eingehende TiSCH-Runs, Chat-Excerpts, etc.
# ---------------------------------------------------------------------------

def _new_urn() -> str:
    slug = uuid.uuid4().hex[:12]
    return f"tisch_shared_core:{slug}"


class MemoryCandidate(BaseModel):
    id: str = Field(default_factory=_new_urn)
    core_id: Literal["tisch_shared_core"] = "tisch_shared_core"
    family_line: list[str] = Field(default_factory=list)
    source_role: SourceRole
    memory_layer: MemoryLayer
    curation_state: CurationState = CurationState.candidate
    approved_for_reuse: bool = False
    canonical: bool = False
    canonical_approved_by: Optional[str] = None
    canonical_approved_at: Optional[datetime] = None
    visibility: Literal["private", "shared", "public"] = "private"
    provenance_chain: list[ProvenanceLink] = Field(default_factory=list)
    reuse_state: ReuseState = Field(default_factory=ReuseState)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Inhaltliche Felder
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    suggested_obsidian_path: Optional[str] = None
    # Herkunfts-Kontext
    origin_app: Optional[str] = None
    project: Optional[str] = None
    task: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# MemoryCard — kuratierte, stabiler als Candidate
# ---------------------------------------------------------------------------

class MemoryCard(MemoryCandidate):
    """Identisch zu MemoryCandidate, aber nur mit curation_state >= reviewed."""
    pass


# ---------------------------------------------------------------------------
# ContextPack — Antwort auf Pre-Run-Anfrage einer TiSCH-App
# ---------------------------------------------------------------------------

class ContextPackRequest(BaseModel):
    app_id: str
    project: Optional[str] = None
    task: Optional[str] = None
    query: str
    max_tokens: int = 1200
    include: list[str] = Field(default_factory=lambda: [
        "canonical_decisions",
        "approved_for_reuse",
        "stable_answers",
        "project_memory",
    ])
    exclude: list[str] = Field(default_factory=lambda: ["raw_chats", "unreviewed_candidates"])
    include_private: bool = False


class ContextPackCard(BaseModel):
    id: str
    title: str
    content: str
    tags: list[str]
    curation_state: CurationState
    source_role: SourceRole
    origin_app: Optional[str]
    created_at: datetime
    score: float = 0.0


class ContextPackResponse(BaseModel):
    app_id: str
    query: str
    cards: list[ContextPackCard]
    total_chars: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
