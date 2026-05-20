"""tisch_shared_core Pydantic-Schemas (MVP).

ID-Konvention: URN-Form, z.B. tisch_shared_core:run_2026_05_18_a1b2c3
Felder exakt nach TISCH_SHARED_CORE_AUFTRAG_PATCH_2026-05-18.md.
"""
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


def _new_urn() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d")
    return f"tisch_shared_core:run_{ts}_{uuid.uuid4().hex[:6]}"


class SourceRole(str, Enum):
    tisch_run_result = "tisch_run_result"
    chat_excerpt = "chat_excerpt"
    kintegrity_synthesis = "kintegrity_synthesis"
    user_authored_note = "user_authored_note"
    curator_decision = "curator_decision"


class CurationState(str, Enum):
    raw = "raw"
    model_generated = "model_generated"
    candidate = "candidate"
    synthesized = "synthesized"
    reviewed = "reviewed"
    curated = "curated"
    approved_for_reuse = "approved_for_reuse"
    canonical = "canonical"


class MemoryLayer(str, Enum):
    personal_memory = "personal_memory"
    project_memory = "project_memory"
    reusable_context = "reusable_context"
    verified_knowledge = "verified_knowledge"   # nur aus knowledge_shared_core
    poetic_surface = "poetic_surface"           # nur aus knowledge_shared_core


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


class ProvenanceLink(BaseModel):
    parent_id: str
    relation: Literal["derived_from", "synthesized_from", "approved_from", "imported_from"]
    at: datetime


class MemoryCandidate(BaseModel):
    id: str = Field(default_factory=_new_urn)
    core_id: Literal["tisch_shared_core"] = "tisch_shared_core"
    family_line: list[str] = ["der-tisch"]
    source_role: SourceRole
    memory_layer: MemoryLayer
    curation_state: CurationState = CurationState.candidate
    approved_for_reuse: bool = False
    canonical: bool = False
    canonical_approved_by: Optional[str] = None
    canonical_approved_at: Optional[datetime] = None
    visibility: Literal["private", "shared", "public"] = "private"
    provenance_chain: list[ProvenanceLink] = []
    reuse_state: ReuseState = Field(default_factory=ReuseState)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Inhaltliche Felder
    title: str
    content: str
    tags: list[str] = []
    suggested_obsidian_path: Optional[str] = None
    # Optionale Analyse-Metadaten (nur bei tisch_run_result)
    origin_app: Optional[str] = None
    project: Optional[str] = None
    task: Optional[str] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    model: Optional[str] = None


class MemoryCard(MemoryCandidate):
    """Ein MemoryCandidate, der mindestens 'reviewed' erreicht hat."""
    pass


class ContextPackRequest(BaseModel):
    app_id: str
    project: Optional[str] = None
    task: Optional[str] = None
    query: str = ""
    max_tokens: int = 1200
    include: list[str] = [
        "canonical_decisions",
        "approved_for_reuse",
        "stable_answers",
        "project_memory",
        "prompts",
    ]
    exclude: list[str] = ["raw_chats", "unreviewed_candidates"]
    include_private: bool = False


class ContextPack(BaseModel):
    app_id: str
    cards: list[MemoryCard]
    total_found: int
    query: str
