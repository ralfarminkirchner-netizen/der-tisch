"""
tisch_shared_core/models.py — Pydantic-Modelle für den TiSCH Shared Core.

Kanonischer Vertrag: HANDOFF_dual_shared_cores_2026-05-18.md.

Alle Records tragen das volle Provenance-Protokoll (Vertrag, Pflicht in beiden
Cores): core_id, family_line, source_role, memory_layer, curation_state,
visibility, moonfingers_use, provenance_chain, optional reuse_state.

- core_id ist fix `tisch_shared_core` (per Literal erzwungen).
- IDs in URN-Form `tisch_shared_core:<short_id>` (per Validator erzwungen).

Hinweis zu `curation_state`: Die Vertrags-Tabelle listet `curated`, die
Vertrags-Prosa (Kurations-Pipeline / Phase 2) spricht zusätzlich von
`curated_draft` als autonom erreichbarem Zustand. Beide Werte sind hier
aufgenommen — die Diskrepanz ist im Phasen-Report als offene Vertrags-Frage
für die Bridge/Knowledge-Seite vermerkt.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

try:  # Python 3.8+: typing.Literal
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

CORE_ID = "tisch_shared_core"
URN_PREFIX = f"{CORE_ID}:"


# ===========================================================================
# Enums — Provenance-Protokoll
# ===========================================================================

class FamilyLine(str, Enum):
    EIN_SEIN = "ein-sein"
    HEIL_SEIN = "heil-sein"
    DEIN_SEIN = "dein-sein"
    IBW = "ibw"
    SPIRAL_MIND_WIKI = "spiral-mind-wiki"
    DER_TISCH = "der-tisch"
    TEAM_TISCH = "team-tisch"
    LITERATEN_TISCH = "literaten-tisch"
    EXPERTISEN_TISCH = "expertisen-tisch"
    INTEGRATIONS_TISCH = "integrations-tisch"
    MOONFINGERS = "moonfingers"
    MINDLAXY = "mindlaxy"


class SourceRole(str, Enum):
    WIKI_MONOGRAPH = "wiki_monograph"
    VERIFIED_RESEARCH = "verified_research"
    IMAGE_ASSET = "image_asset"
    ATLAS_ENTRY = "atlas_entry"
    STORY_SEQUENCE = "story_sequence"
    DOSSIER = "dossier"
    ENTITY = "entity"
    RELATION = "relation"
    TISCH_RUN_RESULT = "tisch_run_result"
    CHAT_EXCERPT = "chat_excerpt"
    MODEL_GENERATED_SYNTHESIS = "model_generated_synthesis"
    KINTEGRITY_SYNTHESIS = "kintegrity_synthesis"
    USER_AUTHORED_NOTE = "user_authored_note"
    CURATOR_DECISION = "curator_decision"
    STABLE_ANSWER = "stable_answer"
    PROMPT = "prompt"


class MemoryLayer(str, Enum):
    VERIFIED_KNOWLEDGE = "verified_knowledge"
    PERSONAL_MEMORY = "personal_memory"
    PROJECT_MEMORY = "project_memory"
    REUSABLE_CONTEXT = "reusable_context"
    POETIC_SURFACE = "poetic_surface"
    ARCHIVE = "archive"


class CurationState(str, Enum):
    RAW = "raw"
    IMPORTED = "imported"
    CANDIDATE = "candidate"
    SYNTHESIZED = "synthesized"
    REVIEWED = "reviewed"
    CURATED_DRAFT = "curated_draft"  # Vertrags-Prosa: autonom erreichbarer Zustand
    CURATED = "curated"
    CANONICAL = "canonical"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"
    MOONFINGERS_ONLY = "moonfingers_only"
    TISCH_ONLY = "tisch_only"


class MoonfingersUse(str, Enum):
    GLOBE_NODE = "globe_node"
    ENTITY_DOSSIER = "entity_dossier"
    RELATION = "relation"
    IMAGE = "image"
    STORY_STEP = "story_step"
    FOG_HINT = "fog_hint"
    PERSONAL_OVERLAY = "personal_overlay"
    CONTEXT_PACK = "context_pack"


class ReuseState(str, Enum):
    APPROVED_FOR_REUSE = "approved_for_reuse"


# Autonom (ohne Nutzerfreigabe) erlaubte curation_state-Zielzustände.
# `canonical` ist bewusst NICHT enthalten — siehe curator.py.
AUTONOMOUS_STATES = frozenset({
    CurationState.RAW,
    CurationState.CANDIDATE,
    CurationState.SYNTHESIZED,
    CurationState.REVIEWED,
    CurationState.CURATED_DRAFT,
})


# ===========================================================================
# Helpers
# ===========================================================================

def now_iso() -> str:
    """UTC-Zeitstempel, ISO-8601, timezone-aware."""
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    """Kleinbuchstaben-ASCII-Slug für ID-Bildung."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def new_short_id(title: str = "", prefix: str = "mem") -> str:
    """Kurze ID `<prefix>_<slug>_<hex8>` — innerhalb des Cores eindeutig."""
    slug = slugify(title)[:40].strip("_")
    suffix = uuid.uuid4().hex[:8]
    parts = [prefix] + ([slug] if slug else []) + [suffix]
    return "_".join(parts)


def new_urn(title: str = "", prefix: str = "mem") -> str:
    """URN-Form-ID `tisch_shared_core:<short_id>` — Außenform (Vertrag Punkt 9)."""
    return URN_PREFIX + new_short_id(title, prefix)


def content_fingerprint(text: str) -> str:
    """Stabiler Hash über normalisierten Inhalt — Basis fürs Dedupe."""
    normalized = " ".join((text or "").lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ===========================================================================
# Provenance
# ===========================================================================

class ProvenanceStep(BaseModel):
    """Ein Transformations-Schritt der provenance_chain."""
    step: str
    origin: str = ""
    timestamp: str = Field(default_factory=now_iso)
    summary: str = ""


class ProvenanceRecord(BaseModel):
    """Basisklasse mit dem vollen Provenance-Protokoll.

    Jeder persistierte Record im TiSCH Shared Core erbt von hier. core_id ist
    fix, die id muss URN-Form haben.
    """
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    id: str = Field(default_factory=new_urn)
    core_id: Literal["tisch_shared_core"] = CORE_ID
    family_line: List[FamilyLine] = Field(
        default_factory=lambda: [FamilyLine.DER_TISCH]
    )
    source_role: SourceRole
    memory_layer: MemoryLayer
    curation_state: CurationState = CurationState.RAW
    visibility: Visibility = Visibility.PRIVATE
    moonfingers_use: List[MoonfingersUse] = Field(default_factory=list)
    provenance_chain: List[ProvenanceStep] = Field(default_factory=list)
    reuse_state: Optional[ReuseState] = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    @field_validator("id")
    @classmethod
    def _validate_urn(cls, v: str) -> str:
        if not v.startswith(URN_PREFIX):
            raise ValueError(
                f"id must be URN-form '{URN_PREFIX}<short_id>', got: {v!r}"
            )
        if v == URN_PREFIX or not v[len(URN_PREFIX):].strip():
            raise ValueError("id needs a non-empty short_id after the core prefix")
        return v

    @property
    def short_id(self) -> str:
        """Kurz-ID ohne Core-Präfix (Innenform, Vertrag Punkt 9)."""
        return self.id[len(URN_PREFIX):]

    def touch(self) -> None:
        """updated_at auf jetzt setzen."""
        self.updated_at = now_iso()

    def add_provenance(self, step: str, origin: str = "", summary: str = "") -> None:
        """Einen Schritt an die provenance_chain anhängen und touchen."""
        self.provenance_chain = self.provenance_chain + [
            ProvenanceStep(step=step, origin=origin, summary=summary)
        ]
        self.touch()


# ===========================================================================
# Kern-Modelle
# ===========================================================================

class MemoryCandidate(ProvenanceRecord):
    """Roh-Capture aus TiSCH-Durchläufen / Chat-Excerpts.

    Eingang der Kurations-Pipeline. curation_state typischerweise `raw` oder
    `candidate`.
    """
    kind: Literal["memory_candidate"] = "memory_candidate"
    title: str
    content: str
    source_app: str = ""  # z.B. "der-tisch", "team-tisch", "integrations-tisch"
    content_hash: str = ""
    raw_payload: Dict = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _ensure_hash(self) -> "MemoryCandidate":
        if not self.content_hash:
            self.content_hash = content_fingerprint(self.content)
        return self


class MemoryCard(ProvenanceRecord):
    """Kuratierte / synthetisierte Wissenskarte — Ergebnis der Pipeline.

    `canonical` ist ein eigenständiges Flag neben `reuse_state`: ein Record
    darf `approved_for_reuse` sein, ohne `canonical` zu sein (Vertrag Punkt 6).
    """
    kind: Literal["memory_card"] = "memory_card"
    title: str
    content: str
    summary: str = ""
    tags: List[str] = Field(default_factory=list)
    content_hash: str = ""
    candidate_id: Optional[str] = None  # URN des Quell-Candidate
    canonical: bool = False
    canonical_approved_by: Optional[str] = None
    canonical_approved_at: Optional[str] = None
    synthesis_meta: Dict = Field(default_factory=dict)  # aber_section, confidence ...

    @model_validator(mode="after")
    def _ensure_hash(self) -> "MemoryCard":
        if not self.content_hash:
            self.content_hash = content_fingerprint(self.content)
        return self

    @model_validator(mode="after")
    def _canonical_consistency(self) -> "MemoryCard":
        # canonical:true und curation_state:canonical müssen zusammenpassen.
        if self.canonical and self.curation_state != CurationState.CANONICAL:
            raise ValueError(
                "canonical=True requires curation_state=canonical"
            )
        if self.canonical and not self.canonical_approved_by:
            raise ValueError(
                "canonical=True requires canonical_approved_by (user gate)"
            )
        return self


class ContextPackEntry(BaseModel):
    """Eine Karte innerhalb eines Context Packs — Provenance bleibt erhalten."""
    card_id: str
    title: str
    content: str
    source_role: SourceRole
    memory_layer: MemoryLayer
    curation_state: CurationState
    reuse_state: Optional[ReuseState] = None
    provenance_chain: List[ProvenanceStep] = Field(default_factory=list)
    relevance_score: float = 0.0
    token_estimate: int = 0


class ContextPack(ProvenanceRecord):
    """Kompakter, wiederverwendbarer Kontext aus mehreren MemoryCards.

    Aggregationsartefakt: source_role `model_generated_synthesis`,
    memory_layer `reusable_context`.
    """
    kind: Literal["context_pack"] = "context_pack"
    source_role: SourceRole = SourceRole.MODEL_GENERATED_SYNTHESIS
    memory_layer: MemoryLayer = MemoryLayer.REUSABLE_CONTEXT
    curation_state: CurationState = CurationState.SYNTHESIZED
    moonfingers_use: List[MoonfingersUse] = Field(
        default_factory=lambda: [MoonfingersUse.CONTEXT_PACK]
    )
    title: str = ""
    task_or_question: str
    entries: List[ContextPackEntry] = Field(default_factory=list)
    token_estimate: int = 0
    max_tokens: int = 2000


class ObsidianNoteExport(ProvenanceRecord):
    """Eine MemoryCard in Obsidian-Markdown-Form (Frontmatter + Body).

    Spiegelt die Provenance der Quell-Card. Maschinenform für den
    Vault-Sync (Vertrag „Obsidian-Regel").
    """
    kind: Literal["obsidian_note_export"] = "obsidian_note_export"
    card_id: str
    vault_relative_path: str
    frontmatter: Dict = Field(default_factory=dict)
    body: str = ""
    rendered_markdown: str = ""


# Mapping kind -> Modellklasse (Rekonstruktion aus dem Store).
RECORD_MODELS = {
    "memory_candidate": MemoryCandidate,
    "memory_card": MemoryCard,
    "context_pack": ContextPack,
    "obsidian_note_export": ObsidianNoteExport,
}


__all__ = [
    "CORE_ID", "URN_PREFIX",
    "FamilyLine", "SourceRole", "MemoryLayer", "CurationState", "Visibility",
    "MoonfingersUse", "ReuseState", "AUTONOMOUS_STATES",
    "now_iso", "slugify", "new_short_id", "new_urn", "content_fingerprint",
    "ProvenanceStep", "ProvenanceRecord",
    "MemoryCandidate", "MemoryCard", "ContextPackEntry", "ContextPack",
    "ObsidianNoteExport", "RECORD_MODELS",
]
