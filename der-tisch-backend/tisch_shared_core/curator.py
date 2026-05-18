"""
tisch_shared_core/curator.py — Curator-Agent mit canonical-Guard.

Kurations-Pipeline (Vertrag):
  Candidate -> (Synthese) -> Dedupe -> Curator -> reviewed / curated_draft
  -> Nutzer-Tor -> canonical

HARTE REGELN (Vertrag + Phase-2-Auftrag):
- Eingang: `MemoryCandidate`s mit curation_state in {raw, candidate, synthesized}.
- Autonom erlaubte Zielzustände: maximal `reviewed` oder `curated_draft`.
  Der Curator darf `reuse_state: approved_for_reuse` setzen.
- `canonical` ist autonom VERBOTEN. Der Übergang erfolgt ausschließlich über
  `promote_to_canonical(...)` mit nicht-leerem `approved_by` — aufgerufen
  durch den expliziten API-Call `POST /api/tisch-memory/curate`
  (`{target_state:"canonical", approved_by:"<user>"}`) oder den
  Obsidian-Frontmatter-Sync (`canonical:true` + `canonical_approved_by`).
- Dedupe: Hash-Gleichheit ODER Token-Overlap-Ähnlichkeit gegen vorhandene
  `MemoryCard`s.

`approved_for_reuse` ≠ `canonical`: Ersteres erlaubt Context-Pack-Nutzung,
Letzteres bestätigt stabile Wahrheit. Schutz gegen Selbstverstärkung
(Vertrag Punkt 6).
"""
from __future__ import annotations

import argparse
import asyncio
import re
from typing import List, Optional, Union

from . import store
from .kintegrity_synthesis import local_tldr
from .models import (
    AUTONOMOUS_STATES,
    CurationState,
    MemoryCandidate,
    MemoryCard,
    ProvenanceStep,
    ReuseState,
    SourceRole,
    new_urn,
    now_iso,
)

# Eingangs-/Ziel-Zustände des autonomen Curators.
CURATOR_INPUT_STATES = frozenset({
    CurationState.RAW,
    CurationState.CANDIDATE,
    CurationState.SYNTHESIZED,
})
CURATOR_AUTONOMOUS_TARGETS = frozenset({
    CurationState.REVIEWED,
    CurationState.CURATED_DRAFT,
})

# Ähnlichkeits-Schwelle fürs Dedupe (Token-Jaccard). Bewusst konservativ.
DEDUPE_SIMILARITY_THRESHOLD = 0.92


class CanonicalGuardError(Exception):
    """Wird ausgelöst, wenn autonome Kuration `canonical` setzen will.

    `canonical` darf nur über `promote_to_canonical` mit Nutzer-Identität
    gesetzt werden — niemals durch den autonomen Curator.
    """


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set:
    return set(re.findall(r"\w+", (text or "").lower()))


def similarity(a: str, b: str) -> float:
    """Token-Jaccard-Ähnlichkeit zweier Texte (0.0–1.0)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


async def find_duplicate_card(
    content: str, content_hash: str, threshold: float = DEDUPE_SIMILARITY_THRESHOLD
) -> Optional[MemoryCard]:
    """Vorhandene MemoryCard finden, die ein Duplikat des Inhalts ist.

    Treffer bei exakter Hash-Gleichheit oder Token-Overlap >= threshold.
    """
    for rec in await store.all_records(store.CARDS):
        if content_hash and rec.get("content_hash") == content_hash:
            return MemoryCard(**rec)
        if similarity(content, rec.get("content", "")) >= threshold:
            return MemoryCard(**rec)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card_with_updates(
    card: MemoryCard, provenance: Optional[ProvenanceStep] = None, **changes
) -> MemoryCard:
    """Eine neue, vollständig validierte MemoryCard mit geänderten Feldern.

    Konstruktion-in-einem-Rutsch statt Feld-für-Feld-Zuweisung — so kann der
    `canonical`-Konsistenz-Validator nie einen transienten Zwischenzustand
    sehen.
    """
    data = card.model_dump(mode="json")
    data.update(changes)
    updated = MemoryCard(**data)
    if provenance is not None:
        updated.provenance_chain = list(updated.provenance_chain) + [provenance]
    updated.updated_at = now_iso()
    return updated


async def get_card(card_id: str) -> Optional[MemoryCard]:
    rec = await store.get(store.CARDS, card_id)
    return MemoryCard(**rec) if rec else None


async def list_cards(
    *, curation_state: Optional[CurationState] = None, limit: int = 200
) -> List[MemoryCard]:
    out: List[MemoryCard] = []
    for rec in await store.all_records(store.CARDS):
        card = MemoryCard(**rec)
        if curation_state is not None and card.curation_state != curation_state:
            continue
        out.append(card)
    return out[:limit]


# ---------------------------------------------------------------------------
# Autonome Kuration: Candidate -> MemoryCard (reviewed / curated_draft)
# ---------------------------------------------------------------------------

async def curate_candidate(
    candidate: MemoryCandidate,
    *,
    target_state: Union[CurationState, str] = CurationState.REVIEWED,
    approved_for_reuse: bool = True,
    dedupe: bool = True,
    persist: bool = True,
) -> MemoryCard:
    """Einen Candidate autonom zu einer MemoryCard kuratieren.

    Stoppt hart bei `reviewed` / `curated_draft`. `canonical` löst
    `CanonicalGuardError` aus.
    """
    if isinstance(target_state, str):
        target_state = CurationState(target_state)

    # --- Guard: canonical ist autonom verboten ---
    if target_state == CurationState.CANONICAL:
        raise CanonicalGuardError(
            "Curator darf 'canonical' nicht autonom setzen. "
            "Nutze promote_to_canonical(card_id, approved_by=...) bzw. "
            "POST /api/tisch-memory/curate mit approved_by."
        )
    if target_state not in CURATOR_AUTONOMOUS_TARGETS:
        raise ValueError(
            f"target_state muss in {sorted(s.value for s in CURATOR_AUTONOMOUS_TARGETS)} "
            f"liegen, nicht {target_state.value!r}."
        )
    if candidate.curation_state not in CURATOR_INPUT_STATES:
        raise ValueError(
            f"Curator-Eingang erfordert curation_state in "
            f"{sorted(s.value for s in CURATOR_INPUT_STATES)}, "
            f"Candidate ist {candidate.curation_state.value!r}."
        )

    # --- Dedupe gegen vorhandene Cards ---
    if dedupe:
        duplicate = await find_duplicate_card(candidate.content, candidate.content_hash)
        if duplicate is not None:
            merged = _card_with_updates(
                duplicate,
                provenance=ProvenanceStep(
                    step="curator_dedupe",
                    origin="curator_agent",
                    summary=f"Inhaltsgleicher Candidate {candidate.id} erneut kuriert — Duplikat.",
                ),
            )
            if persist:
                await store.upsert(store.CARDS, merged.model_dump(mode="json"))
            return merged

    # --- Neue MemoryCard bauen ---
    synthesis_meta = {}
    if isinstance(candidate.raw_payload, dict):
        synthesis_meta = candidate.raw_payload.get("synthesis", {}) or {}

    chain = list(candidate.provenance_chain) + [
        ProvenanceStep(
            step="curator_review",
            origin="curator_agent",
            summary=(
                f"Autonom kuriert nach {target_state.value}; "
                f"reuse_state={'approved_for_reuse' if approved_for_reuse else 'none'}; "
                f"canonical bleibt gesperrt (Nutzer-Tor)."
            ),
        )
    ]

    card = MemoryCard(
        id=new_urn(candidate.title, prefix="card"),
        family_line=candidate.family_line,
        source_role=candidate.source_role,  # Inhaltsherkunft bleibt erhalten
        memory_layer=candidate.memory_layer,
        curation_state=target_state,
        visibility=candidate.visibility,
        moonfingers_use=candidate.moonfingers_use,
        provenance_chain=chain,
        reuse_state=ReuseState.APPROVED_FOR_REUSE if approved_for_reuse else None,
        title=candidate.title,
        content=candidate.content,
        summary=local_tldr(candidate.content, max_sentences=2, max_chars=320),
        content_hash=candidate.content_hash,
        candidate_id=candidate.id,
        canonical=False,
        synthesis_meta=synthesis_meta,
    )
    if persist:
        await store.upsert(store.CARDS, card.model_dump(mode="json"))
    return card


# ---------------------------------------------------------------------------
# Nutzer-Tor: Promotion zu canonical (NICHT autonom)
# ---------------------------------------------------------------------------

async def promote_to_canonical(
    card_id: str,
    *,
    approved_by: str,
    approved_at: Optional[str] = None,
    persist: bool = True,
) -> MemoryCard:
    """Eine MemoryCard zu `canonical` befördern — nur mit Nutzer-Identität.

    `approved_by` MUSS nicht-leer sein. Diese Funktion ist das einzige Tor zu
    `canonical` (API-Call oder Obsidian-Frontmatter-Sync rufen sie auf).
    """
    if not approved_by or not approved_by.strip():
        raise ValueError(
            "Promotion zu 'canonical' erfordert ein nicht-leeres approved_by "
            "(Nutzer-Identität). Modellgenerierte Inhalte werden nie autonom "
            "kanonisiert."
        )
    card = await get_card(card_id)
    if card is None:
        raise ValueError(f"MemoryCard nicht gefunden: {card_id!r}")

    approved_by = approved_by.strip()
    promoted = _card_with_updates(
        card,
        provenance=ProvenanceStep(
            step="canonical_approval",
            origin=approved_by,
            summary=f"Vom Nutzer '{approved_by}' als canonical bestätigt.",
        ),
        curation_state=CurationState.CANONICAL.value,
        canonical=True,
        canonical_approved_by=approved_by,
        canonical_approved_at=approved_at or now_iso(),
    )
    if persist:
        await store.upsert(store.CARDS, promoted.model_dump(mode="json"))
    return promoted


# ---------------------------------------------------------------------------
# Smoke-Test / Demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    from .capture import capture_candidate
    from .kintegrity_synthesis import synthesize_candidate
    from .models import FamilyLine

    print("[demo] Pipeline: capture -> synthesis -> curator -> canonical-Guard")

    candidate = await capture_candidate(
        title="Curator-Demo — Reibung erhalten",
        content=(
            "Eine gute Synthese glättet Widersprüche nicht vorschnell. "
            "Inkommensurabel ist nicht dasselbe wie falsch. Das Ergebnis bleibt "
            "editierbares Arbeitsmaterial. Curator-Demo-Marker " + new_urn("")
        ),
        source_role=SourceRole.TISCH_RUN_RESULT,
        family_line=[FamilyLine.DER_TISCH],
        source_app="der-tisch",
        origin="curator_demo",
    )
    synthesized = await synthesize_candidate(candidate)
    print(f"[demo] synthesized: {synthesized.id} ({synthesized.curation_state})")

    card = await curate_candidate(synthesized, target_state=CurationState.REVIEWED)
    assert card.curation_state == CurationState.REVIEWED, card.curation_state
    assert card.canonical is False
    assert card.reuse_state == ReuseState.APPROVED_FOR_REUSE
    print(f"[demo] kuriert -> Card {card.id}")
    print(f"[demo]   curation_state={card.curation_state.value}  "
          f"reuse_state={card.reuse_state.value}  canonical={card.canonical}")

    # Guard: autonomer canonical-Versuch muss scheitern.
    guard_ok = False
    try:
        await curate_candidate(synthesized, target_state=CurationState.CANONICAL)
    except CanonicalGuardError as exc:
        guard_ok = True
        print(f"[demo] canonical-Guard greift: {exc}")
    assert guard_ok, "FEHLER: canonical-Guard hat NICHT ausgelöst"

    # Promotion ohne approved_by muss scheitern.
    gate_ok = False
    try:
        await promote_to_canonical(card.id, approved_by="")
    except ValueError as exc:
        gate_ok = True
        print(f"[demo] Nutzer-Tor greift (leeres approved_by): {exc}")
    assert gate_ok, "FEHLER: leeres approved_by wurde akzeptiert"

    # Expliziter Nutzer-Call setzt canonical.
    promoted = await promote_to_canonical(card.id, approved_by="ralf")
    assert promoted.canonical is True
    assert promoted.curation_state == CurationState.CANONICAL
    assert promoted.canonical_approved_by == "ralf"
    print(f"[demo] explizite Promotion -> {promoted.id} "
          f"canonical={promoted.canonical} approved_by={promoted.canonical_approved_by}")
    print("[demo] OK — Pipeline + canonical-Guard + Nutzer-Tor verifiziert.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tisch_shared_core.curator",
        description="TiSCH Shared Core — Curator-Agent",
    )
    parser.add_argument("--demo", action="store_true", help="Smoke-Test")
    args = parser.parse_args()
    if args.demo:
        asyncio.run(_demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
