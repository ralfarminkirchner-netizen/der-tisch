"""
tisch_memory/storage.py — JSONL append-only Storage für TiSCH Shared Core.

Dateien:
  data/tisch_shared_core/candidates.jsonl  — alle eingehenden MemoryCandidates
  data/tisch_shared_core/cards.jsonl       — kuratierte MemoryCards (>= reviewed)

Kein /Volumes, keine Datenbank im MVP.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schemas import (
    CurationState,
    MemoryCandidate,
    MemoryCard,
    REUSE_ELIGIBLE_STATES,
)

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent.parent / "data" / "tisch_shared_core"
_CANDIDATES_FILE = _BASE / "candidates.jsonl"
_CARDS_FILE = _BASE / "cards.jsonl"


def _ensure_dirs() -> None:
    _BASE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generische JSONL-Helfer
# ---------------------------------------------------------------------------

def _append(path: Path, record: dict) -> None:
    _ensure_dirs()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_all(path: Path) -> list[dict]:
    _ensure_dirs()
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _filter_records(
    records: list[dict],
    *,
    tags: Optional[list[str]] = None,
    curation_states: Optional[set[str]] = None,
    include_private: bool = False,
    origin_app: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    result = []
    for r in records:
        if not include_private and r.get("visibility", "private") == "private":
            continue
        if curation_states and r.get("curation_state") not in curation_states:
            continue
        if origin_app and r.get("origin_app") != origin_app:
            continue
        if project and r.get("project") != project:
            continue
        if tags:
            record_tags = set(r.get("tags", []))
            if not record_tags.intersection(tags):
                continue
        result.append(r)
    return result[:limit]


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def append_candidate(candidate: MemoryCandidate) -> None:
    """Candidate in candidates.jsonl schreiben."""
    _append(_CANDIDATES_FILE, candidate.model_dump(mode="json"))


def read_candidates(
    *,
    origin_app: Optional[str] = None,
    project: Optional[str] = None,
    include_private: bool = False,
    limit: int = 200,
) -> list[dict]:
    all_records = _read_all(_CANDIDATES_FILE)
    return _filter_records(
        all_records,
        include_private=include_private,
        origin_app=origin_app,
        project=project,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Cards (kuratiert, curation_state >= reviewed)
# ---------------------------------------------------------------------------

def append_card(card: MemoryCard) -> None:
    """Card in cards.jsonl schreiben — nur wenn curation_state reuse-eligible."""
    eligible_str = {s.value for s in REUSE_ELIGIBLE_STATES}
    if card.curation_state.value not in eligible_str:
        raise ValueError(
            f"MemoryCard.curation_state muss in {eligible_str} sein, ist: {card.curation_state}"
        )
    _append(_CARDS_FILE, card.model_dump(mode="json"))


def read_cards(
    *,
    tags: Optional[list[str]] = None,
    include_private: bool = False,
    limit: int = 200,
) -> list[dict]:
    eligible_str = {s.value for s in REUSE_ELIGIBLE_STATES}
    all_records = _read_all(_CARDS_FILE)
    return _filter_records(
        all_records,
        tags=tags,
        curation_states=eligible_str,
        include_private=include_private,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Context-Pack-Ranker
# ---------------------------------------------------------------------------

def rank_cards_for_context(
    query: str,
    *,
    tags: Optional[list[str]] = None,
    include_private: bool = False,
    max_chars: int = 4800,
) -> list[dict]:
    """Liest cards.jsonl, rankiert nach Tag-Match + Recency, schneidet bei max_chars ab."""
    cards = read_cards(tags=tags, include_private=include_private)

    query_words = set(query.lower().split())

    def score(card: dict) -> float:
        card_tags = set(t.lower() for t in card.get("tags", []))
        tag_score = len(card_tags.intersection(query_words)) * 2.0
        tag_score += len(card_tags.intersection(set(tags or []))) * 3.0

        # Recency: newer = higher score
        try:
            created = datetime.fromisoformat(card.get("created_at", "2000-01-01"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - created).days
            recency = max(0.0, 30.0 - age_days) / 30.0
        except (ValueError, TypeError):
            recency = 0.0

        # curation boost
        state_boost = {
            "canonical": 5.0,
            "approved_for_reuse": 4.0,
            "curated": 3.0,
            "reviewed": 2.0,
        }.get(card.get("curation_state", ""), 0.0)

        return tag_score + recency + state_boost

    ranked = sorted(cards, key=score, reverse=True)

    # Sammle bis max_chars
    result: list[dict] = []
    total_chars = 0
    for card in ranked:
        content_len = len(card.get("content", ""))
        if total_chars + content_len > max_chars and result:
            break
        card["score"] = score(card)
        result.append(card)
        total_chars += content_len

    return result
