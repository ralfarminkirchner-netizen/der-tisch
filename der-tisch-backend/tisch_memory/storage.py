"""JSONL append-only Storage für tisch_shared_core (MVP).

Pfade relativ zum Backend-Verzeichnis — kein /Volumes-Zugriff.
"""
from pathlib import Path
from .schemas import MemoryCandidate, MemoryCard, ContextPackRequest, ContextPack, CurationState

_DATA_DIR = Path(__file__).parent.parent / "data" / "tisch_shared_core"
_CANDIDATES_FILE = _DATA_DIR / "candidates.jsonl"
_CARDS_FILE = _DATA_DIR / "cards.jsonl"

_CURATION_ORDER = [
    CurationState.raw,
    CurationState.model_generated,
    CurationState.candidate,
    CurationState.synthesized,
    CurationState.reviewed,
    CurationState.curated,
    CurationState.approved_for_reuse,
    CurationState.canonical,
]


def _ensure_dirs() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def append_candidate(candidate: MemoryCandidate) -> None:
    _ensure_dirs()
    with open(_CANDIDATES_FILE, "a", encoding="utf-8") as f:
        f.write(candidate.model_dump_json() + "\n")


def read_all_candidates() -> list[MemoryCandidate]:
    _ensure_dirs()
    if not _CANDIDATES_FILE.exists():
        return []
    results = []
    with open(_CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(MemoryCandidate.model_validate_json(line))
                except Exception:
                    pass
    return results


def append_card(card: MemoryCard) -> None:
    _ensure_dirs()
    with open(_CARDS_FILE, "a", encoding="utf-8") as f:
        f.write(card.model_dump_json() + "\n")


def read_all_cards() -> list[MemoryCard]:
    _ensure_dirs()
    if not _CARDS_FILE.exists():
        return []
    results = []
    with open(_CARDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(MemoryCard.model_validate_json(line))
                except Exception:
                    pass
    return results


def filter_cards(
    cards: list[MemoryCard],
    include_private: bool = False,
    min_curation: CurationState = CurationState.reviewed,
    tags: list[str] | None = None,
) -> list[MemoryCard]:
    min_idx = _CURATION_ORDER.index(min_curation) if min_curation in _CURATION_ORDER else 0
    result = []
    for card in cards:
        if not include_private and card.visibility == "private":
            continue
        card_idx = _CURATION_ORDER.index(card.curation_state) if card.curation_state in _CURATION_ORDER else 0
        if card_idx < min_idx:
            continue
        if tags and not any(t in card.tags for t in tags):
            continue
        result.append(card)
    return result


def _score_card(card: MemoryCard, query_tags: list[str]) -> float:
    """Tag-Match + Recency-Score für ContextPack-Ranking."""
    tag_score = sum(1.0 for t in query_tags if t in card.tags)
    # neueste Karte zuerst (Unix-timestamp normalisiert, max 10 Punkte)
    import time
    age_seconds = time.time() - card.created_at.timestamp()
    recency_score = max(0.0, 10.0 - age_seconds / 86400)  # 1 Punkt pro Tag Alter abgezogen
    return tag_score * 3.0 + recency_score


def build_context_pack(req: ContextPackRequest) -> ContextPack:
    all_cards = read_all_cards()
    filtered = filter_cards(
        all_cards,
        include_private=req.include_private,
        min_curation=CurationState.reviewed,
    )
    # Tag-Matching auf query + project + task
    query_tags: list[str] = []
    if req.query:
        query_tags += [w.lower() for w in req.query.split() if len(w) > 3]
    if req.project:
        query_tags.append(req.project.lower())
    if req.task:
        query_tags.append(req.task.lower())

    scored = sorted(filtered, key=lambda c: _score_card(c, query_tags), reverse=True)

    # Token-Budget: grobe Schätzung 4 Zeichen ≈ 1 Token
    budget = req.max_tokens * 4
    selected: list[MemoryCard] = []
    used = 0
    for card in scored:
        size = len(card.content)
        if used + size > budget:
            break
        selected.append(card)
        used += size

    return ContextPack(
        app_id=req.app_id,
        cards=selected,
        total_found=len(filtered),
        query=req.query,
    )
