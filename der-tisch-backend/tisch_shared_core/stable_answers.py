"""
tisch_shared_core/stable_answers.py — Index / Lookup stabiler Antworten.

Stabile Antworten sind `MemoryCard`s, die entweder
  - `reuse_state == approved_for_reuse`  ODER
  - `curation_state == canonical`
sind. `approved_for_reuse` reicht für Wiederverwendung; `canonical` ist
zusätzlich bestätigte Wahrheit (Vertrag Punkt 6). Beide gelten als „stabil".

Scoring ist dependency-frei (Token-Overlap, Titel-Bonus, Canonical-Boost) —
keine Embeddings, kein externer Index nötig.
"""
from __future__ import annotations

import argparse
import asyncio
import re
from typing import List

from pydantic import BaseModel, Field

from . import store
from .models import CurationState, MemoryCard, ReuseState


def _tokens(text: str) -> set:
    return set(re.findall(r"\w+", (text or "").lower()))


def is_stable(card: MemoryCard) -> bool:
    """True, wenn die Card als stabile Antwort gilt."""
    return (
        card.reuse_state == ReuseState.APPROVED_FOR_REUSE
        or card.curation_state == CurationState.CANONICAL
    )


def score_card(query: str, card: MemoryCard) -> float:
    """Relevanz einer Card zu einer Query (0.0–1.0).

    Basis: Anteil der Query-Begriffe, die in Titel+Inhalt+Tags vorkommen.
    Titeltreffer und canonical-Status werden leicht geboostet.
    """
    q = _tokens(query)
    if not q:
        return 0.0
    body = _tokens(
        " ".join([card.title or "", card.content or "", " ".join(card.tags)])
    )
    overlap = len(q & body) / len(q)
    title_bonus = 0.3 * (len(q & _tokens(card.title)) / len(q))
    score = overlap + title_bonus
    if card.curation_state == CurationState.CANONICAL:
        score *= 1.15  # bestätigte Wahrheit leicht bevorzugen
    return min(1.0, round(score, 4))


class StableAnswerHit(BaseModel):
    """Ein Treffer der Stable-Answer-Suche."""
    card: MemoryCard
    relevance_score: float = 0.0
    is_canonical: bool = False
    is_stable: bool = True


async def find_stable_answers(
    query: str,
    top_k: int = 5,
    *,
    include_all: bool = False,
) -> List[StableAnswerHit]:
    """Stabile Antworten zu einer Query finden, nach Relevanz sortiert.

    include_all=True hebt den Stable-Filter auf (Suche über ALLE Cards) —
    für eine breitere `/search`-Semantik.
    """
    hits: List[StableAnswerHit] = []
    for rec in await store.all_records(store.CARDS):
        card = MemoryCard(**rec)
        stable = is_stable(card)
        if not include_all and not stable:
            continue
        score = score_card(query, card)
        if score <= 0.0:
            continue
        hits.append(
            StableAnswerHit(
                card=card,
                relevance_score=score,
                is_canonical=(card.curation_state == CurationState.CANONICAL),
                is_stable=stable,
            )
        )
    hits.sort(key=lambda h: h.relevance_score, reverse=True)
    return hits[: max(0, top_k)]


# ---------------------------------------------------------------------------
# Smoke-Test / Demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    from .capture import capture_candidate
    from .curator import curate_candidate
    from .models import FamilyLine, SourceRole

    print("[demo] stable_answers: Card kuratieren, dann suchen ...")
    candidate = await capture_candidate(
        title="Stable-Answer-Demo — Reibungsanalyse",
        content=(
            "Die Reibungsanalyse im TiSCH macht Widersprüche zwischen "
            "Perspektiven sichtbar, statt sie zu glätten. Reibung ist ein "
            "Erkenntniswerkzeug, kein Fehler."
        ),
        source_role=SourceRole.TISCH_RUN_RESULT,
        family_line=[FamilyLine.DER_TISCH],
        source_app="der-tisch",
        origin="stable_demo",
    )
    card = await curate_candidate(candidate, target_state=CurationState.REVIEWED)
    print(f"[demo] kuriert: {card.id} (reuse_state={card.reuse_state})")

    hits = await find_stable_answers("Was macht die Reibungsanalyse?", top_k=3)
    assert hits, "FEHLER: keine stabile Antwort gefunden"
    top = hits[0]
    print(f"[demo] Top-Treffer: {top.card.id}  score={top.relevance_score}  "
          f"canonical={top.is_canonical}")
    print("[demo] OK — Stable-Answer-Index liefert relevante Treffer.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tisch_shared_core.stable_answers",
        description="TiSCH Shared Core — Stable-Answer-Index",
    )
    parser.add_argument("--demo", action="store_true", help="Smoke-Test")
    args = parser.parse_args()
    if args.demo:
        asyncio.run(_demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
