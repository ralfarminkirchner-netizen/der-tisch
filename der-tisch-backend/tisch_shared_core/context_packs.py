"""
tisch_shared_core/context_packs.py — Context-Pack-Erzeugung.

`build_context_pack(task_or_question, top_k, max_tokens)` stellt aus den
relevantesten stabilen `MemoryCard`s einen kompakten, wiederverwendbaren
Kontext zusammen. Provenance bleibt pro Eintrag erhalten; das Token-Budget
wird grob respektiert (Einträge werden notfalls gekürzt).

Token-Schätzung ist bewusst grob und dependency-frei (~1.3 Tokens je Wort).
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from . import store
from .models import (
    ContextPack,
    ContextPackEntry,
    ProvenanceStep,
    new_urn,
)
from .stable_answers import find_stable_answers

_TOKENS_PER_WORD = 1.3
_MIN_ENTRY_TOKENS = 40  # kleiner Rest lohnt keinen gekürzten Eintrag mehr


def estimate_tokens(text: str) -> int:
    """Grobe, dependency-freie Token-Schätzung."""
    words = len((text or "").split())
    return max(1, round(words * _TOKENS_PER_WORD)) if words else 0


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Text auf ungefähr `max_tokens` kürzen (an Wortgrenze)."""
    if estimate_tokens(text) <= max_tokens:
        return text
    n_words = max(1, int(max_tokens / _TOKENS_PER_WORD))
    return " ".join(text.split()[:n_words]).rstrip() + " …"


async def build_context_pack(
    task_or_question: str,
    *,
    top_k: int = 8,
    max_tokens: int = 2000,
    title: str = "",
    persist: bool = True,
) -> ContextPack:
    """Einen Context Pack aus den relevantesten stabilen Cards bauen.

    Es werden nur stabile Cards verwendet (approved_for_reuse oder canonical) —
    `find_stable_answers` setzt diesen Filter. Einträge werden greedy bis zum
    Token-Budget aufgenommen; der letzte Eintrag wird ggf. gekürzt.
    """
    hits = await find_stable_answers(task_or_question, top_k=top_k)

    entries = []
    used_tokens = 0
    for hit in hits:
        card = hit.card
        full_tokens = estimate_tokens(card.content)
        remaining = max_tokens - used_tokens
        if remaining <= 0:
            break

        if full_tokens <= remaining:
            content = card.content
            entry_tokens = full_tokens
        elif remaining >= _MIN_ENTRY_TOKENS:
            content = _truncate_to_tokens(card.content, remaining)
            entry_tokens = estimate_tokens(content)
        else:
            break  # Restbudget zu klein — Pack schließen

        entries.append(
            ContextPackEntry(
                card_id=card.id,
                title=card.title,
                content=content,
                source_role=card.source_role,
                memory_layer=card.memory_layer,
                curation_state=card.curation_state,
                reuse_state=card.reuse_state,
                provenance_chain=card.provenance_chain,  # Provenance erhalten
                relevance_score=hit.relevance_score,
                token_estimate=entry_tokens,
            )
        )
        used_tokens += entry_tokens

    pack = ContextPack(
        id=new_urn(task_or_question, prefix="pack"),
        title=title or f"Context Pack — {task_or_question[:60]}",
        task_or_question=task_or_question,
        entries=entries,
        token_estimate=used_tokens,
        max_tokens=max_tokens,
        provenance_chain=[
            ProvenanceStep(
                step="context_pack_build",
                origin="context_packs",
                summary=(
                    f"{len(entries)} stabile Card(s), ~{used_tokens}/{max_tokens} "
                    f"Tokens, Query: {task_or_question[:80]}"
                ),
            )
        ],
    )
    if persist:
        await store.upsert(store.CONTEXT_PACKS, pack.model_dump(mode="json"))
    return pack


async def get_context_pack(pack_id: str) -> Optional[ContextPack]:
    rec = await store.get(store.CONTEXT_PACKS, pack_id)
    return ContextPack(**rec) if rec else None


# ---------------------------------------------------------------------------
# Smoke-Test / Demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    from .capture import capture_candidate
    from .curator import curate_candidate
    from .models import CurationState, FamilyLine, SourceRole

    print("[demo] context_packs: Cards kuratieren, Pack bauen ...")
    for i, text in enumerate([
        "Der TiSCH orchestriert acht Methoden-Perspektiven zu einer Frage.",
        "Die Reibungsanalyse macht Widersprüche zwischen Perspektiven sichtbar.",
        "Die Synthese integriert Perspektiven, ohne Widersprüche zu glätten.",
    ]):
        cand = await capture_candidate(
            title=f"Pack-Demo Card {i}",
            content=text + " Perspektive Reibung Synthese TiSCH.",
            source_role=SourceRole.TISCH_RUN_RESULT,
            family_line=[FamilyLine.DER_TISCH],
            source_app="der-tisch",
            origin="pack_demo",
        )
        await curate_candidate(cand, target_state=CurationState.REVIEWED)

    pack = await build_context_pack(
        "Wie arbeitet der TiSCH mit Perspektiven und Reibung?",
        top_k=5,
        max_tokens=400,
    )
    assert pack.entries, "FEHLER: Context Pack hat keine Einträge"
    assert pack.token_estimate <= pack.max_tokens, "FEHLER: Token-Budget überschritten"
    for entry in pack.entries:
        assert entry.provenance_chain, "FEHLER: Provenance im Eintrag verloren"
    print(f"[demo] Pack {pack.id}")
    print(f"[demo]   {len(pack.entries)} Eintrag(e), "
          f"~{pack.token_estimate}/{pack.max_tokens} Tokens")
    read_back = await get_context_pack(pack.id)
    assert read_back is not None and read_back.id == pack.id
    print("[demo] OK — Context Pack gebaut, Provenance erhalten, Budget gewahrt.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tisch_shared_core.context_packs",
        description="TiSCH Shared Core — Context Packs",
    )
    parser.add_argument("--demo", action="store_true", help="Smoke-Test")
    args = parser.parse_args()
    if args.demo:
        asyncio.run(_demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
