"""Obsidian-Export für MemoryCards mit curation_state >= reviewed.

Vault-Pfad wird über OBSIDIAN_VAULT_PATH (Env-Var) aufgelöst.
Frontmatter exakt nach Handoff-Spezifikation.
"""
import os
from datetime import timezone
from pathlib import Path
from .schemas import MemoryCard, CurationState

_ELIGIBLE = {
    CurationState.reviewed,
    CurationState.curated,
    CurationState.approved_for_reuse,
    CurationState.canonical,
}


def write_card_to_obsidian(card: MemoryCard) -> Path | None:
    """Schreibt eine MemoryCard als Obsidian-Markdown-Datei.

    Gibt den Pfad der geschriebenen Datei zurück oder None,
    wenn OBSIDIAN_VAULT_PATH nicht gesetzt oder curation_state nicht eligible.
    """
    if card.curation_state not in _ELIGIBLE:
        return None
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    if not vault_path:
        return None

    vault = Path(vault_path)
    sub = card.suggested_obsidian_path or "TiSCH/Memory"
    dest_dir = vault / sub
    dest_dir.mkdir(parents=True, exist_ok=True)

    _unsafe = r'/\:*?"<>|'
    safe_title = card.title
    for ch in _unsafe:
        safe_title = safe_title.replace(ch, "_")
    safe_title = safe_title[:80]
    filepath = dest_dir / f"{safe_title}.md"

    # ISO-8601 UTC
    created_at = card.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    iso_created = created_at.isoformat()

    canonical_by = f'"{card.canonical_approved_by}"' if card.canonical_approved_by else "null"
    canonical_at = f'"{card.canonical_approved_at.isoformat()}"' if card.canonical_approved_at else "null"
    tags_yaml = "[" + ", ".join(f'"{t}"' for t in card.tags) + "]"

    lines = [
        "---",
        f'id: "{card.id}"',
        f'core_id: "{card.core_id}"',
        f'source_role: "{card.source_role.value}"',
        f'memory_layer: "{card.memory_layer.value}"',
        f'curation_state: "{card.curation_state.value}"',
        f"canonical: {str(card.canonical).lower()}",
        f"canonical_approved_by: {canonical_by}",
        f"canonical_approved_at: {canonical_at}",
        f"approved_for_reuse: {str(card.approved_for_reuse).lower()}",
        f'visibility: "{card.visibility}"',
        f'created_at: "{iso_created}"',
        f"tags: {tags_yaml}",
        f'suggested_obsidian_path: "{card.suggested_obsidian_path or sub}"',
        "---",
        "",
        f"# {card.title}",
        "",
        card.content,
    ]
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath
