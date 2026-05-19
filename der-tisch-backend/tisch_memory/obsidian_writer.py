"""
tisch_memory/obsidian_writer.py — Obsidian Markdown Export für kuratierte MemoryCards.

Nimmt eine MemoryCard (curation_state in {reviewed, curated}) und schreibt eine .md-Datei
mit YAML-Frontmatter unter dem suggested_obsidian_path.

Vault-Pfad: Env-Var OBSIDIAN_VAULT_PATH (kein /Volumes-Hardcode).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schemas import CurationState, MemoryCard

_EXPORT_ELIGIBLE = {CurationState.reviewed, CurationState.curated}

OBSIDIAN_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH", "")


def _yaml_str(value: object) -> str:
    if value is None:
        return "null"
    s = str(value)
    if any(c in s for c in (':', '#', '[', ']', '{', '}', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`')):
        return f'"{s}"'
    return s


def card_to_markdown(card: MemoryCard) -> str:
    """Wandelt eine MemoryCard in Obsidian-Markdown mit YAML-Frontmatter um."""
    tags_yaml = "\n".join(f"  - {t}" for t in (card.tags or []))
    prov_yaml = "\n".join(
        f"  - parent: {_yaml_str(p.parent_id)}\n    relation: {p.relation}\n    at: {p.at.isoformat()}"
        for p in (card.provenance_chain or [])
    )

    created_iso = card.created_at.isoformat() if card.created_at else ""

    frontmatter_lines = [
        "---",
        f"id: {_yaml_str(card.id)}",
        f"core_id: {card.core_id}",
        f"title: {_yaml_str(card.title)}",
        f"source_role: {card.source_role.value}",
        f"memory_layer: {card.memory_layer.value}",
        f"curation_state: {card.curation_state.value}",
        f"approved_for_reuse: {str(card.approved_for_reuse).lower()}",
        f"canonical: {str(card.canonical).lower()}",
    ]
    if card.canonical_approved_by:
        frontmatter_lines.append(f"canonical_approved_by: {_yaml_str(card.canonical_approved_by)}")
    if card.canonical_approved_at:
        frontmatter_lines.append(f"canonical_approved_at: {card.canonical_approved_at.isoformat()}")
    frontmatter_lines += [
        f"visibility: {card.visibility}",
        f"origin_app: {_yaml_str(card.origin_app)}",
        f"project: {_yaml_str(card.project)}",
        f"task: {_yaml_str(card.task)}",
        f"created_at: {created_iso}",
    ]
    if tags_yaml:
        frontmatter_lines.append("tags:")
        frontmatter_lines.append(tags_yaml)
    else:
        frontmatter_lines.append("tags: []")
    if card.suggested_obsidian_path:
        frontmatter_lines.append(f"suggested_obsidian_path: {_yaml_str(card.suggested_obsidian_path)}")
    if prov_yaml:
        frontmatter_lines.append("provenance_chain:")
        frontmatter_lines.append(prov_yaml)
    frontmatter_lines.append("---")

    body_parts = [f"# {card.title}", ""]
    if card.input_summary:
        body_parts += ["## Eingabe-Zusammenfassung", "", card.input_summary, ""]
    body_parts += ["## Inhalt", "", card.content, ""]
    if card.output_summary:
        body_parts += ["## Ausgabe-Zusammenfassung", "", card.output_summary, ""]

    return "\n".join(frontmatter_lines) + "\n" + "\n".join(body_parts)


def write_card_to_vault(card: MemoryCard) -> Optional[Path]:
    """
    Schreibt die MemoryCard als .md unter OBSIDIAN_VAULT_PATH / suggested_obsidian_path.
    Gibt None zurück wenn OBSIDIAN_VAULT_PATH nicht gesetzt oder Card nicht export-eligible.
    """
    if card.curation_state not in _EXPORT_ELIGIBLE:
        return None
    if not OBSIDIAN_VAULT_PATH:
        return None

    vault = Path(OBSIDIAN_VAULT_PATH)
    rel_path = card.suggested_obsidian_path or "TiSCH/Cards"
    target_dir = vault / rel_path
    target_dir.mkdir(parents=True, exist_ok=True)

    # Dateiname aus Titel, sanitisiert
    safe_title = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_"
        for c in card.title
    )[:80].strip()
    filename = f"{safe_title}.md"
    target = target_dir / filename

    target.write_text(card_to_markdown(card), encoding="utf-8")
    return target
