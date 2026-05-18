"""
tisch_shared_core/obsidian_export.py — MemoryCard -> Markdown + Frontmatter.

Schreibt eine `MemoryCard` als Markdown-mit-Frontmatter (Schema: Vertrag
Punkt 6) in `OBSIDIAN_VAULT_PATH`.

RAILWAY-SICHERHEITS-REGEL (Vertrag „Verboten"): `OBSIDIAN_VAULT_PATH` ist
OPTIONAL. Fehlt die Env-Var, ist der Schreibvorgang ein No-op mit Warn-Log —
KEIN Fehler. Kein produktiver Pfad darf an einem lokalen Vault hängen.

Round-Trip: Der Card-Inhalt steht zwischen expliziten Content-Markern, damit
`obsidian_import.py` ihn deterministisch zurücklesen kann. Frontmatter ist die
Quelle der Wahrheit für das Provenance-Protokoll. Der YAML-Emitter ist
minimal und dependency-frei (keine pyyaml-Abhängigkeit).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from . import store
from .models import (
    CurationState,
    MemoryCard,
    ObsidianNoteExport,
    ReuseState,
    new_urn,
)

logger = logging.getLogger("tisch_shared_core.obsidian_export")

VAULT_ENV_VAR = "OBSIDIAN_VAULT_PATH"
VAULT_SUBDIR = "tisch-memory"  # Unterordner im Vault

CONTENT_START = "<!-- tisch:content:start -->"
CONTENT_END = "<!-- tisch:content:end -->"


# ---------------------------------------------------------------------------
# Minimaler YAML-Emitter (Frontmatter)
# ---------------------------------------------------------------------------

def _yaml_scalar(val) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    needs_quote = (
        s == ""
        or s != s.strip()
        or ": " in s
        or "\n" in s
        or s in ("null", "true", "false", "~")
        or s[0] in "#&*!|>%@`\"'[]{},"
    )
    return json.dumps(s, ensure_ascii=False) if needs_quote else s


def emit_frontmatter(data: dict) -> str:
    """Flaches dict (Skalare + Listen von Skalaren) als YAML-Frontmatter."""
    lines: List[str] = []
    for key, val in data.items():
        if isinstance(val, list):
            if not val:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                lines.extend(f"  - {_yaml_scalar(item)}" for item in val)
        else:
            lines.append(f"{key}: {_yaml_scalar(val)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Card -> Frontmatter / Markdown
# ---------------------------------------------------------------------------

def build_frontmatter(card: MemoryCard) -> dict:
    """Frontmatter-dict nach Vertrag Punkt 6 (plus Round-Trip-Felder)."""
    return {
        "id": card.id,
        "type": "memory_card",
        "core_id": card.core_id,
        "family_line": [fl.value for fl in card.family_line],
        "source_role": card.source_role.value,
        "memory_layer": card.memory_layer.value,
        "curation_state": card.curation_state.value,
        "approved_for_reuse": card.reuse_state == ReuseState.APPROVED_FOR_REUSE,
        "reuse_state": card.reuse_state.value if card.reuse_state else None,
        "canonical": card.canonical,
        "canonical_approved_by": card.canonical_approved_by,
        "canonical_approved_at": card.canonical_approved_at,
        "visibility": card.visibility.value,
        "moonfingers_use": [mu.value for mu in card.moonfingers_use],
        "tags": list(card.tags),
        "candidate_id": card.candidate_id,
        "content_hash": card.content_hash,
        "created_at": card.created_at,
        "updated_at": card.updated_at,
    }


def render_body(card: MemoryCard) -> str:
    """Markdown-Body (ohne Frontmatter) für eine Card."""
    prov_lines = [
        f"- **{step.step}** · {step.origin or '—'} · {step.timestamp}"
        + (f" — {step.summary}" if step.summary else "")
        for step in card.provenance_chain
    ] or ["- (keine Schritte)"]

    return "\n".join([
        f"# {card.title}",
        "",
        CONTENT_START,
        card.content,
        CONTENT_END,
        "",
        "## Zusammenfassung",
        "",
        card.summary or "—",
        "",
        "## Provenance-Kette",
        "",
        *prov_lines,
        "",
        "## Hinweis",
        "",
        "Maschinen-Round-Trip über Frontmatter + Content-Marker. "
        "`canonical: true` zusammen mit `canonical_approved_by` schließt beim "
        "Re-Import (`obsidian_import.py`) das Approval-Tor.",
    ])


def render_markdown(card: MemoryCard) -> str:
    """Vollständiges Markdown-Dokument (Frontmatter + Body) für eine Card."""
    frontmatter = emit_frontmatter(build_frontmatter(card))
    return f"---\n{frontmatter}\n---\n\n{render_body(card)}\n"


def vault_relative_path(card: MemoryCard) -> str:
    return f"{VAULT_SUBDIR}/{card.short_id}.md"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def vault_path() -> Optional[Path]:
    """Konfigurierter Vault-Pfad oder None, wenn die Env-Var fehlt."""
    raw = os.environ.get(VAULT_ENV_VAR, "").strip()
    return Path(raw) if raw else None


def build_export(card: MemoryCard) -> ObsidianNoteExport:
    """ObsidianNoteExport-Objekt bauen (ohne zu schreiben)."""
    return ObsidianNoteExport(
        id=new_urn(card.title, prefix="obs"),
        family_line=card.family_line,
        source_role=card.source_role,
        memory_layer=card.memory_layer,
        curation_state=card.curation_state,
        visibility=card.visibility,
        moonfingers_use=card.moonfingers_use,
        provenance_chain=card.provenance_chain,
        reuse_state=card.reuse_state,
        card_id=card.id,
        vault_relative_path=vault_relative_path(card),
        frontmatter=build_frontmatter(card),
        body=render_body(card),
        rendered_markdown=render_markdown(card),
    )


def export_card(card: MemoryCard, *, write: bool = True) -> ObsidianNoteExport:
    """Eine Card als Obsidian-Note exportieren.

    Liefert IMMER ein `ObsidianNoteExport` (mit gerendertem Markdown). Nur das
    Schreiben in den Vault hängt an `OBSIDIAN_VAULT_PATH`: fehlt sie, wird ein
    Warn-Log abgesetzt und NICHT geschrieben — niemals ein Fehler geworfen.
    """
    export = build_export(card)
    if not write:
        return export

    vault = vault_path()
    if vault is None:
        logger.warning(
            "%s nicht gesetzt — Obsidian-Export ist No-op (kein Schreiben). "
            "Card %s nur gerendert.", VAULT_ENV_VAR, card.id,
        )
        return export

    target = vault / export.vault_relative_path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(export.rendered_markdown, encoding="utf-8")
        logger.info("Obsidian-Export geschrieben: %s", target)
    except OSError as exc:
        # Auch ein Schreibfehler darf den Core nicht hart fehlschlagen lassen.
        logger.warning("Obsidian-Export fehlgeschlagen (%s): %s", target, exc)
    return export


async def export_all_cards(
    *, curation_state: Optional[CurationState] = None, write: bool = True
) -> List[ObsidianNoteExport]:
    """Alle Cards (optional nach curation_state gefiltert) exportieren."""
    exports: List[ObsidianNoteExport] = []
    for rec in await store.all_records(store.CARDS):
        card = MemoryCard(**rec)
        if curation_state is not None and card.curation_state != curation_state:
            continue
        exports.append(export_card(card, write=write))
    return exports


# ---------------------------------------------------------------------------
# Smoke-Test / Demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    from .capture import capture_candidate
    from .curator import curate_candidate
    from .models import FamilyLine, SourceRole

    print("[demo] obsidian_export: Card kuratieren, rendern, schreiben ...")
    candidate = await capture_candidate(
        title="Obsidian-Export-Demo",
        content="Obsidian ist die menschliche, kuratierte Wissensoberfläche.",
        source_role=SourceRole.USER_AUTHORED_NOTE,
        family_line=[FamilyLine.DEIN_SEIN, FamilyLine.DER_TISCH],
        source_app="der-tisch",
        origin="obsidian_demo",
    )
    card = await curate_candidate(candidate, target_state=CurationState.REVIEWED)

    export = export_card(card)  # ohne Vault: No-op-Schreiben, aber gerendert
    assert export.rendered_markdown.startswith("---\n"), "FEHLER: kein Frontmatter"
    assert CONTENT_START in export.rendered_markdown, "FEHLER: kein Content-Marker"
    assert export.frontmatter["canonical"] is False
    assert export.frontmatter["approved_for_reuse"] is True
    vault = vault_path()
    print(f"[demo] {VAULT_ENV_VAR} = {vault if vault else '(nicht gesetzt -> No-op)'}")
    print(f"[demo] vault_relative_path = {export.vault_relative_path}")
    print("[demo] --- gerendertes Markdown ---")
    print(export.rendered_markdown)
    print("[demo] OK — Card als Obsidian-Markdown gerendert (Schreiben optional).")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tisch_shared_core.obsidian_export",
        description="TiSCH Shared Core — Obsidian-Export",
    )
    parser.add_argument("--demo", action="store_true", help="Smoke-Test")
    args = parser.parse_args()
    if args.demo:
        asyncio.run(_demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
