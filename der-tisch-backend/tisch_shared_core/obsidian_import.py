"""
tisch_shared_core/obsidian_import.py — Markdown + Frontmatter -> MemoryCard.

Liest kuratierte Obsidian-Notes (Frontmatter + Content-Marker) zurück und
mappt sie auf `MemoryCard`s bzw. aktualisiert vorhandene Karten.

Approval-Tor (Vertrag Punkt 6 / Phase 2): Erkennt der Import im Frontmatter
`canonical: true` zusammen mit einem nicht-leeren `canonical_approved_by`, so
schließt das das Approval-Tor — die Karte wird canonical. `canonical: true`
OHNE `canonical_approved_by` ist ein ungültiger Nutzer-Edit und wird
abgelehnt. Modellgenerierte Inhalte werden nie autonom kanonisiert.

DER TiSCH (Store) bleibt operative Wahrheit für die provenance_chain: das
Frontmatter trägt das flache Provenance-Protokoll, die Schrittkette wird beim
Import aus dem vorhandenen Store-Record übernommen und um Import-Schritte
ergänzt. `OBSIDIAN_VAULT_PATH` ist optional (No-op + Warn-Log, kein Fehler).

YAML-Parser ist minimal und dependency-frei — er liest das von
`obsidian_export.py` erzeugte Format zurück.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from . import store
from .curator import get_card
from .models import (
    CurationState,
    MemoryCard,
    ProvenanceStep,
    ReuseState,
    content_fingerprint,
    now_iso,
)
from .obsidian_export import CONTENT_END, CONTENT_START, VAULT_SUBDIR, vault_path

logger = logging.getLogger("tisch_shared_core.obsidian_import")


# ---------------------------------------------------------------------------
# Minimaler Frontmatter-Parser
# ---------------------------------------------------------------------------

def _parse_scalar(raw: str):
    raw = raw.strip()
    if raw in ("", "null", "~"):
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "[]":
        return []
    if raw[0] == '"':
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.strip('"')
    if raw[0] == "'":
        return raw.strip("'")
    return raw


def parse_frontmatter(text: str) -> Tuple[dict, str]:
    """`(frontmatter_dict, body)` aus Markdown-mit-Frontmatter lesen."""
    text = text.lstrip("﻿")
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    fm_lines: List[str] = []
    body_start: Optional[int] = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            body_start = i + 1
            break
        fm_lines.append(lines[i])
    if body_start is None:
        return {}, text

    body = "\n".join(lines[body_start:]).lstrip("\n")
    data: dict = {}
    current_list_key: Optional[str] = None
    for line in fm_lines:
        if not line.strip():
            continue
        if re.match(r"^\s+-\s", line):  # Listenelement
            if current_list_key and isinstance(data.get(current_list_key), list):
                item = line.strip()[1:].strip()
                data[current_list_key].append(_parse_scalar(item))
            continue
        m = re.match(r"^([\w-]+):\s*(.*)$", line)
        if not m:
            continue
        key, rest = m.group(1), m.group(2).rstrip()
        if rest == "":
            data[key] = []  # Listen-Elternzeile
            current_list_key = key
        elif rest == "[]":
            data[key] = []
            current_list_key = None
        else:
            data[key] = _parse_scalar(rest)
            current_list_key = None
    return data, body


def extract_content(body: str) -> str:
    """Card-Inhalt aus dem Body lesen (zwischen Content-Markern)."""
    if CONTENT_START in body and CONTENT_END in body:
        segment = body.split(CONTENT_START, 1)[1].split(CONTENT_END, 1)[0]
        return segment.strip("\n").strip()
    # Fallback für handgeschriebene Notes ohne Marker: Text bis zur 1. ##-Sektion.
    without_title = re.sub(r"^#\s+.*?\n", "", body, count=1)
    return without_title.split("\n## ", 1)[0].strip()


def extract_title(body: str) -> str:
    m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    return m.group(1).strip() if m else "Untitled"


def extract_section(body: str, heading: str) -> str:
    m = re.search(
        rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)",
        body, flags=re.MULTILINE | re.DOTALL,
    )
    if not m:
        return ""
    text = m.group(1).strip()
    return "" if text == "—" else text


# ---------------------------------------------------------------------------
# Markdown -> MemoryCard
# ---------------------------------------------------------------------------

async def import_markdown(text: str, *, persist: bool = True) -> MemoryCard:
    """Eine Obsidian-Note (Markdown-String) zu einer MemoryCard mappen.

    Aktualisiert eine vorhandene Karte gleicher id oder legt eine neue an.
    """
    fm, body = parse_frontmatter(text)
    if not fm.get("id"):
        raise ValueError("Obsidian-Note ohne `id` im Frontmatter — nicht importierbar.")

    card_id = fm["id"]
    title = extract_title(body)
    content = extract_content(body)
    summary = extract_section(body, "Zusammenfassung")

    existing = await get_card(card_id)

    # --- reuse_state ableiten ---
    reuse_state = None
    if fm.get("approved_for_reuse") is True or fm.get("reuse_state") == "approved_for_reuse":
        reuse_state = ReuseState.APPROVED_FOR_REUSE

    # --- canonical / Approval-Tor ---
    canonical = bool(fm.get("canonical"))
    approved_by = fm.get("canonical_approved_by")
    approved_at = fm.get("canonical_approved_at")
    curation_state = fm.get("curation_state") or CurationState.REVIEWED.value
    gate_closed = False
    if canonical:
        if not approved_by or not str(approved_by).strip():
            raise ValueError(
                f"Obsidian-Note {card_id} setzt `canonical: true`, aber "
                "`canonical_approved_by` fehlt — Approval-Tor nicht erfüllt."
            )
        approved_by = str(approved_by).strip()
        curation_state = CurationState.CANONICAL.value  # mit canonical erzwingen
        approved_at = approved_at or now_iso()
        gate_closed = not (existing and existing.canonical)

    # --- provenance_chain: Store ist operative Wahrheit, Import ergänzt ---
    chain = list(existing.provenance_chain) if existing else []
    content_changed = bool(existing) and existing.content_hash != content_fingerprint(content)
    if content_changed:
        chain.append(ProvenanceStep(
            step="obsidian_edit",
            origin="obsidian_vault",
            summary="Inhalt im Obsidian-Vault editiert.",
        ))
    if gate_closed:
        chain.append(ProvenanceStep(
            step="canonical_approval",
            origin=approved_by,
            summary=f"Über Obsidian-Frontmatter durch '{approved_by}' als canonical bestätigt.",
        ))
    chain.append(ProvenanceStep(
        step="obsidian_import",
        origin="obsidian_vault",
        summary=f"Re-Import aus dem Vault ({'aktualisiert' if existing else 'neu'}).",
    ))

    card = MemoryCard(
        id=card_id,
        family_line=fm.get("family_line") or ["der-tisch"],
        source_role=fm.get("source_role") or "user_authored_note",
        memory_layer=fm.get("memory_layer") or "personal_memory",
        curation_state=curation_state,
        visibility=fm.get("visibility") or "private",
        moonfingers_use=fm.get("moonfingers_use") or [],
        provenance_chain=chain,
        reuse_state=reuse_state,
        created_at=fm.get("created_at") or (existing.created_at if existing else now_iso()),
        title=title,
        content=content,
        summary=summary,
        tags=fm.get("tags") or [],
        candidate_id=fm.get("candidate_id"),
        content_hash="",  # frisch aus importiertem Inhalt berechnen
        canonical=canonical,
        canonical_approved_by=approved_by,
        canonical_approved_at=approved_at,
        synthesis_meta=existing.synthesis_meta if existing else {},
    )
    if persist:
        await store.upsert(store.CARDS, card.model_dump(mode="json"))
    return card


async def import_note_file(path: Path, *, persist: bool = True) -> MemoryCard:
    """Eine einzelne `.md`-Datei importieren."""
    return await import_markdown(
        Path(path).read_text(encoding="utf-8"), persist=persist
    )


async def import_vault(*, persist: bool = True) -> List[MemoryCard]:
    """Alle Notes aus `OBSIDIAN_VAULT_PATH/tisch-memory/` importieren.

    Fehlt die Env-Var oder der Ordner: No-op mit Warn-Log, kein Fehler.
    """
    vault = vault_path()
    if vault is None:
        logger.warning("OBSIDIAN_VAULT_PATH nicht gesetzt — import_vault ist No-op.")
        return []
    subdir = vault / VAULT_SUBDIR
    if not subdir.is_dir():
        logger.warning("Vault-Unterordner fehlt: %s — import_vault ist No-op.", subdir)
        return []
    results: List[MemoryCard] = []
    for md in sorted(subdir.glob("*.md")):
        try:
            results.append(await import_note_file(md, persist=persist))
        except Exception as exc:
            logger.warning("Import übersprungen (%s): %s", md, exc)
    return results


# ---------------------------------------------------------------------------
# Smoke-Test / Demo
# ---------------------------------------------------------------------------

async def _demo() -> None:
    from .capture import capture_candidate
    from .curator import curate_candidate
    from .models import FamilyLine, SourceRole
    from .obsidian_export import render_markdown

    print("[demo] obsidian_import: Round-Trip + canonical-Tor über Frontmatter")
    candidate = await capture_candidate(
        title="Obsidian-Import-Demo",
        content="Der lokale Sync schreibt kuratierte Karten und liest sie zurück.",
        source_role=SourceRole.USER_AUTHORED_NOTE,
        family_line=[FamilyLine.DEIN_SEIN, FamilyLine.DER_TISCH],
        source_app="der-tisch",
        origin="obsidian_import_demo",
    )
    card = await curate_candidate(candidate, target_state=CurationState.REVIEWED)
    markdown = render_markdown(card)

    # 1) Round-Trip: rendern -> importieren -> Inhalt/ID gleich.
    reimported = await import_markdown(markdown)
    assert reimported.id == card.id, "FEHLER: id-Round-Trip"
    assert reimported.content == card.content, "FEHLER: content-Round-Trip"
    assert reimported.canonical is False, "FEHLER: unerwartet canonical"
    print(f"[demo] Round-Trip OK — {reimported.id}")

    # 2) Approval-Tor: Frontmatter auf canonical:true + approved_by setzen.
    edited = markdown.replace("canonical: false", "canonical: true")
    edited = edited.replace(
        "canonical_approved_by: null", "canonical_approved_by: ralf"
    )
    promoted = await import_markdown(edited)
    assert promoted.canonical is True, "FEHLER: canonical-Tor nicht geschlossen"
    assert promoted.curation_state == CurationState.CANONICAL
    assert promoted.canonical_approved_by == "ralf"
    print(f"[demo] canonical-Tor über Frontmatter geschlossen — approved_by="
          f"{promoted.canonical_approved_by}")

    # 3) canonical:true OHNE approved_by muss abgelehnt werden.
    bad = markdown.replace("canonical: false", "canonical: true")
    rejected = False
    try:
        await import_markdown(bad, persist=False)
    except ValueError as exc:
        rejected = True
        print(f"[demo] ungültiger canonical-Edit abgelehnt: {exc}")
    assert rejected, "FEHLER: canonical ohne approved_by wurde akzeptiert"
    print("[demo] OK — Import-Round-Trip + Approval-Tor verifiziert.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tisch_shared_core.obsidian_import",
        description="TiSCH Shared Core — Obsidian-Import",
    )
    parser.add_argument("--demo", action="store_true", help="Smoke-Test")
    args = parser.parse_args()
    if args.demo:
        asyncio.run(_demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
