"""
tisch_shared_core/auto_capture.py — Automatik-Schalter.

Speichert TiSCH-Laufergebnisse automatisch als MemoryCandidate im TiSCH
Shared Core. Wird vom api_server nach jedem erfolgreichen Ask-Durchlauf via
asyncio.create_task aufgerufen. Niemals wirft diese Funktion nach außen —
Fehler werden nur geloggt, der Response-Flow bleibt unberührt.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .capture import capture_candidate
from .models import CurationState, FamilyLine, MemoryLayer, SourceRole

logger = logging.getLogger(__name__)


def _build_content(integration: Dict[str, Any]) -> str:
    """Lesbaren Inhalt aus den Integration-Feldern destillieren."""
    parts: list[str] = []
    einfach = (integration.get("einfach_gesagt") or "").strip()
    if einfach:
        parts.append(einfach)
    kurzfassung = integration.get("kurzfassung") or []
    if kurzfassung:
        parts.append("\n".join(f"• {k}" for k in kurzfassung if k))
    fazit = (integration.get("vorlaeufiges_fazit") or "").strip()
    if fazit:
        parts.append(fazit)
    return "\n\n".join(parts)


async def save_tisch_run(
    question: str,
    response_data: Dict[str, Any],
    source_app: str = "der-tisch",
) -> Optional[str]:
    """TiSCH-Laufergebnis als MemoryCandidate im Shared Core speichern.

    Idempotent: gleiche Frage + gleicher Inhalt → kein Duplikat (Dedupe per
    content_hash). Gibt die URN des Candidates zurück, oder None bei Fehler.
    """
    try:
        integration = response_data.get("integration") or {}
        content = _build_content(integration)
        if not content.strip():
            logger.debug("auto_capture: leerer Inhalt, übersprungen (q=%r)", question[:60])
            return None

        title = question.strip()
        if len(title) > 120:
            title = title[:117] + "..."

        candidate = await capture_candidate(
            title=title,
            content=content,
            source_role=SourceRole.TISCH_RUN_RESULT,
            family_line=[FamilyLine.DER_TISCH],
            source_app=source_app,
            memory_layer=MemoryLayer.PROJECT_MEMORY,
            curation_state=CurationState.CANDIDATE,
            raw_payload={"integration": integration},
            dedupe_on_hash=True,
        )
        logger.info("auto_capture: %s gespeichert (q=%r)", candidate.id, question[:60])
        return candidate.id
    except Exception as exc:
        logger.warning("auto_capture: Fehler beim Speichern — %r", exc)
        return None
