"""
tisch_shared_core — TiSCH Shared Core (Dual-Core-Modell).

Persönliches Arbeits- und Antwortgedächtnis der TiSCH-App-Familie.
Host: der-tisch-backend/. Kanonischer Vertrag:
HANDOFF_dual_shared_cores_2026-05-18.md (Stand 2026-05-18).

Module:
  models.py             Pydantic-Modelle + Provenance-Protokoll
  store.py              SQLite-Persistenz (aiosqlite)
  capture.py            Raw Capture -> MemoryCandidate
  kintegrity_synthesis  KiNTEGRiTY-Synthese (gewickelt) -> synthesized
  curator.py            Curator-Agent mit canonical-Guard
  stable_answers.py     Index/Lookup stabiler Antworten
  context_packs.py      Context-Pack-Erzeugung
  obsidian_export.py    MemoryCard -> Markdown + Frontmatter
  obsidian_import.py    Markdown + Frontmatter -> MemoryCard
  api.py                FastAPI-Router /api/tisch-memory/...
"""

__version__ = "0.1.0"
CORE_ID = "tisch_shared_core"
