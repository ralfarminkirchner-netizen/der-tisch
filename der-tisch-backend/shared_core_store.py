"""
shared_core_store.py — Shared Core SQLite Persistence für alle TiSCH-Apps.
Version 1.0 — 2026-04-08

Speichert vollständige Session-Ergebnisse (Perspectives, Friction, Integration)
aller TiSCH-Apps automatisch. Bildet die Datenbasis für den Vault-Sync
und die Pattern-Erkennung.

Pattern: identisch zu moonfingers_store.py — aiosqlite, async, Row-Factory.
"""

import aiosqlite
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# DB-Pfad — neben diesem Modul, persistent auf Railway
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "shared_core.db"

# Interner API-Key für GET /api/sessions (Railway ENV: SHARED_CORE_KEY)
SHARED_CORE_KEY = os.environ.get("SHARED_CORE_KEY", "tisch-shared-core-2026")


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Tabellen anlegen (idempotent — CREATE IF NOT EXISTS)."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Sessions-Tabelle: eine Zeile pro TiSCH-Anfrage
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tisch_sessions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT    NOT NULL UNIQUE,
                source_app        TEXT    NOT NULL DEFAULT 'unknown',
                question          TEXT    NOT NULL,
                lang              TEXT    DEFAULT 'de',
                stil              TEXT    DEFAULT 'philosophisch',
                tone              TEXT    DEFAULT '',
                perspectives_json TEXT,
                friction_json     TEXT,
                integration_json  TEXT,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Pattern-Tabelle: myCEL liest/schreibt erkannte Muster
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tisch_patterns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_key TEXT    NOT NULL UNIQUE,
                data_json   TEXT    NOT NULL,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_app ON tisch_sessions(source_app)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_created ON tisch_sessions(created_at)"
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

async def save_session(
    *,
    source_app: str,
    question: str,
    lang: str,
    stil: str,
    tone: str,
    perspectives: list,
    friction: dict,
    integration: dict,
) -> str:
    """Session speichern. Gibt die neue session_id zurück."""
    session_id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO tisch_sessions
                (session_id, source_app, question, lang, stil, tone,
                 perspectives_json, friction_json, integration_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                source_app,
                question,
                lang,
                stil,
                tone,
                json.dumps(perspectives, ensure_ascii=False),
                json.dumps(friction, ensure_ascii=False),
                json.dumps(integration, ensure_ascii=False),
            ),
        )
        await db.commit()
    return session_id


async def get_sessions(
    *,
    source_app: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Sessions abrufen — optional nach App gefiltert, neueste zuerst."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if source_app:
            cursor = await db.execute(
                """SELECT id, session_id, source_app, question, lang, stil, tone,
                          perspectives_json, friction_json, integration_json, created_at
                   FROM tisch_sessions
                   WHERE source_app = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (source_app, limit, offset),
            )
        else:
            cursor = await db.execute(
                """SELECT id, session_id, source_app, question, lang, stil, tone,
                          perspectives_json, friction_json, integration_json, created_at
                   FROM tisch_sessions
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            )
        rows = await cursor.fetchall()

    result = []
    for row in rows:
        entry = dict(row)
        # JSON-Felder deserialisieren
        for field in ("perspectives_json", "friction_json", "integration_json"):
            raw = entry.pop(field, None)
            key = field.replace("_json", "")
            try:
                entry[key] = json.loads(raw) if raw else None
            except (json.JSONDecodeError, TypeError):
                entry[key] = None
        result.append(entry)
    return result


async def get_session_count(source_app: Optional[str] = None) -> int:
    """Anzahl gespeicherter Sessions (gesamt oder je App)."""
    async with aiosqlite.connect(DB_PATH) as db:
        if source_app:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM tisch_sessions WHERE source_app = ?", (source_app,)
            )
        else:
            cursor = await db.execute("SELECT COUNT(*) FROM tisch_sessions")
        row = await cursor.fetchone()
        return row[0] if row else 0


async def get_app_stats() -> list[dict]:
    """Anzahl Sessions je App — für Übersicht."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT source_app, COUNT(*) as count,
                      MAX(created_at) as last_session
               FROM tisch_sessions
               GROUP BY source_app
               ORDER BY count DESC"""
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Pattern CRUD (myCEL-Interface)
# ---------------------------------------------------------------------------

async def write_pattern(pattern_key: str, data: dict) -> None:
    """Muster schreiben/aktualisieren (Upsert)."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT id FROM tisch_patterns WHERE pattern_key = ?", (pattern_key,)
        )
        existing = await cursor.fetchone()
        if existing:
            await db.execute(
                """UPDATE tisch_patterns
                   SET data_json = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE pattern_key = ?""",
                (json.dumps(data, ensure_ascii=False), pattern_key),
            )
        else:
            await db.execute(
                "INSERT INTO tisch_patterns (pattern_key, data_json) VALUES (?, ?)",
                (pattern_key, json.dumps(data, ensure_ascii=False)),
            )
        await db.commit()


async def read_patterns() -> list[dict]:
    """Alle Muster lesen — neueste zuerst."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT pattern_key, data_json, updated_at FROM tisch_patterns ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            {
                "key": row["pattern_key"],
                "data": json.loads(row["data_json"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]


async def read_pattern(pattern_key: str) -> Optional[dict]:
    """Ein einzelnes Muster lesen."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT pattern_key, data_json, updated_at FROM tisch_patterns WHERE pattern_key = ?",
            (pattern_key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "key": row["pattern_key"],
            "data": json.loads(row["data_json"]),
            "updated_at": row["updated_at"],
        }


# ---------------------------------------------------------------------------
# Export (Vault-Sync-Interface)
# ---------------------------------------------------------------------------

async def export_for_vault(since: Optional[str] = None) -> dict:
    """Vollständiger Export für Vault-Sync.
    since: ISO-Zeitstempel — nur Sessions nach diesem Datum (optional).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if since:
            cursor = await db.execute(
                """SELECT * FROM tisch_sessions WHERE created_at > ?
                   ORDER BY created_at ASC""",
                (since,),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM tisch_sessions ORDER BY created_at ASC"
            )
        session_rows = await cursor.fetchall()

        pattern_cursor = await db.execute(
            "SELECT * FROM tisch_patterns ORDER BY updated_at DESC"
        )
        pattern_rows = await pattern_cursor.fetchall()

    sessions = []
    for row in session_rows:
        entry = dict(row)
        for field in ("perspectives_json", "friction_json", "integration_json"):
            raw = entry.pop(field, None)
            key = field.replace("_json", "")
            try:
                entry[key] = json.loads(raw) if raw else None
            except (json.JSONDecodeError, TypeError):
                entry[key] = None
        sessions.append(entry)

    patterns = [
        {
            "key": row["pattern_key"],
            "data": json.loads(row["data_json"]),
            "updated_at": row["updated_at"],
        }
        for row in pattern_rows
    ]

    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "session_count": len(sessions),
        "sessions": sessions,
        "pattern_count": len(patterns),
        "patterns": patterns,
    }
