"""
tisch_shared_core/store.py — SQLite-Persistenz für den TiSCH Shared Core.

PERSISTENZ: SQLite via `aiosqlite` (async). Dies ist die Auftrags-Präferenz
("bevorzugt SQLite via aiosqlite, im Stil von moonfingers_store.py") und der
Hausstil der bestehenden Stores (`moonfingers_store.py`, `shared_core_store.py`).
`aiosqlite` ist in `requirements.txt` deklariert.

Modell: eine Tabelle je Collection. Jeder Record wird als kompletter
JSON-Blob in der Spalte `data` abgelegt (Schema-frei gegenüber den
Pydantic-Modellen), mit `id` (URN) als Primärschlüssel. Komplexe Felder als
JSON — identisches Muster wie `shared_core_store.py`.

Der DB-Pfad ist modul-relativ (`Path(__file__).parent / "data" /
"tisch_shared_core.db"`) — KEIN /Volumes-Pfad, keine produktive Abhängigkeit
auf lokale Mounts (Vertrag).

Bekannte Grenze: Das Railway-Filesystem ist ephemer — die `.db`-Datei
überlebt keinen Redeploy (dieselbe Grenze wie bei JSONL). Echte Dauerhaftigkeit
(Railway-Volume / Postgres) ist ein separates, späteres Thema. Im
Phasen-Report unter „Offene Risiken" vermerkt.

Die öffentliche API ist async und dict-basiert (dict rein, dict raus). Das
Schema wird beim ersten Zugriff idempotent angelegt (`CREATE TABLE IF NOT
EXISTS`) — kein Startup-Hook nötig (`api_server.py` hat keinen).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import aiosqlite

# --- Speicherort -----------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "tisch_shared_core.db"

# --- Collections (= Tabellennamen) -----------------------------------------
CANDIDATES = "candidates"
CARDS = "cards"
CONTEXT_PACKS = "context_packs"
_COLLECTIONS = (CANDIDATES, CARDS, CONTEXT_PACKS)

_schema_ready = False
_init_lock: Optional[asyncio.Lock] = None
_write_locks: Dict[str, asyncio.Lock] = {}


def _get_init_lock() -> asyncio.Lock:
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


def _write_lock(collection: str) -> asyncio.Lock:
    lock = _write_locks.get(collection)
    if lock is None:
        lock = asyncio.Lock()
        _write_locks[collection] = lock
    return lock


def _check_collection(collection: str) -> None:
    if collection not in _COLLECTIONS:
        raise ValueError(f"unknown collection: {collection!r}")


async def _ensure_schema() -> None:
    """Tabellen idempotent anlegen (einmal je Prozess)."""
    global _schema_ready
    if _schema_ready:
        return
    async with _get_init_lock():
        if _schema_ready:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(DB_PATH) as db:
            for table in _COLLECTIONS:
                await db.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id          TEXT PRIMARY KEY,
                        kind        TEXT,
                        data        TEXT NOT NULL,
                        created_at  TEXT,
                        updated_at  TEXT
                    )
                    """
                )
            await db.commit()
        _schema_ready = True


# --- Async-API -------------------------------------------------------------

async def all_records(collection: str) -> List[dict]:
    """Alle Records einer Collection (Einfüge-Reihenfolge)."""
    _check_collection(collection)
    await _ensure_schema()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(f"SELECT data FROM {collection} ORDER BY rowid ASC")
        rows = await cursor.fetchall()
    return [json.loads(row[0]) for row in rows]


async def get(collection: str, record_id: str) -> Optional[dict]:
    """Einen Record per URN-id; None, wenn nicht vorhanden."""
    _check_collection(collection)
    await _ensure_schema()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            f"SELECT data FROM {collection} WHERE id = ?", (record_id,)
        )
        row = await cursor.fetchone()
    return json.loads(row[0]) if row else None


async def upsert(collection: str, record: dict) -> dict:
    """Record per `id` einfügen oder ersetzen (last-write-wins)."""
    _check_collection(collection)
    rid = record.get("id")
    if not rid:
        raise ValueError("record needs a non-empty 'id'")
    await _ensure_schema()
    payload = json.dumps(record, ensure_ascii=False)
    async with _write_lock(collection):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                f"""
                INSERT INTO {collection} (id, kind, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    data = excluded.data,
                    updated_at = excluded.updated_at
                """,
                (
                    rid,
                    record.get("kind"),
                    payload,
                    record.get("created_at"),
                    record.get("updated_at"),
                ),
            )
            await db.commit()
    return record


async def find(collection: str, predicate: Callable[[dict], bool]) -> List[dict]:
    """Records, die `predicate` erfüllen (Filter in Python)."""
    return [r for r in await all_records(collection) if predicate(r)]


async def count(collection: str) -> int:
    _check_collection(collection)
    await _ensure_schema()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(f"SELECT COUNT(*) FROM {collection}")
        row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def delete(collection: str, record_id: str) -> bool:
    """Record entfernen. True, wenn etwas gelöscht wurde."""
    _check_collection(collection)
    await _ensure_schema()
    async with _write_lock(collection):
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                f"DELETE FROM {collection} WHERE id = ?", (record_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
