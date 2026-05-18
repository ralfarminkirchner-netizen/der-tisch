"""
tisch_shared_core/store.py — JSONL-Persistenz für den TiSCH Shared Core.

PERSISTENZ-ENTSCHEIDUNG (Phase-0-Survey, 2026-05-18) — JSONL statt aiosqlite:
Der Auftrag nennt SQLite via aiosqlite als bevorzugt, erlaubt aber den
JSONL-Fallback „mit klarer Begründung". Begründung:
  1. `aiosqlite` ist im Repo nicht als Dependency deklariert (requirements.txt)
     und lokal nicht installiert. Die aiosqlite-Altmodule (moonfingers_store.py,
     shared_core_store.py) sind von api_server.py NICHT verdrahtet — der
     „Hausstil" ist faktisch toter Code.
  2. SQLite zu wählen erzwänge eine requirements.txt-Änderung plus eine neue
     Railway-Deploy-Abhängigkeit — entgegen der Auftrags-Erwartung „geänderte
     Dateien nur api_server.py + 1 Zeile".
  3. Der Feature-Bedarf (Capture, Hash-Dedupe, Token-Overlap-Suche,
     Context Packs, dateibasierter Obsidian-Round-Trip) braucht kein SQL.
  4. Vertrag Punkt 2 segnet JSONL fürs MVP ab: importierbar, versionierbar,
     prüfbar, ohne externe Platte lauffähig.
Migrationspfad: JSONL -> SQLite/Postgres, sobald Volumen/Concurrency es fordert.

Der Datenpfad ist modul-relativ (`Path(__file__).parent / "data"`) — KEIN
/Volumes-Pfad, keine produktive Abhängigkeit auf lokale Mounts (Vertrag).

Bekannte Grenze: Railway-Filesystem ist ephemer — JSONL überlebt keinen
Redeploy (gleiche Grenze wie das `.db`-Muster der Altmodule). Fürs MVP
akzeptiert; im Phasen-Report unter „Offene Risiken" vermerkt.

Schreibzugriffe sind per asyncio.Lock pro Collection serialisiert; das
Schreiben läuft atomar über tempfile + os.replace.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

# --- Speicherort -----------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

# --- Collections -----------------------------------------------------------
CANDIDATES = "candidates"
CARDS = "cards"
CONTEXT_PACKS = "context_packs"
_COLLECTIONS = {CANDIDATES, CARDS, CONTEXT_PACKS}

_locks: Dict[str, asyncio.Lock] = {}


def _lock(collection: str) -> asyncio.Lock:
    """Lazy pro-Collection-Lock (im laufenden Event-Loop erzeugt)."""
    lock = _locks.get(collection)
    if lock is None:
        lock = asyncio.Lock()
        _locks[collection] = lock
    return lock


def _path(collection: str) -> Path:
    if collection not in _COLLECTIONS:
        raise ValueError(f"unknown collection: {collection!r}")
    return DATA_DIR / f"{collection}.jsonl"


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- Synchroner Kern (läuft in asyncio.to_thread) --------------------------

def _read_all_sync(collection: str) -> List[dict]:
    path = _path(collection)
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Eine korrupte Zeile darf den Rest nicht killen.
                continue
    return records


def _write_all_sync(collection: str, records: List[dict]) -> None:
    _ensure_dir()
    path = _path(collection)
    fd, tmp = tempfile.mkstemp(dir=str(DATA_DIR), suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False))
                fh.write("\n")
        os.replace(tmp, path)  # atomarer Tausch
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --- Async-API -------------------------------------------------------------

async def all_records(collection: str) -> List[dict]:
    """Alle Records einer Collection (Reihenfolge wie in der Datei)."""
    return await asyncio.to_thread(_read_all_sync, collection)


async def get(collection: str, record_id: str) -> Optional[dict]:
    """Einen Record per URN-id; None wenn nicht vorhanden."""
    for rec in await all_records(collection):
        if rec.get("id") == record_id:
            return rec
    return None


async def upsert(collection: str, record: dict) -> dict:
    """Record per `id` einfügen oder ersetzen (last-write-wins)."""
    rid = record.get("id")
    if not rid:
        raise ValueError("record needs a non-empty 'id'")
    async with _lock(collection):
        records = await asyncio.to_thread(_read_all_sync, collection)
        for i, rec in enumerate(records):
            if rec.get("id") == rid:
                records[i] = record
                break
        else:
            records.append(record)
        await asyncio.to_thread(_write_all_sync, collection, records)
    return record


async def find(collection: str, predicate: Callable[[dict], bool]) -> List[dict]:
    """Records, die `predicate` erfüllen."""
    return [r for r in await all_records(collection) if predicate(r)]


async def count(collection: str) -> int:
    return len(await all_records(collection))


async def delete(collection: str, record_id: str) -> bool:
    """Record entfernen. True, wenn etwas gelöscht wurde."""
    async with _lock(collection):
        records = await asyncio.to_thread(_read_all_sync, collection)
        kept = [r for r in records if r.get("id") != record_id]
        if len(kept) == len(records):
            return False
        await asyncio.to_thread(_write_all_sync, collection, kept)
        return True
