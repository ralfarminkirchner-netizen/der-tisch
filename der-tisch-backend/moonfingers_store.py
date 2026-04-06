"""
moonfingers_store.py — SQLite-based persistence for the MOONFiNGERS app.
Uses aiosqlite for async DB access. Import and use from api_server.py.

Dependency: aiosqlite
"""

import aiosqlite
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

# ---------------------------------------------------------------------------
# DB path — sits next to this module
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "moonfingers.db"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class NoteCreate(BaseModel):
    entity_id: str
    content: str


class NoteUpdate(BaseModel):
    content: Optional[str] = None
    is_bookmark: Optional[bool] = None


class NoteResponse(BaseModel):
    id: int
    entity_id: str
    content: str
    is_bookmark: bool
    created_at: str
    updated_at: str


class UserContentCreate(BaseModel):
    entity_id: str
    content_type: str  # 'text' or 'image_url'
    content: str
    title: Optional[str] = None


class UserContentResponse(BaseModel):
    id: int
    entity_id: str
    content_type: str
    content: str
    title: Optional[str]
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_note(row: aiosqlite.Row) -> dict:
    return {
        "id": row["id"],
        "entity_id": row["entity_id"],
        "content": row["content"],
        "is_bookmark": bool(row["is_bookmark"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_user_content(row: aiosqlite.Row) -> dict:
    return {
        "id": row["id"],
        "entity_id": row["entity_id"],
        "content_type": row["content_type"],
        "content": row["content"],
        "title": row["title"],
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Initialize the database and create tables if they do not exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id   TEXT NOT NULL,
                content     TEXT NOT NULL,
                is_bookmark BOOLEAN DEFAULT 0,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_content (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id    TEXT NOT NULL,
                content_type TEXT NOT NULL,
                content      TEXT NOT NULL,
                title        TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


# ---------------------------------------------------------------------------
# Notes CRUD
# ---------------------------------------------------------------------------

async def get_notes(entity_id: Optional[str] = None) -> list:
    """Return all notes, optionally filtered by entity_id."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if entity_id:
            cursor = await db.execute(
                "SELECT * FROM notes WHERE entity_id = ? ORDER BY created_at DESC",
                (entity_id,),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM notes ORDER BY created_at DESC"
            )
        rows = await cursor.fetchall()
        return [_row_to_note(row) for row in rows]


async def create_note(entity_id: str, content: str) -> dict:
    """Insert a new note and return the created record."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "INSERT INTO notes (entity_id, content) VALUES (?, ?)",
            (entity_id, content),
        )
        await db.commit()
        row = await (await db.execute(
            "SELECT * FROM notes WHERE id = ?", (cursor.lastrowid,)
        )).fetchone()
        return _row_to_note(row)


async def update_note(
    note_id: int,
    content: Optional[str] = None,
    is_bookmark: Optional[bool] = None,
) -> dict:
    """
    Update a note's content and/or bookmark flag.
    Returns the updated record, or raises ValueError if not found.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Build SET clause dynamically
        fields = []
        values = []
        if content is not None:
            fields.append("content = ?")
            values.append(content)
        if is_bookmark is not None:
            fields.append("is_bookmark = ?")
            values.append(int(is_bookmark))

        if not fields:
            # Nothing to update — return existing record
            row = await (await db.execute(
                "SELECT * FROM notes WHERE id = ?", (note_id,)
            )).fetchone()
            if row is None:
                raise ValueError(f"Note {note_id} not found")
            return _row_to_note(row)

        fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(note_id)

        await db.execute(
            f"UPDATE notes SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        await db.commit()

        row = await (await db.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        )).fetchone()
        if row is None:
            raise ValueError(f"Note {note_id} not found")
        return _row_to_note(row)


async def delete_note(note_id: int) -> bool:
    """Delete a note by id. Returns True if a row was deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM notes WHERE id = ?", (note_id,)
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_bookmarks() -> list:
    """Return all bookmarked notes."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM notes WHERE is_bookmark = 1 ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [_row_to_note(row) for row in rows]


# ---------------------------------------------------------------------------
# User Content CRUD
# ---------------------------------------------------------------------------

async def get_user_content(entity_id: str) -> list:
    """Return all user_content entries for a given entity_id."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM user_content WHERE entity_id = ? ORDER BY created_at DESC",
            (entity_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_user_content(row) for row in rows]


async def add_user_content(
    entity_id: str,
    content_type: str,
    content: str,
    title: Optional[str] = None,
) -> dict:
    """Insert a new user_content entry and return the created record."""
    if content_type not in ("text", "image_url"):
        raise ValueError("content_type must be 'text' or 'image_url'")
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "INSERT INTO user_content (entity_id, content_type, content, title) VALUES (?, ?, ?, ?)",
            (entity_id, content_type, content, title),
        )
        await db.commit()
        row = await (await db.execute(
            "SELECT * FROM user_content WHERE id = ?", (cursor.lastrowid,)
        )).fetchone()
        return _row_to_user_content(row)


async def delete_user_content(content_id: int) -> bool:
    """Delete a user_content entry by id. Returns True if a row was deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM user_content WHERE id = ?", (content_id,)
        )
        await db.commit()
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

async def export_notebook() -> dict:
    """Export all notes and bookmarks as a JSON-serialisable dict."""
    all_notes = await get_notes()
    bookmarks = [n for n in all_notes if n["is_bookmark"]]
    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "notes": all_notes,
        "bookmarks": bookmarks,
        "total_notes": len(all_notes),
        "total_bookmarks": len(bookmarks),
    }
