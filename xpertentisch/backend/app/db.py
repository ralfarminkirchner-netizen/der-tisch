"""SQLite-Persistenz für XPERTENTiSCH.

Alles, was eine Sitzung ausmacht (Funken, Aufträge, Antworten, Marker,
Ereignisse), liegt in SQLite. Dadurch überlebt der Zustand einen
Serverneustart, und der Ereignisstrom kann nach einem SSE-Abbruch
lückenlos nachgeliefert werden.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'offen',
    created_at   REAL NOT NULL,
    closed_at    REAL,
    closing_note TEXT
);

CREATE TABLE IF NOT EXISTS sparks (
    id                TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq               INTEGER NOT NULL,
    prompt            TEXT NOT NULL,
    client_request_id TEXT NOT NULL,
    created_at        REAL NOT NULL,
    -- funke | antwort | weitergabe | gegenposition | vertiefung | szenario
    kind              TEXT NOT NULL DEFAULT 'funke',
    -- Ausdrücklich gewählte Bezugsbeiträge (JSON-Liste von Beitragskennungen)
    refs              TEXT NOT NULL DEFAULT '[]'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sparks_idempotency
    ON sparks(session_id, client_request_id);
CREATE INDEX IF NOT EXISTS idx_sparks_session ON sparks(session_id, seq);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    spark_id    TEXT NOT NULL REFERENCES sparks(id) ON DELETE CASCADE,
    model_id    TEXT NOT NULL,
    label       TEXT NOT NULL,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    status      TEXT NOT NULL,
    text        TEXT NOT NULL DEFAULT '',
    error       TEXT,
    partial     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    latency_ms  INTEGER,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    -- Kosten in Millionstel der Währung, in der die Preise eingetragen sind.
    cost_micro  INTEGER,
    -- 'berechnet' oder 'unbekannt' — nie geraten.
    cost_source TEXT NOT NULL DEFAULT 'unbekannt'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_spark_model ON jobs(spark_id, model_id);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id);

CREATE TABLE IF NOT EXISTS markers (
    id             TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    spark_id       TEXT NOT NULL REFERENCES sparks(id) ON DELETE CASCADE,
    job_id         TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    related_job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL,
    start_offset   INTEGER NOT NULL,
    end_offset     INTEGER NOT NULL,
    quote          TEXT NOT NULL,
    note           TEXT NOT NULL DEFAULT '',
    -- Die Begriffe, an denen diese Fundstelle hängt (JSON-Liste). Grundlage
    -- der Themen-Linse: sie zeigt, WORAN sich die Stimmen treffen.
    topics         TEXT NOT NULL DEFAULT '[]',
    created_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_markers_spark ON markers(spark_id);

CREATE TABLE IF NOT EXISTS assessments (
    spark_id   TEXT PRIMARY KEY REFERENCES sparks(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, id);

CREATE TABLE IF NOT EXISTS settings (
    name       TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- Was ein Auftrag tatsächlich zu sehen bekam. Unveränderlich ab Auftragsbeginn:
-- ein späterer Einwurf darf nicht rückwirkend zum Kenntnisstand erklärt werden.
CREATE TABLE IF NOT EXISTS context_snapshots (
    job_id      TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    session_id  TEXT NOT NULL,
    rendered    TEXT NOT NULL,
    message_ids TEXT NOT NULL,
    rule        TEXT NOT NULL,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    truncated   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

-- Beziehungen zwischen Beiträgen. Herkunft und Stand bleiben unterscheidbar:
-- was die Maschine vorschlägt, ist kein bestätigter Befund.
CREATE TABLE IF NOT EXISTS relations (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    from_id    TEXT NOT NULL,
    to_id      TEXT NOT NULL,
    type       TEXT NOT NULL,
    origin     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'bestaetigt',
    note       TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_relations_session ON relations(session_id);

CREATE TABLE IF NOT EXISTS providers (
    id         TEXT PRIMARY KEY,
    label      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    base_url   TEXT,
    model      TEXT NOT NULL,
    api_key    TEXT NOT NULL DEFAULT '',
    -- Preis je Million Token, in selbst gewählter Währung. NULL heißt unbekannt.
    price_in   REAL,
    price_out  REAL,
    enabled    INTEGER NOT NULL DEFAULT 0,
    is_preset  INTEGER NOT NULL DEFAULT 0,
    position   INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
"""

JOB_NOT_REQUESTED = "not_requested"
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_ERROR = "error"
JOB_INTERRUPTED = "interrupted"
JOB_CANCELLED = "cancelled"
JOB_STREAMING = "streaming"
OPEN_JOB_STATES = (JOB_QUEUED, JOB_RUNNING, JOB_STREAMING)

SESSION_OPEN = "offen"
SESSION_CLOSED = "abgeschlossen"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def now() -> float:
    return time.time()


def _spark_row(row: Any) -> dict[str, Any]:
    """Bezüge kommen als Liste heraus, nicht als JSON-Text."""
    spark = dict(row)
    roh = spark.get("refs") or "[]"
    try:
        spark["refs"] = json.loads(roh) if isinstance(roh, str) else list(roh)
    except ValueError:
        spark["refs"] = []
    spark.setdefault("kind", "funke")
    return spark


def _marker_row(row: Any) -> dict[str, Any]:
    """Themen kommen als Liste heraus, nicht als JSON-Text."""
    marker = dict(row)
    roh = marker.get("topics") or "[]"
    try:
        marker["topics"] = json.loads(roh) if isinstance(roh, str) else list(roh)
    except ValueError:
        marker["topics"] = []
    return marker


class Store:
    """Dünne, bewusst explizite Datenzugriffsschicht."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    # ---------------------------------------------------------------- Lifecycle

    async def connect(self) -> None:
        if self.db_path.parent and str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._migrate()
        await self._conn.commit()

    async def _migrate(self) -> None:
        """Ergänzt Spalten, die es in älteren Datenbanken noch nicht gab."""
        async with self._conn.execute("PRAGMA table_info(sparks)") as cur:  # type: ignore[union-attr]
            spalten = {r["name"] for r in await cur.fetchall()}
        if "kind" not in spalten:
            await self._conn.execute(  # type: ignore[union-attr]
                "ALTER TABLE sparks ADD COLUMN kind TEXT NOT NULL DEFAULT 'funke'"
            )
        if "refs" not in spalten:
            await self._conn.execute(  # type: ignore[union-attr]
                "ALTER TABLE sparks ADD COLUMN refs TEXT NOT NULL DEFAULT '[]'"
            )

        async with self._conn.execute("PRAGMA table_info(markers)") as cur:  # type: ignore[union-attr]
            marker_spalten = {r["name"] for r in await cur.fetchall()}
        if "topics" not in marker_spalten:
            await self._conn.execute(  # type: ignore[union-attr]
                "ALTER TABLE markers ADD COLUMN topics TEXT NOT NULL DEFAULT '[]'"
            )

        async with self._conn.execute("PRAGMA table_info(jobs)") as cur:  # type: ignore[union-attr]
            job_spalten = {r["name"] for r in await cur.fetchall()}
        for name, typ in (
            ("tokens_in", "INTEGER"),
            ("tokens_out", "INTEGER"),
            ("cost_micro", "INTEGER"),
            ("cost_source", "TEXT NOT NULL DEFAULT 'unbekannt'"),
        ):
            if name not in job_spalten:
                await self._conn.execute(  # type: ignore[union-attr]
                    f"ALTER TABLE jobs ADD COLUMN {name} {typ}"
                )

        async with self._conn.execute("PRAGMA table_info(providers)") as cur:  # type: ignore[union-attr]
            anbieter_spalten = {r["name"] for r in await cur.fetchall()}
        for name in ("price_in", "price_out"):
            if name not in anbieter_spalten:
                await self._conn.execute(  # type: ignore[union-attr]
                    f"ALTER TABLE providers ADD COLUMN {name} REAL"
                )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Store ist nicht verbunden.")
        return self._conn

    async def healthy(self) -> bool:
        try:
            async with self.conn.execute("SELECT 1") as cur:
                await cur.fetchone()
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------- Sessions

    async def create_session(self, title: str) -> dict[str, Any]:
        sid = new_id("ses")
        ts = now()
        await self.conn.execute(
            "INSERT INTO sessions (id, title, status, created_at) VALUES (?,?,?,?)",
            (sid, title, SESSION_OPEN, ts),
        )
        await self.conn.commit()
        return {"id": sid, "title": title, "status": SESSION_OPEN, "created_at": ts,
                "closed_at": None, "closing_note": None}

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def close_session(self, session_id: str, note: str) -> None:
        await self.conn.execute(
            "UPDATE sessions SET status=?, closed_at=?, closing_note=? WHERE id=?",
            (SESSION_CLOSED, now(), note, session_id),
        )
        await self.conn.commit()

    # ------------------------------------------------------------------ Sparks

    async def find_spark_by_request_id(
        self, session_id: str, client_request_id: str
    ) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM sparks WHERE session_id=? AND client_request_id=?",
            (session_id, client_request_id),
        ) as cur:
            row = await cur.fetchone()
        return _spark_row(row) if row else None

    async def insert_spark(
        self,
        session_id: str,
        prompt: str,
        client_request_id: str,
        kind: str = "funke",
        refs: list[str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Legt einen Funken an. Gibt (Funke, war_neu) zurück.

        Die Eindeutigkeit von (session_id, client_request_id) verhindert,
        dass eine doppelt übertragene Anfrage doppelte Modellaufrufe erzeugt.
        """
        existing = await self.find_spark_by_request_id(session_id, client_request_id)
        if existing:
            return existing, False

        async with self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM sparks WHERE session_id=?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        seq = int(row["next"])

        spark = {
            "id": new_id("fnk"),
            "session_id": session_id,
            "seq": seq,
            "prompt": prompt,
            "client_request_id": client_request_id,
            "created_at": now(),
            "kind": kind,
            # Nach außen immer eine Liste. In der Datenbank steht JSON-Text;
            # wer hier eine Zeichenkette durchreicht, iteriert sonst Buchstaben.
            "refs": list(refs or []),
        }
        try:
            await self.conn.execute(
                "INSERT INTO sparks (id, session_id, seq, prompt, client_request_id,"
                " created_at, kind, refs) VALUES (:id, :session_id, :seq, :prompt,"
                " :client_request_id, :created_at, :kind, :refs)",
                {**spark, "refs": json.dumps(spark["refs"], ensure_ascii=False)},
            )
            await self.conn.commit()
        except aiosqlite.IntegrityError:
            # Wettlauf zweier identischer Anfragen: der zuerst eingetragene gewinnt.
            await self.conn.rollback()
            existing = await self.find_spark_by_request_id(session_id, client_request_id)
            if existing is None:
                raise
            return existing, False
        return spark, True

    async def get_spark(self, spark_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM sparks WHERE id=?", (spark_id,)
        ) as cur:
            row = await cur.fetchone()
        return _spark_row(row) if row else None

    async def list_sparks(self, session_id: str) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM sparks WHERE session_id=? ORDER BY seq", (session_id,)
        ) as cur:
            return [_spark_row(r) for r in await cur.fetchall()]

    # -------------------------------------------------------------------- Jobs

    async def insert_jobs(self, jobs: Iterable[dict[str, Any]]) -> None:
        await self.conn.executemany(
            "INSERT OR IGNORE INTO jobs "
            "(id, session_id, spark_id, model_id, label, provider, model, status, text,"
            " error, partial, created_at) "
            "VALUES (:id, :session_id, :spark_id, :model_id, :label, :provider, :model,"
            " :status, :text, :error, :partial, :created_at)",
            list(jobs),
        )
        await self.conn.commit()

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        async with self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def list_jobs(self, session_id: str) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM jobs WHERE session_id=? ORDER BY created_at", (session_id,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def list_jobs_for_spark(self, spark_id: str) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM jobs WHERE spark_id=? ORDER BY created_at", (spark_id,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def mark_job_running(self, job_id: str) -> float:
        """Setzt den Auftrag auf „läuft" und gibt den festgehaltenen Zeitpunkt zurück.

        Der Rückgabewert geht in das Ereignis: sonst kennte die Oberfläche die
        gemessenen Zeiten erst nach dem nächsten vollständigen Laden — und die
        Zeitlinse zeigte während der Runde ein falsches Bild.
        """
        ts = now()
        await self.conn.execute(
            "UPDATE jobs SET status=?, started_at=? WHERE id=?",
            (JOB_RUNNING, ts, job_id),
        )
        await self.conn.commit()
        return ts

    async def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        text: str = "",
        error: str | None = None,
        partial: bool = False,
        latency_ms: int | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_micro: int | None = None,
        cost_source: str = "unbekannt",
    ) -> float:
        """Schließt den Auftrag ab und gibt den festgehaltenen Zeitpunkt zurück."""
        ts = now()
        await self.conn.execute(
            "UPDATE jobs SET status=?, text=?, error=?, partial=?, finished_at=?,"
            " latency_ms=?, tokens_in=?, tokens_out=?, cost_micro=?, cost_source=?"
            " WHERE id=?",
            (status, text, error, 1 if partial else 0, ts, latency_ms,
             tokens_in, tokens_out, cost_micro, cost_source, job_id),
        )
        await self.conn.commit()
        return ts

    async def append_job_text(self, job_id: str, text: str) -> None:
        """Hält den Zwischenstand fest, während die Antwort noch entsteht."""
        await self.conn.execute("UPDATE jobs SET text=? WHERE id=?", (text, job_id))
        await self.conn.commit()

    async def set_job_status(
        self, job_id: str, status: str, error: str | None = None
    ) -> float:
        """Setzt einen Endzustand und gibt den festgehaltenen Zeitpunkt zurück."""
        ts = now()
        await self.conn.execute(
            "UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?",
            (status, error, ts, job_id),
        )
        await self.conn.commit()
        return ts

    async def mark_open_jobs_interrupted(self) -> list[dict[str, Any]]:
        """Nach einem Neustart: laufende Aufträge als unterbrochen kennzeichnen."""
        placeholders = ",".join("?" for _ in OPEN_JOB_STATES)
        async with self.conn.execute(
            f"SELECT * FROM jobs WHERE status IN ({placeholders})", OPEN_JOB_STATES
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        if not rows:
            return []
        await self.conn.execute(
            f"UPDATE jobs SET status=?, error=?, finished_at=? WHERE status IN ({placeholders})",
            (
                JOB_INTERRUPTED,
                "Durch Serverneustart unterbrochen.",
                now(),
                *OPEN_JOB_STATES,
            ),
        )
        await self.conn.commit()
        return rows

    # ----------------------------------------------------------------- Marker

    async def replace_markers(
        self, spark_id: str, session_id: str, markers: list[dict[str, Any]]
    ) -> None:
        await self.conn.execute("DELETE FROM markers WHERE spark_id=?", (spark_id,))
        if markers:
            await self.conn.executemany(
                "INSERT INTO markers (id, session_id, spark_id, job_id, related_job_id,"
                " kind, start_offset, end_offset, quote, note, topics, created_at) "
                "VALUES (:id, :session_id, :spark_id, :job_id, :related_job_id, :kind,"
                " :start_offset, :end_offset, :quote, :note, :topics, :created_at)",
                [{**m, "topics": json.dumps(m.get("topics") or [])} for m in markers],
            )
        await self.conn.commit()

    async def list_markers(self, session_id: str) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM markers WHERE session_id=? ORDER BY created_at", (session_id,)
        ) as cur:
            return [_marker_row(r) for r in await cur.fetchall()]

    async def list_markers_for_spark(self, spark_id: str) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM markers WHERE spark_id=? ORDER BY created_at", (spark_id,)
        ) as cur:
            return [_marker_row(r) for r in await cur.fetchall()]

    # ------------------------------------------------------------ Assessments

    async def save_assessment(
        self, spark_id: str, session_id: str, payload: dict[str, Any]
    ) -> None:
        await self.conn.execute(
            "INSERT INTO assessments (spark_id, session_id, payload, created_at) "
            "VALUES (?,?,?,?) ON CONFLICT(spark_id) DO UPDATE SET payload=excluded.payload,"
            " created_at=excluded.created_at",
            (spark_id, session_id, json.dumps(payload, ensure_ascii=False), now()),
        )
        await self.conn.commit()

    async def list_assessments(self, session_id: str) -> dict[str, Any]:
        async with self.conn.execute(
            "SELECT spark_id, payload FROM assessments WHERE session_id=?", (session_id,)
        ) as cur:
            rows = await cur.fetchall()
        return {r["spark_id"]: json.loads(r["payload"]) for r in rows}

    # ------------------------------------------------------------ Einstellungen

    async def all_settings(self) -> dict[str, str]:
        async with self.conn.execute("SELECT name, value FROM settings") as cur:
            return {r["name"]: r["value"] for r in await cur.fetchall()}

    async def set_setting(self, name: str, value: str) -> None:
        await self.conn.execute(
            "INSERT INTO settings (name, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET value=excluded.value,"
            " updated_at=excluded.updated_at",
            (name, value, now()),
        )
        await self.conn.commit()

    async def delete_setting(self, name: str) -> None:
        await self.conn.execute("DELETE FROM settings WHERE name=?", (name,))
        await self.conn.commit()

    async def settings_updated_at(self) -> dict[str, float]:
        async with self.conn.execute("SELECT name, updated_at FROM settings") as cur:
            return {r["name"]: r["updated_at"] for r in await cur.fetchall()}

    # ------------------------------------------------------ Kontext-Schnappschuss

    async def save_context_snapshot(self, snapshot: dict[str, Any]) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO context_snapshots (job_id, session_id, rendered,"
            " message_ids, rule, provider, model, truncated, created_at) "
            "VALUES (:job_id, :session_id, :rendered, :message_ids, :rule, :provider,"
            " :model, :truncated, :created_at)",
            {**snapshot, "created_at": now()},
        )
        await self.conn.commit()

    async def get_context_snapshot(self, job_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM context_snapshots WHERE job_id=?", (job_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        snapshot = dict(row)
        snapshot["message_ids"] = json.loads(snapshot["message_ids"])
        snapshot["truncated"] = bool(snapshot["truncated"])
        return snapshot

    async def list_context_snapshots(self, session_id: str) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM context_snapshots WHERE session_id=?", (session_id,)
        ) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            snapshot = dict(r)
            snapshot["message_ids"] = json.loads(snapshot["message_ids"])
            snapshot["truncated"] = bool(snapshot["truncated"])
            out.append(snapshot)
        return out

    # ---------------------------------------------------------------- Beziehungen

    async def insert_relation(self, relation: dict[str, Any]) -> dict[str, Any]:
        record = {**relation, "id": relation.get("id") or new_id("bez"), "created_at": now()}
        await self.conn.execute(
            "INSERT INTO relations (id, session_id, from_id, to_id, type, origin, status,"
            " note, created_at) VALUES (:id, :session_id, :from_id, :to_id, :type, :origin,"
            " :status, :note, :created_at)",
            record,
        )
        await self.conn.commit()
        return record

    async def list_relations(self, session_id: str) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM relations WHERE session_id=? ORDER BY created_at", (session_id,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def delete_machine_relations(self, session_id: str, job_ids: list[str]) -> None:
        """Entfernt frühere Vorschläge zu diesen Aufträgen, menschliche bleiben."""
        if not job_ids:
            return
        platzhalter = ",".join("?" for _ in job_ids)
        await self.conn.execute(
            f"DELETE FROM relations WHERE session_id=? AND origin='maschine' "
            f"AND (from_id IN ({platzhalter}) OR to_id IN ({platzhalter}))",
            (session_id, *job_ids, *job_ids),
        )
        await self.conn.commit()

    async def set_relation_status(self, relation_id: str, status: str) -> None:
        await self.conn.execute(
            "UPDATE relations SET status=? WHERE id=?", (status, relation_id)
        )
        await self.conn.commit()

    # ----------------------------------------------------------------- Anbieter

    async def list_providers(self) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM providers ORDER BY position, label"
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_provider(self, provider_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM providers WHERE id=?", (provider_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def insert_provider(self, record: dict[str, Any]) -> None:
        await self.conn.execute(
            "INSERT INTO providers (id, label, kind, base_url, model, api_key, enabled,"
            " is_preset, position, price_in, price_out, updated_at) VALUES (:id, :label,"
            " :kind, :base_url, :model, :api_key, :enabled, :is_preset, :position,"
            " :price_in, :price_out, :updated_at)",
            {"price_in": None, "price_out": None, **record, "updated_at": now()},
        )
        await self.conn.commit()

    async def update_provider(self, provider_id: str, felder: dict[str, Any]) -> None:
        """Ändert genau die übergebenen Spalten."""
        erlaubt = {"label", "kind", "base_url", "model", "api_key", "enabled",
                   "position", "price_in", "price_out"}
        gesetzt = {k: v for k, v in felder.items() if k in erlaubt}
        if not gesetzt:
            return
        zuweisung = ", ".join(f"{k}=:{k}" for k in gesetzt)
        await self.conn.execute(
            f"UPDATE providers SET {zuweisung}, updated_at=:updated_at WHERE id=:id",
            {**gesetzt, "id": provider_id, "updated_at": now()},
        )
        await self.conn.commit()

    async def delete_provider(self, provider_id: str) -> None:
        await self.conn.execute("DELETE FROM providers WHERE id=?", (provider_id,))
        await self.conn.commit()

    async def ensure_presets(
        self, presets: Iterable[Any], env_keys: dict[str, str | None]
    ) -> list[str]:
        """Legt fehlende mitgelieferte Anbieter an. Bestehende bleiben unangetastet."""
        vorhanden = {p["id"] for p in await self.list_providers()}
        neu: list[str] = []
        for index, preset in enumerate(presets):
            if preset.id in vorhanden:
                continue
            await self.insert_provider(
                {
                    "id": preset.id,
                    "label": preset.label,
                    "kind": preset.kind,
                    "base_url": preset.base_url,
                    "model": preset.default_model,
                    "api_key": "",
                    # Ohne Schlüssel in der Umgebung startet ein Anbieter abgeschaltet,
                    # damit der Tisch nicht mit Fehlermeldungen beginnt.
                    "enabled": 1 if env_keys.get(preset.key_env) else 0,
                    "is_preset": 1,
                    "position": index,
                }
            )
            neu.append(preset.id)
        return neu

    # ---------------------------------------------------------------- Events

    async def append_event(
        self, session_id: str, type_: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        ts = now()
        cur = await self.conn.execute(
            "INSERT INTO events (session_id, type, payload, created_at) VALUES (?,?,?,?)",
            (session_id, type_, json.dumps(payload, ensure_ascii=False), ts),
        )
        await self.conn.commit()
        return {
            "id": int(cur.lastrowid),
            "session_id": session_id,
            "type": type_,
            "payload": payload,
            "created_at": ts,
        }

    async def events_since(
        self, session_id: str, last_event_id: int, limit: int = 1000
    ) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM events WHERE session_id=? AND id > ? ORDER BY id LIMIT ?",
            (session_id, last_event_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "type": r["type"],
                "payload": json.loads(r["payload"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def last_event_id(self, session_id: str) -> int:
        async with self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM events WHERE session_id=?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["m"])
