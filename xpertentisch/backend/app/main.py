"""XPERTENTiSCH — HTTP-Schnittstelle."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .catalog import CUSTOM_HINTS, KIND_LABELS, KIND_OPENAI, KINDS
from .config import Settings, load_settings
from .db import SESSION_CLOSED, Store, now
from .events import EventBus
from .provider_table import ProviderTable
from .providers import ProviderError, ProviderRegistry
from .reports import render_html, render_markdown
from .runner import Runner

log = logging.getLogger("xpertentisch")

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class SessionCreate(BaseModel):
    title: str = Field(default="Neue Sitzung", max_length=200)


class SparkCreate(BaseModel):
    prompt: str = Field(min_length=1)
    client_request_id: str = Field(min_length=4, max_length=100)
    model_ids: list[str] | None = None


class SessionClose(BaseModel):
    note: str = Field(default="", max_length=5000)


class SettingsUpdate(BaseModel):
    """Allgemeine Einstellungen. `None` heißt unverändert."""

    request_timeout_s: int | None = Field(default=None, ge=5, le=600)


class ProviderCreate(BaseModel):
    """Ein selbst eingetragener Anbieter."""

    label: str = Field(min_length=1, max_length=60)
    kind: str = Field(default=KIND_OPENAI)
    base_url: str | None = Field(default=None, max_length=300)
    model: str = Field(min_length=1, max_length=160)
    api_key: str = Field(default="", max_length=400)
    enabled: bool = True


class ProviderUpdate(BaseModel):
    """Änderung an einem Anbieter. `None` heißt unverändert.

    Ein leerer `api_key` löscht den hinterlegten Schlüssel; danach greift
    wieder die Umgebungsvariable, falls es eine gibt.
    """

    label: str | None = Field(default=None, min_length=1, max_length=60)
    base_url: str | None = Field(default=None, max_length=300)
    model: str | None = Field(default=None, min_length=1, max_length=160)
    api_key: str | None = Field(default=None, max_length=400)
    enabled: bool | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = Store(settings.db_path)
    bus = EventBus(store)
    table = ProviderTable(settings)
    registry = ProviderRegistry(settings, table)
    runner = Runner(store, bus, registry, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.connect()
        # Auf der Einstellungsseite hinterlegte Werte haben Vorrang vor der Umgebung.
        settings.overrides = await store.all_settings()
        await table.reload(store)
        # Ein Neustart darf keine Aufträge im Schwebezustand hinterlassen.
        interrupted = await store.mark_open_jobs_interrupted()
        for job in interrupted:
            await bus.publish(
                job["session_id"],
                "auftrag.unterbrochen",
                {
                    "job_id": job["id"],
                    "model_id": job["model_id"],
                    "label": job["label"],
                    "error": "Durch Serverneustart unterbrochen.",
                },
            )
        for spark_id in {j["spark_id"] for j in interrupted}:
            job_row = next(j for j in interrupted if j["spark_id"] == spark_id)
            await runner.recompute_assessment(job_row["session_id"], spark_id)
        if interrupted:
            log.warning("%d Aufträge als unterbrochen gekennzeichnet", len(interrupted))
        if settings.allow_fake_providers:
            log.warning(
                "TEST-PROVIDER AKTIV (XT_ALLOW_FAKE_PROVIDERS) — Umgebung: %s", settings.env
            )
        try:
            yield
        finally:
            await runner.shutdown()
            await registry.aclose()
            await store.close()

    app = FastAPI(title="XPERTENTiSCH", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.bus = bus
    app.state.registry = registry
    app.state.runner = runner
    app.state.table = table

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------- Grundlagen

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        db_ok = await store.healthy()
        providers = registry.availability()
        ready = db_ok and any(p["ready"] for p in providers.values())
        return {
            "status": "ok" if ready else "degraded",
            "env": settings.env,
            "database": "ok" if db_ok else "fehler",
            "providers": providers,
            "fake_providers_enabled": settings.allow_fake_providers,
            "missing_credentials": table.missing_keys(),
            "time": now(),
        }

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        """Öffentliche Konfiguration. Enthält keine Zugangsdaten."""
        return {
            "models": [
                {"id": m.id, "label": m.label, "provider": m.provider, "model": m.model}
                for m in table.as_models()
            ],
            "max_prompt_chars": settings.max_prompt_chars,
            "fake_providers_enabled": settings.allow_fake_providers,
            "env": settings.env,
            # Sagt nur, OB ein Zugangswort eingerichtet ist — nie welches.
            "settings_available": bool(settings.admin_token),
        }

    # -------------------------------------------------------------- Sitzungen

    @app.post("/api/sessions", status_code=201)
    async def create_session(body: SessionCreate) -> dict[str, Any]:
        session = await store.create_session(body.title.strip() or "Neue Sitzung")
        await bus.publish(session["id"], "sitzung.angelegt", {"session": session})
        return session

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:
        return {"sessions": await store.list_sessions()}

    async def _require_session(session_id: str) -> dict[str, Any]:
        session = await store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Sitzung nicht gefunden.")
        return session

    async def _bundle(session_id: str) -> dict[str, Any]:
        session = await _require_session(session_id)
        sparks = await store.list_sparks(session_id)
        jobs = await store.list_jobs(session_id)
        markers = await store.list_markers(session_id)
        summaries = await store.list_assessments(session_id)
        by_spark: dict[str, dict[str, Any]] = {
            s["id"]: {"spark": s, "jobs": [], "markers": [], "summary": summaries.get(s["id"])}
            for s in sparks
        }
        for job in jobs:
            entry = by_spark.get(job["spark_id"])
            if entry:
                job = dict(job)
                job["partial"] = bool(job["partial"])
                entry["jobs"].append(job)
        for marker in markers:
            entry = by_spark.get(marker["spark_id"])
            if entry:
                entry["markers"].append(marker)
        return {
            "session": session,
            "sparks": [by_spark[s["id"]] for s in sparks],
            "last_event_id": await store.last_event_id(session_id),
            "exported_at": now(),
        }

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        return await _bundle(session_id)

    @app.post("/api/sessions/{session_id}/close")
    async def close_session(session_id: str, body: SessionClose) -> dict[str, Any]:
        session = await _require_session(session_id)
        if session["status"] == SESSION_CLOSED:
            return session
        await store.close_session(session_id, body.note.strip())
        session = await _require_session(session_id)
        await bus.publish(session_id, "sitzung.abgeschlossen", {"session": session})
        return session

    # ------------------------------------------------------------------ Funken

    @app.post("/api/sessions/{session_id}/sparks", status_code=201)
    async def create_spark(session_id: str, body: SparkCreate, response: Response) -> dict[str, Any]:
        session = await _require_session(session_id)
        if session["status"] == SESSION_CLOSED:
            raise HTTPException(status_code=409, detail="Sitzung ist abgeschlossen.")

        prompt = body.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="Der Funke ist leer.")
        if len(prompt) > settings.max_prompt_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Der Funke ist länger als {settings.max_prompt_chars} Zeichen.",
            )

        models = table.as_models()
        if body.model_ids:
            wanted = set(body.model_ids)
            models = [m for m in models if m.id in wanted]
        if not models:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Kein einsatzbereiter Anbieter. In den Einstellungen mindestens "
                    "einen Anbieter einschalten und mit einem Schlüssel versehen."
                ),
            )

        spark, created = await store.insert_spark(session_id, prompt, body.client_request_id)
        if not created:
            # Doppelte Übertragung: bestehender Stand, keine neuen Modellaufrufe.
            response.status_code = 200
            return {
                "spark": spark,
                "jobs": await store.list_jobs_for_spark(spark["id"]),
                "duplicate": True,
            }

        jobs = await runner.launch_spark(session_id, spark, models)
        return {"spark": spark, "jobs": jobs, "duplicate": False}

    # -------------------------------------------------------------------- SSE

    @app.get("/api/sessions/{session_id}/events")
    async def events(
        session_id: str,
        last_event_id: int = 0,
        last_event_id_header: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        await _require_session(session_id)
        cursor = last_event_id
        if last_event_id_header:
            try:
                cursor = max(cursor, int(last_event_id_header))
            except ValueError:
                pass

        async def generate():
            # Trennt der Client, bricht Starlette diesen Generator ab; der
            # finally-Block in EventBus.stream meldet den Abonnenten ab.
            yield b"retry: 3000\n\n"
            try:
                async for event in bus.stream(session_id, cursor):
                    if event is None:
                        yield b": keepalive\n\n"
                        continue
                    data = json.dumps(event["payload"], ensure_ascii=False)
                    chunk = f"id: {event['id']}\nevent: {event['type']}\ndata: {data}\n\n"
                    yield chunk.encode("utf-8")
            except asyncio.CancelledError:  # pragma: no cover
                raise

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------------------------------------------------------- Berichte

    @app.get("/api/sessions/{session_id}/report.html", response_class=HTMLResponse)
    async def report_html(session_id: str) -> HTMLResponse:
        bundle = await _bundle(session_id)
        html_text = render_html(bundle)
        filename = f"xpertentisch-{session_id}.html"
        return HTMLResponse(
            html_text,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            },
        )

    @app.get("/api/sessions/{session_id}/report.md", response_class=PlainTextResponse)
    async def report_markdown(session_id: str) -> PlainTextResponse:
        bundle = await _bundle(session_id)
        filename = f"xpertentisch-{session_id}.md"
        return PlainTextResponse(
            render_markdown(bundle),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ------------------------------------------------------------ Einstellungen

    def _require_admin(token: str | None) -> None:
        """Lässt nur mit dem eingerichteten Einstellungs-Zugangswort durch."""
        if not settings.admin_token:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Die Einstellungen sind gesperrt, weil kein Zugangswort eingerichtet "
                    "ist. Setze die Umgebungsvariable XT_ADMIN_TOKEN auf ein selbst "
                    "gewähltes, langes Wort und starte den Dienst neu."
                ),
            )
        if not token or not secrets.compare_digest(token, settings.admin_token):
            raise HTTPException(status_code=401, detail="Zugangswort stimmt nicht.")

    def _admin_view() -> dict[str, Any]:
        return {
            "providers": [r.public() for r in table.all()],
            "kinds": [{"id": k, "label": KIND_LABELS[k]} for k in KINDS],
            "custom_hints": list(CUSTOM_HINTS),
            "request_timeout_s": settings.resolved_timeout_s,
            "timeout_source": settings.timeout_source,
            "env": settings.env,
            "fake_providers_enabled": settings.allow_fake_providers,
            "editable": not settings.models,
        }

    async def _after_change() -> dict[str, Any]:
        await table.reload(store)
        # Der nächste Auftrag soll die geänderten Zugangsdaten verwenden.
        registry.invalidate()
        return _admin_view()

    def _neue_id(label: str, vergeben: set[str]) -> str:
        roh = "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-")
        basis = re.sub(r"-+", "-", roh)[:30] or "anbieter"
        if basis not in vergeben:
            return basis
        for n in range(2, 100):
            if f"{basis}-{n}" not in vergeben:
                return f"{basis}-{n}"
        return f"{basis}-{secrets.token_hex(3)}"

    @app.get("/api/admin/providers")
    async def read_providers(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _require_admin(x_admin_token)
        return _admin_view()

    @app.post("/api/admin/providers", status_code=201)
    async def create_provider(
        body: ProviderCreate,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _require_admin(x_admin_token)
        if settings.models:
            raise HTTPException(
                status_code=409,
                detail="Im Test- und Entwicklungsbetrieb steht der Tisch fest.",
            )
        if body.kind not in KINDS:
            raise HTTPException(status_code=422, detail=f"Unbekannte Art: {body.kind}")
        if body.kind == KIND_OPENAI and not (body.base_url or "").strip():
            raise HTTPException(
                status_code=422,
                detail="Ein OpenAI-kompatibler Anbieter braucht eine Basis-Adresse.",
            )

        vergeben = {r.id for r in table.all()}
        neue_id = _neue_id(body.label, vergeben)
        await store.insert_provider(
            {
                "id": neue_id,
                "label": body.label.strip(),
                "kind": body.kind,
                "base_url": (body.base_url or "").strip() or None,
                "model": body.model.strip(),
                "api_key": body.api_key.strip(),
                "enabled": 1 if body.enabled else 0,
                "is_preset": 0,
                "position": 100 + len(vergeben),
            }
        )
        log.info("Anbieter angelegt: %s (%s)", neue_id, body.kind)
        return {"created": neue_id, **(await _after_change())}

    @app.post("/api/admin/providers/{provider_id}")
    async def update_provider(
        provider_id: str,
        body: ProviderUpdate,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _require_admin(x_admin_token)
        record = table.by_id(provider_id)
        if record is None or settings.models:
            raise HTTPException(status_code=404, detail="Anbieter nicht gefunden.")

        felder: dict[str, Any] = {}
        if body.label is not None:
            felder["label"] = body.label.strip()
        if body.model is not None:
            felder["model"] = body.model.strip()
        if body.base_url is not None:
            felder["base_url"] = body.base_url.strip() or None
        if body.enabled is not None:
            felder["enabled"] = 1 if body.enabled else 0
        if body.api_key is not None:
            # Leerer Text löscht den Schlüssel; danach greift wieder die Umgebung.
            felder["api_key"] = body.api_key.strip()
        if not felder:
            raise HTTPException(status_code=422, detail="Nichts zu ändern.")

        await store.update_provider(provider_id, felder)
        log.info("Anbieter geändert: %s (%s)", provider_id, ", ".join(sorted(felder)))
        return {"changed": sorted(felder), **(await _after_change())}

    @app.delete("/api/admin/providers/{provider_id}")
    async def delete_provider(
        provider_id: str,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _require_admin(x_admin_token)
        record = table.by_id(provider_id)
        if record is None or settings.models:
            raise HTTPException(status_code=404, detail="Anbieter nicht gefunden.")
        if record.is_preset:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Mitgelieferte Anbieter lassen sich abschalten, aber nicht löschen — "
                    "sonst wären sie beim nächsten Start wieder da."
                ),
            )
        await store.delete_provider(provider_id)
        log.info("Anbieter gelöscht: %s", provider_id)
        return {"deleted": provider_id, **(await _after_change())}

    @app.post("/api/admin/providers/{provider_id}/test")
    async def test_provider(
        provider_id: str,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        """Prüft einen Anbieter mit einem einzigen, sehr kurzen echten Aufruf."""
        _require_admin(x_admin_token)
        record = table.by_id(provider_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Anbieter nicht gefunden.")

        start_zeit = time.monotonic()
        try:
            provider = registry.get(record.id)
            antwort = await provider.complete(
                prompt="Antworte mit genau einem Wort: bereit.",
                model=record.model,
                timeout_s=min(settings.resolved_timeout_s, 30),
            )
        except ProviderError as exc:
            return {"ok": False, "detail": str(exc), "latency_ms": None}
        except Exception as exc:  # pragma: no cover - unerwartete Fehlerklasse
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "latency_ms": None}
        return {
            "ok": True,
            "detail": antwort.text.strip()[:120],
            "latency_ms": int((time.monotonic() - start_zeit) * 1000),
        }

    @app.get("/api/admin/settings")
    async def read_settings(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _require_admin(x_admin_token)
        return _admin_view()

    @app.post("/api/admin/settings")
    async def write_settings(
        body: SettingsUpdate,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _require_admin(x_admin_token)
        if body.request_timeout_s is None:
            raise HTTPException(status_code=422, detail="Nichts zu ändern.")
        await store.set_setting("request_timeout_s", str(body.request_timeout_s))
        settings.overrides = await store.all_settings()
        return {"changed": ["request_timeout_s"], **(await _after_change())}

    # ---------------------------------------------------------------- Oberfläche

    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
    else:

        @app.get("/", response_class=HTMLResponse)
        async def placeholder() -> HTMLResponse:
            return HTMLResponse(
                "<!DOCTYPE html><html lang=de><meta charset=utf-8>"
                "<title>XPERTENTiSCH</title><body style='font-family:system-ui;padding:40px'>"
                "<h1>XPERTENTiSCH</h1><p>Die Oberfläche ist noch nicht gebaut. "
                "<code>npm --prefix frontend run build</code> ausführen.</p>"
                "<p><a href='/api/health'>/api/health</a></p>",
                status_code=200,
            )

    return app


app = create_app()
