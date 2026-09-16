"""XPERTENTiSCH — HTTP-Schnittstelle."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .db import SESSION_CLOSED, Store, now
from .events import EventBus
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
    """Änderungen an den Einstellungen.

    `None` heißt unverändert, ein leerer Text löscht den Wert und gibt damit
    wieder der Umgebungsvariablen den Vorrang.
    """

    openai_api_key: str | None = Field(default=None, max_length=400)
    anthropic_api_key: str | None = Field(default=None, max_length=400)
    openai_model: str | None = Field(default=None, max_length=120)
    anthropic_model: str | None = Field(default=None, max_length=120)
    request_timeout_s: int | None = Field(default=None, ge=5, le=600)


class ProviderTest(BaseModel):
    provider: str = Field(min_length=2, max_length=40)


SECRET_FIELDS = ("openai_api_key", "anthropic_api_key")
SETTING_FIELDS = SECRET_FIELDS + ("openai_model", "anthropic_model", "request_timeout_s")


def _hint(value: str) -> str:
    """Erkennungshilfe für einen hinterlegten Schlüssel — nie der Schlüssel selbst."""
    return f"…{value[-4:]}" if len(value) >= 8 else "gesetzt"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = Store(settings.db_path)
    bus = EventBus(store)
    registry = ProviderRegistry(settings)
    runner = Runner(store, bus, registry, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.connect()
        # Auf der Einstellungsseite hinterlegte Werte haben Vorrang vor der Umgebung.
        settings.overrides = await store.all_settings()
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
            "missing_credentials": settings.missing_credentials(),
            "time": now(),
        }

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        """Öffentliche Konfiguration. Enthält keine Zugangsdaten."""
        return {
            "models": [
                {"id": m.id, "label": m.label, "provider": m.provider, "model": m.model}
                for m in settings.enabled_models()
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

        models = settings.enabled_models()
        if body.model_ids:
            wanted = set(body.model_ids)
            models = [m for m in models if m.id in wanted]
        if not models:
            raise HTTPException(status_code=422, detail="Kein Modell ausgewählt.")

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
        """Lässt nur mit dem eingerichteten Einstellungs-Token durch."""
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

    def _settings_view() -> dict[str, Any]:
        verfuegbar = registry.availability()
        provider_zeilen = []
        gesehen: set[str] = set()
        for modell in settings.enabled_models():
            if modell.provider in gesehen:
                continue
            gesehen.add(modell.provider)
            braucht_schluessel = modell.provider in ("openai", "anthropic")
            feld = f"{modell.provider}_api_key" if braucht_schluessel else ""
            wert = settings.overrides.get(feld) or {
                "openai": settings.openai_api_key,
                "anthropic": settings.anthropic_api_key,
            }.get(modell.provider)
            zustand = verfuegbar.get(modell.provider, {"ready": False, "reason": ""})
            provider_zeilen.append(
                {
                    "provider": modell.provider,
                    "label": modell.label,
                    "model": modell.model,
                    "model_field": f"{modell.provider}_model",
                    "key_field": feld,
                    "needs_key": braucht_schluessel,
                    "key_source": settings.source_of(feld) if feld in SECRET_FIELDS else "fehlt",
                    "key_hint": _hint(wert) if wert else "",
                    "ready": bool(zustand["ready"]),
                    "reason": str(zustand["reason"]),
                }
            )
        return {
            "providers": provider_zeilen,
            "request_timeout_s": settings.resolved_timeout_s,
            "timeout_source": (
                "einstellungen" if settings.overrides.get("request_timeout_s") else "umgebung"
            ),
            "env": settings.env,
            "fake_providers_enabled": settings.allow_fake_providers,
        }

    @app.get("/api/admin/settings")
    async def read_settings(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _require_admin(x_admin_token)
        return _settings_view()

    @app.post("/api/admin/settings")
    async def write_settings(
        body: SettingsUpdate,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _require_admin(x_admin_token)

        geaendert: list[str] = []
        for feld in SETTING_FIELDS:
            wert = getattr(body, feld)
            if wert is None:
                continue
            text = str(wert).strip()
            if text:
                await store.set_setting(feld, text)
            else:
                await store.delete_setting(feld)
            geaendert.append(feld)

        settings.overrides = await store.all_settings()
        # Die nächsten Aufträge sollen die neuen Zugangsdaten verwenden.
        registry.invalidate()
        if geaendert:
            log.info("Einstellungen geändert: %s", ", ".join(geaendert))
        return {"changed": geaendert, **_settings_view()}

    @app.post("/api/admin/test")
    async def test_provider(
        body: ProviderTest,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        """Prüft einen Provider mit einem einzigen, sehr kurzen echten Aufruf."""
        _require_admin(x_admin_token)
        modell = next(
            (m for m in settings.enabled_models() if m.provider == body.provider), None
        )
        if modell is None:
            raise HTTPException(status_code=404, detail="Unbekannter Provider.")

        import time as _time

        start = _time.monotonic()
        try:
            provider = registry.get(modell.provider)
            antwort = await provider.complete(
                prompt="Antworte mit genau einem Wort: bereit.",
                model=modell.model,
                timeout_s=min(settings.resolved_timeout_s, 30),
            )
        except ProviderError as exc:
            return {"ok": False, "detail": str(exc), "latency_ms": None}
        except Exception as exc:  # pragma: no cover - unerwartete Fehlerklasse
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "latency_ms": None}
        return {
            "ok": True,
            "detail": antwort.text.strip()[:120],
            "latency_ms": int((_time.monotonic() - start) * 1000),
        }

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
