"""Auftragsausführung: ein Funke, mehrere unabhängige Modellaufträge.

Jeder Auftrag läuft in einer eigenen Task. Der Ausfall eines Providers
beendet oder verzögert die anderen Aufträge nicht — Fehler werden pro
Auftrag gefangen und als Zustand des jeweiligen Auftrags festgehalten.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .analysis import analyse
from .config import ModelConfig, Settings
from .db import (
    JOB_DONE,
    JOB_ERROR,
    JOB_QUEUED,
    Store,
    new_id,
    now,
)
from .events import EventBus
from .providers import ProviderError, ProviderRegistry

log = logging.getLogger("xpertentisch.runner")


class Runner:
    def __init__(
        self,
        store: Store,
        bus: EventBus,
        registry: ProviderRegistry,
        settings: Settings,
    ) -> None:
        self.store = store
        self.bus = bus
        self.registry = registry
        self.settings = settings
        self._tasks: set[asyncio.Task] = set()
        #: Funken, deren Aufträge in diesem Prozess bereits gestartet wurden.
        self._started_sparks: set[str] = set()

    # ------------------------------------------------------------------ Start

    async def launch_spark(
        self, session_id: str, spark: dict[str, Any], models: list[ModelConfig]
    ) -> list[dict[str, Any]]:
        """Legt die Aufträge an und startet sie. Doppelte Starts sind wirkungslos."""
        if spark["id"] in self._started_sparks:
            return await self.store.list_jobs_for_spark(spark["id"])
        self._started_sparks.add(spark["id"])

        ts = now()
        job_rows = [
            {
                "id": new_id("auf"),
                "session_id": session_id,
                "spark_id": spark["id"],
                "model_id": m.id,
                "label": m.label,
                "provider": m.provider,
                "model": m.model,
                "status": JOB_QUEUED,
                "text": "",
                "error": None,
                "partial": 0,
                "created_at": ts,
            }
            for m in models
        ]
        await self.store.insert_jobs(job_rows)
        # Nach INSERT OR IGNORE zählt der gespeicherte Stand, nicht der lokale.
        jobs = await self.store.list_jobs_for_spark(spark["id"])

        await self.bus.publish(
            session_id,
            "funke.angelegt",
            {"spark": spark, "jobs": jobs},
        )

        by_model = {m.id: m for m in models}
        for job in jobs:
            if job["status"] != JOB_QUEUED:
                continue
            model = by_model.get(job["model_id"])
            if model is None:
                continue
            task = asyncio.create_task(self._run_job(job, model, spark["prompt"]))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        return jobs

    # -------------------------------------------------------------- Ein Auftrag

    async def _run_job(self, job: dict[str, Any], model: ModelConfig, prompt: str) -> None:
        session_id = job["session_id"]
        started = time.monotonic()
        try:
            await self.store.mark_job_running(job["id"])
            await self.bus.publish(
                session_id, "auftrag.laeuft", {"job_id": job["id"], "model_id": job["model_id"]}
            )

            provider = self.registry.get(model.provider)
            response = await provider.complete(
                prompt=prompt, model=model.model, timeout_s=self.settings.resolved_timeout_s
            )
            latency = int((time.monotonic() - started) * 1000)
            await self.store.finish_job(
                job["id"],
                status=JOB_DONE,
                text=response.text,
                partial=response.partial,
                latency_ms=latency,
            )
            await self.bus.publish(
                session_id,
                "auftrag.fertig",
                {
                    "job_id": job["id"],
                    "model_id": job["model_id"],
                    "label": job["label"],
                    "text": response.text,
                    "partial": response.partial,
                    "note": response.note,
                    "latency_ms": latency,
                },
            )
        except ProviderError as exc:
            await self._fail_job(job, exc, started, partial_text=exc.partial_text)
        except asyncio.CancelledError:  # pragma: no cover - Abbruch beim Herunterfahren
            raise
        except Exception as exc:  # Unerwartetes bleibt auf diesen Auftrag beschränkt
            log.exception("Auftrag %s unerwartet fehlgeschlagen", job["id"])
            await self._fail_job(job, exc, started)
        finally:
            await self._maybe_finalise(job["session_id"], job["spark_id"])

    async def _fail_job(
        self, job: dict[str, Any], exc: BaseException, started: float, partial_text: str = ""
    ) -> None:
        latency = int((time.monotonic() - started) * 1000)
        message = str(exc) or type(exc).__name__
        await self.store.finish_job(
            job["id"],
            status=JOB_ERROR,
            text=partial_text,
            error=message,
            partial=bool(partial_text),
            latency_ms=latency,
        )
        await self.bus.publish(
            job["session_id"],
            "auftrag.fehler",
            {
                "job_id": job["id"],
                "model_id": job["model_id"],
                "label": job["label"],
                "error": message,
                "text": partial_text,
                "partial": bool(partial_text),
                "latency_ms": latency,
            },
        )

    # ----------------------------------------------------------- Abschlusslauf

    async def _maybe_finalise(self, session_id: str, spark_id: str) -> None:
        """Berechnet die Einschätzungen, sobald alle Aufträge des Funkens ruhen."""
        jobs = await self.store.list_jobs_for_spark(spark_id)
        if any(j["status"] in ("queued", "running") for j in jobs):
            return
        await self.recompute_assessment(session_id, spark_id, jobs)

    async def recompute_assessment(
        self, session_id: str, spark_id: str, jobs: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        jobs = jobs if jobs is not None else await self.store.list_jobs_for_spark(spark_id)
        markers, summary = analyse(session_id, spark_id, jobs)
        await self.store.replace_markers(spark_id, session_id, markers)
        await self.store.save_assessment(spark_id, session_id, summary)
        await self.bus.publish(
            session_id,
            "einschaetzung.fertig",
            {"spark_id": spark_id, "summary": summary, "markers": markers},
        )
        return summary

    # -------------------------------------------------------------- Herunterfahren

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
