"""Auftragsausführung: ein Funke, mehrere unabhängige Modellaufträge.

Jeder Auftrag läuft in einer eigenen Task. Der Ausfall eines Providers
beendet oder verzögert die anderen Aufträge nicht — Fehler werden pro
Auftrag gefangen und als Zustand des jeweiligen Auftrags festgehalten.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from . import context as kontext
from .analysis import analyse
from .config import ModelConfig, Settings
from .db import (
    JOB_DONE,
    JOB_ERROR,
    JOB_NOT_REQUESTED,
    JOB_QUEUED,
    Store,
    new_id,
    now,
)
from .events import EventBus
from .providers import ProviderError, ProviderRegistry

log = logging.getLogger("xpertentisch.runner")

def _fingerprint(jobs: list[dict[str, Any]]) -> tuple:
    """Kennzeichnet den Stand der Aufträge eines Funkens."""
    return tuple(sorted((j["id"], j["status"], len(j.get("text") or "")) for j in jobs))


#: Welche Beziehungsart eine Eingabeart setzt.
BEZUG_TYP = {
    "funke": "antwortet_auf",
    "antwort": "antwortet_auf",
    "weitergabe": "abgeleitet_aus",
    "gegenposition": "widerspricht",
    "vertiefung": "vertieft",
}


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
        #: Je Funke ein Schloss. Enden zwei Aufträge gleichzeitig, liefe der
        #: Abschlusslauf sonst doppelt und legte doppelte Bezüge an.
        self._finalise_locks: dict[str, asyncio.Lock] = {}
        #: Stand, der zuletzt ausgewertet wurde — verhindert Doppelläufe.
        self._finalised: dict[str, tuple] = {}

    # ------------------------------------------------------------------ Start

    async def launch_spark(
        self,
        session_id: str,
        spark: dict[str, Any],
        models: list[ModelConfig],
        untouched: list[ModelConfig] | None = None,
    ) -> list[dict[str, Any]]:
        """Legt die Aufträge an und startet sie. Doppelte Starts sind wirkungslos.

        `untouched` sind einsatzbereite Anbieter, die für diesen Funken bewusst
        nicht angefragt wurden. Sie bekommen eine Zeile mit dem Zustand
        `not_requested` — „nicht gefragt“ ist etwas anderes als „keine Antwort“.
        """
        if spark["id"] in self._started_sparks:
            return await self.store.list_jobs_for_spark(spark["id"])
        self._started_sparks.add(spark["id"])

        ts = now()

        def zeile(m: ModelConfig, status: str) -> dict[str, Any]:
            return {
                "id": new_id("auf"),
                "session_id": session_id,
                "spark_id": spark["id"],
                "model_id": m.id,
                "label": m.label,
                "provider": m.provider,
                "model": m.model,
                "status": status,
                "text": "",
                "error": None,
                "partial": 0,
                "created_at": ts,
            }

        job_rows = [zeile(m, JOB_QUEUED) for m in models]
        job_rows += [zeile(m, JOB_NOT_REQUESTED) for m in (untouched or [])]
        await self.store.insert_jobs(job_rows)

        # Ausdrücklich gewählte Bezüge sind menschlich gesetzte Beziehungen.
        for ref in spark.get("refs") or []:
            await self.store.insert_relation(
                {
                    "session_id": session_id,
                    "from_id": spark["id"],
                    "to_id": ref,
                    "type": BEZUG_TYP.get(spark.get("kind", "funke"), "antwortet_auf"),
                    "origin": "mensch",
                    "status": "bestaetigt",
                    "note": "",
                }
            )
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

            # Der Kontext wird jetzt festgehalten — was danach eingeworfen wird,
            # gehörte nicht zum Kenntnisstand dieses Auftrags.
            gesendet = await self._freeze_context(job, model, prompt)

            provider = self.registry.get(model.provider)
            response = await provider.complete(
                prompt=gesendet, model=model.model, timeout_s=self.settings.resolved_timeout_s
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

    async def _freeze_context(
        self, job: dict[str, Any], model: ModelConfig, prompt: str
    ) -> str:
        """Baut den Gesprächsauszug und schreibt den Schnappschuss fest."""
        spark = await self.store.get_spark(job["spark_id"])
        if spark is None:  # pragma: no cover - der Funke existiert immer
            return prompt
        sparks = await self.store.list_sparks(job["session_id"])
        jobs = await self.store.list_jobs(job["session_id"])
        gesendet, eintraege, gekuerzt = kontext.build(
            spark=spark, sparks=sparks, jobs=jobs
        )
        await self.store.save_context_snapshot(
            {
                "job_id": job["id"],
                "session_id": job["session_id"],
                "rendered": gesendet,
                "message_ids": json.dumps(
                    [e.public() for e in eintraege], ensure_ascii=False
                ),
                "rule": kontext.REGEL,
                "provider": model.provider,
                "model": model.model,
                "truncated": 1 if gekuerzt else 0,
            }
        )
        return gesendet

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
        schloss = self._finalise_locks.setdefault(spark_id, asyncio.Lock())
        async with schloss:
            jobs = await self.store.list_jobs_for_spark(spark_id)
            if any(j["status"] in ("queued", "running") for j in jobs):
                return
            if self._finalised.get(spark_id) == _fingerprint(jobs):
                # Derselbe Stand wurde schon ausgewertet.
                return
            await self.recompute_assessment(session_id, spark_id, jobs)
            self._finalised[spark_id] = _fingerprint(jobs)

    async def recompute_assessment(
        self, session_id: str, spark_id: str, jobs: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        jobs = jobs if jobs is not None else await self.store.list_jobs_for_spark(spark_id)
        markers, summary = analyse(session_id, spark_id, jobs)
        await self.store.replace_markers(spark_id, session_id, markers)
        await self.store.save_assessment(spark_id, session_id, summary)
        await self._suggest_relations(session_id, jobs, summary)
        await self.bus.publish(
            session_id,
            "einschaetzung.fertig",
            {
                "spark_id": spark_id,
                "summary": summary,
                "markers": markers,
                "relations": await self.store.list_relations(session_id),
            },
        )
        return summary

    async def _suggest_relations(
        self, session_id: str, jobs: list[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        """Schreibt die gefundenen Bezüge — ausdrücklich als Vorschlag.

        Was die Auswertung findet, ist eine Lesehilfe. Erst eine menschliche
        Bestätigung macht daraus einen Befund; bis dahin steht der Status auf
        `vorschlag`.
        """
        job_ids = [j["id"] for j in jobs]
        await self.store.delete_machine_relations(session_id, job_ids)
        for paar in summary.get("pairs", []):
            for art, anzahl in (
                ("uebereinstimmung", paar.get("agreements", 0)),
                ("widerspricht", paar.get("contradictions", 0)),
            ):
                if not anzahl:
                    continue
                await self.store.insert_relation(
                    {
                        "session_id": session_id,
                        "from_id": paar["a_job_id"],
                        "to_id": paar["b_job_id"],
                        "type": art,
                        "origin": "maschine",
                        "status": "vorschlag",
                        "note": f"{anzahl} Fundstelle(n): {', '.join(paar.get('topics', [])[:3])}",
                    }
                )

    # -------------------------------------------------------------- Herunterfahren

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
