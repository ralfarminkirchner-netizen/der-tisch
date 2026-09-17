"""Auftragsausführung: Warteschlange je Anbieter, unabhängige Läufe.

Jeder Anbieter hat seine eigene Warteschlange. Ein freier Anbieter beginnt den
nächsten passenden Auftrag, während ein anderer noch am vorherigen arbeitet —
es gibt keine gemeinsame Wartebarriere. Der Ausfall eines Anbieters beendet
oder verzögert die anderen nicht.

Antworten werden gestreamt, wo der Adapter es kann: Textstücke erscheinen,
während sie entstehen, und werden in Abständen festgeschrieben. Nach einem
Verbindungsabbruch ist der Zwischenstand darum wieder sichtbar.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from . import context as kontext
from .analysis import analyse, szenario_block
from .config import ModelConfig, Settings
from .db import (
    JOB_CANCELLED,
    JOB_DONE,
    JOB_ERROR,
    JOB_NOT_REQUESTED,
    JOB_QUEUED,
    JOB_STREAMING,
    Store,
    new_id,
    now,
)
from .events import EventBus
from .providers import ProviderError, ProviderRegistry

log = logging.getLogger("xpertentisch.runner")

#: Wie oft der Zwischenstand einer laufenden Antwort festgeschrieben wird.
FLUSH_INTERVAL_S = 0.4
#: Ab so vielen neuen Zeichen wird auch vor Ablauf der Zeit geschrieben.
FLUSH_CHARS = 240


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
    # Ein Szenario leitet Folgen aus einer Aussage ab — es widerspricht ihr nicht.
    "szenario": "abgeleitet_aus",
    "pingpong": "antwortet_auf",
    "kuratierung": "abgeleitet_aus",
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
        #: Je Anbieter eine Warteschlange und eigene Arbeitskräfte.
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: dict[str, list[asyncio.Task]] = {}
        #: Laufende Aufträge, damit sich einzelne abbrechen lassen.
        self._current: dict[str, asyncio.Task] = {}
        self._cancel_requested: set[str] = set()
        #: Funken, deren Aufträge in diesem Prozess bereits gestartet wurden.
        self._started_sparks: set[str] = set()
        #: Je Funke ein Schloss. Enden zwei Aufträge gleichzeitig, liefe der
        #: Abschlusslauf sonst doppelt und legte doppelte Bezüge an.
        self._finalise_locks: dict[str, asyncio.Lock] = {}
        self._finalised: dict[str, tuple] = {}
        self._closing = False

    # ------------------------------------------------------------ Warteschlange

    def _queue_for(self, provider_id: str) -> asyncio.Queue:
        if provider_id in self._queues:
            return self._queues[provider_id]
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[provider_id] = queue
        self._workers[provider_id] = [
            asyncio.create_task(self._worker(provider_id, queue))
            for _ in range(max(1, self.settings.provider_concurrency))
        ]
        return queue

    async def _worker(self, provider_id: str, queue: asyncio.Queue) -> None:
        """Arbeitet die Aufträge eines Anbieters der Reihe nach ab."""
        while not self._closing:
            job, model, prompt = await queue.get()
            try:
                if job["id"] in self._cancel_requested:
                    self._cancel_requested.discard(job["id"])
                    await self._maybe_finalise(job["session_id"], job["spark_id"])
                    continue
                aufgabe = asyncio.create_task(self._run_job(job, model, prompt))
                self._current[job["id"]] = aufgabe
                try:
                    await aufgabe
                except asyncio.CancelledError:
                    # Nur dieser Auftrag wurde abgebrochen, nicht der Arbeiter.
                    if self._closing:
                        raise
                finally:
                    self._current.pop(job["id"], None)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - der Arbeiter darf nie sterben
                log.exception("Arbeiter %s: unerwarteter Fehler", provider_id)
            finally:
                queue.task_done()

    # ------------------------------------------------------------------ Start

    async def launch_spark(
        self,
        session_id: str,
        spark: dict[str, Any],
        models: list[ModelConfig],
        untouched: list[ModelConfig] | None = None,
    ) -> list[dict[str, Any]]:
        """Legt die Aufträge an und reiht sie ein. Doppelte Starts sind wirkungslos.

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

        jobs = await self.store.list_jobs_for_spark(spark["id"])
        await self.bus.publish(session_id, "funke.angelegt", {"spark": spark, "jobs": jobs})

        by_model = {m.id: m for m in models}
        for job in jobs:
            if job["status"] != JOB_QUEUED:
                continue
            model = by_model.get(job["model_id"])
            if model is None:
                continue
            await self._queue_for(model.provider).put((job, model, spark["prompt"]))

        return jobs

    # ---------------------------------------------------------------- Abbruch

    async def cancel_job(self, job_id: str) -> bool:
        """Bricht genau einen Auftrag ab. Andere laufen weiter."""
        job = await self.store.get_job(job_id)
        if job is None or job["status"] not in (JOB_QUEUED, "running", JOB_STREAMING):
            return False

        self._cancel_requested.add(job_id)
        aufgabe = self._current.get(job_id)
        if aufgabe is not None:
            aufgabe.cancel()
            return True

        # Noch in der Warteschlange: Zustand jetzt setzen, der Arbeiter überspringt ihn.
        beendet = await self.store.set_job_status(
            job_id, JOB_CANCELLED, "Von dir abgebrochen."
        )
        await self.bus.publish(
            job["session_id"], "auftrag.abgebrochen",
            {"job_id": job_id, "model_id": job["model_id"], "label": job["label"],
             "finished_at": beendet},
        )
        return True

    # -------------------------------------------------------------- Ein Auftrag

    async def _run_job(self, job: dict[str, Any], model: ModelConfig, prompt: str) -> None:
        session_id = job["session_id"]
        started = time.monotonic()
        puffer: list[str] = []
        geschrieben = 0

        try:
            begonnen = await self.store.mark_job_running(job["id"])
            await self.bus.publish(
                session_id,
                "auftrag.laeuft",
                {"job_id": job["id"], "model_id": job["model_id"], "started_at": begonnen},
            )

            # Der Kontext wird jetzt festgehalten — was danach eingeworfen wird,
            # gehörte nicht zum Kenntnisstand dieses Auftrags.
            gesendet = await self._freeze_context(job, model, prompt)

            provider = self.registry.get(model.provider)

            def on_delta(stueck: str) -> None:
                puffer.append(stueck)

            spuelen = asyncio.create_task(self._flush_loop(job, puffer))
            try:
                response = await provider.complete(
                    prompt=gesendet,
                    model=model.model,
                    timeout_s=self.settings.resolved_timeout_s,
                    on_delta=on_delta if provider.streams else None,
                )
            finally:
                spuelen.cancel()
                geschrieben = len("".join(puffer))

            latency = int((time.monotonic() - started) * 1000)
            kosten, quelle = await self._kosten(model.provider, response)
            beendet = await self.store.finish_job(
                job["id"],
                status=JOB_DONE,
                text=response.text,
                partial=response.partial,
                latency_ms=latency,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                cost_micro=kosten,
                cost_source=quelle,
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
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "cost_micro": kosten,
                    "cost_source": quelle,
                    "finished_at": beendet,
                },
            )
        except asyncio.CancelledError:
            if self._closing:
                raise
            await self._cancel_job_record(job, "".join(puffer), started)
            return
        except ProviderError as exc:
            await self._fail_job(
                job, exc, started, partial_text=exc.partial_text or "".join(puffer)
            )
        except Exception as exc:  # Unerwartetes bleibt auf diesen Auftrag beschränkt
            log.exception("Auftrag %s unerwartet fehlgeschlagen", job["id"])
            await self._fail_job(job, exc, started, partial_text="".join(puffer))
        finally:
            self._cancel_requested.discard(job["id"])
            _ = geschrieben
            await self._maybe_finalise(job["session_id"], job["spark_id"])

    async def _flush_loop(self, job: dict[str, Any], puffer: list[str]) -> None:
        """Schreibt den Zwischenstand fest, solange die Antwort entsteht."""
        zuletzt = 0
        gemeldet = False
        try:
            while True:
                await asyncio.sleep(FLUSH_INTERVAL_S)
                stand = "".join(puffer)
                if len(stand) == zuletzt:
                    continue
                if not gemeldet:
                    await self.store.set_job_status(job["id"], JOB_STREAMING)
                    gemeldet = True
                await self.store.append_job_text(job["id"], stand)
                await self.bus.publish(
                    job["session_id"],
                    "auftrag.teilstueck",
                    {
                        "job_id": job["id"],
                        "model_id": job["model_id"],
                        "text": stand,
                        "chars": len(stand),
                    },
                )
                zuletzt = len(stand)
        except asyncio.CancelledError:
            return

    async def _kosten(self, provider_id: str, response: Any) -> tuple[int | None, str]:
        """Rechnet die Kosten — nur mit hinterlegten Preisen, sonst unbekannt."""
        if response.tokens_in is None and response.tokens_out is None:
            return None, "unbekannt"
        zeile = await self.store.get_provider(provider_id)
        preis_in = (zeile or {}).get("price_in")
        preis_out = (zeile or {}).get("price_out")
        if preis_in is None and preis_out is None:
            return None, "unbekannt"
        betrag = 0.0
        betrag += (response.tokens_in or 0) / 1_000_000 * (preis_in or 0.0)
        betrag += (response.tokens_out or 0) / 1_000_000 * (preis_out or 0.0)
        return int(round(betrag * 1_000_000)), "berechnet"

    async def _cancel_job_record(
        self, job: dict[str, Any], teiltext: str, started: float
    ) -> None:
        latency = int((time.monotonic() - started) * 1000)
        beendet = await self.store.finish_job(
            job["id"],
            status=JOB_CANCELLED,
            text=teiltext,
            error="Von dir abgebrochen.",
            partial=bool(teiltext),
            latency_ms=latency,
        )
        await self.bus.publish(
            job["session_id"],
            "auftrag.abgebrochen",
            {
                "job_id": job["id"], "model_id": job["model_id"], "label": job["label"],
                "text": teiltext, "partial": bool(teiltext), "finished_at": beendet,
            },
        )

    async def _freeze_context(
        self, job: dict[str, Any], model: ModelConfig, prompt: str
    ) -> str:
        """Baut den Gesprächsauszug und schreibt den Schnappschuss fest."""
        spark = await self.store.get_spark(job["spark_id"])
        if spark is None:  # pragma: no cover - der Funke existiert immer
            return prompt
        sparks = await self.store.list_sparks(job["session_id"])
        jobs = await self.store.list_jobs(job["session_id"])
        gesendet, eintraege, gekuerzt = kontext.build(spark=spark, sparks=sparks, jobs=jobs)
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
        beendet = await self.store.finish_job(
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
                "finished_at": beendet,
            },
        )

    # ----------------------------------------------------------- Abschlusslauf

    async def _maybe_finalise(self, session_id: str, spark_id: str) -> None:
        """Berechnet die Einschätzungen, sobald alle Aufträge des Funkens ruhen."""
        schloss = self._finalise_locks.setdefault(spark_id, asyncio.Lock())
        async with schloss:
            jobs = await self.store.list_jobs_for_spark(spark_id)
            if any(j["status"] in (JOB_QUEUED, "running", JOB_STREAMING) for j in jobs):
                return
            if self._finalised.get(spark_id) == _fingerprint(jobs):
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
                # Die Konsequenzkarte reist mit: sonst kennte die Oberfläche
                # ein gerade durchgespieltes Szenario erst nach dem nächsten
                # vollständigen Laden.
                "szenario": await self._szenario(session_id, spark_id, jobs),
            },
        )
        return summary

    async def _szenario(
        self, session_id: str, spark_id: str, jobs: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Die Konsequenzkarte dieses Funkens — oder None, wenn es keiner ist."""
        spark = await self.store.get_spark(spark_id)
        if spark is None or spark.get("kind") != "szenario":
            return None
        ausgang = None
        for ref in spark.get("refs") or []:
            treffer = await self.store.get_job(ref)
            if treffer is not None:
                ausgang = treffer
                break
        return szenario_block(spark, jobs, ausgang=ausgang)

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
        self._closing = True
        for aufgaben in self._workers.values():
            for aufgabe in aufgaben:
                aufgabe.cancel()
        for aufgabe in list(self._current.values()):
            aufgabe.cancel()
        alle = [t for liste in self._workers.values() for t in liste]
        alle += list(self._current.values())
        if alle:
            await asyncio.gather(*alle, return_exceptions=True)
        self._workers.clear()
        self._queues.clear()
        self._current.clear()
