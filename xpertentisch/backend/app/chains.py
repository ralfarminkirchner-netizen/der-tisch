"""Ketten: Kuratierung und begrenztes Ping-Pong.

Beides sind Abläufe über mehrere Funken hinweg, die der Server steuert. Zwei
Grundsätze:

* **Nichts läuft heimlich weiter.** Ping-Pong hat eine feste Obergrenze, die
  vor dem Start feststeht und sichtbar ist; ein Stopp beendet genau diesen
  Strang, nicht die Sitzung.
* **Die Kuratierung ersetzt nichts.** Sie ist ein eigener, als maschinell
  erkennbarer Beitrag. Der Originalfunke bleibt unverändert; scheitert die
  Kuratierung, läuft die Runde trotzdem — mit sichtbarem Hinweis.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .config import ModelConfig
from .db import JOB_QUEUED, JOB_STREAMING, Store, new_id

log = logging.getLogger("xpertentisch.chains")

OFFEN = (JOB_QUEUED, "running", JOB_STREAMING)

KURATOR_AUFTRAG = (
    "Du bist Kuratorin dieser Runde, nicht Beantworterin.\n"
    "Lies den folgenden Gedanken und liefere in höchstens sechs Zeilen:\n"
    "1. die erkennbare Frage oder Denkbewegung,\n"
    "2. einen knappen Untersuchungsfokus,\n"
    "3. die Unklarheiten — ausdrücklich als Unklarheiten benannt.\n"
    "Beantworte die Frage nicht und formuliere den Gedanken nicht um.\n\n"
    "GEDANKE:\n"
)

#: So lange wartet eine Kette höchstens auf einen Funken.
WARTEGRENZE_S = 300


async def warte_auf_funken(store: Store, spark_id: str, grenze: float = WARTEGRENZE_S) -> list[dict[str, Any]]:
    """Wartet, bis alle Aufträge eines Funkens ruhen."""
    ende = asyncio.get_event_loop().time() + grenze
    while asyncio.get_event_loop().time() < ende:
        jobs = await store.list_jobs_for_spark(spark_id)
        if jobs and not any(j["status"] in OFFEN for j in jobs):
            return jobs
        await asyncio.sleep(0.2)
    return await store.list_jobs_for_spark(spark_id)


class Chains:
    """Führt Kuratierung und Ping-Pong aus."""

    def __init__(self, store: Store, bus, runner, settings) -> None:
        self.store = store
        self.bus = bus
        self.runner = runner
        self.settings = settings
        self._tasks: set[asyncio.Task] = set()
        #: Laufende Ping-Pong-Läufe je Kennung.
        self.runs: dict[str, dict[str, Any]] = {}

    def _start(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ------------------------------------------------------------ Kuratierung

    async def curate(
        self, session_id: str, spark: dict[str, Any], kurator: ModelConfig
    ) -> dict[str, Any]:
        """Legt einen Kuratierungsfunken an und lässt ihn laufen.

        Gibt den angelegten Funken zurück. Der Aufrufer wartet nicht darauf —
        die Kuratierung blockiert die Sitzung nicht.
        """
        kurationsfunke, neu = await self.store.insert_spark(
            session_id,
            f"{KURATOR_AUFTRAG}{spark['prompt']}",
            f"kur-{spark['id']}",
            kind="kuratierung",
            refs=[spark["id"]],
        )
        if neu:
            await self.runner.launch_spark(session_id, kurationsfunke, [kurator], [])
        return kurationsfunke

    # -------------------------------------------------------------- Ping-Pong

    async def start_pingpong(
        self,
        session_id: str,
        *,
        prompt: str,
        refs: list[str],
        participants: list[ModelConfig],
        max_turns: int,
    ) -> dict[str, Any]:
        """Startet ein begrenztes Wechselgespräch zwischen Modellen."""
        run_id = new_id("pp")
        lauf = {
            "id": run_id,
            "session_id": session_id,
            "status": "laeuft",
            "turn": 0,
            "max_turns": max_turns,
            "participants": [m.id for m in participants],
            "labels": [m.label for m in participants],
            "prompt": prompt,
            "refs": list(refs),
            "stopped_reason": "",
        }
        self.runs[run_id] = lauf
        await self.bus.publish(session_id, "pingpong.gestartet", dict(lauf))
        self._start(self._pingpong_loop(lauf, participants))
        return lauf

    def stop_pingpong(self, run_id: str) -> bool:
        lauf = self.runs.get(run_id)
        if lauf is None or lauf["status"] != "laeuft":
            return False
        lauf["status"] = "gestoppt"
        lauf["stopped_reason"] = "von dir gestoppt"
        return True

    async def _pingpong_loop(
        self, lauf: dict[str, Any], participants: list[ModelConfig]
    ) -> None:
        session_id = lauf["session_id"]
        bezuege = list(lauf["refs"])
        try:
            for runde in range(lauf["max_turns"]):
                if lauf["status"] != "laeuft":
                    break
                sprecher = participants[runde % len(participants)]
                funke, neu = await self.store.insert_spark(
                    session_id,
                    lauf["prompt"],
                    f"{lauf['id']}-{runde}",
                    kind="pingpong",
                    refs=bezuege,
                )
                if not neu:  # pragma: no cover - dieselbe Runde zweimal
                    continue
                lauf["turn"] = runde + 1
                await self.bus.publish(session_id, "pingpong.runde", dict(lauf))
                await self.runner.launch_spark(session_id, funke, [sprecher], [])

                jobs = await warte_auf_funken(self.store, funke["id"])
                geglueckt = [j for j in jobs if j["status"] == "done" and (j["text"] or "").strip()]
                if not geglueckt:
                    lauf["status"] = "beendet"
                    lauf["stopped_reason"] = "keine verwertbare Antwort"
                    break
                bezuege = [geglueckt[0]["id"]]
            else:
                lauf["status"] = "beendet"
                lauf["stopped_reason"] = f"Obergrenze von {lauf['max_turns']} Beiträgen erreicht"
        except asyncio.CancelledError:  # pragma: no cover - Herunterfahren
            lauf["status"] = "beendet"
            lauf["stopped_reason"] = "abgebrochen"
            raise
        except Exception as exc:  # pragma: no cover - unerwartet
            log.exception("Ping-Pong %s gescheitert", lauf["id"])
            lauf["status"] = "beendet"
            lauf["stopped_reason"] = f"Fehler: {exc}"
        finally:
            if lauf["status"] == "laeuft":
                lauf["status"] = "beendet"
            await self.bus.publish(lauf["session_id"], "pingpong.ende", dict(lauf))

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
