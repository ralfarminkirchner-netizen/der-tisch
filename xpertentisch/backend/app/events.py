"""Ereignisbus für den SSE-Strom.

Jedes Ereignis wird zuerst in SQLite geschrieben und bekommt dadurch eine
monoton steigende Kennung. Genau diese Kennung geht als `id:` in den
SSE-Strom. Ein Client, der die Verbindung verliert, schickt sie beim
Wiederverbinden als `Last-Event-ID` zurück und bekommt die verpassten
Ereignisse nachgeliefert — ohne dass ein neuer LLM-Auftrag entsteht.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from .db import Store

QUEUE_SIZE = 256


class EventBus:
    def __init__(self, store: Store) -> None:
        self._store = store
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    async def publish(
        self, session_id: str, type_: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        event = await self._store.append_event(session_id, type_, payload)
        for queue in list(self._subscribers.get(session_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Langsamer Abonnent: er holt den Rest über Last-Event-ID nach.
                pass
        return event

    def subscribe(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(session_id)
        if not subs:
            return
        subs.discard(queue)
        if not subs:
            self._subscribers.pop(session_id, None)

    def subscriber_count(self, session_id: str) -> int:
        return len(self._subscribers.get(session_id, ()))

    async def stream(
        self, session_id: str, last_event_id: int, keepalive_s: float = 15.0
    ) -> AsyncIterator[dict[str, Any] | None]:
        """Liefert zuerst die verpassten, dann die laufenden Ereignisse.

        `None` steht für einen Keepalive-Tick.
        """
        queue = self.subscribe(session_id)
        try:
            cursor = last_event_id
            while True:
                missed = await self._store.events_since(session_id, cursor)
                if not missed:
                    break
                for event in missed:
                    cursor = event["id"]
                    yield event

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=keepalive_s)
                except asyncio.TimeoutError:
                    # Lücke schließen, falls die Queue übergelaufen war.
                    missed = await self._store.events_since(session_id, cursor)
                    for ev in missed:
                        cursor = ev["id"]
                        yield ev
                    yield None
                    continue
                if event["id"] <= cursor:
                    continue
                cursor = event["id"]
                yield event
        finally:
            self.unsubscribe(session_id, queue)
