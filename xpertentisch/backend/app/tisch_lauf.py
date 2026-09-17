"""TiSCH-Lauf: Perspektiven parallel → Reibung → Integration.

Kein Siegel, kein Auto-canonical. Modelle kommen aus der Konfiguration,
nicht aus einem fest verdrahteten Modellnamen. Memory nur bei TISCH_CORE_BASE_URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from .catalog.extract import extract
from .catalog.schema import Katalog, Tisch
from types import SimpleNamespace

from .db import new_id, now
from .providers.fake import FakeProvider

log = logging.getLogger("xpertentisch.tisch_lauf")

_KAT: Katalog | None = None


def katalog() -> Katalog:
    global _KAT
    if _KAT is None:
        _KAT = extract()
    return _KAT


def tisch_by_id(tisch_id: str) -> Tisch:
    for t in katalog().tische:
        if t.id == tisch_id:
            return t
    raise KeyError(tisch_id)


def _parse_perspective(text: str, rolle: str) -> dict[str, str]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "kernanalyse" in data:
            data.setdefault("rolle", rolle)
            data.setdefault("anspruchstyp", "")
            data.setdefault("evidenz", "")
            data.setdefault("blinder_fleck", "")
            return {k: str(data.get(k, "") or "") for k in (
                "rolle", "anspruchstyp", "kernanalyse", "evidenz", "blinder_fleck"
            )}
    except json.JSONDecodeError:
        pass
    return {
        "rolle": rolle,
        "anspruchstyp": "",
        "kernanalyse": text,
        "evidenz": "",
        "blinder_fleck": "",
    }


def _parse_friction(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "echte_widersprueche" in data:
            return {
                "uebersetzungsfehler": list(data.get("uebersetzungsfehler") or []),
                "echte_widersprueche": list(data.get("echte_widersprueche") or []),
                "uebersehenes": str(data.get("uebersehenes") or ""),
            }
    except json.JSONDecodeError:
        pass
    return {
        "uebersetzungsfehler": [],
        "echte_widersprueche": [],
        "uebersehenes": text,
    }


def _parse_integration(text: str) -> dict[str, Any]:
    keys = (
        "anspruchskarte", "uebersetzbare_bruecken", "echte_unvereinbarkeiten",
        "praktische_optionen", "offene_pruefpfade", "vorlaeufiges_fazit",
        "entscheidungshilfe", "kurzfassung", "einfach_gesagt",
        "herzmensch", "kopfmensch", "maennlich", "weiblich",
    )
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "vorlaeufiges_fazit" in data:
            out = {k: data.get(k, "" if k not in (
                "uebersetzbare_bruecken", "echte_unvereinbarkeiten",
                "praktische_optionen", "offene_pruefpfade",
                "entscheidungshilfe", "kurzfassung",
            ) else []) for k in keys}
            return out
    except json.JSONDecodeError:
        pass
    return {
        "anspruchskarte": "",
        "uebersetzbare_bruecken": [],
        "echte_unvereinbarkeiten": [],
        "praktische_optionen": [],
        "offene_pruefpfade": [],
        "vorlaeufiges_fazit": text,
        "entscheidungshilfe": [],
        "kurzfassung": [],
        "einfach_gesagt": "",
        "herzmensch": "",
        "kopfmensch": "",
        "maennlich": "",
        "weiblich": "",
    }


def agent_prompt(rolle: str) -> str:
    """Systemprompt wörtlich aus api_server.AGENTS_DE, sonst leer."""
    return katalog().agenten_de.get(rolle, "")


async def _memory_opt_in(payload: dict[str, Any]) -> None:
    base = (os.environ.get("TISCH_CORE_BASE_URL") or "").strip().rstrip("/")
    if not base:
        log.warning("TISCH_CORE_BASE_URL fehlt — Memory no-op")
        return
    url = f"{base}/api/tisch-memory/candidates"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json=payload)
    except Exception as exc:  # noqa: BLE001 — Memory darf den Lauf nicht kippen
        log.warning("Memory-POST fehlgeschlagen: %s", type(exc).__name__)


class TischLaeufe:
    def __init__(self, app) -> None:
        self._app = app
        self._laeufe: dict[str, dict[str, Any]] = {}
        self._cancel: dict[str, set[str]] = {}
        self._phase_events: dict[str, dict[str, asyncio.Event]] = {}

    def get(self, lauf_id: str) -> dict[str, Any]:
        if lauf_id not in self._laeufe:
            raise KeyError(lauf_id)
        return self._laeufe[lauf_id]

    async def start(
        self,
        *,
        tisch_id: str,
        modus: str,
        question: str,
        model_id: str | None = None,
        reibung_model_id: str | None = None,
        integration_model_id: str | None = None,
    ) -> dict[str, Any]:
        tisch = tisch_by_id(tisch_id)
        settings = self._app.state.settings
        models = list(settings.models)
        if not models:
            raise RuntimeError("kein Modell konfiguriert")
        default = next((m for m in models if m.id == model_id), models[0])
        reibung_m = next((m for m in models if m.id == reibung_model_id), default)
        integ_m = next((m for m in models if m.id == integration_model_id), default)
        lauf_id = new_id("lauf")
        persps = list(tisch.perspectives)
        if not persps:
            # der-tisch u. a.: AGENTS_DE-Namen
            persps = [
                SimpleNamespace(id=_slug(n), de=n)
                for n in list(katalog().agenten_de)[:4]
            ]
        rec = {
            "id": lauf_id,
            "tisch_id": tisch_id,
            "modus": modus,
            "question": question,
            "status": "running",
            "phase": "perspektiven",
            "perspectives": [],
            "friction": None,
            "integration": None,
            "created_at": now(),
            "models": {
                "perspektive": default.id,
                "reibung": reibung_m.id,
                "integration": integ_m.id,
            },
        }
        self._laeufe[lauf_id] = rec
        self._cancel[lauf_id] = set()
        self._phase_events[lauf_id] = {
            "perspektiven_done": asyncio.Event(),
            "reibung_started": asyncio.Event(),
            "reibung_done": asyncio.Event(),
            "integration_started": asyncio.Event(),
        }
        asyncio.create_task(
            self._run(lauf_id, persps, default, reibung_m, integ_m, question),
            name=f"tisch-lauf-{lauf_id}",
        )
        return rec

    def cancel_perspective(self, lauf_id: str, perspective_id: str) -> None:
        self._cancel[lauf_id].add(perspective_id)

    async def _run(self, lauf_id, persps, persp_model, reibung_m, integ_m, question) -> None:
        rec = self._laeufe[lauf_id]
        registry = self._app.state.registry
        store = self._app.state.store
        events = self._phase_events[lauf_id]

        async def eine(p) -> dict[str, Any] | None:
            pid = getattr(p, "id", None) or p["id"]
            rolle = getattr(p, "de", None) or getattr(p, "id", pid)
            if pid in self._cancel[lauf_id]:
                return None
            system = agent_prompt(rolle)
            rendered = (system + "\n\n" + question).strip()
            job_id = new_id("job")
            rec.setdefault("snapshots", []).append(
                {
                    "job_id": job_id,
                    "phase": "perspektive",
                    "perspective_id": pid,
                    "rendered": rendered,
                    "model": persp_model.model,
                }
            )
            provider = registry.get(persp_model.id)
            task = asyncio.create_task(
                provider.complete(prompt=rendered, model=persp_model.model, timeout_s=30)
            )
            while not task.done():
                if pid in self._cancel[lauf_id]:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    return None
                await asyncio.sleep(0.02)
            resp = task.result()
            return _parse_perspective(resp.text, rolle)

        results = await asyncio.gather(*[eine(p) for p in persps], return_exceptions=True)
        rec["perspectives"] = [r for r in results if isinstance(r, dict)]
        events["perspektiven_done"].set()
        rec["phase"] = "reibung"
        events["reibung_started"].set()

        friction_prompt = json.dumps(rec["perspectives"], ensure_ascii=False) + "\n" + question
        provider = registry.get(reibung_m.id)
        fr = await provider.complete(
            prompt=friction_prompt, model=reibung_m.model, timeout_s=30
        )
        rec["friction"] = _parse_friction(fr.text)
        events["reibung_done"].set()
        rec["phase"] = "integration"
        events["integration_started"].set()

        integ_prompt = json.dumps(
            {"perspectives": rec["perspectives"], "friction": rec["friction"], "question": question},
            ensure_ascii=False,
        )
        provider = registry.get(integ_m.id)
        ig = await provider.complete(
            prompt=integ_prompt, model=integ_m.model, timeout_s=30
        )
        rec["integration"] = _parse_integration(ig.text)
        rec["phase"] = "done"
        rec["status"] = "done"
        rec["table"] = {
            "perspectives": rec["perspectives"],
            "friction": rec["friction"],
            "integration": rec["integration"],
        }
        await _memory_opt_in({"lauf_id": lauf_id, "raw": True, "tisch_id": rec["tisch_id"]})


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-")


# FakeProvider imported so tests can configure without a new mill
_ = FakeProvider
