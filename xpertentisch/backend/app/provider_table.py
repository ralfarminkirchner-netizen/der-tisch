"""Der Tisch: wer sitzt daran, und womit spricht er.

Die Anbieter kommen aus der Datenbank, nicht aus dem Quelltext. Diese Schicht
hält den aktuellen Stand im Speicher, damit die Anfragebehandlung ihn ohne
Datenbankzugriff lesen kann, und löst Zugangsdaten gegen die Umgebung auf.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .catalog import PRESET_BY_ID, PRESETS
from .config import ModelConfig, Settings
from .db import Store

KIND_FAKE = "fake"


@dataclass
class ProviderRecord:
    id: str
    label: str
    kind: str
    base_url: str | None
    model: str
    #: In der Datenbank hinterlegter Schlüssel. Leer heißt: aus der Umgebung.
    stored_key: str
    enabled: bool
    is_preset: bool
    position: int = 0
    #: Preis je Million Token. None heißt: unbekannt, es wird nichts geraten.
    price_in: float | None = None
    price_out: float | None = None

    @property
    def key_env(self) -> str:
        preset = PRESET_BY_ID.get(self.id)
        return preset.key_env if preset else f"{self.id.upper()}_API_KEY"

    @property
    def api_key(self) -> str:
        return self.stored_key or (os.environ.get(self.key_env) or "")

    @property
    def key_source(self) -> str:
        if self.stored_key:
            return "einstellungen"
        return "umgebung" if os.environ.get(self.key_env) else "fehlt"

    @property
    def needs_key(self) -> bool:
        return self.kind != KIND_FAKE

    @property
    def ready(self) -> bool:
        """Kann dieser Anbieter jetzt antworten?"""
        if not self.enabled:
            return False
        return bool(self.api_key) or not self.needs_key

    @property
    def reason(self) -> str:
        """Was diesem Anbieter fehlt.

        Der fehlende Schlüssel wiegt schwerer als das Abschalten: dass ein
        Anbieter abgeschaltet ist, steht ohnehin am Schalter daneben.
        """
        if self.needs_key and not self.api_key:
            return (
                f"Kein Schlüssel hinterlegt — in den Einstellungen eintragen "
                f"oder {self.key_env} setzen."
            )
        if not self.enabled:
            return "Abgeschaltet."
        return ""

    def as_model(self) -> ModelConfig:
        return ModelConfig(id=self.id, label=self.label, provider=self.id, model=self.model)

    def public(self) -> dict[str, Any]:
        """Darstellung für die Einstellungsseite — ohne den Schlüssel selbst."""
        preset = PRESET_BY_ID.get(self.id)
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "base_url": self.base_url,
            "model": self.model,
            "enabled": self.enabled,
            "is_preset": self.is_preset,
            "needs_key": self.needs_key,
            "key_source": self.key_source,
            "key_hint": _hint(self.api_key) if self.api_key else "",
            "key_env": self.key_env,
            "ready": self.ready,
            "reason": self.reason,
            "key_url": preset.key_url if preset else "",
            "models_url": preset.models_url if preset else "",
            "price_in": self.price_in,
            "price_out": self.price_out,
        }


def _hint(value: str) -> str:
    """Erkennungshilfe für einen Schlüssel — nie der Schlüssel selbst."""
    return f"…{value[-4:]}" if len(value) >= 8 else "gesetzt"


def _from_row(row: dict[str, Any]) -> ProviderRecord:
    return ProviderRecord(
        id=row["id"],
        label=row["label"],
        kind=row["kind"],
        base_url=row["base_url"],
        model=row["model"],
        stored_key=row["api_key"] or "",
        enabled=bool(row["enabled"]),
        is_preset=bool(row["is_preset"]),
        position=int(row["position"]),
        price_in=row["price_in"] if "price_in" in row.keys() else None,
        price_out=row["price_out"] if "price_out" in row.keys() else None,
    )


class ProviderTable:
    """Der aktuelle Stand der Anbieter, im Speicher gehalten."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records: list[ProviderRecord] = []

    async def reload(self, store: Store) -> None:
        if self._settings.models:
            # Test- und Entwicklungsbetrieb: der Tisch steht fest im Quelltext.
            self._records = [
                ProviderRecord(
                    id=m.id, label=m.label, kind=m.provider, base_url=None, model=m.model,
                    stored_key="", enabled=m.enabled, is_preset=False, position=i,
                )
                for i, m in enumerate(self._settings.models)
            ]
            return
        await store.ensure_presets(
            PRESETS, {p.key_env: os.environ.get(p.key_env) for p in PRESETS}
        )
        self._records = [_from_row(r) for r in await store.list_providers()]

    # ------------------------------------------------------------------ Lesen

    def all(self) -> list[ProviderRecord]:
        return list(self._records)

    def by_id(self, provider_id: str) -> ProviderRecord | None:
        return next((r for r in self._records if r.id == provider_id), None)

    def ready(self) -> list[ProviderRecord]:
        """Anbieter, die jetzt an den Tisch können."""
        return [r for r in self._records if r.ready]

    def as_models(self) -> list[ModelConfig]:
        return [r.as_model() for r in self.ready()]

    def missing_keys(self) -> list[str]:
        return [
            r.key_env for r in self._records
            if r.enabled and r.needs_key and not r.api_key
        ]

    def curator(self) -> ModelConfig | None:
        """Wer die Kuratierung übernimmt — oder niemand.

        Es wird nie heimlich ein Modell zugeschaltet: ohne ausdrückliche
        Einstellung gibt es keinen Kurator.
        """
        gewaehlt = (self._settings.overrides.get("curator") or "").strip()
        if not gewaehlt:
            return None
        record = self.by_id(gewaehlt)
        return record.as_model() if record and record.ready else None

    def availability(self) -> dict[str, dict[str, Any]]:
        return {
            r.id: {"ready": r.ready, "reason": r.reason, "label": r.label}
            for r in self._records
            if r.enabled
        }
