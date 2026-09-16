"""Gemeinsame Schnittstelle aller Provider-Adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderResponse:
    text: str
    partial: bool = False
    note: str = ""


class ProviderError(Exception):
    """Fehler eines einzelnen Providers.

    Wird pro Auftrag gefangen; andere Provider laufen unbeeinflusst weiter.
    """

    def __init__(self, message: str, *, partial_text: str = "") -> None:
        super().__init__(message)
        self.partial_text = partial_text


class Provider:
    name = "base"

    async def complete(self, *, prompt: str, model: str, timeout_s: int) -> ProviderResponse:
        raise NotImplementedError

    async def aclose(self) -> None:  # pragma: no cover - Standardfall ohne Ressourcen
        return None
