"""Gemeinsame Schnittstelle aller Provider-Adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

#: Wird mit jedem Textstück aufgerufen, sobald es eintrifft.
OnDelta = Callable[[str], None]


@dataclass
class ProviderResponse:
    text: str
    partial: bool = False
    note: str = ""
    #: Verbrauch, sofern der Anbieter ihn meldet. None heißt: unbekannt.
    tokens_in: int | None = None
    tokens_out: int | None = None


class ProviderError(Exception):
    """Fehler eines einzelnen Providers.

    Wird pro Auftrag gefangen; andere Provider laufen unbeeinflusst weiter.
    """

    def __init__(self, message: str, *, partial_text: str = "") -> None:
        super().__init__(message)
        self.partial_text = partial_text


class Provider:
    name = "base"
    #: Liefert dieser Adapter Textstücke, während sie entstehen?
    streams = False

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        timeout_s: int,
        on_delta: OnDelta | None = None,
    ) -> ProviderResponse:
        """Holt die Antwort.

        Ist `on_delta` gesetzt und kann der Adapter streamen, wird jedes
        Textstück sofort weitergereicht. Die vollständige Antwort kommt
        trotzdem als Rückgabewert.
        """
        raise NotImplementedError

    async def aclose(self) -> None:  # pragma: no cover - Standardfall ohne Ressourcen
        return None
