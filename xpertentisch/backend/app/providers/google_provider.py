"""Adapter für die Generative-Language-Schnittstelle von Google (Gemini).

Bewusst direkt über HTTP statt über ein weiteres SDK: die Schnittstelle ist
klein, und jede zusätzliche Abhängigkeit will gepflegt werden.
"""

from __future__ import annotations

import httpx

from .base import Provider, ProviderError, ProviderResponse
from .openai_provider import SYSTEM_PROMPT

STANDARD_BASIS = "https://generativelanguage.googleapis.com/v1beta"


class GoogleProvider(Provider):
    name = "google"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = (base_url or STANDARD_BASIS).rstrip("/")

    async def complete(self, *, prompt: str, model: str, timeout_s: int) -> ProviderResponse:
        ziel = f"{self._base_url}/models/{model}:generateContent"
        nutzlast = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                antwort = await client.post(
                    ziel, json=nutzlast, headers={"x-goog-api-key": self._api_key}
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Zeitüberschreitung nach {timeout_s}s") from exc
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        if antwort.status_code >= 400:
            raise ProviderError(_fehlertext(antwort))

        try:
            daten = antwort.json()
        except ValueError as exc:
            raise ProviderError("Antwort war kein JSON.") from exc

        kandidaten = daten.get("candidates") or []
        if not kandidaten:
            grund = (daten.get("promptFeedback") or {}).get("blockReason")
            raise ProviderError(
                f"Keine Antwort erhalten (blockiert: {grund})." if grund
                else "Keine Antwort erhalten."
            )

        kandidat = kandidaten[0]
        teile = ((kandidat.get("content") or {}).get("parts")) or []
        text = "".join(t.get("text", "") for t in teile)
        abbruch = kandidat.get("finishReason")
        partial = abbruch not in (None, "STOP")
        if not text.strip():
            raise ProviderError(f"Leere Antwort erhalten (Abbruchgrund: {abbruch}).")
        return ProviderResponse(
            text=text,
            partial=partial,
            note=f"Abbruchgrund: {abbruch}" if partial else "",
        )


def _fehlertext(antwort: httpx.Response) -> str:
    """Macht aus einer Fehlerantwort eine Meldung, die weiterhilft."""
    try:
        fehler = (antwort.json() or {}).get("error") or {}
        meldung = fehler.get("message") or antwort.text[:200]
    except ValueError:
        meldung = antwort.text[:200]
    if antwort.status_code in (401, 403):
        return f"Zugangsdaten abgelehnt ({antwort.status_code}): {meldung}"
    if antwort.status_code == 404:
        return f"Modell oder Adresse nicht gefunden ({antwort.status_code}): {meldung}"
    if antwort.status_code == 429:
        return f"Zu viele Anfragen ({antwort.status_code}): {meldung}"
    return f"Fehler {antwort.status_code}: {meldung}"
