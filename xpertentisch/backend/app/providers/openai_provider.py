"""Adapter für die OpenAI-API."""

from __future__ import annotations

import asyncio

from .base import Provider, ProviderError, ProviderResponse

SYSTEM_PROMPT = (
    "Du sitzt an einem Expertentisch. Antworte eigenständig, sachlich und in der "
    "Sprache der Frage. Nenne Unsicherheiten offen. Antworte ohne Rückfragen."
)


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - Abhängigkeit fehlt
            raise ProviderError(f"openai-Paket nicht installiert: {exc}") from exc
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def complete(self, *, prompt: str, model: str, timeout_s: int) -> ProviderResponse:
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError(f"Zeitüberschreitung nach {timeout_s}s") from exc
        except Exception as exc:  # Provider-eigene Fehlerklassen bleiben isoliert
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "") if choice else ""
        partial = bool(choice and choice.finish_reason not in (None, "stop"))
        note = f"Abbruchgrund: {choice.finish_reason}" if partial and choice else ""
        if not text.strip():
            raise ProviderError("Leere Antwort erhalten.")
        return ProviderResponse(text=text, partial=partial, note=note)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
