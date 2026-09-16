"""Adapter für die Anthropic-API."""

from __future__ import annotations

import asyncio

from .base import Provider, ProviderError, ProviderResponse
from .openai_provider import SYSTEM_PROMPT


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self, api_key: str, base_url: str | None = None, max_tokens: int = 4096
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - Abhängigkeit fehlt
            raise ProviderError(f"anthropic-Paket nicht installiert: {exc}") from exc
        kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._max_tokens = max_tokens

    async def complete(self, *, prompt: str, model: str, timeout_s: int) -> ProviderResponse:
        try:
            message = await asyncio.wait_for(
                self._client.messages.create(
                    model=model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError(f"Zeitüberschreitung nach {timeout_s}s") from exc
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        partial = message.stop_reason == "max_tokens"
        note = "Antwort an der Längengrenze abgeschnitten." if partial else ""
        if not text.strip():
            raise ProviderError("Leere Antwort erhalten.")
        return ProviderResponse(text=text, partial=partial, note=note)

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
