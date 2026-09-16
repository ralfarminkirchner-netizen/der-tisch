"""Adapter für die Messages-Schnittstelle von Anthropic."""

from __future__ import annotations

import asyncio

from .base import OnDelta, Provider, ProviderError, ProviderResponse
from .openai_provider import SYSTEM_PROMPT


class AnthropicProvider(Provider):
    name = "anthropic"
    streams = True

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

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        timeout_s: int,
        on_delta: OnDelta | None = None,
    ) -> ProviderResponse:
        try:
            if on_delta is not None:
                return await asyncio.wait_for(
                    self._stream(prompt, model, on_delta), timeout=timeout_s
                )
            nachricht = await asyncio.wait_for(
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
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        text = "".join(
            block.text for block in nachricht.content
            if getattr(block, "type", "") == "text"
        )
        return self._fertig(text, nachricht.stop_reason, getattr(nachricht, "usage", None))

    async def _stream(self, prompt: str, model: str, on_delta: OnDelta) -> ProviderResponse:
        teile: list[str] = []
        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as strom:
                async for stueck in strom.text_stream:
                    teile.append(stueck)
                    on_delta(stueck)
                fertig = await strom.get_final_message()
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}",
                                partial_text="".join(teile)) from exc
        return self._fertig("".join(teile), fertig.stop_reason, getattr(fertig, "usage", None))

    def _fertig(self, text: str, stop_reason: str | None, verbrauch: object) -> ProviderResponse:
        if not text.strip():
            raise ProviderError("Leere Antwort erhalten.")
        partial = stop_reason == "max_tokens"
        return ProviderResponse(
            text=text,
            partial=partial,
            note="Antwort an der Längengrenze abgeschnitten." if partial else "",
            tokens_in=getattr(verbrauch, "input_tokens", None),
            tokens_out=getattr(verbrauch, "output_tokens", None),
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
