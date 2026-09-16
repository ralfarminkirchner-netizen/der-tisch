"""Adapter für die OpenAI-Schnittstelle — und alles, was sie spricht."""

from __future__ import annotations

import asyncio

from .base import OnDelta, Provider, ProviderError, ProviderResponse

SYSTEM_PROMPT = (
    "Du sitzt an einem Expertentisch. Antworte eigenständig, sachlich und in der "
    "Sprache der Frage. Nenne Unsicherheiten offen. Antworte ohne Rückfragen."
)


class OpenAIProvider(Provider):
    name = "openai"
    streams = True

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - Abhängigkeit fehlt
            raise ProviderError(f"openai-Paket nicht installiert: {exc}") from exc
        kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        timeout_s: int,
        on_delta: OnDelta | None = None,
    ) -> ProviderResponse:
        nachrichten = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            if on_delta is not None:
                return await asyncio.wait_for(
                    self._stream(nachrichten, model, on_delta), timeout=timeout_s
                )
            antwort = await asyncio.wait_for(
                self._client.chat.completions.create(model=model, messages=nachrichten),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError(f"Zeitüberschreitung nach {timeout_s}s") from exc
        except ProviderError:
            raise
        except Exception as exc:  # Provider-eigene Fehlerklassen bleiben isoliert
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        wahl = antwort.choices[0] if antwort.choices else None
        text = (wahl.message.content or "") if wahl else ""
        partial = bool(wahl and wahl.finish_reason not in (None, "stop"))
        verbrauch = getattr(antwort, "usage", None)
        if not text.strip():
            raise ProviderError("Leere Antwort erhalten.")
        return ProviderResponse(
            text=text,
            partial=partial,
            note=f"Abbruchgrund: {wahl.finish_reason}" if partial and wahl else "",
            tokens_in=getattr(verbrauch, "prompt_tokens", None),
            tokens_out=getattr(verbrauch, "completion_tokens", None),
        )

    async def _stream(
        self, nachrichten: list[dict[str, str]], model: str, on_delta: OnDelta
    ) -> ProviderResponse:
        """Sammelt den Strom ein und reicht jedes Stück sofort weiter."""
        teile: list[str] = []
        abbruchgrund: str | None = None
        tokens_in: int | None = None
        tokens_out: int | None = None

        strom = await self._client.chat.completions.create(
            model=model,
            messages=nachrichten,
            stream=True,
            stream_options={"include_usage": True},
        )
        try:
            async for stueck in strom:
                verbrauch = getattr(stueck, "usage", None)
                if verbrauch is not None:
                    tokens_in = getattr(verbrauch, "prompt_tokens", tokens_in)
                    tokens_out = getattr(verbrauch, "completion_tokens", tokens_out)
                if not stueck.choices:
                    continue
                wahl = stueck.choices[0]
                if wahl.finish_reason:
                    abbruchgrund = wahl.finish_reason
                inhalt = getattr(wahl.delta, "content", None)
                if inhalt:
                    teile.append(inhalt)
                    on_delta(inhalt)
        except Exception as exc:
            # Was schon ankam, bleibt als Teilantwort erhalten.
            raise ProviderError(f"{type(exc).__name__}: {exc}",
                                partial_text="".join(teile)) from exc

        text = "".join(teile)
        if not text.strip():
            raise ProviderError("Leere Antwort erhalten.")
        partial = abbruchgrund not in (None, "stop")
        return ProviderResponse(
            text=text,
            partial=partial,
            note=f"Abbruchgrund: {abbruchgrund}" if partial else "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()
