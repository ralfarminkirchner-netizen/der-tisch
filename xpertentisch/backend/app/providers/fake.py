"""Kontrollierbarer Test-Provider.

Nur für reproduzierbare Tests und lokale Entwicklung. Er wird ausschließlich
instanziiert, wenn `XT_ALLOW_FAKE_PROVIDERS` gesetzt ist (siehe config.py);
im Produktionsmodus bricht die Konfigurationsprüfung vorher ab.

Steuerbar sind Verzögerung, Teilantwort und Fehler — je Modellname.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field

from .base import OnDelta, Provider, ProviderError, ProviderResponse


@dataclass
class FakeScenario:
    delay_s: float = 0.0
    text: str | None = None
    partial: bool = False
    error: str | None = None
    #: Text, der bei einem Fehler bereits angefallen ist (Teilantwort).
    partial_text: str = ""
    #: Zählt die tatsächlichen Aufrufe – Grundlage für den Doppelaufruf-Test.
    calls: list[str] = field(default_factory=list)
    #: In so viele Stücke zerfällt die Antwort beim Streamen.
    chunks: int = 1
    #: Pause zwischen den Stücken.
    chunk_delay_s: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None


class FakeProvider(Provider):
    name = "fake"
    streams = True

    #: Szenarien je Modellname, global setzbar (Tests).
    scenarios: dict[str, FakeScenario] = {}

    @classmethod
    def configure(cls, model: str, scenario: FakeScenario) -> FakeScenario:
        cls.scenarios[model] = scenario
        return scenario

    @classmethod
    def reset(cls) -> None:
        cls.scenarios = {}

    @classmethod
    def call_count(cls, model: str) -> int:
        scenario = cls.scenarios.get(model)
        return len(scenario.calls) if scenario else 0

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        timeout_s: int,
        on_delta: OnDelta | None = None,
    ) -> ProviderResponse:
        scenario = self.scenarios.get(model) or FakeScenario()
        scenario.calls.append(prompt)

        if scenario.delay_s:
            await asyncio.sleep(scenario.delay_s)

        if scenario.error:
            if on_delta and scenario.partial_text:
                on_delta(scenario.partial_text)
            raise ProviderError(scenario.error, partial_text=scenario.partial_text)

        text = scenario.text if scenario.text is not None else _canned_answer(prompt, model)

        if on_delta and scenario.chunks > 1:
            groesse = max(1, len(text) // scenario.chunks)
            for start in range(0, len(text), groesse):
                on_delta(text[start:start + groesse])
                if scenario.chunk_delay_s:
                    await asyncio.sleep(scenario.chunk_delay_s)
        elif on_delta:
            on_delta(text)

        return ProviderResponse(
            text=text, partial=scenario.partial,
            tokens_in=scenario.tokens_in, tokens_out=scenario.tokens_out,
        )


def _canned_answer(prompt: str, model: str) -> str:
    """Deterministische Ersatzantwort ohne Netzwerk."""
    digest = hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()[:8]
    return (
        f"Testantwort von {model} (Signatur {digest}). "
        f"Die Frage lautete: {prompt.strip()[:200]} "
        "Dies ist eine künstliche Antwort ohne inhaltliche Aussagekraft."
    )
