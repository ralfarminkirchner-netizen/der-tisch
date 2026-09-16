"""Provider-Registry.

Gebaut wird nach der *Art* der Schnittstelle, nicht nach dem Namen des
Anbieters: eine OpenAI-kompatible Adresse ist eine OpenAI-kompatible Adresse,
gleich ob dahinter DeepSeek, Mistral, Groq oder ein lokaler Server steht.
"""

from __future__ import annotations

from ..catalog import KIND_ANTHROPIC, KIND_GOOGLE, KIND_OPENAI
from ..config import Settings
from .base import Provider, ProviderError, ProviderResponse
from .fake import FakeProvider, FakeScenario

__all__ = [
    "Provider",
    "ProviderError",
    "ProviderResponse",
    "FakeProvider",
    "FakeScenario",
    "ProviderRegistry",
]


class ProviderRegistry:
    """Erzeugt Adapter träge und genau einmal je Anbieter."""

    def __init__(self, settings: Settings, table) -> None:
        self._settings = settings
        self._table = table
        self._instances: dict[str, Provider] = {}
        self._errors: dict[str, str] = {}

    def get(self, provider_id: str) -> Provider:
        if provider_id in self._instances:
            return self._instances[provider_id]
        if provider_id in self._errors:
            raise ProviderError(self._errors[provider_id])

        try:
            provider = self._build(provider_id)
        except ProviderError as exc:
            self._errors[provider_id] = str(exc)
            raise
        self._instances[provider_id] = provider
        return provider

    def _build(self, provider_id: str) -> Provider:
        record = self._table.by_id(provider_id)
        if record is None:
            raise ProviderError(f"Unbekannter Anbieter: {provider_id}")

        if record.kind == "fake":
            if not self._settings.allow_fake_providers:
                raise ProviderError(
                    "Test-Provider 'fake' ist nicht freigeschaltet "
                    "(XT_ALLOW_FAKE_PROVIDERS fehlt)."
                )
            return FakeProvider()

        if not record.api_key:
            raise ProviderError(record.reason or "Kein Schlüssel hinterlegt.")

        if record.kind == KIND_OPENAI:
            from .openai_provider import OpenAIProvider

            return OpenAIProvider(record.api_key, record.base_url)
        if record.kind == KIND_ANTHROPIC:
            from .anthropic_provider import AnthropicProvider

            return AnthropicProvider(record.api_key, base_url=record.base_url)
        if record.kind == KIND_GOOGLE:
            from .google_provider import GoogleProvider

            return GoogleProvider(record.api_key, record.base_url)
        raise ProviderError(f"Unbekannte Schnittstellenart: {record.kind}")

    def availability(self) -> dict[str, dict[str, object]]:
        """Bereitschaft je eingeschaltetem Anbieter, ohne Netzwerkaufruf."""
        ergebnis: dict[str, dict[str, object]] = {}
        for record in self._table.all():
            if not record.enabled:
                continue
            if not record.ready:
                ergebnis[record.id] = {"ready": False, "reason": record.reason,
                                       "label": record.label}
                continue
            try:
                self.get(record.id)
                ergebnis[record.id] = {"ready": True, "reason": "", "label": record.label}
            except ProviderError as exc:
                ergebnis[record.id] = {"ready": False, "reason": str(exc),
                                       "label": record.label}
        return ergebnis

    def invalidate(self) -> None:
        """Verwirft zwischengespeicherte Adapter nach einer Änderung."""
        self._instances.clear()
        self._errors.clear()

    async def aclose(self) -> None:
        for provider in self._instances.values():
            try:
                await provider.aclose()
            except Exception:  # pragma: no cover - Abbau darf nie stören
                pass
        self._instances.clear()
