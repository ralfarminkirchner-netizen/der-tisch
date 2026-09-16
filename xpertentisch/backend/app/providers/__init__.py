"""Provider-Registry."""

from __future__ import annotations

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
    """Erzeugt Provider-Instanzen träge und genau einmal."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._instances: dict[str, Provider] = {}
        self._errors: dict[str, str] = {}

    def get(self, name: str) -> Provider:
        if name in self._instances:
            return self._instances[name]
        if name in self._errors:
            raise ProviderError(self._errors[name])

        try:
            provider = self._build(name)
        except ProviderError as exc:
            self._errors[name] = str(exc)
            raise
        self._instances[name] = provider
        return provider

    def _build(self, name: str) -> Provider:
        s = self._settings
        if name == "fake":
            if not s.allow_fake_providers:
                raise ProviderError(
                    "Test-Provider 'fake' ist nicht freigeschaltet "
                    "(XT_ALLOW_FAKE_PROVIDERS fehlt)."
                )
            return FakeProvider()
        if name == "openai":
            if not s.resolved_openai_key:
                raise ProviderError(
                    "Kein OpenAI-Schlüssel hinterlegt — in den Einstellungen eintragen "
                    "oder OPENAI_API_KEY setzen."
                )
            from .openai_provider import OpenAIProvider

            return OpenAIProvider(s.resolved_openai_key, s.openai_base_url)
        if name == "anthropic":
            if not s.resolved_anthropic_key:
                raise ProviderError(
                    "Kein Anthropic-Schlüssel hinterlegt — in den Einstellungen eintragen "
                    "oder ANTHROPIC_API_KEY setzen."
                )
            from .anthropic_provider import AnthropicProvider

            return AnthropicProvider(s.resolved_anthropic_key)
        raise ProviderError(f"Unbekannter Provider: {name}")

    def availability(self) -> dict[str, dict[str, object]]:
        """Bereitschaft je konfiguriertem Provider, ohne Netzwerkaufruf."""
        result: dict[str, dict[str, object]] = {}
        for model in self._settings.enabled_models():
            if model.provider in result:
                continue
            try:
                self.get(model.provider)
                result[model.provider] = {"ready": True, "reason": ""}
            except ProviderError as exc:
                result[model.provider] = {"ready": False, "reason": str(exc)}
        return result

    def invalidate(self) -> None:
        """Verwirft zwischengespeicherte Provider.

        Nach einer Änderung auf der Einstellungsseite muss der nächste Auftrag
        mit den neuen Zugangsdaten gebaut werden.
        """
        self._instances.clear()
        self._errors.clear()

    async def aclose(self) -> None:
        for provider in self._instances.values():
            try:
                await provider.aclose()
            except Exception:  # pragma: no cover - Abbau darf nie stören
                pass
        self._instances.clear()
