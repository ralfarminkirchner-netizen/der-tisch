"""Anbieterverzeichnis.

Ein Anbieter ist hier kein fest verdrahteter Sonderfall, sondern ein Datensatz:
Art der Schnittstelle, Basis-Adresse, Modellname, Zugangsdaten. Die mitgelieferten
Voreinstellungen sind nur Startpunkte — jeder Wert lässt sich in den Einstellungen
ändern, und beliebige weitere Anbieter kommen über „Eigener Endpunkt“ dazu.

Drei Arten von Schnittstellen decken praktisch das gesamte Feld ab:

* ``openai``    — die OpenAI-Schnittstelle. Viele Anbieter sprechen sie
                  (DeepSeek, Mistral, xAI, Groq, OpenRouter, Together, Fireworks,
                  vLLM, Ollama …); der Unterschied ist allein die Basis-Adresse.
* ``anthropic`` — die Messages-Schnittstelle von Anthropic.
* ``google``    — die Generative-Language-Schnittstelle von Google.
"""

from __future__ import annotations

from dataclasses import dataclass

KIND_OPENAI = "openai"
KIND_ANTHROPIC = "anthropic"
KIND_GOOGLE = "google"
KINDS = (KIND_OPENAI, KIND_ANTHROPIC, KIND_GOOGLE)

KIND_LABELS = {
    KIND_OPENAI: "OpenAI-kompatibel",
    KIND_ANTHROPIC: "Anthropic",
    KIND_GOOGLE: "Google",
}


@dataclass(frozen=True)
class ProviderPreset:
    """Ein mitgelieferter Anbieter. Alle Werte sind änderbare Vorschläge."""

    id: str
    label: str
    kind: str
    base_url: str | None
    default_model: str
    key_env: str
    key_url: str
    models_url: str


PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        id="openai",
        label="OpenAI",
        kind=KIND_OPENAI,
        base_url=None,
        default_model="gpt-4.1",
        key_env="OPENAI_API_KEY",
        key_url="https://platform.openai.com/api-keys",
        models_url="https://platform.openai.com/docs/models",
    ),
    ProviderPreset(
        id="anthropic",
        label="Anthropic",
        kind=KIND_ANTHROPIC,
        base_url=None,
        default_model="claude-sonnet-5",
        key_env="ANTHROPIC_API_KEY",
        key_url="https://console.anthropic.com/settings/keys",
        models_url="https://docs.anthropic.com/en/docs/about-claude/models",
    ),
    ProviderPreset(
        id="google",
        label="Google Gemini",
        kind=KIND_GOOGLE,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-2.5-flash",
        key_env="GOOGLE_API_KEY",
        key_url="https://aistudio.google.com/apikey",
        models_url="https://ai.google.dev/gemini-api/docs/models",
    ),
    ProviderPreset(
        id="deepseek",
        label="DeepSeek",
        kind=KIND_OPENAI,
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        key_env="DEEPSEEK_API_KEY",
        key_url="https://platform.deepseek.com/api_keys",
        models_url="https://api-docs.deepseek.com/quick_start/pricing",
    ),
    ProviderPreset(
        id="mistral",
        label="Mistral",
        kind=KIND_OPENAI,
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        key_env="MISTRAL_API_KEY",
        key_url="https://console.mistral.ai/api-keys",
        models_url="https://docs.mistral.ai/getting-started/models/models_overview/",
    ),
    ProviderPreset(
        id="xai",
        label="xAI Grok",
        kind=KIND_OPENAI,
        base_url="https://api.x.ai/v1",
        default_model="grok-4",
        key_env="XAI_API_KEY",
        key_url="https://console.x.ai",
        models_url="https://docs.x.ai/docs/models",
    ),
)

PRESET_BY_ID = {p.id: p for p in PRESETS}

#: Vorschläge für den Dialog „Eigener Endpunkt“. Reine Bequemlichkeit.
CUSTOM_HINTS = (
    {"label": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4.1"},
    {"label": "Together", "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    {"label": "Ollama (lokal)", "base_url": "http://127.0.0.1:11434/v1", "model": "llama3.1"},
)
