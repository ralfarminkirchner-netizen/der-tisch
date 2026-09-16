"""Konfiguration für XPERTENTiSCH.

Die Grundeinstellungen kommen aus Umgebungsvariablen. Zugangsdaten und
Modellnamen lassen sich zusätzlich zur Laufzeit über die Einstellungsseite
setzen; solche Werte liegen in der Datenbank und haben Vorrang vor der
Umgebung (`overrides`).

Ein zentrales Prinzip bleibt: Test-Fakes (`fake`-Provider) dürfen im
Produktionsmodus niemals unbemerkt für echte Provider einspringen. Dafür
sorgt `Settings.validate()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

TRUE_VALUES = {"1", "true", "yes", "on", "ja"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ModelConfig:
    """Ein am Tisch sitzendes Modell."""

    id: str
    label: str
    provider: str
    model: str
    enabled: bool = True


@dataclass
class Settings:
    env: str = "production"
    db_path: Path = Path("xpertentisch.sqlite3")
    host: str = "0.0.0.0"
    port: int = 8000
    allow_fake_providers: bool = False
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    request_timeout_s: int = 120
    max_prompt_chars: int = 20000
    cors_origins: list[str] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    #: Token für die Einstellungsseite. Ohne dieses Token bleibt sie gesperrt.
    admin_token: str | None = None
    #: Zur Laufzeit gesetzte Werte aus der Datenbank. Vorrang vor der Umgebung.
    overrides: dict[str, str] = field(default_factory=dict)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    # ------------------------------------------------- Aufgelöste Einstellungen

    @property
    def resolved_timeout_s(self) -> int:
        roh = (self.overrides.get("request_timeout_s") or "").strip()
        if roh.isdigit() and 5 <= int(roh) <= 600:
            return int(roh)
        return self.request_timeout_s

    @property
    def timeout_source(self) -> str:
        return "einstellungen" if self.overrides.get("request_timeout_s") else "umgebung"

    def enabled_models(self) -> list[ModelConfig]:
        """Nur noch für den Test- und Entwicklungsbetrieb mit festem Tisch."""
        return [m for m in self.models if m.enabled]

    def model_by_id(self, model_id: str) -> ModelConfig | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def validate(self) -> None:
        """Bricht ab, wenn die Konfiguration unsicher oder unbrauchbar ist."""
        if self.is_production and self.allow_fake_providers:
            raise RuntimeError(
                "XT_ALLOW_FAKE_PROVIDERS ist im Produktionsmodus gesetzt. "
                "Test-Fakes dürfen im Produktionsbetrieb nicht aktiv sein. "
                "Entweder XT_ENV=development setzen oder XT_ALLOW_FAKE_PROVIDERS entfernen."
            )
        for m in self.models:
            if m.provider == "fake" and not self.allow_fake_providers:
                raise RuntimeError(
                    f"Modell '{m.id}' nutzt den Test-Provider 'fake', "
                    "aber XT_ALLOW_FAKE_PROVIDERS ist nicht gesetzt."
                )

FAKE_MODELS = [
    ModelConfig(id="fake-a", label="Fake A", provider="fake", model="fake-fast"),
    ModelConfig(id="fake-b", label="Fake B", provider="fake", model="fake-slow"),
]


def load_settings(env: dict[str, str] | None = None) -> Settings:
    if env is not None:
        os.environ.update(env)

    app_env = os.environ.get("XT_ENV", "production").strip().lower()
    if app_env not in {"production", "development", "test"}:
        app_env = "production"

    allow_fakes = _env_bool("XT_ALLOW_FAKE_PROVIDERS", False)
    use_fake_models = _env_bool("XT_USE_FAKE_MODELS", False)

    if use_fake_models and not allow_fakes:
        raise RuntimeError(
            "XT_USE_FAKE_MODELS verlangt zusätzlich XT_ALLOW_FAKE_PROVIDERS=1."
        )

    # Leere Liste heißt: der Tisch kommt aus der Anbietertabelle der Datenbank.
    models = list(FAKE_MODELS) if use_fake_models else []

    cors_raw = os.environ.get("XT_CORS_ORIGINS", "").strip()
    cors = [o.strip() for o in cors_raw.split(",") if o.strip()]

    settings = Settings(
        env=app_env,
        db_path=Path(os.environ.get("XT_DB_PATH", "xpertentisch.sqlite3")),
        host=os.environ.get("XT_HOST", "0.0.0.0"),
        port=_env_int("PORT", _env_int("XT_PORT", 8000)),
        allow_fake_providers=allow_fakes,
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        request_timeout_s=_env_int("XT_REQUEST_TIMEOUT_S", 120),
        max_prompt_chars=_env_int("XT_MAX_PROMPT_CHARS", 20000),
        cors_origins=cors,
        models=models,
        admin_token=(os.environ.get("XT_ADMIN_TOKEN") or "").strip() or None,
    )
    settings.validate()
    return settings
