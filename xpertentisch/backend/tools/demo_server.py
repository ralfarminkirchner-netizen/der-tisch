"""Startet den Tisch mit steuerbaren Test-Providern — nur für die Browserprüfung.

Die Test-Provider springen niemals unbemerkt ein: dieser Starter setzt
XT_ALLOW_FAKE_PROVIDERS ausdrücklich und läuft damit im Entwicklungsmodus.
Im Produktionsmodus bricht die Konfigurationsprüfung vorher ab.

    python tools/demo_server.py 8077
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("XT_ENV", "development")
os.environ.setdefault("XT_ALLOW_FAKE_PROVIDERS", "1")
os.environ.setdefault("XT_USE_FAKE_MODELS", "1")
os.environ.setdefault("XT_ADMIN_TOKEN", "demo-zugangswort")

import uvicorn  # noqa: E402

from app.main import create_app  # noqa: E402
from app.config import ModelConfig  # noqa: E402
from app.providers.fake import FakeProvider, FakeScenario  # noqa: E402

LANGTEXT = (
    "Eine tägliche Sicherung ist sinnvoll, sobald der Verlust eines Tages "
    "spürbar wäre. Entscheidend ist nicht der Rhythmus allein, sondern die "
    "Frage, wie viel Arbeit zwischen zwei Sicherungen entstehen darf. "
    "Wichtig ist außerdem, die Wiederherstellung regelmäßig zu proben: eine "
    "Sicherung, die nie zurückgespielt wurde, ist ein Versprechen ohne Beleg."
)

KURZTEXT = (
    "Täglich ist ein guter Ausgangswert. Wichtig ist, dass die Sicherung "
    "außerhalb desselben Servers liegt."
)

# Ein schneller und ein langsamer Anbieter — so wird sichtbar, dass die Karten
# unabhängig voneinander laufen und dass sich ein Auftrag abbrechen lässt.
FakeProvider.configure("fake-fast", FakeScenario(
    text=LANGTEXT, chunks=8, chunk_delay_s=0.25, tokens_in=180, tokens_out=140,
))
FakeProvider.configure("fake-slow", FakeScenario(
    delay_s=1.0, text=LANGTEXT.replace("sinnvoll", "nicht in jedem Fall nötig"),
    chunks=30, chunk_delay_s=1.0, tokens_in=180, tokens_out=150,
))


def grosser_tisch() -> None:
    """Fünf Stimmen mit unterschiedlichem Verhalten — der Belastungsfall.

    Nur für die Gestaltungsprüfung: mehrere Karten laufen unterschiedlich
    schnell ein, eine scheitert mit Teiltext, eine antwortet ohne Text. Das
    sind künstliche Testdaten, keine Aussagen von Modellen.
    """
    from app import config

    config.FAKE_MODELS = [
        ModelConfig(id="fake-a", label="Alto", provider="fake", model="fake-fast"),
        ModelConfig(id="fake-b", label="Basso", provider="fake", model="fake-slow"),
        ModelConfig(id="fake-c", label="Cantus", provider="fake", model="fake-bruch"),
        ModelConfig(id="fake-d", label="Discant", provider="fake", model="fake-stumm"),
        ModelConfig(id="fake-e", label="Echo", provider="fake", model="fake-mittel"),
    ]
    FakeProvider.configure("fake-bruch", FakeScenario(
        delay_s=0.4, chunks=3, chunk_delay_s=0.5,
        error="Der Anbieter hat die Verbindung nach 2 s geschlossen.",
        partial_text="Eine tägliche Sicherung halte ich für zu selten, wenn",
    ))
    FakeProvider.configure("fake-stumm", FakeScenario(
        delay_s=2.2, text="", tokens_in=140, tokens_out=0,
    ))
    FakeProvider.configure("fake-mittel", FakeScenario(
        delay_s=0.6, text=KURZTEXT, chunks=6, chunk_delay_s=0.6,
        tokens_in=150, tokens_out=40,
    ))


if os.environ.get("XT_DEMO_TISCH", "").strip().lower() == "gross":
    grosser_tisch()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
