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
from app.providers.fake import FakeProvider, FakeScenario  # noqa: E402

LANGTEXT = (
    "Eine tägliche Sicherung ist sinnvoll, sobald der Verlust eines Tages "
    "spürbar wäre. Entscheidend ist nicht der Rhythmus allein, sondern die "
    "Frage, wie viel Arbeit zwischen zwei Sicherungen entstehen darf. "
    "Wichtig ist außerdem, die Wiederherstellung regelmäßig zu proben: eine "
    "Sicherung, die nie zurückgespielt wurde, ist ein Versprechen ohne Beleg."
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

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
