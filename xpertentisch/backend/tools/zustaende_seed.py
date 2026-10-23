"""Legt eine Sitzung an, in der alle acht Auftragszustände nebeneinander stehen.

Nur für die Gestaltungsprüfung. Die Inhalte sind **künstliche Testdaten** und
keine Aussagen irgendeines Modells; jede Karte sagt das auch selbst. Manche
Zustände lassen sich im normalen Lauf nicht auf Kommando herstellen —
`interrupted` entsteht etwa nur durch einen Serverneustart mitten im Auftrag.
Darum werden die Zeilen hier direkt geschrieben.

    python tools/zustaende_seed.py /pfad/zur/demo.sqlite3

Die Datei muss eine Demo-Datenbank sein. Gegen eine Produktionsdatenbank ist
dieses Werkzeug nicht zu benutzen.
"""

from __future__ import annotations

import sqlite3
import sys
import time
import uuid

HINWEIS = "Künstliche Testdaten zur Gestaltungsprüfung — keine Modellaussage."

# (Modell, Anzeigename, Zustand, Text, Fehler, Teilantwort, Dauer, Token ein/aus)
ZUSTAENDE = [
    ("z-a", "Alto", "not_requested", "", None, 0, None, None, None),
    ("z-b", "Basso", "queued", "", None, 0, None, None, None),
    ("z-c", "Cantus", "running", "", None, 0, None, None, None),
    ("z-d", "Discant", "streaming",
     f"{HINWEIS} Dieser Text läuft gerade ein und ist noch nicht zu Ende",
     None, 1, None, 150, 38),
    ("z-e", "Echo", "done",
     f"{HINWEIS} Eine vollständige Antwort mit zwei Sätzen. Der zweite Satz "
     "steht hier, damit die Karte ihre normale Höhe zeigt.",
     None, 0, 2310, 180, 140),
    ("z-f", "Faux", "done", "", None, 0, 1980, 140, 0),
    ("z-g", "Grave", "error", "",
     "Der Anbieter hat die Verbindung nach 2 s geschlossen.", 0, 2040, None, None),
    ("z-h", "Hymnus", "interrupted",
     f"{HINWEIS} Dieser Text war zur Hälfte da, als der Dienst neu startete",
     "Durch einen Neustart des Dienstes unterbrochen.", 1, None, 160, 52),
    ("z-i", "Introit", "cancelled",
     f"{HINWEIS} Bis hierhin war die Antwort gekommen, dann kam der Abbruch",
     "Von dir abgebrochen.", 1, 1450, 150, 44),
]


def main(pfad: str) -> None:
    jetzt = time.time()
    db = sqlite3.connect(pfad)
    db.execute("PRAGMA foreign_keys = ON")

    sitzung = f"ses_{uuid.uuid4().hex[:16]}"
    db.execute(
        "INSERT INTO sessions (id, title, status, created_at) VALUES (?,?,?,?)",
        (sitzung, "Zustandsschau (Testdaten)", "offen", jetzt),
    )

    funke = f"fun_{uuid.uuid4().hex[:16]}"
    db.execute(
        "INSERT INTO sparks (id, session_id, seq, prompt, client_request_id, created_at,"
        " kind, refs) VALUES (?,?,?,?,?,?,?,?)",
        (funke, sitzung, 1,
         "Wie sehen die acht Zustände eines Auftrags nebeneinander aus?",
         f"seed-{uuid.uuid4().hex[:8]}", jetzt, "funke", "[]"),
    )

    for modell, name, zustand, text, fehler, teil, dauer, ein, aus in ZUSTAENDE:
        db.execute(
            "INSERT INTO jobs (id, session_id, spark_id, model_id, label, provider, model,"
            " status, text, error, partial, created_at, started_at, finished_at, latency_ms,"
            " tokens_in, tokens_out, cost_micro, cost_source)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"auf_{uuid.uuid4().hex[:16]}", sitzung, funke, modell, name,
                "fake", f"fake-{modell}", zustand, text, fehler, teil, jetzt,
                jetzt if zustand != "queued" else None,
                jetzt if dauer else None, dauer, ein, aus, None, "unbekannt",
            ),
        )

    db.commit()
    db.close()
    print(f"Zustandsschau angelegt: Sitzung {sitzung} in {pfad}")
    print("Alle Inhalte sind künstliche Testdaten.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
