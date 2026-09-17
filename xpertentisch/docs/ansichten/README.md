# Ansichtsnachweis

Elf Ansichten, jede hell und dunkel, jede bei 390 px und 1440 px — **44
Belege**. Alle zeigen tatsächliche Anwendungszustände; nichts ist nachträglich
zusammengesetzt oder retuschiert.

| Ansicht | Was darauf zu sehen ist |
| --- | --- |
| `leerer-tisch` | vor dem ersten Funken |
| `einlaufen` | mehrere Antworten entstehen gleichzeitig, eine ist gescheitert |
| `auswertung` | fertiger Funke mit Vergleich und gefundenen Bezügen |
| `linse-stimmen` | wer sich mit wem trifft und wo widersprochen wird |
| `linse-themen` | woran die Fundstellen hängen |
| `linse-szenario` | die Konsequenzkarte: links die Ausgangsaussage, rechts die genannten Folgen |
| `linse-herkunft` | was bestätigt ist, was Vorschlag, was verworfen |
| `linse-zeit` | die Runde als Zeitbild: Warten und Schreiben getrennt |
| `einstellungen` | Anbieter, Preise, Zeitgrenze, Kurator |
| `bericht` | der exportierte HTML-Bericht, über `file://` geöffnet |
| `zustaende` | alle acht Auftragszustände nebeneinander |

Der **Verlauf** hat keinen eigenen Beleg: er steht auf jedem `linse-*`-Bild
links, weil er sitzungsweit gilt und die Linsenfläche ihn dauerhaft zeigt.

Die Linsenbilder entstehen aus zwei Runden — einem Funken und einem daraus
durchgespielten Szenario. Ohne die zweite Runde hätte die Folgen-Linse nichts
zu zeigen, und der Beleg zeigte eine leere Fläche.

## Womit sie entstanden sind

- Chromium über Playwright 1.63.0, `deviceScaleFactor: 2`, danach für dieses
  Verzeichnis auf 50 % verkleinert und als JPEG gespeichert.
- Testdaten aus den Werkzeugen des Projekts — **keine echten Modellaufrufe**:
  - `backend/tools/demo_server.py` mit `XT_DEMO_TISCH=gross` (fünf Stimmen:
    schnell, langsam, mit Fehler, ohne Text, mittel)
  - `backend/tools/zustaende_seed.py` für `zustaende` (die acht Zustände; in
    den Karten steht ausdrücklich, dass es Testdaten sind)
- Jede Theme-/Breiten-Kombination lief in einem eigenen Browserkontext und
  damit in einer eigenen, frischen Sitzung.

## Neu erzeugen

```bash
cd xpertentisch/backend
XT_DEMO_TISCH=gross XT_DB_PATH=/tmp/demo.sqlite3 .venv/bin/python tools/demo_server.py 8077
.venv/bin/python tools/zustaende_seed.py /tmp/demo.sqlite3

cd ../frontend
PW_CHROMIUM=<pfad/zu/chromium> node e2e/ansichten.mjs http://127.0.0.1:8077 ./ansichten
```

Die Belege in diesem Verzeichnis sind Dokumentation, kein Prüfmittel: ob die
Gestaltung trägt, prüfen die vier Skripte unter `frontend/e2e/`.
