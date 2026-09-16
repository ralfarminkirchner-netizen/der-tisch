# XPERTENTiSCH

Mehrere Sprachmodelle beantworten denselben **Funken** (die Frage) unabhängig
voneinander. Die Antworten stehen gleichberechtigt nebeneinander, werden
regelbasiert verglichen und lassen sich als Bericht ausgeben.

Es gibt **keine Rangfolge der Modelle, keinen Konsenszwang und keine
verbindliche Lehre**. Die Einordnung bleibt beim Menschen.

---

## Aufbau

```
xpertentisch/
├─ backend/                FastAPI + SQLite
│  ├─ app/
│  │  ├─ config.py         Konfiguration; trennt Produktion und Test-Fakes
│  │  ├─ db.py             SQLite-Schema und Zugriffe (Sitzungen, Funken, Aufträge, Marker, Ereignisse)
│  │  ├─ events.py         Ereignisbus mit Nachlieferung (Last-Event-ID)
│  │  ├─ runner.py         Ein Funke → je Modell ein unabhängiger Auftrag
│  │  ├─ analysis.py       Automatische Einschätzungen (ohne LLM-Aufrufe)
│  │  ├─ reports.py        Berichtsexporte (HTML offline, Markdown)
│  │  ├─ providers/        Adapter: openai, anthropic, fake (nur Test)
│  │  └─ main.py           HTTP-Schnittstelle inkl. SSE und Healthcheck
│  └─ tests/               pytest
├─ frontend/               Vite + TypeScript, ohne Framework
│  ├─ src/                 Glasoberfläche, Modellkarten, Tabelle, Beziehungsnetz
│  ├─ tests/               vitest
│  ├─ e2e/browser-check.mjs Browserprüfung (Playwright)
│  └─ scripts/check-build.mjs Prüft das Bundle auf Zugangsdaten
├─ Dockerfile              Zweistufiges Produktionsabbild
└─ railway.json            Railway-Deploy (Dockerfile + Healthcheck)
```

## Funktionsumfang

- **Funken-Eingabe** mit Modellauswahl; jede Übertragung trägt eine
  Anfragekennung. Eine doppelt gesendete Anfrage erzeugt keine zweiten
  Modellaufrufe (Eindeutigkeit in der Datenbank).
- **Unabhängige Modellkarten**: Zustand, Dauer, Teilantwort und Fehler je
  Modell. Der Ausfall eines Providers hält die anderen nicht auf.
- **Ereignisstrom (SSE)** mit lückenloser Nachlieferung nach Abbruch; Neuladen
  stellt den Stand aus der Datenbank her, ohne neue Aufträge zu starten.
- **Serverneustart**: Sitzungsdaten bleiben erhalten, unterbrochene Aufträge
  werden als `unterbrochen` gekennzeichnet und im Ereignisstrom vermerkt.
- **Automatische Einschätzungen** (regelbasiert, deterministisch, ohne weitere
  Modellaufrufe): Übereinstimmung, Widerspruch, Einzigartig. Marker verweisen
  über Zeichenpositionen auf den **unveränderten** Antworttext und führen ein
  wörtliches Textbelegzitat mit.
- **Vergleichstabelle** und **Beziehungsnetz**; ein Klick auf Knoten oder Kante
  öffnet genau die zugehörigen Antworten.
- **Sitzungsabschluss** mit freiwilliger Abschlussnotiz.
- **Zwei Berichtsexporte**: eigenständiges HTML (offline, druckbar, ohne
  Skripte, alle Inhalte maskiert) und Markdown.

### Widerspruch vs. Unterschiedlichkeit

Ein Widerspruch wird nur markiert, wenn zwei Sätze **dasselbe Thema** betreffen
(Begriffsüberschneidung über Schwellwert) **und** entgegengesetzte Polarität
haben — Verneinung oder ein bekanntes Gegensatzpaar. Bloß unterschiedliche
Aussagen gelten als *einzigartig*, nicht als widersprüchlich.

---

## Start und Build

### Entwicklung

```bash
# Backend (mit Test-Modellen, kein Netzwerk, keine Kosten)
cd xpertentisch/backend
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
XT_ENV=development XT_ALLOW_FAKE_PROVIDERS=1 XT_USE_FAKE_MODELS=1 \
  .venv/bin/uvicorn app.main:app --reload --port 8000

# Oberfläche (zweites Terminal)
cd xpertentisch/frontend
npm install
npm run dev            # http://localhost:5173, /api wird auf Port 8000 gespiegelt
```

Mit echten Modellen stattdessen `OPENAI_API_KEY` und/oder `ANTHROPIC_API_KEY`
setzen und die Fake-Schalter weglassen.

### Produktionsbuild

```bash
cd xpertentisch/frontend && npm ci && npm run build
#   führt aus: tsc --noEmit && vite build && node scripts/check-build.mjs

cd ../backend && pip install -r requirements.txt
XT_ENV=production uvicorn app.main:app --app-dir . --host 0.0.0.0 --port 8000
```

Der Server liefert `frontend/dist` unter `/` aus, sobald das Verzeichnis
existiert; die Schnittstelle liegt unter `/api`.

### Docker

```bash
docker build -t xpertentisch xpertentisch/
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=... -e ANTHROPIC_API_KEY=... \
  -v xpertentisch-data:/data \
  xpertentisch
```

### Tests

```bash
cd xpertentisch/backend && XT_ENV=test .venv/bin/python -m pytest        # Fakes, kein Netzwerk
cd xpertentisch/backend && .venv/bin/python -m pytest -m live            # echte APIs, nur mit Schlüsseln
cd xpertentisch/frontend && npm run build && npx vitest run              # Build zuerst: der Bundle-Test prüft dist/
cd xpertentisch/frontend && npm i -D playwright && PW_CHROMIUM=<pfad/zu/chromium> \
  node e2e/browser-check.mjs http://127.0.0.1:8000   # Browserprüfung; playwright ist bewusst keine feste Abhängigkeit
```

---

## Konfiguration

| Variable | Bedeutung | Vorgabe |
| --- | --- | --- |
| `XT_ENV` | `production`, `development` oder `test` | `production` |
| `OPENAI_API_KEY` | Zugangsdaten OpenAI | — |
| `ANTHROPIC_API_KEY` | Zugangsdaten Anthropic | — |
| `XT_OPENAI_MODEL` | Modellname OpenAI | `gpt-4.1` |
| `XT_ANTHROPIC_MODEL` | Modellname Anthropic | `claude-sonnet-4-5` |
| `XT_DB_PATH` | Pfad der SQLite-Datei | `xpertentisch.sqlite3` |
| `XT_REQUEST_TIMEOUT_S` | Zeitgrenze je Modellaufruf | `120` |
| `XT_MAX_PROMPT_CHARS` | Längengrenze eines Funkens | `20000` |
| `XT_CORS_ORIGINS` | Kommaliste erlaubter Ursprünge | leer |
| `PORT` | Port (von Railway gesetzt) | `8000` |
| `XT_ALLOW_FAKE_PROVIDERS` | schaltet den Test-Provider frei | aus |
| `XT_USE_FAKE_MODELS` | besetzt den Tisch mit Test-Modellen | aus |

**Test-Fakes können nicht unbemerkt einspringen.** Ist `XT_ENV=production` und
zugleich `XT_ALLOW_FAKE_PROVIDERS` gesetzt, bricht der Start mit einem Fehler
ab. Ohne die Freischaltung wird der Test-Provider nicht einmal gebaut. Läuft er
in der Entwicklung, weist die Oberfläche sichtbar darauf hin, und `/api/health`
meldet `fake_providers_enabled: true`.

## Railway

- **Root Directory**: `xpertentisch`
- **Builder**: Dockerfile (`railway.json` ist hinterlegt)
- **Healthcheck-Pfad**: `/api/health`
- **Variablen**: `XT_ENV=production`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `XT_DB_PATH=/data/xpertentisch.sqlite3`
- **Volume**: unter `/data` einhängen — ohne Volume sind die Sitzungsdaten nach
  jedem Neustart weg.
- `PORT` setzt Railway selbst; der Startbefehl übernimmt ihn.

`/api/health` meldet `degraded` (nicht `ok`), solange kein Provider einsatzbereit
ist, und nennt die fehlenden Zugangsdaten.

## Schnittstelle

| Methode | Pfad | Zweck |
| --- | --- | --- |
| GET | `/api/health` | Zustand von Datenbank und Providern |
| GET | `/api/config` | Modelle und Grenzen (ohne Zugangsdaten) |
| POST | `/api/sessions` | Sitzung anlegen |
| GET | `/api/sessions/{id}` | vollständiger Sitzungsstand |
| POST | `/api/sessions/{id}/sparks` | Funke setzen (`client_request_id` pflicht) |
| GET | `/api/sessions/{id}/events` | SSE-Strom (`Last-Event-ID` wird beachtet) |
| POST | `/api/sessions/{id}/close` | Sitzung abschließen |
| GET | `/api/sessions/{id}/report.html` | Bericht als eigenständiges HTML |
| GET | `/api/sessions/{id}/report.md` | Bericht als Markdown |

Ein wiederholter POST auf `/sparks` mit derselben `client_request_id` liefert
`200` und `duplicate: true` — es entstehen keine neuen Modellaufrufe.

---

## Ausdrücklich nicht enthalten

Obsidian-/Vault-Zugriff, Live-Sync oder Pflichtintegration mit anderen
Systemen, Nutzerkonten oder Login, automatische Veröffentlichung von
Nutzerinhalten, verbindliche Lehren aus Modellkonsens, Rankings der Modelle.

## Bekannte Einschränkungen

- **Kein Zugriffsschutz.** Wer die Sitzungskennung kennt, sieht die Sitzung.
  Ohne Login ist das so gewollt — die Anwendung gehört deshalb nicht ungeschützt
  ins offene Netz, wenn die Inhalte vertraulich sind.
- **Die Einschätzungen sind rein sprachstatistisch.** Sie zählen Begriffe und
  Verneinungen; sie verstehen den Inhalt nicht. Umschreibungen ohne gemeinsame
  Wörter bleiben unerkannt, ironische oder mehrgliedrige Widersprüche ebenso.
  Die Marker sind Lesehilfen, keine Bewertung.
- **Keine Streaming-Ausgabe der Modelle.** Eine Antwort erscheint vollständig,
  sobald sie fertig ist; Teilantworten entstehen nur bei Längenabbruch oder
  Fehler.
- **SQLite und Prozessspeicher.** Der Ereignisbus lebt im Prozess. Mehrere
  Instanzen hinter einem Lastverteiler teilen ihn nicht — XPERTENTiSCH läuft als
  eine Instanz.
- **Kein Wiederaufnehmen unterbrochener Aufträge.** Nach einem Neustart sind sie
  als `unterbrochen` gekennzeichnet; die Frage muss neu gestellt werden.
- **Berichts-HTML ohne Skript.** Marker sind sichtbar, aber nicht filterbar.
