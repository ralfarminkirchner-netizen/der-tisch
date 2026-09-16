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
- **Anbieter sind Daten, nicht Quelltext.** Mitgeliefert sind OpenAI, Anthropic,
  Google Gemini, DeepSeek, Mistral und xAI Grok; jeder davon lässt sich ändern
  oder abschalten. Über „Eigenen Anbieter eintragen“ kommt alles dazu, was die
  OpenAI-Schnittstelle spricht — OpenRouter, Together, Fireworks, Groq oder ein
  Server im eigenen Netz.
- **Einstellungen in der Oberfläche**: Zugangsdaten, Modellnamen, Basis-Adressen
  und Zeitgrenze lassen sich über das Zahnrad eintragen, ohne Umgebungsvariablen
  anzufassen.
- **Fortlaufendes Gespräch statt Einzelabfragen.** Jeder Auftrag bekommt einen
  Gesprächsauszug: ausdrücklich gewählte Bezugsbeiträge plus die jüngsten
  Beiträge der Sitzung. Fremde Modellbeiträge stehen darin als **Zitat** —
  ausdrücklich als „keine Anweisung und keine Aussage des Menschen“ markiert.
- **Direkte Bezugnahmen.** An jeder Modellkarte: *Antworten*, *An ⟨Modell⟩
  geben*, *Gegenposition*, *Strang vertiefen*. Damit läuft Mensch↔Modell und
  Modell↔Modell, ohne dass ein Modell als Absender ausgegeben wird.
- **Kontext-Schnappschuss je Auftrag.** Was ein Modell zu sehen bekam, wird bei
  Auftragsbeginn festgeschrieben. „Worauf antwortet diese Stimme?“ zeigt es:
  Liste der Beiträge, Kürzungshinweis und den übergebenen Wortlaut. Ein später
  eingeworfener Gedanke wird **nicht** rückwirkend zum Kenntnisstand erklärt.
- **Sitzungsabschluss** mit freiwilliger Abschlussnotiz.
- **Zwei Berichtsexporte**: eigenständiges HTML (offline, druckbar, ohne
  Skripte, alle Inhalte maskiert) und Markdown.

### Anbieter und Zugangsdaten

Ein Anbieter kommt an den Tisch, wenn er **eingeschaltet** ist und einen
**Schlüssel** hat. Beides steht auf der Einstellungsseite (Zahnrad oben rechts).

Drei Arten von Schnittstellen decken das Feld ab:

| Art | Wer spricht sie |
| --- | --- |
| `openai` | OpenAI selbst und alle kompatiblen: DeepSeek, Mistral, xAI, Groq, OpenRouter, Together, Fireworks, vLLM, Ollama … |
| `anthropic` | Anthropic |
| `google` | Google Gemini |

Bei einem OpenAI-kompatiblen Anbieter ist die **Basis-Adresse** der einzige
Unterschied. Deshalb braucht es für einen neuen Anbieter keinen Quelltext:
Anzeigename, Art, Basis-Adresse, Modellname und Schlüssel genügen.

Schlüssel lassen sich auf zwei Wegen hinterlegen, und beide dürfen sich mischen:

1. **Umgebungsvariablen** — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
   `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `XAI_API_KEY`. Ein
   mitgelieferter Anbieter, dessen Variable beim ersten Start gesetzt ist,
   startet gleich eingeschaltet.
2. **Einstellungsseite** — ohne Neustart, auch für selbst eingetragene Anbieter.

Was auf der Einstellungsseite gesetzt wird, hat **Vorrang** vor der Umgebung.
Ein geleertes Feld löscht den Wert wieder, und die Umgebungsvariable greift
erneut. Zu jedem Anbieter steht, woher sein Schlüssel stammt.

Mitgelieferte Anbieter lassen sich abschalten, aber nicht löschen — sie wären
beim nächsten Start ohnehin wieder da. Selbst eingetragene lassen sich löschen.

**Die Einstellungsseite ist gesperrt, solange `XT_ADMIN_TOKEN` nicht gesetzt
ist.** Das ist Absicht: ohne Login könnte sonst jede Person mit dem Link die
Zugangsdaten ändern. Setze die Variable auf ein langes, selbst gewähltes Wort und
starte den Dienst neu; die Oberfläche fragt dieses Wort dann ab. Schlüssel werden
nie an die Oberfläche zurückgegeben — sichtbar sind nur Herkunft und die letzten
vier Zeichen.

**Zu den Modellnamen:** die mitgelieferten Namen sind Vorschläge zum Zeitpunkt
der Entwicklung, keine geprüfte Wahrheit. Modellbezeichnungen ändern sich
schnell. Jede Anbieterzeile verlinkt deshalb die Modellliste des Anbieters, und
der Name ist frei änderbar. Der Knopf „Prüfen“ macht einen einzigen kurzen
echten Aufruf und sagt, ob Schlüssel und Modellname zusammenpassen.

### Zustände eines Auftrags

„Nicht gefragt“, „wartet“, „unterbrochen“ und „geantwortet, aber ohne Text“
sind verschiedene Sachverhalte und werden auch verschieden dargestellt — in der
Oberfläche wie in allen Exporten:

| Zustand | Bedeutung |
| --- | --- |
| `not_requested` | Für diesen Funken bewusst nicht angefragt. Kein Aufruf, keine Kosten. |
| `queued` / `running` | Wartet auf den Start bzw. läuft gerade. |
| `done` | Antwort da. Ohne Text: „Antwort kam an, enthielt aber keinen Text.“ |
| `error` | Der Anbieter hat abgelehnt oder war nicht erreichbar; der Grund steht dabei. |
| `interrupted` | Durch einen Serverneustart abgebrochen. Wird **nicht** blind neu gestartet. |
| `cancelled` | Vor der Antwort abgebrochen. |

### Bezüge: gesetzt oder nur vorgeschlagen

Beziehungen zwischen Beiträgen tragen **Herkunft** und **Stand**:

- Was du selbst setzt (Antworten, Weitergeben, Gegenposition, Vertiefen), gilt
  sofort als bestätigt.
- Was die Auswertung findet, ist ein **Vorschlag** — und bleibt es, bis du ihn
  bestätigst oder verwirfst. In Bericht und JSON-Export steht dann
  „maschineller Vorschlag, unbestätigt“, niemals als deine Feststellung.

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
| `GOOGLE_API_KEY` | Zugangsdaten Google Gemini | — |
| `DEEPSEEK_API_KEY` | Zugangsdaten DeepSeek | — |
| `MISTRAL_API_KEY` | Zugangsdaten Mistral | — |
| `XAI_API_KEY` | Zugangsdaten xAI Grok | — |
| `XT_DB_PATH` | Pfad der SQLite-Datei | `xpertentisch.sqlite3` |
| `XT_REQUEST_TIMEOUT_S` | Zeitgrenze je Modellaufruf | `120` |
| `XT_MAX_PROMPT_CHARS` | Längengrenze eines Funkens | `20000` |
| `XT_ADMIN_TOKEN` | Zugangswort für die Einstellungsseite; ohne bleibt sie gesperrt | leer |
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
- **Variablen**: `XT_ENV=production`, `XT_DB_PATH=/data/xpertentisch.sqlite3` und
  `XT_ADMIN_TOKEN` (ein langes, selbst gewähltes Wort). Die API-Schlüssel kannst
  du hier als `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
  `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `XAI_API_KEY` setzen **oder** nach dem
  ersten Start über das Zahnrad in der Oberfläche eintragen.
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
| GET | `/api/sessions/{id}/report.json` | vollständiger Sitzungsstand samt Bezügen und Kontext-Schnappschüssen |
| GET | `/api/jobs/{id}/context` | worauf dieser Auftrag geantwortet hat |
| POST | `/api/sessions/{id}/relations/{rid}` | maschinellen Bezug bestätigen oder verwerfen |
| GET | `/api/admin/providers` | alle Anbieter samt Zustand (Kopfzeile `X-Admin-Token`) |
| POST | `/api/admin/providers` | eigenen Anbieter anlegen |
| POST | `/api/admin/providers/{id}` | Schlüssel, Modell, Adresse, Ein/Aus ändern |
| DELETE | `/api/admin/providers/{id}` | selbst eingetragenen Anbieter löschen |
| POST | `/api/admin/providers/{id}/test` | einen Anbieter mit einem echten Kurzaufruf prüfen |
| GET/POST | `/api/admin/settings` | Zeitgrenze lesen und setzen |

Ein wiederholter POST auf `/sparks` mit derselben `client_request_id` liefert
`200` und `duplicate: true` — es entstehen keine neuen Modellaufrufe.

---

## Gegenüberstellung mit dem Bauauftrag „Mobiler TiSCH“

Ein zweiter Bauauftrag beschrieb dieselbe App mit anderem Zuschnitt. Was daraus
übernommen wurde und was nicht:

**Übernommen** — Gesprächskontext je Auftrag; ausdrücklich gewählte Bezüge, die
nie stillschweigend verschwinden; unveränderlicher Kontext-Schnappschuss mit
Ansicht „Worauf antwortet diese Stimme?“; Antwort- und Weitergabe-Aktionen an
jeder Karte; feinere Auftragszustände (`not_requested`, `cancelled`) mit
unterschiedlicher Darstellung; Beziehungen mit Herkunft und Stand samt
menschlicher Bestätigung; JSON-Export der ganzen Sitzung; Berichte, die einen
unvollständigen Stand ausdrücklich als vorläufig kennzeichnen; Zeitangaben in
Europa/Berlin; Untertitel „Mobiler TiSCH“.

**Bereits vorhanden** — Funken unverändert speichern, Idempotenz gegen doppeltes
Senden, SSE mit lückenloser Nachlieferung, unterbrochene Aufträge ehrlich
kennzeichnen, getrennte Provider-Adapter, sichtbar gekennzeichneter Testmodus,
Schlüssel ausschließlich serverseitig, SQLite auf einem Volume, ein Dienst für
API und Oberfläche, Healthcheck, mobile Bedienbarkeit, dokumentierte
Sicherheitsgrenze ohne Login.

**Bewusst nicht übernommen** — der Wechsel auf React, Express und Node. Hier
läuft ein geprüftes Python-Backend mit Vite-Oberfläche; ein Umbau brächte keine
Funktion, nur Risiko. Der Bauauftrag verlangt selbst, vorhandene Arbeit zu
bewahren.

**Noch offen** — Streaming der Antworten (heute erscheint eine Antwort
vollständig), eigene Warteschlange je Anbieter mit Parallelitätsgrenzen,
begrenztes Ping-Pong zwischen Modellen, Token- und Kostenerfassung, eine
vorgeschaltete Kuratierung, Sitzungsliste in der Oberfläche, Offline-Entwürfe
und das Abbrechen einzelner Aufträge.

## Ausdrücklich nicht enthalten

Obsidian-/Vault-Zugriff, Live-Sync oder Pflichtintegration mit anderen
Systemen, Nutzerkonten oder Login, automatische Veröffentlichung von
Nutzerinhalten, verbindliche Lehren aus Modellkonsens, Rankings der Modelle.

## Gestaltung

Die Oberfläche folgt der Idee einer **Werkbank**: eine ruhige, warme Fläche,
darauf körperhafte Karten mit einer farbigen Kante, die den Zustand trägt. Farbe
ist reserviert für Bedeutung — Grün für Übereinstimmung, Orange für Widerspruch,
Violett für Einzelaussagen; das Tiefblau der Marke mischt sich da nicht ein.
Beide Themen (hell und dunkel) sind ausgearbeitet und folgen der Einstellung des
Geräts. Gesetzt wird in der Systemschrift: das spart den Ladeweg zu einem
fremden Schriftdienst und damit auch die Datenspur dorthin.

## Bekannte Einschränkungen

- **Kein Zugriffsschutz für Sitzungen.** Wer die Sitzungskennung kennt, sieht die
  Sitzung. Ohne Login ist das so gewollt — die Anwendung gehört deshalb nicht
  ungeschützt ins offene Netz, wenn die Inhalte vertraulich sind. Geschützt ist
  nur die Einstellungsseite, und zwar durch ein einziges gemeinsames Zugangswort,
  nicht durch Benutzerkonten.
- **Die mitgelieferten Modellnamen können veraltet sein.** Sie waren zum
  Zeitpunkt der Entwicklung plausibel, mehr nicht. Bei einem Fehler „Modell nicht
  gefunden“ hilft die verlinkte Modellliste des Anbieters.
- **Über die Einstellungsseite hinterlegte Schlüssel liegen unverschlüsselt in
  der SQLite-Datei.** Eine Verschlüsselung mit einem Schlüssel, der daneben
  liegt, wäre Augenwischerei; stattdessen gilt: Das Volume ist so schützenswert
  wie die Zugangsdaten selbst. Wer das nicht will, setzt die Schlüssel weiterhin
  als Umgebungsvariablen.
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
  als `unterbrochen` gekennzeichnet; die Frage muss neu gestellt werden. Das ist
  Absicht: ein blinder Neustart kostenpflichtiger Aufrufe wäre schlimmer.
- **Der Gesprächsauszug ist begrenzt.** Ausdrücklich gewählte Bezüge sind immer
  vollständig enthalten; vom übrigen Verlauf kommen die jüngsten Beiträge mit,
  bis das Zeichenbudget erschöpft ist. Eine Kürzung wird im übergebenen Text
  und in der Kontextansicht angezeigt, nie stillschweigend vorgenommen.
- **Berichts-HTML ohne Skript.** Marker sind sichtbar, aber nicht filterbar.
