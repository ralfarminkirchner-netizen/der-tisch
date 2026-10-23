# Bauauftrag: Das Beziehungssystem zu Ende bauen

Du übernimmst eine laufende, geprüfte Anwendung und baust ihr Erkenntniswerkzeug
aus. Setze im vorhandenen Projekt um — liefere keinen Plan und keinen weiteren
Prompt.

**Basis:** Zweig `claude/xpertentisch-gestaltung`, Commit `5ad9eb5` (offener
Entwurfs-PR #65), Hauptzweig `ea75cd4`. Arbeite auf einem eigenen Zweig und
öffne einen Pull Request.

---

## 1. Was die Anwendung ist

XPERTENTiSCH ist ein Expertentisch für mehrere Sprachmodelle. Der Mensch wirft
einen **Funken** (eine Frage) ein; mehrere **Stimmen** — OpenAI, Anthropic,
Google, DeepSeek, Mistral, xAI und beliebige eigene Endpunkte — antworten
**unabhängig voneinander**. Die Antworten stehen nebeneinander, nicht
untereinander verrechnet.

Die Haltung ist nicht verhandelbar. Sie ist der Grund, warum es die App gibt:

- **Es gibt keine Rangfolge.** Kein Modell gewinnt. Nichts wird bewertet.
- **Unterschied ist nicht Widerspruch.** Ein Widerspruch wird nur ausgewiesen,
  wenn zwei Aussagen dasselbe Thema betreffen **und** entgegengesetzte
  Polarität haben.
- **Der Originaltext bleibt unangetastet.** Marker sind eine Auflage darüber,
  keine Änderung daran.
- **Die Anwendung erfindet nichts.** Keine geschätzten Kosten, keine
  Fortschrittsprozente, keine ausgedachten Szenarien, keine
  Wahrscheinlichkeiten. Wo etwas unbekannt ist, steht „unbekannt".
- **Aus Konsens wird keine Wahrheit.** Mehrere Stimmen, die dasselbe sagen,
  sind mehrere Stimmen — kein Beweis.

Oberfläche, Code, Kommentare und Bezeichner sind deutsch und bleiben es:
`renderSparkBlock`, `ruesteKarteAus`, `baueZweige`, `themenAusMarkern`,
`--flaeche`, `zustand-streaming`.

## 2. Technischer Stand

**Frontend:** Vite + TypeScript, kein Framework, kein JSX, kein
State-Management. DOM über einen kleinen `el()`-Helfer, Diagramme als
handgeschriebenes SVG.
**Backend:** FastAPI + SQLite (WAL), liefert `/api` und den gebauten
Frontend-Stand aus einem Dienst.
**Ereignisse:** Server-Sent Events mit lückenloser Nachlieferung nach Abbruch.

| Datei | Inhalt |
| --- | --- |
| `frontend/src/linsen.ts` | `themenAusMarkern()`, `renderThemen()`, `baueZweige()`, `renderVerlauf()` |
| `frontend/src/graph.ts` | Stimmen-Linse; exportiert `SVG_NS`, `bedienbar()`, `initialen()`, `shorten()` |
| `frontend/src/main.ts` | Gerüst, `linsenPanel()`, `kartenAktionen()`, Ereignisse, Themenschalter |
| `frontend/src/render.ts` | Karten, Vergleichstabelle, Legende, Signet |
| `frontend/src/markers.ts` | Marker-Auflage über dem Antworttext (maskiert) |
| `frontend/src/styles.css` | Gestaltungssystem „DER SAAL"; die Position steht im Kopf der Datei |
| `backend/app/analysis.py` | regelbasierte Auswertung, `STOPWORDS`, Marker mit `topics` |
| `backend/app/main.py` | `SPARK_KINDS`, Endpunkte |
| `backend/app/runner.py` | `BEZUG_TYP`, Warteschlangen je Anbieter, Streaming, Abbruch |
| `backend/app/reports.py` | HTML- und Markdown-Export |

## 3. Was schon steht — nicht neu bauen

**Funktion:** unabhängige Aufträge mit Warteschlange je Anbieter, Streamen,
Abbrechen einzelner Aufträge, Antworten/Weitergabe/Gegenposition/Vertiefung,
unveränderliche Kontext-Schnappschüsse je Auftrag, begrenztes Wechselgespräch
zwischen Modellen, Kuratierung als eigener Beitrag, Sitzungswechsel,
Offline-Entwürfe, Token und Kosten ohne Schätzung, Einstellungsseite mit frei
eintragbaren Anbietern, HTML- und Markdown-Bericht.

**Gestaltung („DER SAAL"):** ein ruhiger Raum, jede Antwort ein beleuchtetes
Pult. Licht bedeutet Aufmerksamkeit, nicht Qualität. Farbe spricht nur, wo die
Auswertung spricht — Grün Übereinstimmung, Orange Widerspruch, Violett
Einzelaussage; das Blau ist Bedienung, nie Befund. Herkunft trägt ein Signet
aus Anfangsbuchstaben, keine eigene Farbe. Zwei Schriften, zwei Ränge: der
Funke in der Serife (Newsreader), alles Maschinelle in der Grotesk (Inter),
beide im Repository unter `frontend/src/schrift/`. Drei Theme-Zustände mit
Schalter in der Kopfzeile.

**Linsen:** drei Blicke auf dieselben Daten, umschaltbar, die Wahl gilt für
alle Funkenblöcke und überdauert das Neuladen.

- **Stimmen** — wer trifft sich mit wem, wo wird widersprochen. Bögen statt
  paralleler Geraden, damit beide Kanten eines Paares getrennt treffbar bleiben.
- **Themen** — links die Stimmen, rechts die Begriffe, an denen ihre
  Fundstellen hängen. Ein Faden je Fundstelle in der Farbe seiner Bedeutung.
  Eine Stimme ohne Faden bleibt gestrichelt stehen — auch das ist ein Befund.
- **Verlauf** — die Sitzung als Baum. Jeder Zweig ist ein Beitrag, der aus der
  Antwort eines anderen hervorging (aus `spark.refs`). Ringe in den Bezügen
  werden abgefangen.

**Szenarien:** Beitragsart `szenario` und die Kartenaktion „Folgen
durchspielen" fragen die Stimmen nach Konsequenzen einer Aussage. Die Folgen
kommen von den Modellen.

**Datengrundlage:** `markers.topics` (Spalte mit Migration) führt je Fundstelle
die Begriffe mit, an denen sie hängt. Die Stoppwortliste wurde um rund sechzig
Funktions- und Füllwörter erweitert, weil „allein", „außerdem" oder „darf"
sonst als vermeintliche Themen oben landeten.

## 4. Was du bauen sollst

Ziel: aus dem Beziehungssystem ein Denkwerkzeug machen. Reihenfolge nach Wert.

### 4.1 Die Szenario-Linse — der Kern deines Auftrags

Heute erscheint ein Szenario als gewöhnlicher Beitrag im Verlauf. Es soll eine
eigene Linse bekommen, die aus den Szenario-Antworten eine lesbare
**Konsequenzkarte** macht:

- Nimm alle Beiträge der Art `szenario` und deren Antworten.
- Zerlege sie mit der vorhandenen Satzzerlegung aus `analysis.py` in einzeln
  genannte Folgen. Gruppiere Folgen, die mehrere Stimmen nennen, über dieselbe
  Ähnlichkeitsrechnung, die auch Übereinstimmungen findet. **Erfinde keine neue
  Heuristik; benutze die vorhandene.**
- Zeige: die Ausgangsaussage links, die genannten Folgen rechts, und je Folge,
  **welche** Stimmen sie nennen.
- Eine Folge, die mehrere Stimmen unabhängig nennen, wird gewichtiger
  dargestellt — **ausschließlich über die Zahl der Stimmen**. Das ist **keine
  Wahrscheinlichkeit** und darf nirgends so heißen. Beschriftung: „von 3 von 5
  Stimmen genannt".
- Widersprechen sich zwei Folgen, gilt dieselbe Regel wie überall: nur bei
  gleichem Thema und entgegengesetzter Polarität.
- Jede Folge ist anklickbar und öffnet die Antworten, in denen sie steht, mit
  markierter Fundstelle.

### 4.2 Die Herkunfts-Linse

Beziehungen tragen bereits `origin` (mensch|maschine) und `status`
(vorschlag|bestaetigt|abgelehnt). Baue daraus eine Linse, die zeigt, was der
Mensch **selbst festgestellt** hat und was bloß maschineller Vorschlag ist:
bestätigte Bezüge durchgezogen, Vorschläge gestrichelt, verworfene ausgegraut.
In einem Werkzeug, das keine Wahrheit behauptet, ist das die wichtigste Linse
überhaupt — sie beantwortet „worauf kann ich mich hier berufen?".

### 4.3 Die Zeitlinse

Aufträge tragen `created_at`, `started_at`, `finished_at`, `latency_ms`. Zeige
eine Runde als Zeitbild: wer wann anfing, wie lange schrieb, wer abbrach, wo
die Warteschlange bremste. Das macht die Warteschlangen je Anbieter erstmals
sichtbar und hilft beim Einstellen von `XT_PROVIDER_CONCURRENCY`.

### 4.4 Die Linsen aus dem Funkenblock lösen

Drei Linsen im Panel jedes Funkens werden mit fünf zu eng. Entwirf, wie das
skaliert: eine eigene Fläche, die den ganzen Verlauf und den gewählten Funken
zusammen zeigt. Der Verlauf ist ohnehin sitzungsweit und sitzt im Funkenblock
nur zu Gast.

### 4.5 Die Auswertung besser machen

Die Themen-Linse ist nur so gut wie die Begriffe darunter. Heute werden
Wortstämme nicht zusammengeführt — „Sicherung" und „Sicherungen" sind zwei
Begriffe — und die Stoppwortliste ist handgepflegt. Verbessere das, aber
**ausschließlich regelbasiert und deterministisch**, ohne zusätzliche
Modellaufrufe und ohne schwere neue Abhängigkeit. Ein einfacher deutscher
Stemmer im Projekt ist in Ordnung; ein Sprachmodell für die Auswertung ist es
nicht: die Auswertung muss nachvollziehbar und kostenlos bleiben.

### 4.6 Der Bericht kennt die Linsen nicht

Themen und Verlauf fehlen im Export. Der Bericht bleibt dabei ein
eigenständiges, offline nutzbares, druckbares HTML-Dokument ohne Skripte — ein
SVG darin ist erlaubt, ein Nachladen von außen nicht.

## 5. Unverhandelbar

Abnahmebedingungen. Ein Entwurf, der sie verletzt, ist abgelehnt, egal wie gut
er aussieht.

1. **Kein Framework-Umbau.** Kein React, Vue, Svelte, Tailwind, keine
   Diagrammbibliothek. Vanilla TypeScript und handgeschriebenes SVG bleiben,
   auch für neue Linsen. Die SVG-Bausteine sind exportiert; benutze sie.
2. **Die Anwendung sagt nichts voraus und schätzt nichts.** Keine
   Wahrscheinlichkeiten, Konfidenzen, Prognosen, Risikowerte oder Punktzahlen.
   Häufigkeit („von 3 von 5 Stimmen genannt") ist erlaubt und muss als
   Häufigkeit beschriftet sein.
3. **Keine Rangfolge der Modelle**, in keiner Linse — auch nicht implizit über
   Größe, Reihenfolge, Position oder Farbe.
4. **Farbe bleibt bedeutungstragend:** Grün Übereinstimmung, Orange
   Widerspruch, Violett Einzelaussage, Blau Bedienung. Identität und Herkunft
   werden über Signet, Strichart und Form unterschieden, nie über eine eigene
   Farbe.
5. **Keine Seitenüberbreite** bei 320, 390, 768 und 1440 px — auch nicht in
   geöffneten Panels. Wird geprüft.
6. **Zugänglichkeit ist Teil der Gestaltung:** jede Linse per Tastatur
   bedienbar, sichtbarer Fokus über `:focus-visible`, `aria-label` an jedem
   bedienbaren Element, Trefferflächen, die sich nicht gegenseitig verdecken,
   Kontrast mindestens 4,5:1 in beiden Themen, und zu jeder Bewegung eine
   Entsprechung unter `prefers-reduced-motion: reduce`.
7. **Der HTML-Bericht** bleibt offline benutzbar, druckbar, ohne Skripte und
   gegen Injection geschützt. Keine fremden CDNs zur Laufzeit, auch nicht für
   Schriften.
8. **Kein Textinhalt wird ungeprüft ins DOM geschrieben:** Text über
   `textContent`, die Marker-Auflage ausschließlich über `markers.ts`.
9. **Keine Nutzerkonten, kein Login**, keine automatische Veröffentlichung von
   Nutzerinhalten.
10. **Kein Test wird gelöscht, übersprungen oder stillgelegt**, um grün zu
    werden.

## 6. Start- und Prüfkommandos

```bash
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO/xpertentisch/backend"
test -x .venv/bin/python || python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
cd "$REPO/xpertentisch/frontend" && npm ci && npm run build && npm test
cd "$REPO/xpertentisch/backend" && XT_ENV=test .venv/bin/python -m pytest -m 'not live'
```

`npm run build` umfasst `tsc --noEmit`, `vite build` und
`scripts/check-build.mjs`, das prüft, dass keine Zugangsdaten im
Auslieferungsstand liegen. Der Build muss **vor** den Frontend-Tests laufen,
weil ein Test das erzeugte Bundle prüft. Ausgefilterte Live-Tests führst du als
**NICHT AUSGEFÜHRT** auf, nicht als bestanden.

Demo-Server mit steuerbaren Test-Providern. Test-Fakes brauchen
`XT_ALLOW_FAKE_PROVIDERS` und können im Produktionsmodus nicht starten; der
Starter setzt das selbst:

```bash
cd "$REPO/xpertentisch/backend"
XT_DEMO_TISCH=gross XT_DB_PATH=/tmp/demo.sqlite3 .venv/bin/python tools/demo_server.py 8077
.venv/bin/python tools/zustaende_seed.py /tmp/demo.sqlite3
```

`XT_DEMO_TISCH=gross` setzt fünf Stimmen an den Tisch: schnell, langsam, mit
Fehler und Teiltext, ohne Text, mittel. `zustaende_seed.py` legt eine Sitzung
mit allen acht Auftragszuständen an, auch `interrupted`, das sonst nur ein
Serverneustart mitten im Auftrag erzeugt. Beide erzeugen **künstliche
Testdaten** und sagen das in den Daten selbst.

Playwright steht bewusst nicht in der Paketdatei: `npm i --no-save playwright`,
Browserpfad über `PW_CHROMIUM`. Halte Paket- und Browserversion fest und nimm
keine unbemerkten Abhängigkeitsänderungen in den Commit.

Vier Browserprüfungen, jede gegen einen **frisch gestarteten** Server mit
eigener temporärer Datenbank — sonst stimmen Annahmen wie „noch keine Preise
eingetragen" nicht mehr:

```bash
cd "$REPO/xpertentisch/frontend"
node e2e/browser-check.mjs   http://127.0.0.1:8077   # Layout, Graphklick, Kantenklick, Überbreite
node e2e/gespraech-check.mjs http://127.0.0.1:8077   # Streamen, Abbrechen, Wechselgespräch, Entwürfe
node e2e/bericht-check.mjs   http://127.0.0.1:8077   # der Bericht über file://, ohne Server und ohne Netz
node e2e/zugang-check.mjs    http://127.0.0.1:8077   # Kontrast gemessen, Tastatur, Fokus, Themenvorrang
```

Ansichtsnachweis:
`node e2e/ansichten.mjs http://127.0.0.1:8077 ./ansichten`

Protokolliere den unveränderten Ausgangsstand, **bevor** du etwas änderst.
Schlägt eine Ausgangsprüfung fehl, trenne vorhandenen Fehler, Testdatenproblem
und neue Regression, statt das Ergebnis nachträglich zu glätten.

## 7. Der Stand, den du vorfindest

Frontend **80 Tests**, Backend **83 Tests** (2 abgewählt: echte APIs, nicht
ausgeführt), Build 0. Browser: **14/14** Layout, **21/21** Ablauf, **11/11**
Bericht, **62/62** Zugang. Alle Fließtexte, Nebenangaben, Tabellenzellen, alle
acht Zustandsschilder und die Linsenbeschriftungen erreichen in beiden Themen
mindestens 4,5:1.

**Nicht geprüft und darum nicht behauptet:** echte Anbieter-APIs (keine
Zugangsdaten), `docker build` (kein Daemon), andere Browser als Chromium.

**Gestalterisch ungelöst:** bei fünf unterschiedlich hohen Karten entstehen
Rasterlücken (Masonry wäre die Antwort, ist aber nicht verlässlich verfügbar);
der Bericht bindet die Schriften bewusst nicht ein.

## 8. Wie hier geprüft wird

Übernimm diese Haltung. Sie ist der Grund, warum man dem Stand trauen kann.

- **Eine Zusicherung, die auch über leeren Daten besteht, ist keine
  Zusicherung.** Vier solche Fälle wurden gefunden und behoben: ein
  Kantenklick-Test, der mit „keine Kante vorhanden" auf `true` lief; eine
  Überlagerungsprüfung, die stillschweigend entfiel; ein `every(...)` über
  einem leeren Feld; und eine Kontrastmessung, die `color` statt `fill` las und
  damit bei SVG den vererbten Wert des Elternelements maß. Suche solche Stellen
  in dem, was du dazuschreibst.
- **Miss, statt zu behaupten.** `zugang-check` bestimmt Kontraste über ein
  Canvas-Pixel gegen den tatsächlich zusammengerechneten Untergrund und prüft
  den Tastaturweg mit der echten Tabulatortaste: programmatisches `focus()`
  löst `:focus-visible` nicht aus und prüfte etwas anderes, als ein Mensch
  erlebt. **Jede neue Linse gehört in diese Prüfung aufgenommen.**
- **Prüfe jede Linse mit leeren Daten, mit einer Stimme, mit fünf und mit
  widersprüchlichen Daten.** Eine Linse, die nur mit den Demo-Daten schön
  aussieht, ist nicht fertig.
- Für jede neue Linse: **Was zeigt sie, das die anderen nicht zeigen?** Wenn du
  das nicht in einem Satz sagen kannst, baue sie nicht.

## 9. Abnahme

- Pull Request mit der **Begründung deiner Entscheidungen**: warum diese
  Linsen, warum diese Kodierung, was sie sichtbar machen, das vorher unsichtbar
  war.
- **Ansichtsbelege je Linse**, hell und dunkel, bei 390 px und 1440 px, aus
  tatsächlichen Anwendungszuständen. Bilder in der PR-Beschreibung als
  `<img src="..." width="900">` einbetten, **nicht** als Markdown-Bildsyntax:
  das GitHub-Werkzeug verstümmelt sonst einzelne URLs mit Backticks. Danach
  nachlesen, ob der Text unverfälscht ankam.
- Die **tatsächlichen Ergebnisse** aller Läufe mit Versionen und Exit-Codes.
- **Jede geänderte Zusicherung einzeln begründen:** welche Formannahme wurde
  ersetzt, und wie wird dasselbe Schutzziel weiterhin geprüft?
- Trenne ausdrücklich „implementiert und geprüft", „implementiert, aber nicht
  geprüft" und „nicht implementiert, blockiert oder ungelöst".

**Zur Ehrlichkeit über die eigene Arbeit:** Behaupte nicht, alle Tests seien
bestanden, wenn du sie nicht ausgeführt hast. Konntest du etwas nicht prüfen —
kein Browser, keine Zugangsdaten, kein Docker —, schreib genau das hin, statt
es zu überspringen und zu verschweigen. Dasselbe gilt für gestalterische
Kompromisse: sag, was du nicht gelöst bekommen hast. **In diesem Projekt ist
eine ehrlich ausgewiesene Lücke mehr wert als eine geglättete Bilanz.**
