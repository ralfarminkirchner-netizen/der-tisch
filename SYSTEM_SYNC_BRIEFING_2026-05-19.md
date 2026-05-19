# MOONFiNGERS — System-Sync-Briefing (alle Workspaces)

Stand: 2026-05-19 · Status: System „in the making", mehrere Instanzen parallel.

> **Zweck.** Dieses Dokument bringt jede Instanz in jedem Workspace auf
> denselben Wissensstand. Wer in irgendeinem der unten gelisteten Repos zu
> arbeiten beginnt, liest **zuerst dieses Briefing**, dann den kanonischen
> Vertrag `HANDOFF_dual_shared_cores_2026-05-18.md`, dann das eigene
> Workspace-`SURVEY`/`REPORT`. Ziel ist **ein zusammenhängendes, großes,
> modulares System** — viele unabhängige Module, föderiert über geteilte
> Verträge, nicht ein Monolith und nicht ein Wildwuchs.

---

## 1. Das große Bild

Familien-Hierarchie: **AiTHENTEC → Ki-NTEGRiTY → MiNDLAXY → { MOONFiNGERS,
DER TiSCH, EiN SEiN, … }**.

MOONFiNGERS ist die poetisch-visuelle Oberfläche über einem **föderierten
Mehr-Core-System**. Kein Core ist „der eine Ursprung". Jeder Core ist ein
eigenständiges Modul mit eigenem Host, eigener Persistenz, eigener API. Sie
sprechen ein **gemeinsames Provenance-Protokoll** und werden über **HTTP**
(nicht über geteilte DBs) zu einer begehbaren Wissens-/Gedächtnislandschaft
verbunden.

**Modular heißt hier:** Module kennen einander nur über HTTP-APIs und das
Provenance-Protokoll. Kein Modul importiert den Code oder die DB eines
anderen. Austauschbar, einzeln testbar, einzeln deploybar.

---

## 2. Workspace-Landkarte

| Workspace | Pfad | Git-Remote | Rolle |
|---|---|---|---|
| **der-tisch** | `/Users/ralfkirchner/Documents/der-tisch/` | `ralfarminkirchner-netizen/der-tisch` | TiSCH-App-Backend; hostet `tisch_shared_core`, `memory_shared_core`, Personen-Bibliothek |
| **moonfingers-recovery** | `/Users/ralfkirchner/moonfingers-recovery/` | `ralfarminkirchner-netizen/moonfingers` | MOONFiNGERS-App; hostet `knowledge_shared_core` + die Bridge |
| **shared-core** | `/Volumes/ThunderBolt4_2TB/MeineApps/shared-core/` | `kiNTEGRiTY/shared-core` | Zentraler sprachneutraler Wissens-Core: 40 Personen, 7 Brücken, 23 Praktiken (TS=Wahrheit, JSON=Derivat) |
| **knowledge-core** | `/Volumes/ThunderBolt4_2TB/MeineApps/knowledge-core/` | (kein Top-Git) | Zwei-Core-Repo: **Core A** = Obsidian-Vault `core-a/entries/`; **Core B** = Node-Engine (Port 4788) |
| **shared-memory** | `/Volumes/ThunderBolt4_2TB/MeineApps/shared-memory/` | (kein Top-Git) | Legacy Node-API (Port 4788, JSONL+BM25) — soll in `knowledge-core/core-b` aufgehen |
| **mindlaxy** | `/Volumes/ThunderBolt4_2TB/MeineApps/mindlaxy/` | `kiNTEGRiTY/mindlaxy` | Familien-Umbrella; `core/shared-memory.ts` ≈ 98 Records → Export in den Knowledge Core |
| **moonfingers (scratch)** | `/Volumes/ThunderBolt4_2TB/MeineApps/moonfingers/` | — | Claude-Code-Arbeits-/Steuer-Checkout — keine App, treibt die Implementierung |

Ignorieren: `/Volumes/ThunderBolt4_2TB/MeineApps/der-tisch-main/` ist
vermutlich eine veraltete ZIP-Kopie von der-tisch. **Autoritativ ist
`/Users/ralfkirchner/Documents/der-tisch/`.**

---

## 3. Aktueller Stand (was ist gebaut, Stand 2026-05-19)

### der-tisch — drei Module nebeneinander
- **`tisch_shared_core/`** ✅ fertig (Phasen 0–4). Persönliches Arbeits-/
  Antwortgedächtnis mit Kurations-Pipeline (`Candidate → Synthesis → Curator
  → reviewed/curated_draft → (Nutzer-Tor) → canonical`). Persistenz: SQLite
  via `aiosqlite` (Pfad via `TISCH_CORE_DB_PATH` überschreibbar). API:
  `/api/tisch-memory/*` (6 Endpoints). Volles Provenance-Protokoll, URN-IDs
  `tisch_shared_core:<short_id>`. Obsidian-Export/-Import vorhanden. Risiken-
  Fixes gelandet (`anthropic` deklariert, OpenAI-Import tolerant).
- **`memory_shared_core/`** ⚠️ neu (2026-05-19, andere Instanz). Schlankes
  Notiz-/Chat-Gedächtnis (persönliche Notizen, Chat-Ausschnitte). Persistenz:
  stdlib `sqlite3`. API: `/api/memory/*` (read-only: stats, folders, search,
  notes/{id}). **Bewusst OHNE** Provenance-Protokoll/Kurations-Pipeline.
  Commit-Status unklar (siehe §6.E).
- **Personen-Bibliothek** ✅ fertig. 40 Denker + 11 Disziplinen aus
  `shared-core`, ausführliche (modellgenerierte) Beschreibungen. API:
  `/api/bibliothek*`, Seite `/bibliothek`.
- `api_server.py` bindet `tisch_shared_core`- und `memory_shared_core`-Router
  ein. Die volle App bootet end-to-end (verifiziert).

### moonfingers-recovery — Knowledge Core + Bridge
- **`moonfingers-backend/knowledge_shared_core/`** ✅ gebaut (Phasen 2–3,
  Codex). Modelle (5 Record-Typen, volles Provenance-Protokoll, URN-IDs
  `knowledge_shared_core:<short_id>`), `legacy_mapping.py`, FTS5-Suche, 6
  Importer. Materialisierte JSONL-Snapshots: **758 documents, 6856 chunks,
  1017 entities, 1010 relations** (media leer). API: `/api/knowledge/*`.
- **`moonfingers-backend/moonfingers_dual_core_bridge/`** ✅ gebaut (Phase 4,
  Codex). Resolver, API `/api/moonfingers/*`. Ruft den TiSCH-Core über HTTP
  (`TISCH_CORE_BASE_URL`, Default `http://localhost:8001`). Hält Knowledge-
  und TiSCH-Daten in der Antwort getrennt; `include_private=false` Default.
- Alles sauber committet (11 Commits, 2026-05-19).

### shared-core / knowledge-core / shared-memory / mindlaxy
- **shared-core** ✅ stabil — Datenquelle, kein laufender Dienst.
- **knowledge-core** 🚧 eigenes Zwei-Core-Konzept (Core A Obsidian-Vault +
  Core B Node-Engine + Unified API). Eigene Architektur, **noch nicht** mit
  dem `knowledge_shared_core` aus dem Vertrag verschmolzen — siehe §6.B.
- **shared-memory** 🚧 Legacy, Migration nach `knowledge-core/core-b` geplant.
- **mindlaxy** — `core/shared-memory.ts` (≈98 Records) wartet auf ein
  Export-Script → JSONL → Import in den Knowledge Core (Vertrags-Phase 3b,
  noch offen).

---

## 4. Die geteilten Verträge (verbindlich für ALLE Workspaces)

1. **Kanonischer Vertrag:** `HANDOFF_dual_shared_cores_2026-05-18.md`. Liegt
   in der-tisch/ und moonfingers-recovery/. Gewinnt bei Konflikten.
2. **Provenance-Protokoll** — jeder persistierte Record jedes Cores trägt:
   `core_id`, `family_line`, `source_role`, `memory_layer`, `curation_state`,
   `visibility`, `moonfingers_use`, `provenance_chain`, optional `reuse_state`.
   Die Enum-**Werte** müssen über alle Cores **identisch** sein.
   (Ausnahme: `memory_shared_core` führt es bewusst nicht — siehe §6.A.)
3. **IDs in URN-Form** `<core_id>:<short_id>` — Außenform an allen API-Grenzen.
4. **Föderation nur über HTTP.** Kein Core importiert Code oder DB eines
   anderen. Die Bridge ruft den TiSCH-Core über die `/api/tisch-memory/*`-API.
5. **Keine `/Volumes`-Laufzeitabhängigkeit.** `/Volumes` ist nur Build-Zeit-
   Importquelle. Produktion (Railway) muss ohne externe Platte laufen.
6. **`include_private=false`** ist Default für alle MOONFiNGERS-Endpunkte.
7. **Modellgenerierte Inhalte werden nie automatisch `canonical`.** Nur über
   Nutzer-Tor (explizit / Obsidian-Frontmatter).
8. **Scoped Commits.** Jede Instanz committet nur ihre eigenen Dateien
   (`git add <datei>`, nie `git add -A`). Fremde uncommitted Änderungen
   bleiben unangetastet.
9. **Verträge nicht eigenmächtig ändern.** Fehlt etwas an einer fremden API,
   wird das als „offene API-Frage" im eigenen Report dokumentiert — der
   Nutzer entscheidet.

---

## 5. Bekannte Spannungen & Klärungsbedarf

> Das ist der wichtigste Abschnitt. Diese Punkte gefährden die Kohärenz des
> Systems, wenn jede Instanz blind weiterbaut.

**A. Drei Gedächtnis-Module in der-tisch, weiche Zuständigkeitsnaht.**
`tisch_shared_core` (Kurations-Pipeline) und `memory_shared_core` (schlankes
Notiz-Gedächtnis) beanspruchen beide „Chat-Ausschnitte / KI-Antworten".
`tisch_shared_core` hat `source_role: chat_excerpt` als Default; die
`memory_shared_core`-README zählt Chat-Ausschnitte zu ihrem Revier. Keine
Routing-Regel trennt sie. → Klare Regel nötig: *Was* gehört in welches Modul?

**B. Die Knowledge-Seite ist selbst gespalten (größtes Kohärenzrisiko).**
Es gibt **vier** Dinge mit „Knowledge/Memory"-Anspruch über Weltwissen:
`knowledge_shared_core` (moonfingers-recovery, Vertrags-Design, gebaut),
`knowledge-core` (eigenes Zwei-Core-Repo, Core A+B), `shared-memory` (Legacy
Node-API), `shared-core` (TS-Datenpaket). → Es muss geklärt werden, welches
**das** kanonische Weltwissens-Modul ist und wie die anderen sich dazu
verhalten (Quelle? Vorgänger? zu verschmelzen? eigenständig?).

**C. Obsidian-Vault.** Der einzige echte Vault liegt unter
`/Volumes/ThunderBolt4_2TB/MeineApps/knowledge-core/core-a/entries/`
(„Core A — Eigen-Content-Vault"). Der TiSCH-Core erwartet via
`OBSIDIAN_VAULT_PATH` einen Vault mit Unterordner `tisch-memory/`.
`OBSIDIAN_VAULT_PATH` ist aktuell **nirgends dauerhaft gesetzt** (nur
Testläufe gegen `/tmp`). → Entscheidung: Schreibt der TiSCH-Core in
`core-a/entries/tisch-memory/`, in einen eigenen Vault, oder gar nicht?

**D. Schema-Divergenz.** `tisch_shared_core` und `memory_shared_core`
definieren gleichnamige Felder unterschiedlich: `visibility` (5 Werte vs. 2),
`content_hash` (verschiedene Hash-Eingaben), `id` (strikte URN vs. freie
Form). Eine spätere übergreifende Suche muss das von Hand versöhnen.

**E. `memory_shared_core` Commit-Status unbestätigt.** `api_server.py`
importiert es bereits — falls das Verzeichnis untracked ist, scheitert ein
Deploy des committeten Standes. Muss per `git status` geprüft und sauber
committet werden.

**F. `curated_draft` vs. `curated`.** Die Vertrags-Tabelle listet `curated`,
die Vertrags-Prosa `curated_draft`. `tisch_shared_core` und
`knowledge_shared_core` führen beide Werte. Vertrag sollte einen festlegen.

**G. TiSCH-Core-Port.** Die Bridge erwartet den TiSCH-Core unter
`http://localhost:8001` (`TISCH_CORE_BASE_URL`). Lokale TiSCH-Smoke-Server
liefen auf 8000. → Konvention: TiSCH-Core lokal auf **8001**, oder
`TISCH_CORE_BASE_URL` explizit setzen.

---

## 6. Bauregeln für das modulare System

- **Bleib in deinem Workspace.** Andere Workspaces nur lesen (zur Information)
  oder per HTTP aufrufen — nie deren Dateien ändern oder committen.
- **Provenance-Protokoll wertgleich** halten (siehe §4.2). Bei Enum-Bedarf:
  Vertrags-Tabelle ist die Wahrheit.
- **Föderation per HTTP**, keine DB-/FS-Kopplung zwischen Cores.
- **Pro Phase committen**, scoped, mit Phasen-Report.
- **Neue Module** unter klarem Pfad, mit `SURVEY` (Phase 0) und Schluss-
  `REPORT`. Kein neues Repo ohne Nutzer-Freigabe.
- **Offene Punkte melden, nicht raten.** Cross-Workspace-Bedarf → „offene
  API-Fragen" im Report. §5 dieses Briefings ist die laufende Liste.
- **Vor jedem Schreibvorgang:** dieses Briefing + Vertrag + eigenes Survey
  gelesen?

---

## 7. Pro-Workspace-Auftrag (Kurzfassung)

- **der-tisch** — TiSCH-Core ist fertig; offene Risiken adressiert. Nächste
  Arbeit nur nach Klärung von §5.A/§5.E (memory_shared_core sauberstellen,
  Zuständigkeitsnaht definieren). Nicht weitere Cores erfinden.
- **moonfingers-recovery** — Knowledge Core + Bridge fertig; offen: Mindlaxy-
  Export (Phase 3b), Media-Katalog, leere Manifeste, Tests, Legacy-Routen-
  Aufräumen. Bridge gegen den **echten** TiSCH-Core verifizieren.
- **knowledge-core** — Klärung mit §5.B abwarten: Verhältnis zu
  `knowledge_shared_core` definieren, bevor Core A/B weitergebaut werden.
- **shared-memory** — keine Neuentwicklung; nur als Migrationsquelle für
  `knowledge-core/core-b` behandeln.
- **mindlaxy** — `core/shared-memory.ts` → Export-Script → JSONL für den
  Knowledge-Core-Importer `mindlaxy_snapshot.py`.
- **shared-core** — stabil halten; nur über das Export-Skript pflegen.

---

## 8. Offene Entscheidungen — nur Ralf kann sie treffen

1. **Knowledge-Konsolidierung (§5.B):** Welches Modul ist das kanonische
   Weltwissen — `knowledge_shared_core` (Vertrag) oder `knowledge-core`
   (Core A/B)? Sind sie zu verschmelzen, oder ist eines Quelle des anderen?
2. **Zuständigkeit TiSCH- vs. Memory-Core (§5.A):** klare Regel, was ein
   „TiSCH-Lauf" ist und in welchen Topf Chat/Notiz/KI-Antwort gehört.
3. **Obsidian-Vault (§5.C):** Welcher Vault, und wird `OBSIDIAN_VAULT_PATH`
   produktiv gesetzt? Schreibt der TiSCH-Core in `core-a/entries/`?
4. **`curated_draft` vs. `curated` (§5.F):** einen Wert kanonisieren.
5. **`memory_shared_core` (§5.E):** ist es ein dauerhaftes Modul? Dann
   committen + ins Provenance-Protokoll einordnen (mind. `core_id`).

---

## 9. Confidence / Lücken dieses Briefings

Erstellt aus drei parallelen Read-only-Surveys (2026-05-19). Sicher:
der-tisch- und moonfingers-recovery-Stände, Obsidian-Vault-Ort,
Workspace-Remotes. Unsicher / per Bash zu verifizieren: `memory_shared_core`-
Commit-Status; die internen Details von `knowledge-core` Core A/B (nur
Survey-Zusammenfassung); ob `der-tisch-main` ein eigenes Git hat.
