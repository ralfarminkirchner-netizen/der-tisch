# Auftrag für Codex — Knowledge Shared Core + MOONFiNGERS-Bridge

Du arbeitest im Repo `/Users/ralfkirchner/moonfingers-recovery/`. Du baust die
**Knowledge-Seite** des Dual-Core-Modells **und** die **MOONFiNGERS-Bridge**.

Eine zweite Instanz (Claude Code) hat parallel die **TiSCH-Seite** im Repo
`/Users/ralfkirchner/Documents/der-tisch/` gebaut — **diese ist fertig** (Phasen
0–4, alle Smoke-Tests grün, committet). Den TiSCH-Pfad fasst du NICHT an. Alles,
was du über die TiSCH-Seite wissen musst, steht unten in diesem Prompt — du musst
das der-tisch-Repo nicht lesen.

---

## Kanonischer Vertrag

Lies zuerst vollständig:

```
/Users/ralfkirchner/moonfingers-recovery/HANDOFF_dual_shared_cores_2026-05-18.md
```

Bei jedem Konflikt zwischen Vertrag und Codebasis: Vertrag gewinnt für neue
Module unter `knowledge_shared_core/` und `moonfingers_dual_core_bridge/`.
Bestehender Code bleibt unangetastet — siehe „Verboten".

---

## Stand der TiSCH-Seite (fertig — deine Integrationsfakten)

Der TiSCH Shared Core liegt unter `der-tisch-backend/tisch_shared_core/` und ist
fertig. Die Bridge integriert gegen ihn. Was du brauchst:

### A-API (TiSCH-Endpoints — die Bridge ruft sie)

```
POST /api/tisch-memory/candidates        -> 201, MemoryCandidate
GET  /api/tisch-memory/search?q=&top_k=&include_all=  -> List[StableAnswerHit]
POST /api/tisch-memory/context-pack      -> ContextPack
GET  /api/tisch-memory/cards/{card_id}   -> MemoryCard  (URN ODER Kurz-ID)
POST /api/tisch-memory/curate            -> MemoryCard
GET  /api/tisch-memory/obsidian/export   -> {vault_configured, written, count, exports[]}
```

### Datenformen

- `StableAnswerHit` = `{card: MemoryCard, relevance_score: float, is_canonical: bool, is_stable: bool}` — `GET /search` liefert diese Hülle, **nicht** nackte Cards.
- `MemoryCard` trägt das volle Provenance-Protokoll + `title, content, summary, tags, canonical, canonical_approved_by, synthesis_meta, candidate_id`.
- IDs in **URN-Form** `tisch_shared_core:<short_id>` (Vertrag Punkt 9).
- TiSCH-Persistenz: SQLite via `aiosqlite`. (Für den Knowledge-Core gilt
  abweichend **JSONL** — Vertrag Punkt 2; das ist kein Widerspruch.)

### Offene API-Fragen, die die TiSCH-Seite gemeldet hat — du musst sie kennen

1. **`curated_draft` vs. `curated`** — die Vertrags-Tabelle listet `curated`,
   die Vertrags-Prosa spricht von `curated_draft`. Die TiSCH-`models.py` führt
   **beide**. Deine Provenance-Enums **müssen wertgleich** sein — sonst spricht
   die Bridge zwei Dialekte. Bis der Nutzer entscheidet: beide Werte zulassen.
2. `ContextPack` nutzt `source_role: model_generated_synthesis` (es gibt keinen
   eigenen Enum-Wert für Aggregations-Artefakte).
3. Die TiSCH-A-API bietet **keinen entity-keyed Lookup**. Die Bridge muss den
   TiSCH-Core mit dem Entity-Label als Query abfragen
   (`GET /api/tisch-memory/search?q=<label>`). Brauchst du entity-genaue
   TiSCH-Treffer, ist das eine A-API-Lücke → im Report melden, Vertrag NICHT
   eigenmächtig ändern.

### Wie die Bridge die TiSCH-Seite ruft

Ausschließlich über **HTTP gegen die A-API**. KEINE direkte DB-/FS-Kopplung.
TiSCH-Basis-URL über Env-Var `TISCH_CORE_BASE_URL` konfigurierbar (Default
`http://localhost:8000` für Dev). In Phase 1 sind Mocks/Snapshots zulässig.

---

## Phase 0 — Survey (vor jedem Schreibvorgang)

1. Pfad-Checks:
   ```bash
   test -d /Users/ralfkirchner/moonfingers-recovery
   test -d /Users/ralfkirchner/moonfingers-recovery/moonfingers-backend
   ```
   Fehlt ein Pfad: **stoppen**, als Blocker melden, nicht woanders scaffolden,
   kein neues Repo anlegen.
2. `git status` und `git log -3 --oneline`.
3. Read-only lesen: FastAPI-Struktur von `moonfingers-backend/`, Persistenz-
   Pattern, `requirements.txt`, Procfile/Dockerfile.
4. **Codex-Quarantäne** (Vertrag „Erste Arbeitsphase" Punkt 3 + Implementation
   Defaults Punkt 5). Diese Dateien lesen, aber **nicht blind committen**:
   - `moonfingers-backend/api/shared_memory.py`
   - `moonfingers-backend/models/shared_memory.py`
   - `moonfingers-backend/services/shared_memory_scanner.py`
   - `moonfingers-backend/services/shared_memory_validator.py`
   - `moonfingers-backend/shared-memory/`

   Kopie nach `artifacts/quarantine/codex_knowledge_core_2026-05-18/`,
   Bewertungsnotiz `artifacts/CODEX_KNOWLEDGE_CORE_QUARANTINE_2026-05-18.md`
   (welche Dateien, was sie wollten, welche Ideen brauchbar, warum nicht blind
   übernommen). Regel: **Codex-Code darf Ideen liefern, nicht die Struktur setzen.**
5. Survey-Notiz `knowledge_shared_core/SURVEY_2026-05-18.md` (FastAPI-Struktur,
   Persistenz-Pattern, Pydantic-Pattern, Codex-Stellen, alles seit dem letzten
   Audit Veränderte).

**Stoppen** und Rückfrage stellen, falls: ein Pfad fehlt; die Railway-Config
abweicht; ein paralleler Branch / uncommitted Changes auf Knowledge-Code zeigen;
die Codex-Dateien an mehreren Stellen liegen und die Wahl unklar ist.

---

## Phase 1 — Architektur-Doku + Knowledge-Modelle + legacy_mapping

- Architekturentscheidung dokumentieren:
  `MOONFINGERS_DUAL_SHARED_CORE_ARCHITECTURE_2026-05-18.md`.
- `knowledge_shared_core/models.py` — Pydantic-Modelle `KnowledgeDocument`,
  `KnowledgeChunk`, `KnowledgeEntity`, `KnowledgeRelation`, `MediaAsset`. Alle
  tragen das **volle Provenance-Protokoll**: `core_id` (fix
  `knowledge_shared_core`), `family_line`, `source_role`, `memory_layer`,
  `curation_state`, `visibility`, `moonfingers_use`, `provenance_chain`,
  optional `reuse_state`. IDs in URN-Form `knowledge_shared_core:<short_id>`.
  **Die Provenance-Enums müssen wertgleich zur TiSCH-Seite sein** (Vertrags-
  Tabelle „Provenance-Protokoll"; bei Werte-Zweifeln gewinnt die Vertrags-Tabelle
  plus die `curated_draft`-Ausnahme).
- `legacy_mapping.py` — alte flache Quellenfelder → Provenance-Protokoll
  (Vertrag „Mapping bestehender Quellen" + Implementation Defaults Punkt 4).
  Die 754 bereits persistierten Dokumente müssen darüber einordbar sein.
- `shared-memory/GENEALOGY_MAPPING_2026-05-18.md` (Implementation Defaults Punkt 4).
- Smoke-Test (Demo-Record erzeugen + zurücklesen).
- Commit: `feat(knowledge-core): models + legacy mapping`.

## Phase 2 — Importer-Skelette + Daten-Snapshot

- `knowledge_shared_core/importers/`: `einsein_salvage.py`, `restart_atlas.py`,
  `story_sequence.py`, `shared_core_seed.py`, `mindlaxy_snapshot.py` — Skelette
  mit korrekten Signaturen, kein Live-Import nötig.
- Daten-Snapshot nach `data/knowledge_shared_core/` als **JSONL** (Vertrag
  Punkt 2 — JSONL ist für den Knowledge-Core fürs MVP ausdrücklich vorgesehen):
  `source_manifest.json`, `documents.jsonl`, `chunks.jsonl`, `entities.jsonl`,
  `relations.jsonl`, `media_manifest.jsonl`, `genealogy_mapping.json`.
- `/Volumes`-Pfade sind **nur Build-Zeit-Importquelle**, nie Laufzeit.
- Bilder nur als **Metadaten** (Vertrag Punkt 2: `image_url`,
  `local_source_path`, `attribution`, `entity_id`, `exists_locally`,
  `moonfingers_use`) — keine Binärdateien.
- Mindlaxy-Records nicht in Python nachbauen — Export-Script → JSONL-Snapshot
  mit `composition_source`-Feld (Vertrag Punkt 3).
- Commit: `feat(knowledge-core): importer skeletons + jsonl snapshot`.

## Phase 3 — Knowledge-Suche + Knowledge-API

- `search.py` — Suche/Lookup über die JSONL-Snapshots.
- `api.py` — FastAPI-Router mit der **B-API**:
  ```
  GET /api/knowledge/search
  GET /api/knowledge/entities/{entity_id}
  GET /api/knowledge/media/{entity_id}
  GET /api/knowledge/dossiers/{id}
  GET /api/knowledge/stories
  ```
- Einbindung in das `moonfingers-backend`-FastAPI-App über **eine**
  `include_router`-Zeile.
- Commit: `feat(knowledge-core): search + knowledge api`.

## Phase 4 — MOONFiNGERS-Bridge

- `moonfingers_dual_core_bridge/`: `dual_core_resolver.py`, `world_view.py`,
  `entity_view.py`, `story_view.py`, `private_overlay.py`, `api.py`.
  Physischer Ort: im `moonfingers-backend/` (Vertrag Implementation Defaults
  Punkt 8 — Knowledge-seitig).
- **C-API**:
  ```
  GET /api/moonfingers/world
  GET /api/moonfingers/entity/{entity_id}
  GET /api/moonfingers/knowledge/search
  GET /api/moonfingers/stories
  GET /api/moonfingers/private-overlay
  ```
- Der Resolver ruft den Knowledge-Core direkt und den TiSCH-Core über HTTP
  (A-API). Die Antwort weist Knowledge- und TiSCH-Daten **getrennt** aus:
  `verified_dossiers`, `images`, `relations`, `stories`, `personal_notes`,
  `tisch_syntheses`, `context_cards`, `provenance` — nie implizit vermischt.
- **`include_private=false` ist Default** (Vertrag Punkt 7). Private Overlays
  nur mit explizitem `?include_private=true` im authentifizierten/lokalen Kontext.
- Alle IDs in Bridge-Antworten in URN-Form mit Core-Präfix (Vertrag Punkt 9).
- Commit: `feat(bridge): dual-core resolver + /api/moonfingers routes`.

---

## Verboten

- Kein Anfassen von `/Users/ralfkirchner/Documents/der-tisch/`. Das ist die
  TiSCH-Instanz.
- Kein neues Backend-Repo. Kein TypeScript-Backend, kein Fastify.
- Keine `/Volumes`-Laufzeitabhängigkeit (nur Build-Zeit-Importquelle).
- Keine direkte DB-/FS-Kopplung zum TiSCH-Core — Bridge spricht nur HTTP gegen
  die A-API.
- Keine automatische Kanonisierung modellgenerierter Inhalte.
- Keine blinde Übernahme der Codex-Dateien.
- Keine flache Datenbank ohne Provenance.

---

## Koordination mit der TiSCH-Instanz

- Berührungspunkt ist die A-API. Wenn dir auffällt, dass die A-API ein
  Feld/Endpoint nicht liefert, das die Bridge braucht (z.B. entity-keyed
  Lookup), oder die Provenance-Form nicht passt → dokumentiere das im
  Phasen-Report unter **„Offene API-Fragen für die TiSCH-Seite"**. Du änderst
  den Vertrag NICHT eigenmächtig — der Nutzer entscheidet.
- Provenance-Enums beider Cores wertgleich halten. Die `curated_draft`/
  `curated`-Diskrepanz ist offen; bis zur Nutzer-Entscheidung beide Werte
  zulassen (so wie die TiSCH-Seite).

---

## Bericht

Pro Phase committen — nicht alles in einem Mega-Commit. Schluss-Report
`knowledge_shared_core/PHASE1_REPORT_2026-05-18.md` mit:

- ausgeführte Phasen (0–4) mit Status (✓ / ⚠ / ✗)
- Liste aller neu angelegten und geänderten Dateien
- Liste aller Commits mit Hash und Message
- Inhalt von `CODEX_KNOWLEDGE_CORE_QUARANTINE_2026-05-18.md`
- neue Datenmodelle + Mapping-Tabelle (`legacy_mapping.py`)
- Beispiele: `KnowledgeDocument`, MOONFiNGERS-Dual-Core-Antwort
  (`GET /api/moonfingers/entity/{id}?include_private=true`)
- Smoke-Test-Output für die B- und C-API
- offene Risiken
- offene API-Fragen für die TiSCH-Seite

---

## Akzeptanzkriterien (Vertrag)

- Zwei Cores klar getrennt, jeder mit eigenem Host.
- Jeder Eintrag trägt das volle Provenance-Protokoll; IDs in URN-Form.
- Die 754 Dokumente sind über `legacy_mapping.py` einordbar.
- MOONFiNGERS-Antworten weisen Knowledge- und TiSCH-Daten getrennt aus.
- `include_private=false` ist Default; private Daten nur mit explizitem Flag.
- Bridge ruft den TiSCH-Core nur über HTTP — keine DB-Kopplung.
- Keine `/Volumes`-/Obsidian-Pfade in produktiver Runtime.
