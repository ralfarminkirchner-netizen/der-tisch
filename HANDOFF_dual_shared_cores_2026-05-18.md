# MOONFiNGERS — Dual Shared Cores Handoff (kanonisch)

Stand: 2026-05-18 (Revision nach Nutzer-Korrektur)
Ersetzt: `HANDOFF_moonfinger-memory-backend.md` (TypeScript-Variante, verworfen)

---

## Leitsatz

> MOONFiNGERS ist das genealogische Kind aus SEiN-Familie, Spiral Mind Wiki und TiSCH-Apps. Der Wissens-Core hält Weltwissen. Der TiSCH-Core hält persönliches Arbeits- und Antwortgedächtnis. MOONFiNGERS verwandelt beides in eine begehbare Denklandschaft.

MOONFiNGERS ist **nicht** der eine Memory-Ursprung und **nicht** in einen einzigen Host eingelassen. MOONFiNGERS ist die poetisch-visuelle Dual-Core-Oberfläche über zwei **separate**, **föderierte** Shared Cores, die ein gemeinsames Herkunfts-/Provenance-Protokoll sprechen, aber nicht zwingend dieselbe Datenbank teilen.

---

## Architektur in einem Bild

```
Wissens-App-Familie                          TiSCH-App-Familie
─────────────────────                        ────────────────────────────
EiN SEiN, HEiL SEiN,                         DER TiSCH, TEAM TiSCH,
DEiN SEiN, IBW,                              LiTERATEN, EXPERTiSEN,
Spiral Mind Wiki,                            iNTEGRATiONS TiSCH,
Wiki-/Monographie-Dumps,                     Chats, Prompts, Antworten,
Bilder, Atlas, Dossiers,                     KiNTEGRiTY-Synthesen,
Entitäten, Relationen,                       kuratierte Cards,
Wissensgeschichten                           Obsidian-Notizen
        │                                              │
        ▼                                              ▼
┌────────────────────────────┐         ┌────────────────────────────────┐
│ Knowledge Shared Core      │         │ TiSCH Shared Core              │
│ Host: moonfingers-backend/ │         │ Host: der-tisch-backend/       │
│  verified_knowledge        │         │  personal_memory               │
│  poetic_surface            │         │  project_memory                │
│                            │         │  reusable_context              │
└─────────────┬──────────────┘         └─────────────┬──────────────────┘
              │   gemeinsames Provenance-Protokoll   │
              │   (core_id, family_line, source_role,│
              │    memory_layer, curation_state,     │
              │    visibility, moonfingers_use,      │
              │    provenance_chain)                 │
              └────────────────┬─────────────────────┘
                               ▼
                    ┌──────────────────────────┐
                    │ moonfingers_dual_core_   │
                    │ bridge — Resolver mit    │
                    │ Herkunftsmarkierung      │
                    └────────────┬─────────────┘
                                 ▼
                     MOONFiNGERS Frontend
                     (Globus, Story, Bild,
                      private Overlay, Fog)
```

**Föderation, keine Verschmelzung.** Jeder Eintrag in jedem Core trägt das volle Provenance-Protokoll. MOONFiNGERS zeigt beides zusammen, aber **nie** ohne Herkunftsmarkierung.

---

## Hosts und Repos

### Knowledge Shared Core

- Startpunkt: `/Users/ralfkirchner/moonfingers-recovery/moonfingers-backend/`
- Sprache: Python / FastAPI
- Aktion: Der existierende Ordner wird **semantisch neu gerahmt** als „Knowledge Shared Core / MOONFiNGERS World Backend". Kein neues Repo. Kein TypeScript-Backend. Der Name `moonfinger-memory-backend` wird als Übergangsbezeichnung aufgegeben.
- Arbeitsname im Code: `knowledge_shared_core`.

### TiSCH Shared Core

- Startpunkt: `/Users/ralfkirchner/Documents/der-tisch/der-tisch-backend/`
- Sprache: Python / FastAPI, deployed auf Railway (`der-tisch` im Projekt `clever-gentleness`)
- Aktion: Bestehende Funktionalität bleibt intakt. Neue Module unter `tisch_shared_core/` (oder unter `services/`, falls Struktur das vorzieht).

### Bridge

- Arbeitsname: `moonfingers_dual_core_bridge`
- Eine logische Schicht: ein Resolver, der MOONFiNGERS-Endpunkte gegen beide Cores ausspielt und Herkunft mitführt.
- Physischer Ort: Default in `moonfingers-backend/` (siehe Implementation Defaults Punkt 8).

---

## Verzeichnis-Struktur (Vorschlag)

### Knowledge Shared Core (`moonfingers-backend/`)

```
moonfingers-backend/
  knowledge_shared_core/
    models.py            # KnowledgeDocument, KnowledgeChunk, KnowledgeEntity, KnowledgeRelation, MediaAsset
    importers/
      einsein_salvage.py
      restart_atlas.py
      story_sequence.py
      shared_core_seed.py
      mindlaxy_snapshot.py
    legacy_mapping.py    # alte Felder → Provenance-Protokoll
    search.py
    api.py               # /api/knowledge/...

  moonfingers_dual_core_bridge/
    dual_core_resolver.py
    world_view.py
    entity_view.py
    story_view.py
    private_overlay.py
    api.py               # /api/moonfingers/...

  data/
    knowledge_shared_core/
      source_manifest.json
      documents.jsonl
      chunks.jsonl
      entities.jsonl
      relations.jsonl
      media_manifest.jsonl
      genealogy_mapping.json
```

### TiSCH Shared Core (`der-tisch-backend/`)

```
der-tisch-backend/
  tisch_shared_core/
    models.py            # MemoryCandidate, MemoryCard, ContextPack, ObsidianNoteExport
    capture.py
    kintegrity_synthesis.py
    curator.py
    stable_answers.py
    context_packs.py
    obsidian_export.py
    search.py
    api.py               # /api/tisch-memory/...
```

---

## Provenance-Protokoll (Pflicht in beiden Cores)

| Feld | Werte |
|---|---|
| `core_id` | `knowledge_shared_core`, `tisch_shared_core` |
| `family_line` (Liste) | `ein-sein`, `heil-sein`, `dein-sein`, `ibw`, `spiral-mind-wiki`, `der-tisch`, `team-tisch`, `literaten-tisch`, `expertisen-tisch`, `integrations-tisch`, `moonfingers`, `mindlaxy` |
| `source_role` | `wiki_monograph`, `verified_research`, `image_asset`, `atlas_entry`, `story_sequence`, `dossier`, `entity`, `relation`, `tisch_run_result`, `chat_excerpt`, `model_generated_synthesis`, `kintegrity_synthesis`, `user_authored_note`, `curator_decision`, `stable_answer`, `prompt` |
| `memory_layer` | `verified_knowledge`, `personal_memory`, `project_memory`, `reusable_context`, `poetic_surface`, `archive` |
| `curation_state` | `raw`, `imported`, `candidate`, `synthesized`, `reviewed`, `curated`, `canonical`, `deprecated`, `archived` |
| `visibility` | `public`, `private`, `internal`, `moonfingers_only`, `tisch_only` |
| `moonfingers_use` (Liste) | `globe_node`, `entity_dossier`, `relation`, `image`, `story_step`, `fog_hint`, `personal_overlay`, `context_pack` |
| `provenance_chain` | Liste von Transformations-Schritten |
| `reuse_state` (optional) | z.B. `approved_for_reuse` — erlaubt Wiederverwendung als Context Pack **ohne** als kanonische Wahrheit zu gelten |

### Beispiel mit Provenance-Chain

```json
{
  "id": "mem_moonfingers_dual_core_architecture_2026_05_18",
  "core_id": "tisch_shared_core",
  "family_line": ["der-tisch", "moonfingers"],
  "source_role": "kintegrity_synthesis",
  "memory_layer": "project_memory",
  "curation_state": "reviewed",
  "visibility": "private",
  "reuse_state": "approved_for_reuse",
  "provenance_chain": [
    { "step": "chat_answer", "origin": "chatgpt", "timestamp": "2026-05-18", "summary": "Dual-Core-Architektur formuliert" },
    { "step": "kintegrity_synthesis", "origin": "der-tisch", "timestamp": "2026-05-18", "summary": "Verdichtet und auf Wiederverwendung geprüft" },
    { "step": "curator_proposal", "origin": "curator_agent", "timestamp": "2026-05-18", "summary": "Als Obsidian-Projektnotiz vorgeschlagen" }
  ]
}
```

---

## Mapping bestehender Quellen

```
einsein-salvage / Wiki-Dumps
  core_id: knowledge_shared_core
  family_line: [ein-sein, spiral-mind-wiki]
  source_role: wiki_monograph
  memory_layer: verified_knowledge
  curation_state: imported
  moonfingers_use: [entity_dossier, globe_node, story_step]

EiN SEiN Restart-Atlas / Storydaten
  core_id: knowledge_shared_core
  family_line: [ein-sein]
  source_role: atlas_entry | story_sequence | dossier
  memory_layer: verified_knowledge
  curation_state: imported
  moonfingers_use: [story_step, entity_dossier, globe_node]

EiN SEiN Bilder
  core_id: knowledge_shared_core
  family_line: [ein-sein, spiral-mind-wiki]
  source_role: image_asset
  memory_layer: verified_knowledge
  curation_state: imported
  moonfingers_use: [image, entity_dossier, story_step]

MOONFiNGERS shared-core seed.json
  core_id: knowledge_shared_core
  family_line: [moonfingers]
  source_role: entity | relation
  memory_layer: poetic_surface
  curation_state: imported
  moonfingers_use: [globe_node, relation, fog_hint]

TiSCH-Durchläufe
  core_id: tisch_shared_core
  family_line: [der-tisch]
  source_role: tisch_run_result
  memory_layer: project_memory
  curation_state: candidate
  moonfingers_use: [personal_overlay, context_pack]

Chat-Texte
  core_id: tisch_shared_core
  family_line: [der-tisch]
  source_role: chat_excerpt
  memory_layer: personal_memory
  curation_state: raw | candidate
  moonfingers_use: [personal_overlay, context_pack]

KiNTEGRiTY-Synthesen
  core_id: tisch_shared_core
  family_line: [der-tisch, moonfingers]
  source_role: kintegrity_synthesis
  memory_layer: reusable_context
  curation_state: synthesized | reviewed
  moonfingers_use: [context_pack, personal_overlay]

Obsidian-Notizen
  core_id: tisch_shared_core
  family_line: [dein-sein, der-tisch]
  source_role: user_authored_note | curator_decision
  memory_layer: personal_memory | project_memory
  curation_state: curated | canonical
  moonfingers_use: [personal_overlay, context_pack]
```

---

## Kurations-Pipeline

```
TiSCH-Durchlauf / Chat / Antwort / Artefakt
  → Raw Capture
  → MemoryCandidate
  → KiNTEGRiTY-Zusammenfassung
  → Dedupe gegen vorhandene MemoryCards
  → Curator Agent
  → reviewed / curated_draft
  → Nutzerfreigabe oder explizite Regel
  → canonical
  → Obsidian Note / MemoryCard
  → Indexierung
  → ContextPack für spätere TiSCH-Apps
```

**Harte Regeln:**

- `model_generated` darf **nie** direkt `canonical` werden.
- Der Agent darf autonom bis `reviewed` / `curated_draft` gehen und darf `approved_for_reuse` setzen.
- Der Übergang zu `canonical` ist Nutzer-Tor oder explizite, deklarierte Regel.
- `approved_for_reuse` ist ausdrücklich **nicht** dasselbe wie `canonical`: gut genug für Context Packs, nicht „Wahrheit".

---

## API-Gruppen

### A. TiSCH Shared Core (`der-tisch-backend`)

```
POST /api/tisch-memory/candidates
GET  /api/tisch-memory/search
POST /api/tisch-memory/context-pack
GET  /api/tisch-memory/cards/{card_id}
POST /api/tisch-memory/curate
GET  /api/tisch-memory/obsidian/export
```

### B. Knowledge Shared Core (`moonfingers-backend`)

```
GET /api/knowledge/search
GET /api/knowledge/entities/{entity_id}
GET /api/knowledge/media/{entity_id}
GET /api/knowledge/dossiers/{id}
GET /api/knowledge/stories
```

### C. MOONFiNGERS Dual-Core (über Bridge)

```
GET /api/moonfingers/world
GET /api/moonfingers/entity/{entity_id}
GET /api/moonfingers/knowledge/search
GET /api/moonfingers/stories
GET /api/moonfingers/private-overlay
```

Beispiel:

```http
GET /api/moonfingers/entity/David_Chalmers?include_private=true
```

```json
{
  "entity": { "id": "knowledge_shared_core:David_Chalmers", "label": "David Chalmers", "core_id": "knowledge_shared_core" },
  "verified_dossiers": [],
  "images": [],
  "relations": [],
  "stories": [],
  "personal_notes": [],
  "tisch_syntheses": [],
  "context_cards": [],
  "provenance": []
}
```

---

## Obsidian-Regel

Obsidian ist die menschliche, kuratierte Wissensoberfläche für persönliches und projektbezogenes Wissen.

- DER TiSCH bleibt operative Wahrheit für Candidates, Workflow, Index, Context Packs, Provenance.
- Ein **lokaler Sync/Writer** schreibt kuratierte `MemoryCard`s als Markdown-mit-Frontmatter in den Vault — und liest sie wieder ein.
- Railway / produktive Pfade dürfen **nicht** direkt vom lokalen Obsidian-Vault oder von `/Volumes`-Pfaden abhängen. Maschinenlesbare Spiegelung läuft über Snapshots, JSONL oder DB.
- Kein Obsidian-Plugin als MVP. Erst stabile Markdown-IDs und Export/Import.

---

## Erste Arbeitsphase

1. Architekturentscheidung dokumentieren:
   `/Users/ralfkirchner/moonfingers-recovery/MOONFINGERS_DUAL_SHARED_CORE_ARCHITECTURE_2026-05-18.md`

2. Git-Status und vorhandene Dateien prüfen in:
   - `/Users/ralfkirchner/moonfingers-recovery/`
   - `/Users/ralfkirchner/moonfingers-recovery/moonfingers-backend/`
   - `/Users/ralfkirchner/Documents/der-tisch/der-tisch-backend/`

3. **Codex-Quarantäne.** Diese Dateien lesen, aber **nicht blind committen**:
   - `moonfingers-recovery/moonfingers-backend/api/shared_memory.py`
   - `moonfingers-recovery/moonfingers-backend/models/shared_memory.py`
   - `moonfingers-recovery/moonfingers-backend/services/shared_memory_scanner.py`
   - `moonfingers-recovery/moonfingers-backend/services/shared_memory_validator.py`
   - `moonfingers-recovery/moonfingers-backend/shared-memory/`

   Bewertung in:
   `/Users/ralfkirchner/moonfingers-recovery/artifacts/CODEX_KNOWLEDGE_CORE_QUARANTINE_2026-05-18.md`

4. **Knowledge Shared Core** minimal anlegen:
   - Pydantic-Modelle für `KnowledgeDocument`, `KnowledgeChunk`, `KnowledgeEntity`, `KnowledgeRelation`, `MediaAsset` mit vollem Provenance-Protokoll
   - `legacy_mapping.py` für alte flache Felder → Provenance-Protokoll (siehe Mapping-Tabelle oben)
   - Import-Skelette für `einsein-salvage`, Wiki-Dumps, `shared-core seed.json`, Atlas-/Storydaten, Bilder

5. **TiSCH Shared Core** minimal anlegen:
   - `MemoryCandidate`, `MemoryCard`, `ContextPack`, `ObsidianNoteExport`
   - Candidate Capture
   - KiNTEGRiTY-Synthesis-Stub oder Integration
   - Curator-Proposal mit harter Grenze bei `canonical`
   - `ContextPack`-Erzeugung
   - Obsidian-Markdown-Export mit Frontmatter

6. **Bridge** minimal anlegen (Mocks/Snapshots in Phase 1 zulässig):
   - `dual_core_resolver` mit Herkunftsmarkierung
   - `GET /api/moonfingers/entity/{entity_id}?include_private=true`

7. **Nicht tun:**
   - kein neues Backend-Repo
   - kein neues TypeScript-Backend, kein Fastify
   - keine flache Datenbank ohne Provenance
   - keine automatische Kanonisierung modellgenerierter Inhalte
   - keine Railway-Abhängigkeit auf lokale Pfade
   - keine blinde Übernahme der Codex-Dateien

---

## Akzeptanzkriterien

- Zwei Cores sind klar getrennt, jeder mit eigenem Host.
- MOONFiNGERS ist als Dual-Core-App modelliert, Brücke vorhanden.
- Jeder Eintrag in beiden Cores trägt das volle Provenance-Protokoll.
- Die 754 schon persistierten Dokumente sind über `legacy_mapping.py` einordbar.
- TiSCH-Antworten laufen über `Candidate → Synthesis → Curator → reviewed/curated_draft → (User-Gate) → canonical`.
- `approved_for_reuse` funktioniert als Wiederverwendungs-Flag ohne `canonical`-Promotion.
- Obsidian-Export funktioniert als Markdown mit Frontmatter und ist re-importierbar.
- MOONFiNGERS-Antworten weisen Knowledge- und TiSCH-Daten getrennt aus.
- `include_private=false` ist Default, private Daten nur mit explizitem Flag.
- Bestehende der-tisch-Funktionalität bleibt intakt.
- Keine `/Volumes`-/Obsidian-Pfade in produktiver Runtime.

---

## Bericht am Ende

- getroffene Architekturentscheidungen
- geänderte und neu angelegte Dateien
- Inhalt von `CODEX_KNOWLEDGE_CORE_QUARANTINE_2026-05-18.md`
- neue Datenmodelle
- Mapping-Tabelle (umgesetzt in `legacy_mapping.py`)
- Beispiele: `KnowledgeDocument`, `MemoryCandidate`, `MemoryCard`, `ContextPack`, MOONFiNGERS Dual-Core-Antwort
- ausgeführte Checks
- offene Risiken

---

## Implementation Defaults für offene Kanten

Diese Punkte sind keine offenen Fragen mehr, sondern gelten als Defaults für Claude Code.

### 1. Lokaler `der-tisch`-Pfad

Kanonischer Pfad für den TiSCH Shared Core:

```
/Users/ralfkirchner/Documents/der-tisch/der-tisch-backend
```

Claude Code prüft diesen Pfad zuerst:

```bash
test -d /Users/ralfkirchner/Documents/der-tisch/der-tisch-backend
```

Wenn der Pfad fehlt: **kein Ersatzbackend erzeugen, keinen Ersatzpfad erfinden, kein neues Repo anlegen.** Fallback-Suche:

```bash
find /Users/ralfkirchner/Documents -maxdepth 4 -type d -name der-tisch 2>/dev/null
find /Users/ralfkirchner/Documents/GitHub -maxdepth 4 -type d -name der-tisch 2>/dev/null
```

Wenn nichts gefunden: als Blocker dokumentieren und im `moonfingers-recovery` nur Architektur-/Quarantäne-Dateien schreiben.

### 2. Produktionssichere Speicherung der 754 Wiki-/Shared-Memory-Dokumente

`/Volumes`-Pfade sind nur lokale Importquellen. Produktive Runtime darf nie von `/Volumes` abhängen.

Für den Knowledge Shared Core wird ein materialisierter Snapshot erzeugt:

```
/Users/ralfkirchner/moonfingers-recovery/moonfingers-backend/data/knowledge_shared_core/
  source_manifest.json
  documents.jsonl
  chunks.jsonl
  entities.jsonl
  relations.jsonl
  media_manifest.jsonl
```

Für MVP reicht JSONL. Später kann daraus SQLite/Postgres werden. Wichtig ist erstmal: importierbar, versionierbar, prüfbar, ohne externe Platte lauffähig.

Bilder werden zunächst **nicht** als Binärdateien übernommen. Nur Metadaten:

```
image_url
local_source_path
attribution
entity_id
exists_locally
moonfingers_use
```

### 3. Mindlaxy-Records

Die ~98 Mindlaxy-Records werden **nicht** in Python nachgebaut.

```
mindlaxy/core/shared-memory.ts
  → Export-Script
  → JSONL-Snapshot
  → Python-Importer im Knowledge Shared Core
```

Die Herkunft aus `sharedMemoryAcceptanceCriteria`, `ibvSharedPractices`, `sharedPersonalities`, `sharedBridges` bleibt erhalten — ideal als einzelne Exporte, ansonsten Snapshot mit `composition_source`-Feld:

```json
{
  "id": "...",
  "title": "...",
  "content": "...",
  "origin_module": "mindlaxy/core/shared-memory.ts",
  "composition_source": "sharedMemoryAcceptanceCriteria | ibvSharedPractices | sharedPersonalities | sharedBridges",
  "core_id": "knowledge_shared_core",
  "family_line": ["mindlaxy"],
  "source_role": "...",
  "memory_layer": "verified_knowledge",
  "curation_state": "imported"
}
```

### 4. Mapping alter flacher Quellenfelder

Feste Genealogie-Mapping-Tabelle:

```
/Users/ralfkirchner/moonfingers-recovery/moonfingers-backend/shared-memory/GENEALOGY_MAPPING_2026-05-18.md
```

Optional später maschinenlesbar:

```
moonfingers-backend/data/knowledge_shared_core/genealogy_mapping.json
```

Kanonisches Mapping (formal):

```
sourceId: einsein-salvage
  core_id: knowledge_shared_core
  family_line: [ein-sein, spiral-mind-wiki]
  source_role: wiki_monograph
  memory_layer: verified_knowledge
  curation_state: imported
  moonfingers_use: [entity_dossier, globe_node, story_step]

sourcePath: */wiki-app/public/data*.json#*
  core_id: knowledge_shared_core
  family_line: [ein-sein, spiral-mind-wiki]
  source_role: wiki_monograph
  memory_layer: verified_knowledge

restart-atlas.ts
  core_id: knowledge_shared_core
  family_line: [ein-sein]
  source_role: atlas_entry
  memory_layer: verified_knowledge
  moonfingers_use: [story_step, globe_node, entity_dossier]

story-sequence.ts
  core_id: knowledge_shared_core
  family_line: [ein-sein]
  source_role: story_sequence
  memory_layer: verified_knowledge
  moonfingers_use: [story_step]

atlas-content.ts
  core_id: knowledge_shared_core
  family_line: [ein-sein]
  source_role: dossier
  memory_layer: verified_knowledge
  moonfingers_use: [entity_dossier, story_step]

public/assets/imported/commons/restart/*
  core_id: knowledge_shared_core
  family_line: [ein-sein, spiral-mind-wiki]
  source_role: image_asset
  memory_layer: verified_knowledge
  moonfingers_use: [image, entity_dossier, story_step]

shared-core/entities/seed.json entities
  core_id: knowledge_shared_core
  family_line: [moonfingers]
  source_role: entity
  memory_layer: poetic_surface
  curation_state: imported
  moonfingers_use: [globe_node, entity_dossier]

shared-core/entities/seed.json relations
  core_id: knowledge_shared_core
  family_line: [moonfingers]
  source_role: relation
  memory_layer: poetic_surface
  curation_state: imported
  moonfingers_use: [relation, fog_hint]

TiSCH run result
  core_id: tisch_shared_core
  family_line: [der-tisch]
  source_role: tisch_run_result
  memory_layer: project_memory
  curation_state: candidate
  moonfingers_use: [personal_overlay, context_pack]

Chat excerpt
  core_id: tisch_shared_core
  family_line: [der-tisch]
  source_role: chat_excerpt
  memory_layer: personal_memory
  curation_state: raw
  moonfingers_use: [personal_overlay, context_pack]

KiNTEGRiTY synthesis
  core_id: tisch_shared_core
  family_line: [der-tisch, moonfingers]
  source_role: kintegrity_synthesis
  memory_layer: reusable_context
  curation_state: synthesized
  moonfingers_use: [context_pack, personal_overlay]

Obsidian note
  core_id: tisch_shared_core
  family_line: [dein-sein, der-tisch]
  source_role: user_authored_note | curator_decision
  memory_layer: personal_memory | project_memory
  curation_state: curated | canonical
  moonfingers_use: [personal_overlay, context_pack]
```

### 5. Codex-Untracked-Files in `moonfingers-backend/`

Werden **nicht** blind committet und **nicht** als kanonische Architektur übernommen. Inspiziert und quarantäniert unter:

```
/Users/ralfkirchner/moonfingers-recovery/artifacts/quarantine/codex_knowledge_core_2026-05-18/
```

Bewertungsnotiz:

```
/Users/ralfkirchner/moonfingers-recovery/artifacts/CODEX_KNOWLEDGE_CORE_QUARANTINE_2026-05-18.md
```

Inhalt:

- welche Dateien Codex erzeugt hat
- was sie tun wollten
- welche Konzepte nützlich sind
- warum sie nicht blind übernommen werden
- was später nach Genealogie-/Provenance-Schema portiert werden darf

Regel: **Codex-Code darf Ideen liefern, aber nicht die Struktur setzen.**

### 6. Approval-Tor für `canonical`

MVP-Approval sitzt in Obsidian-Frontmatter. Keine eigene MOONFiNGERS-Kurations-UI an Tag eins.

Agent und KiNTEGRiTY dürfen autonom bis `reviewed` / `curated_draft` / `approved_for_reuse` gehen. Bei `model_generated`-Inhalten **nie** automatisch `canonical` setzen. `canonical` nur durch Nutzerfreigabe oder explizite spätere Regel.

Frontmatter-Beispiel für eine reviewed Card:

```yaml
---
id: mem_example
type: memory_card
core_id: tisch_shared_core
family_line:
  - der-tisch
  - moonfingers
source_role: kintegrity_synthesis
memory_layer: project_memory
curation_state: reviewed
approved_for_reuse: true
canonical: false
canonical_approved_by:
canonical_approved_at:
visibility: private
---
```

Für echte Kanonisierung:

```yaml
curation_state: canonical
canonical: true
canonical_approved_by: ralf
canonical_approved_at: 2026-05-18
```

Wichtiger Unterschied:

- `approved_for_reuse: true` → darf für Context Packs genutzt werden
- `canonical: true` → als stabile Wahrheit / Entscheidung bestätigt

Beides muss explizit gesetzt sein, eines schließt das andere nicht ein. Schutz gegen Selbstverstärkung.

### 7. Privacy-Default für die MOONFiNGERS-Bridge

Persönliches TiSCH-/Obsidian-Wissen darf **nie** standardmäßig wie öffentliches Weltwissen ausgeliefert werden.

Default für alle MOONFiNGERS-Endpunkte:

```
include_private=false
```

Private Overlays nur mit explizitem Flag und im authentifizierten/lokalen Kontext:

```http
GET /api/moonfingers/entity/{entity_id}?include_private=true
```

Auch dann muss die Antwort die Felder **getrennt** ausweisen (`verified_dossiers`, `images`, `relations`, `stories`, `personal_notes`, `tisch_syntheses`, `context_cards`, `provenance`) — keine implizite Vermischung von öffentlichem Weltwissen und privatem Material.

### 8. Bridge-Physik

Default-Ort für `moonfingers_dual_core_bridge`: als Modul im `moonfingers-backend/` (Knowledge-seitig). Die `/api/moonfingers/...`-Routen sind dort am natürlichsten verankert, und es schmälert die `der-tisch`-Verantwortung sauber. Die Bridge ruft den TiSCH Shared Core über die A-API (`/api/tisch-memory/...`) auf — keine direkte DB-Kopplung.

Alternative Erwägung (für später, nicht MVP): dünner Proxy-Service. Frontend-direkt-zwei-Hosts wird vermieden, weil das die Provenance-Markierung an der Bridge-Schicht aufweicht.

### 9. Cross-Core-ID-Konvention

Default: **URN-Form mit Core-Präfix.** IDs in Provenance-Chains, in der Bridge und in MOONFiNGERS-Antworten tragen den Core mit:

```
knowledge_shared_core:David_Chalmers
tisch_shared_core:mem_chalmers_chat_2026_05_12
```

Vorteil: keine Dopplung in der Bridge, Provenance-Chains sind eindeutig, IDs sind über Cores hinweg unique. Innerhalb eines Cores darf intern die kurze ID (`David_Chalmers`) verwendet werden — die URN-Form ist die Außenform.
