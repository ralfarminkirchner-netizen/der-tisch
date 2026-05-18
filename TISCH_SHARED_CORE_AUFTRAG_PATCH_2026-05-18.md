# TiSCH Shared Core — Patch zur Direktive vom 2026-05-18

**Gilt zusätzlich zu** `TISCH_SHARED_CORE_AUFTRAG_2026-05-18.md`.
**Quelle der Erweiterungen:** parallele Architektursession 2026-05-18, Handoff-Datei `~/Documents/Claude/Projects/MOONFiNGERS/HANDOFF_dual_shared_cores_2026-05-18.md`.
**Kopierziel:** `~/Documents/der-tisch/TISCH_SHARED_CORE_AUFTRAG_PATCH_2026-05-18.md`, neben die Hauptdirektive.

## Warum es diese Patch-Notiz gibt

Die Hauptdirektive beschreibt den TiSCH-Anteil eines größeren Dual-Core-Vertrags. Der Gesamtvertrag wurde in der Parallelsession ausverhandelt und enthält Festlegungen, die unsere Implementierung binden — sonst weicht der TiSCH-Pilot vom Vertrag ab, und die Bridge zwischen den beiden Cores funktioniert nicht.

## Konfliktregel

Wenn diese Patch-Notiz oder die Hauptdirektive im Widerspruch zum Handoff `HANDOFF_dual_shared_cores_2026-05-18.md` stehen, **gewinnt der Handoff**. Die TiSCH-Seite implementiert den App-spezifischen Ausschnitt eines übergreifenden Vertrags, nicht eine eigenständige Architektur.

## Architektur ist Dual-Core, zwei Backends

```
┌────────────────────────┐         ┌────────────────────────┐
│  moonfingers-backend   │◀──────▶│   der-tisch-backend     │
│  knowledge_shared_core │ Bridge  │   tisch_shared_core     │
│  (Weltwissen, Wiki,    │         │   (TiSCH-Runs, Chat-    │
│   Entities, Bilder)    │         │    Excerpts, KiNTEGRiTY,│
│                        │         │    Obsidian-Notes)      │
└────────────────────────┘         └────────────────────────┘
         ▲                                    ▲
         │                                    │
         └─── moonfingers_dual_core_bridge ◀──┘
              (knowledge-seitig, Default)
```

- Zwei getrennte Backends. Zwei Hosts. Kein gemeinsames Repo, kein gemeinsamer Prozess.
- Die Bridge lebt knowledge-seitig (im `moonfingers-backend/`), nicht in `der-tisch-backend/`.
- DER TiSCH und die anderen TiSCH-Apps reden direkt nur mit `tisch_shared_core`. Wenn sie Weltwissen brauchen, gehen sie über die Bridge — nicht direkt in `knowledge_shared_core`.

## ID-Konvention: URN-Form

Alle Cross-Core-IDs sind URN-prefixed:

- `knowledge_shared_core:David_Chalmers`
- `knowledge_shared_core:einsein_salvage_0042`
- `tisch_shared_core:run_2026_05_18_d3a1`
- `tisch_shared_core:chat_excerpt_2026_05_18_aaa`

Begründung: macht Provenance-Ketten eindeutig, verhindert versehentliche Dopplungen. Konsequenz: alle bestehenden flachen IDs müssen beim Import migriert werden (im MVP nur für neu erzeugte Records, Migration der Altdaten kommt später).

## MemoryCandidate / MemoryCard — Pflichtfelder erweitert

Die in der Hauptdirektive skizzierten Schemas sind **unvollständig**. Verbindlich für die Pydantic-Modelle in `der-tisch-backend/tisch_memory/schemas.py`:

```python
class MemoryCandidate(BaseModel):
    id: str                          # URN, z.B. tisch_shared_core:run_xxx
    core_id: Literal["tisch_shared_core"]
    family_line: list[str]           # z.B. ["der-tisch"]
    source_role: SourceRole          # siehe unten
    memory_layer: MemoryLayer        # siehe unten
    curation_state: CurationState    # siehe unten
    approved_for_reuse: bool = False
    canonical: bool = False
    canonical_approved_by: str | None = None
    canonical_approved_at: datetime | None = None
    visibility: Literal["private", "shared", "public"] = "private"
    provenance_chain: list[ProvenanceLink]
    reuse_state: ReuseState          # siehe unten
    created_at: datetime
    # Inhaltliche Felder
    title: str
    content: str
    tags: list[str] = []
    suggested_obsidian_path: str | None = None
```

### Kanonische `source_role`-Werte TiSCH-seitig

(aus der Genealogy-Mapping-Tabelle im Handoff — nicht erfinden, nur diese verwenden):

- `tisch_run_result` — Ergebnis eines TiSCH-Laufs, `memory_layer=project_memory`, `curation_state=candidate`
- `chat_excerpt` — Auszug aus Chat / Konversation, `memory_layer=personal_memory`, `curation_state=raw`
- `kintegrity_synthesis` — KiNTEGRiTY-verdichtete Antwort, `memory_layer=reusable_context`, `curation_state=synthesized`, `family_line` darf `["der-tisch", "moonfingers"]` sein
- `user_authored_note` — vom Nutzer geschriebene Obsidian-Note, `memory_layer=personal_memory`, `curation_state=curated`
- `curator_decision` — vom Nutzer freigegebene Entscheidung, `memory_layer=project_memory`, `curation_state=canonical`

### `curation_state`-Enum

```python
class CurationState(str, Enum):
    raw = "raw"
    model_generated = "model_generated"
    candidate = "candidate"
    synthesized = "synthesized"
    reviewed = "reviewed"
    curated = "curated"
    approved_for_reuse = "approved_for_reuse"
    canonical = "canonical"
```

`canonical` darf **niemals** automatisch gesetzt werden. Nur durch Nutzer-Freigabe via Obsidian-Frontmatter (`canonical: true` + `canonical_approved_by` + `canonical_approved_at`).

### `memory_layer`-Enum

```python
class MemoryLayer(str, Enum):
    personal_memory = "personal_memory"
    project_memory = "project_memory"
    reusable_context = "reusable_context"
    verified_knowledge = "verified_knowledge"  # kommt nur aus knowledge_shared_core
    poetic_surface = "poetic_surface"           # kommt nur aus knowledge_shared_core
```

### `reuse_state` und `provenance_chain`

```python
class ReuseState(BaseModel):
    used_in_context_packs: int = 0
    last_used_at: datetime | None = None
    moonfingers_use: list[Literal[
        "personal_overlay",
        "context_pack",
        "entity_dossier",
        "globe_node",
        "story_step",
        "fog_hint",
        "relation",
        "image",
    ]] = []

class ProvenanceLink(BaseModel):
    parent_id: str        # URN
    relation: Literal["derived_from", "synthesized_from", "approved_from", "imported_from"]
    at: datetime
```

## Runtime-Pfade: `/Volumes` ist tabu

Die Hauptdirektive hat das nicht hart genug gemacht. Verbindlich:

- `/Volumes/...` darf **nur** von lokalen Import-/Snapshot-Scripts gelesen werden.
- Die produktive Runtime (lokal wie Railway) darf nicht von `/Volumes` abhängen.
- TiSCH-Storage liegt unter `der-tisch-backend/data/tisch_shared_core/` (JSONL für MVP, später SQLite/Postgres).
- Wenn ein Script doch `/Volumes` braucht, ist es ein Import-Script und muss `import_only=True` in den Metadaten dokumentieren.

## Privacy-Default

- Default für alle Endpoints (TiSCH-Seite **und** Bridge): `include_private=false`.
- Antwort-Trennung in Response-Objekten ist verbindlich, auch wenn ein Block leer ist:

```
{
  "verified_dossiers": [...],
  "images": [...],
  "relations": [...],
  "stories": [...],
  "personal_notes": [...],
  "tisch_syntheses": [...],
  "context_cards": [...],
  "provenance": [...]
}
```

- `personal_notes`, `tisch_syntheses`, `context_cards` aus `tisch_shared_core` mit `visibility=private` werden nur gefüllt, wenn `include_private=true` **und** der Caller authentifiziert ist.

## Anschluss-Punkte für DER TiSCH

Wenn DER TiSCH als Pilot startet (Schritt 1: `schemas.py`), gilt zusätzlich:

1. Schemas exakt mit den oben genannten Pflichtfeldern. Nicht weglassen, nicht erfinden.
2. JSONL-Dateien append-only unter `der-tisch-backend/data/tisch_shared_core/`. Keine Datenbank im MVP.
3. Zwei Endpoints zuerst: `POST /api/tisch-memory/candidates`, `POST /api/tisch-memory/context-pack`. Keine Bridge im MVP — die kommt mit der MOONFiNGERS-Seite.
4. `/api/ask` (vorhandener Endpoint) bekommt Pre-Run-Hook (`context-pack` lesen) und Post-Run-Hook (`candidates` schreiben). Beide Hooks sind ein-/abschaltbar via Feature-Flag.
5. Obsidian-Export erst nachdem Schritte 1–4 stehen. Frontmatter exakt wie im Handoff Abschnitt 6.

## Was diese Patch-Notiz **nicht** ändert

- Pilot-Entscheidung bleibt DER TiSCH.
- Minimal-First-Regel bleibt: kein Big-Bang-Refactor, eine Pilot-App.
- Die Hauptdirektive bleibt gültig — diese Notiz erweitert, ersetzt nicht.
