# TiSCH Shared Core — Cowork-Auftrag

**Datum:** 2026-05-18
**Quelle:** Ralf, in Cowork-Session
**Status:** Direktive für nächste Implementierungsrunde — noch nicht umgesetzt
**Pilot-App (entschieden 2026-05-18):** **DER TiSCH** — höchster Memory-Hebel, meiste Entscheidungen, meiste Runs.
**Bezug:** Hängt zusammen mit der Dual-Shared-Core-Architektur (`knowledge_shared_core` + `tisch_shared_core`), die parallel in der Moonfingers-Recovery-Session entschieden wurde.

---

## Kernsatz

Die TiSCH-Apps sollen nicht einzeln "intelligenter" gemacht werden, sondern gemeinsam an den `tisch_shared_core` angeschlossen werden. Jede TiSCH-App wird dadurch gleichzeitig Memory-Konsument und Memory-Produzent.

**Die TiSCH-Apps sollen lernen, ohne sich selbst blind zu glauben.**

---

## Pipeline

```
Vor einem Durchlauf:
  TiSCH-App fragt nach relevantem Context Pack.

Nach einem Durchlauf:
  TiSCH-App reicht gute Ergebnisse als MemoryCandidate ein.

Danach:
  KiNTEGRiTY verdichtet.
  Curator Agent prüft.
  Obsidian bekommt kuratierbare Notes.
  Erst nach Review wird daraus stabile Wiederverwendung.
```

Harte Regel:
```
model_generated → candidate → synthesized → reviewed → approved_for_reuse → canonical (nur durch Nutzerfreigabe)
```

`model_generated` darf nie direkt `canonical` werden.

---

## Die TiSCH-Familie

- DER TiSCH
- TEAM TiSCH
- LiTERATEN TiSCH
- EXPERTiSEN TiSCH
- iNTEGRATiONS TiSCH
- weitere TiSCH-Apps / TiSCH-Durchläufe

Diese Apps hängen **nicht** direkt am `knowledge_shared_core` (SEiN-/Wiki-Familie). Sie hängen primär am eigenen `tisch_shared_core`.

| Core | Verantwortung |
|---|---|
| `knowledge_shared_core` | Weltwissen, Dossiers, Bilder, Entities, Relationen, Atlas, Wiki-/Monographie-Wissen |
| `tisch_shared_core` | Arbeitswissen, Projektwissen, Durchlauf-Ergebnisse, gute Antworten, Prompts, Entscheidungen, Obsidian-Notizen, KiNTEGRiTY-Synthesen, wiederverwendbare Context Packs |

---

## Minimale Schnittstelle (zwei Hooks pro App)

### A) Pre-run Context Hook

```
POST /api/tisch-memory/context-pack
```

Beispiel-Request:
```json
{
  "app_id": "literaten-tisch",
  "project": "moonfingers",
  "task": "frontend_polish",
  "query": "Welche stabilen Entscheidungen gelten für MOONFiNGERS Frontend und Dual-Core-Architektur?",
  "max_tokens": 1200,
  "include": [
    "canonical_decisions",
    "approved_for_reuse",
    "stable_answers",
    "project_memory",
    "prompts"
  ],
  "exclude": [
    "raw_chats",
    "unreviewed_candidates"
  ]
}
```

Die App nimmt nur den kompakten Context Pack in den Prompt — nicht alte Chatlogs oder ganze Dumps.

### B) Post-run Memory Candidate Hook

```
POST /api/tisch-memory/candidates
```

Beispiel-Request:
```json
{
  "origin_app": "literaten-tisch",
  "project": "moonfingers",
  "task": "frontend_polish",
  "content": "... Ergebnis / Antwort / Entscheidung ...",
  "input_summary": "... worum ging es? ...",
  "output_summary": "... was wurde entschieden oder erzeugt? ...",
  "proposed_tags": ["moonfingers", "frontend", "design", "tisch"],
  "source_class": "model_generated",
  "memory_layer": "project_memory",
  "curation_state": "candidate",
  "visibility": "private",
  "suggested_reuse": true,
  "suggested_obsidian_path": "20_Projects/MOONFiNGERS/"
}
```

---

## Pflicht-Metadaten für jeden MemoryCandidate

- `origin_app`
- `run_id`
- `project`
- `task`
- `input_summary`
- `output_summary`
- `content`
- `source_class`
- `model`/`provider`, falls bekannt
- `created_at`
- `proposed_tags`
- `memory_layer`
- `curation_state`
- `visibility`
- `suggested_reuse`
- `suggested_obsidian_path`
- `provenance_chain`

### Pflichtwerte

```
core_id:        tisch_shared_core
source_class:   model_generated | user_authored | runtime_artifact
source_role:    tisch_run_result | chat_excerpt | model_generated_synthesis | prompt | stable_answer
memory_layer:   project_memory | personal_memory | reusable_context | archive
curation_state: raw | candidate | synthesized | reviewed | curated | canonical | archived
visibility:     private | internal | tisch_only | moonfingers_only
```

---

## Privacy-Regel

TiSCH-Memory ist standardmäßig privat oder intern.
MOONFiNGERS darf TiSCH-/Obsidian-Daten nie standardmäßig öffentlich anzeigen.

```
Default:   include_private=false
Explizit:  include_private=true
```

---

## App-spezifische Rollen

| App | Fokus |
|---|---|
| DER TiSCH | allgemeiner Arbeits- und Entscheidungstisch — besonders viel Context-Pack + MemoryCandidate-Verkehr |
| TEAM TiSCH | Team-/Rollen-/Koordinationswissen, Übergaben, Zuständigkeiten |
| LiTERATEN TiSCH | Text, Sprache, Stil, Narrativ, Leitmetaphern, gute Formulierungen |
| EXPERTiSEN TiSCH | Fachlogik, Prüfung, Architektur, Risiken, technische Entscheidungen |
| iNTEGRATiONS TiSCH | Verbindung von Apps, APIs, Backends, Endpoints, Mapping-Logik |

MOONFiNGERS nutzt diese TiSCH-Spuren später als private Overlays / Context Cards / Denkspuren — aus `tisch_shared_core`, **nicht** aus `knowledge_shared_core`.

---

## Akzeptanzkriterien für ersten Ausbau (MVP)

1. Gemeinsames Datenmodell für `MemoryCandidate`, `MemoryCard`, `ContextPack`.
2. Pre-run Context-Pack-Endpunkt.
3. Post-run Candidate-Endpunkt.
4. Beispielanschluss für **eine** TiSCH-App.
5. Obsidian-Markdown-Export für `reviewed`/`curated` Cards.
6. Keine automatische Kanonisierung.
7. Bestehende TiSCH-App-Funktionalität bleibt intakt.

**Anti-Goal:** keine große Refaktorierung aller TiSCH-Apps. Erst minimale gemeinsame Memory-Schnittstelle bauen, dann App für App anschließen.

---

## Kurzform (für andere Sessions)

> Die TiSCH-Apps sollen als Familie an einen gemeinsamen `tisch_shared_core` angeschlossen werden. Jede App fragt vor einem Run einen kompakten Context Pack ab und reicht nach einem guten Run einen MemoryCandidate ein. KiNTEGRiTY und Curator verdichten daraus wiederverwendbare MemoryCards. Obsidian ist das menschliche Review-/Kurationsfenster. Keine TiSCH-App darf eigene Modellantworten automatisch canonical setzen. Ziel ist weniger Wiederholung, weniger Tokenverbrauch und stabilere Antworten über alle TiSCH-Apps hinweg.

---

## Operativer Schlüsselsatz

**Cowork soll nicht alle TiSCH-Apps neu bauen, sondern eine gemeinsame Memory-Schnittstelle definieren und dann zuerst eine App exemplarisch anschließen.**

---

## Nächste Session — Startpunkt

In dieser Reihenfolge:

1. **Datenmodell.** `der-tisch-backend/tisch_memory/schemas.py` mit Pydantic-Schemas für `MemoryCandidate`, `MemoryCard`, `ContextPack`. Pflichtfelder aus dem Abschnitt "Pflicht-Metadaten" oben, plus `id` (UUID), `created_at` (ISO-8601 UTC), `provenance_chain` (Liste).
2. **Storage.** JSONL append-only auf Disk:
   ```
   der-tisch-backend/data/tisch_shared_core/candidates.jsonl
   der-tisch-backend/data/tisch_shared_core/cards.jsonl
   ```
   Kein `/Volumes`-Pfad. Storage-Modul `tisch_memory/storage.py` mit `append()`, `read_all()`, `filter()`.
3. **Endpoints.**
   - `POST /api/tisch-memory/candidates` → schreibt nach `candidates.jsonl`
   - `POST /api/tisch-memory/context-pack` → liest `cards.jsonl`, Ranker = Tag-Match + Recency + Filter `curation_state >= reviewed`. Kein Embedding im MVP.
4. **Pilot-Anschluss DER TiSCH.** In `api_server.py` (oder im Frontend-Call zu `der-tisch.html`): vor jedem `/api/ask` ein Pre-Run-Context-Pack-Call, nach jedem erfolgreichen `/api/ask` ein Post-Run-Candidate-Call. Bestehende `/api/ask`-Funktionalität bleibt unverändert.
5. **Obsidian-Export.** Modul `tisch_memory/obsidian_writer.py`: nimmt eine `MemoryCard` mit `curation_state` in `{reviewed, curated}` und schreibt sie als `.md` mit Frontmatter unter `suggested_obsidian_path`. Pfad wird über Config (Env-Var `OBSIDIAN_VAULT_PATH`) aufgelöst.

**Was NICHT zu tun ist in der ersten Implementierungsrunde:**
- Keine anderen TiSCH-Apps anschließen (erst DER TiSCH stabil)
- Keine automatische Kanonisierung
- Keine Embedding/Vektor-Suche (Tag+Recency reicht)
- Keine UI für Curation (Obsidian ist das Review-Fenster)
- Keine `/Volumes`-Abhängigkeit

**Reentry-Prompt für nächste Session:**
```
Lies zuerst:
/Users/ralfkirchner/Documents/der-tisch/TISCH_SHARED_CORE_AUFTRAG_2026-05-18.md
Pilot-App ist DER TiSCH. Starte mit Schritt 1 (Pydantic-Schemas in der-tisch-backend/tisch_memory/schemas.py).
```

