# TiSCH Shared Core — Phasen-Report

Auftrag: `PROMPT_claude_code_tisch_core.md` · Vertrag: `HANDOFF_dual_shared_cores_2026-05-18.md`
Repo: `/Users/ralfkirchner/Documents/der-tisch/` · Branch `main`
Erstellt: 2026-05-19 (Datei trägt den Auftrags-Namen `PHASE1_REPORT_2026-05-18.md`).

---

## 1. Ausgeführte Phasen (0–4)

| Phase | Inhalt | Status |
|---|---|---|
| 0 | Survey: Pfad-Check, Git, Code-Lesung, KiNTEGRiTY-Ortung, SURVEY-Notiz | ✓ |
| 1 | Modelle + Persistenz (`models.py`, `store.py`, `capture.py`) | ✓ |
| 2 | KiNTEGRiTY-Synthese (gewickelt) + Curator mit canonical-Guard | ✓ |
| 3 | Stable-Answer-Index, Context Packs, Obsidian Export/Import | ✓ |
| 4 | API-Router + Integration in `api_server.py` | ✓ |

Alle Smoke-Tests grün (7 Modul-`--demo`-Tests + curl gegen alle 6 Endpoints).

**Persistenz-Hinweis:** Phase 0 hatte zunächst JSONL gewählt (vom Auftrag als
Fallback erlaubt, weil `aiosqlite` keine deklarierte Dependency war). Nach
ausdrücklicher Nutzer-Freigabe wurde `aiosqlite` in `requirements.txt`
aufgenommen und die Persistenz auf die Auftrags-Präferenz **SQLite via
aiosqlite** umgestellt (Commit `43f6035`). Die async dict-Schnittstelle von
`store.py` blieb dabei identisch — kein Caller-Modul war betroffen.

---

## 2. Neu angelegte Dateien

Alle unter `der-tisch-backend/tisch_shared_core/`:

| Datei | Zweck |
|---|---|
| `__init__.py` | Paket-Marker, Versions-/Core-ID-Konstanten |
| `.gitignore` | ignoriert `data/` (Laufzeit-DB, nicht versioniert) |
| `SURVEY_2026-05-18.md` | Phase-0-Survey |
| `models.py` | Pydantic-Modelle + Provenance-Protokoll (Enums, URN-IDs) |
| `store.py` | SQLite-Persistenz via aiosqlite (async dict-API) |
| `capture.py` | Raw Capture → `MemoryCandidate` |
| `kintegrity_synthesis.py` | KiNTEGRiTY gewickelt + lokaler TL;DR-Fallback |
| `curator.py` | Curator-Agent, canonical-Guard, Dedupe |
| `stable_answers.py` | Stable-Answer-Index/Lookup |
| `context_packs.py` | Context-Pack-Erzeugung |
| `obsidian_export.py` | `MemoryCard` → Markdown + Frontmatter |
| `obsidian_import.py` | Markdown + Frontmatter → `MemoryCard` |
| `api.py` | FastAPI-Router (6 Endpoints) + Standalone-Smoke-Server |
| `PHASE1_REPORT_2026-05-18.md` | dieser Report |

---

## 3. Geänderte Dateien (außerhalb `tisch_shared_core/`)

| Datei | Änderung | Autorisierung |
|---|---|---|
| `der-tisch-backend/api_server.py` | +4 Zeilen: Kommentar + `import` + `include_router` + Leerzeile (2 funktionale Zeilen). Keine andere Änderung. | Auftrag Phase 4 |
| `der-tisch-backend/requirements.txt` | +1 Zeile: `aiosqlite>=0.20.0` | Nutzer ausdrücklich freigegeben (latenter Deploy-Bug-Fix) |
| `HANDOFF_dual_shared_cores_2026-05-18.md`, `PROMPT_claude_code_tisch_core.md` (Repo-Root) | neu committet (waren untracked) | Nutzer ausdrücklich freigegeben (Doc-Commit) |

`api_server.py`-Diff (vollständig):

```diff
 client = OpenAI()
 
+# --- TiSCH Shared Core — Dual-Core-Memory-Router (HANDOFF 2026-05-18) ---
+from tisch_shared_core.api import router as tisch_shared_core_router
+app.include_router(tisch_shared_core_router, prefix="")
+
 NO_CACHE = {...}
```

**Nicht angefasst** (fremde, schon vor Arbeitsbeginn vorhandene
Working-Tree-Änderungen): `CODEX_HANDOFF.md`, 10 `*.html`-Dateien,
`der-tisch-backend/tisch-responsive.css`. Diese bleiben unverändert im
`git status`. Es wurde durchweg gezielt (`git add <datei>`) committet, nie
`git add -A`. `moonfingers_store.py` wurde nicht angefasst.

---

## 4. Commits (chronologisch)

| Hash | Message |
|---|---|
| `4705bac` | `docs(tisch-core): phase 0 survey of der-tisch-backend` |
| `7f58873` | `feat(tisch-core): models for candidate/card/contextpack + capture pipeline` |
| `e20f5ba` | `feat(tisch-core): kintegrity synthesis + curator with canonical guard` |
| `7cdc878` | `feat(tisch-core): stable-answer index, context packs, obsidian export/import` |
| `661cc65` | `docs: add dual-shared-cores handoff contract` |
| `8999a06` | `chore: add aiosqlite to requirements.txt` |
| `43f6035` | `refactor(tisch-core): back the store with aiosqlite (replaces JSONL)` |
| `5a5b4d1` | `feat(tisch-core): api router + integration into api_server` |

**Reihenfolge-Hinweis:** Die vom Nutzer skizzierte Ideal-Ordnung (docs →
survey → aiosqlite → Phasen) weicht von der tatsächlichen History ab, weil
Phase 0–4 bereits abgeschlossen waren, als die Doc-/aiosqlite-Anweisung kam.
Inhaltlich ist alles vorhanden. Die Commits sind **lokal** (nicht gepusht,
`origin/main` steht auf `c20e167`) — sie können bei Bedarf noch
umsortiert/gesquasht werden.

---

## 5. Beispiele

### 5.1 MemoryCandidate (POST /api/tisch-memory/candidates)

```json
{
  "id": "tisch_shared_core:mem_demo_f96286a5",
  "core_id": "tisch_shared_core",
  "family_line": ["der-tisch"],
  "source_role": "chat_excerpt",
  "memory_layer": "personal_memory",
  "curation_state": "raw",
  "visibility": "private",
  "moonfingers_use": ["personal_overlay", "context_pack"],
  "provenance_chain": [
    {"step": "raw_capture", "origin": "api",
     "timestamp": "2026-05-19T00:59:46.998328+00:00",
     "summary": "Captured as chat_excerpt (raw)"}
  ],
  "reuse_state": null,
  "created_at": "2026-05-19T00:59:46.998614+00:00",
  "updated_at": "2026-05-19T00:59:46.998618+00:00",
  "kind": "memory_candidate",
  "title": "Demo", "content": "Hello", "source_app": "",
  "content_hash": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
  "raw_payload": {}, "notes": ""
}
```

### 5.2 MemoryCard (nach Curator → `reviewed`)

```json
{
  "id": "tisch_shared_core:card_demo_16b4f1a7",
  "core_id": "tisch_shared_core",
  "family_line": ["der-tisch"],
  "source_role": "chat_excerpt",
  "memory_layer": "personal_memory",
  "curation_state": "reviewed",
  "visibility": "private",
  "moonfingers_use": ["personal_overlay", "context_pack"],
  "provenance_chain": [
    {"step": "raw_capture", "origin": "api", "timestamp": "...", "summary": "..."},
    {"step": "curator_review", "origin": "curator_agent", "timestamp": "...",
     "summary": "Autonom kuriert nach reviewed; reuse_state=approved_for_reuse; canonical bleibt gesperrt (Nutzer-Tor)."}
  ],
  "reuse_state": "approved_for_reuse",
  "kind": "memory_card",
  "title": "Demo", "content": "Hello", "summary": "Hello",
  "tags": [], "candidate_id": "tisch_shared_core:mem_demo_f96286a5",
  "canonical": false, "canonical_approved_by": null, "canonical_approved_at": null,
  "synthesis_meta": {}
}
```

Nach expliziter Promotion (`POST /curate {target_state:canonical, approved_by:ralf}`):
`curation_state: "canonical"`, `canonical: true`, `canonical_approved_by: "ralf"`,
`canonical_approved_at` gesetzt, zusätzlicher `canonical_approval`-Provenance-Schritt.

### 5.3 ContextPack (POST /api/tisch-memory/context-pack)

```json
{
  "id": "tisch_shared_core:pack_hello_fc3b9133",
  "core_id": "tisch_shared_core",
  "source_role": "model_generated_synthesis",
  "memory_layer": "reusable_context",
  "curation_state": "synthesized",
  "moonfingers_use": ["context_pack"],
  "provenance_chain": [
    {"step": "context_pack_build", "origin": "context_packs",
     "summary": "1 stabile Card(s), ~1/500 Tokens, Query: Hello"}
  ],
  "kind": "context_pack",
  "title": "Context Pack — Hello",
  "task_or_question": "Hello",
  "entries": [
    {"card_id": "tisch_shared_core:card_demo_16b4f1a7", "title": "Demo",
     "content": "Hello", "source_role": "chat_excerpt",
     "memory_layer": "personal_memory", "curation_state": "reviewed",
     "reuse_state": "approved_for_reuse", "provenance_chain": [ ... ],
     "relevance_score": 1.0, "token_estimate": 1}
  ],
  "token_estimate": 1, "max_tokens": 500
}
```

### 5.4 Obsidian-Frontmatter-Export

```markdown
---
id: tisch_shared_core:card_obsidian_export_demo_787c18d3
type: memory_card
core_id: tisch_shared_core
family_line:
  - dein-sein
  - der-tisch
source_role: user_authored_note
memory_layer: personal_memory
curation_state: reviewed
approved_for_reuse: true
reuse_state: approved_for_reuse
canonical: false
canonical_approved_by: null
canonical_approved_at: null
visibility: private
moonfingers_use:
  - personal_overlay
tags: []
candidate_id: tisch_shared_core:mem_obsidian_export_demo_3e12b094
content_hash: 4e2fdadb474450e4ea24934a12f6fb3dbce85fdbdc8dddbfba6c7850ceebb749
created_at: 2026-05-18T22:26:56.136326+00:00
updated_at: 2026-05-18T22:26:56.136348+00:00
---

# Obsidian-Export-Demo

<!-- tisch:content:start -->
Obsidian ist die menschliche, kuratierte Wissensoberfläche.
<!-- tisch:content:end -->

## Zusammenfassung
...
## Provenance-Kette
- **raw_capture** · obsidian_demo · ...
- **curator_review** · curator_agent · ...
```

Für echte Kanonisierung setzt der Nutzer im Frontmatter `canonical: true` +
`canonical_approved_by: <name>`; `obsidian_import.py` schließt damit das
Approval-Tor (`curation_state` wird auf `canonical` gezogen).

---

## 6. Smoke-Test-Output (curl, alle 6 Endpoints)

Standalone-Router-Server: `python -m tisch_shared_core.api --port 8000`.

| # | Endpoint | Aufruf | Ergebnis |
|---|---|---|---|
| 1 | `POST /api/tisch-memory/candidates` | `{"title":"Demo","content":"Hello","source_role":"chat_excerpt","family_line":["der-tisch"]}` | **HTTP 201**, `id: tisch_shared_core:mem_demo_f96286a5` |
| 2 | `POST /api/tisch-memory/curate` | `{"target_state":"reviewed","candidate_id":"…mem_demo_f96286a5"}` | HTTP 200, Card `…card_demo_16b4f1a7`, `curation_state=reviewed`, `reuse_state=approved_for_reuse` |
| 3 | `GET /api/tisch-memory/cards/{card_id}` | `…/cards/tisch_shared_core:card_demo_16b4f1a7` | HTTP 200, Card zurückgegeben |
| 4 | `GET /api/tisch-memory/search` | `?q=Hello&top_k=5` | HTTP 200, 1 `StableAnswerHit`, `relevance_score` gesetzt |
| 5 | `POST /api/tisch-memory/context-pack` | `{"task_or_question":"Hello","top_k":5,"max_tokens":500}` | HTTP 200, Pack mit 1 Eintrag, Provenance erhalten |
| 6 | `GET /api/tisch-memory/obsidian/export` | `?card_id=…card_demo_16b4f1a7` | HTTP 200, `vault_configured=false`, `written=false`, `count=1` |

Zusatz-Checks am `curate`-Endpoint (canonical-Tor):

| Aufruf | Ergebnis |
|---|---|
| `POST /curate {target_state:canonical, card_id:…, approved_by:"ralf"}` | HTTP 200, `canonical=true`, `curation_state=canonical`, `canonical_approved_by=ralf` |
| `POST /curate {target_state:canonical, card_id:…}` (ohne `approved_by`) | **HTTP 400** — „Promotion zu 'canonical' erfordert ein nicht-leeres approved_by" |

Die 7 Modul-Smoke-Tests (`python -m tisch_shared_core.<modul> --demo`) für
`capture`, `kintegrity_synthesis`, `curator`, `stable_answers`,
`context_packs`, `obsidian_export`, `obsidian_import` liefen ebenfalls grün
(nach dem aiosqlite-Umbau erneut verifiziert).

---

## 7. Akzeptanzkriterien (TiSCH-Seite)

- ✓ TiSCH-Antworten laufen `Candidate → Synthesis → Curator → reviewed/curated_draft → (User-Gate) → canonical`.
- ✓ Jeder Record trägt das volle Provenance-Protokoll; IDs in URN-Form.
- ✓ `approved_for_reuse` funktioniert als Wiederverwendungs-Flag ohne `canonical`-Promotion (eigenständige Felder, Konsistenz-Validator).
- ✓ Modellgenerierte Inhalte werden nie autonom `canonical` (CanonicalGuardError; Promotion nur mit `approved_by`).
- ✓ Obsidian-Export = Markdown + Frontmatter, re-importierbar (Round-Trip verifiziert).
- ✓ Keine `/Volumes`-/Obsidian-Pfade in produktiver Runtime (`OBSIDIAN_VAULT_PATH` optional, DB modul-relativ).
- ✓ Bestehende `der-tisch`-Funktionalität unangetastet (nur die eine `include_router`-Einbindung).
- ✓ Keine direkte DB-/FS-Kopplung zum Knowledge Core.

---

## 8. Offene Risiken

1. **Railway-Filesystem ist ephemer.** Die SQLite-`.db` (wie zuvor JSONL,
   wie das `.db`-Muster der Altmodule) überlebt keinen Redeploy. Für MVP
   akzeptiert. Echte Dauerhaftigkeit braucht ein Railway-Volume oder Postgres.
2. **`kintegrity.py` ist nicht railway-sicher.** Es importiert `anthropic`
   auf Modulebene; `anthropic` ist **nicht** in `requirements.txt`. Der
   Wrapper `kintegrity_synthesis.py` fängt das defensiv ab (lokaler
   TL;DR-Fallback) — der TiSCH-Core bleibt lauffähig, aber echte KiNTEGRiTY-
   Synthese läuft auf Railway erst, wenn `anthropic` deklariert und
   `ANTHROPIC_API_KEY` gesetzt ist. Bewusst NICHT eigenmächtig geändert
   (Auftrag: nur die eine `include_router`-Zeile + freigegebene Punkte).
3. **`api_server.py` lädt `OpenAI()` auf Modulebene** (Zeile 17) — ohne
   `OPENAI_API_KEY` schlägt schon der Import fehl. Dadurch konnte die
   *voll integrierte* App lokal nicht gebootet werden; der Router wurde über
   den Standalone-Server (`python -m tisch_shared_core.api`) verifiziert, die
   Integrationszeile ist importgeprüft. Pre-existing, nicht TiSCH-Core-Sache.
4. **`find()` lädt je Aufruf alle Records** und filtert in Python. Für MVP-
   Volumen unkritisch; bei Wachstum auf indizierte SQL-Queries umstellen.
5. **Dedupe nutzt Token-Jaccard** (Schwelle 0.92), keine semantische
   Ähnlichkeit — bewusst dependency-frei. Embeddings später nachrüstbar.
6. **`OBSIDIAN_VAULT_PATH`-Schreiben ist ungetestet gegen einen echten
   Vault** (lokal nicht gesetzt). Render-/Parse-Round-Trip ist verifiziert;
   das tatsächliche Schreiben/Lesen im Vault sollte einmal manuell geprüft
   werden.

---

## 9. Offene API-Fragen für die Bridge / Knowledge-Seite

Der Vertrag wurde NICHT eigenmächtig geändert. Folgende Punkte zur
Nutzer-Entscheidung:

1. **`curated_draft` vs. `curated`.** Die Vertrags-Tabelle „Provenance-
   Protokoll" listet `curated` als `curation_state`; die Vertrags-Prosa
   (Kurations-Pipeline + Phase-2-Auftrag) spricht von `curated_draft` als
   autonom erreichbarem Zustand. `models.py` nimmt **beide** Werte auf, der
   Curator kann autonom `reviewed` oder `curated_draft` setzen. Falls die
   Bridge gegen die Tabellen-Liste validiert, schlägt `curated_draft` fehl.
   → Vertrag sollte einen der beiden Werte kanonisieren.
2. **`source_role` für Context Packs.** Die `source_role`-Liste hat keinen
   Wert für ein Aggregations-Artefakt. `ContextPack` nutzt
   `model_generated_synthesis`. Falls die Bridge Context Packs gesondert
   erkennen will, fehlt ggf. ein Wert wie `context_pack` in der Liste
   (analog zu `moonfingers_use: context_pack`).
3. **ID-Form an der Bridge.** Alle IDs sind URN-Form
   (`tisch_shared_core:<short_id>`). `GET /api/tisch-memory/cards/{card_id}`
   akzeptiert URN-Form *und* Kurz-ID (Prefix wird ggf. ergänzt). Der Doppel-
   punkt im Pfad funktioniert (Test #3) — die Bridge kann URN-IDs direkt
   durchreichen.
4. **`GET /search` liefert `StableAnswerHit`-Objekte** (`{card, relevance_score,
   is_canonical, is_stable}`), nicht nackte Cards. Default = nur stabile
   Cards; `?include_all=true` sucht über alle. Die Bridge sollte die
   Hit-Hülle kennen.
5. **`GET /obsidian/export`** liefert `{vault_configured, written, count,
   exports[]}`; `write` ist per Default `false` (GET ohne Seiteneffekt),
   `?write=true` schreibt zusätzlich in den Vault (No-op ohne Vault).

---

## 10. Wie es weitergeht (Vorschlag, nicht ausgeführt)

- `anthropic` deklarieren + `ANTHROPIC_API_KEY` setzen → echte KiNTEGRiTY-
  Synthese statt TL;DR-Fallback.
- Railway-Volume oder Postgres für dauerhafte Persistenz.
- Einmaliger manueller Vault-Schreib-/Lese-Test mit gesetztem
  `OBSIDIAN_VAULT_PATH`.
- Vertrags-Klärung zu Punkt 9.1 / 9.2 durch den Nutzer.
