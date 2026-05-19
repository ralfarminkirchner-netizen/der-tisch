# Auftrag für Claude Code — TiSCH Shared Core in `der-tisch-backend/`

Du arbeitest im Repo `/Users/ralfkirchner/Documents/der-tisch/`. Du baust die TiSCH-Seite des Dual-Core-Modells. Eine zweite Instanz (Codex) arbeitet parallel im völlig anderen Repo `/Users/ralfkirchner/moonfingers-recovery/` am Knowledge Shared Core und an der MOONFiNGERS-Bridge. Diesen zweiten Pfad fasst du NICHT an.

---

## Kanonischer Vertrag

Lies zuerst vollständig:

```
/Users/ralfkirchner/Documents/Claude/Projects/MOONFiNGERS/HANDOFF_dual_shared_cores_2026-05-18.md
```

Bei jedem Konflikt zwischen Vertrag und Codebasis: Vertrag gewinnt für neue Module unter `tisch_shared_core/`. Bestehender Code in `api_server.py`, `moonfingers_store.py` etc. bleibt unangetastet — siehe „Verboten".

---

## Phase 0 — Survey (vor jedem Schreibvorgang)

1. Pfad-Check:
   ```bash
   test -d /Users/ralfkirchner/Documents/der-tisch/der-tisch-backend
   ```
   Falls Pfad fehlt:
   ```bash
   find /Users/ralfkirchner/Documents -maxdepth 4 -type d -name der-tisch 2>/dev/null
   find /Users/ralfkirchner/Documents/GitHub -maxdepth 4 -type d -name der-tisch 2>/dev/null
   ```
   Wenn nichts gefunden: **stoppen**, als Blocker melden, nicht woanders scaffolden, kein neues Repo anlegen.
2. `git status` und `git log -3 --oneline` in `der-tisch`.
3. Lies vollständig (read-only):
   - `der-tisch-backend/api_server.py` (besonders MOONFiNGERS-Block ab Zeile ~2109)
   - `der-tisch-backend/moonfingers_store.py`
   - `der-tisch-backend/requirements.txt`
   - `der-tisch-backend/Procfile`
   - `der-tisch-backend/Dockerfile` (falls vorhanden)
4. Suche im ganzen Repo nach `kintegrity`:
   ```bash
   grep -rn "kintegrity" /Users/ralfkirchner/Documents/der-tisch --include="*.py" 2>/dev/null
   ```
   Wenn KiNTEGRiTY-Code im Repo existiert, wickelst du ihn später ein. Du schreibst ihn nicht neu.
5. Survey-Notiz: `der-tisch-backend/tisch_shared_core/SURVEY_2026-05-18.md` mit:
   - bestehende FastAPI-Struktur (wie sind Router organisiert)
   - bestehende Persistenz (SQLite/aiosqlite, Pattern)
   - bestehende Pydantic-Pattern (Modelle, Validators)
   - existierende KiNTEGRiTY-Stellen
   - alles, was sich seit dem letzten Codex-Audit verändert haben könnte

**Stoppen** und Rückfrage stellen, falls:
- der `der-tisch`-Pfad nicht da ist
- Railway-Service-Konfiguration im Repo abweicht vom bekannten Stand (`d532fcfe`)
- ein paralleler Branch oder uncommitted Changes auf TiSCH-Code zeigen
- die KiNTEGRiTY-Implementierung an mehreren Stellen liegt und die Wahl unklar ist

---

## Phase 1 — Modelle und Persistenz

Anlegen unter `der-tisch-backend/tisch_shared_core/`:

- `models.py` — Pydantic-Modelle für `MemoryCandidate`, `MemoryCard`, `ContextPack`, `ObsidianNoteExport`. Alle tragen das volle Provenance-Protokoll: `core_id` (fix `tisch_shared_core`), `family_line`, `source_role`, `memory_layer`, `curation_state`, `visibility`, `moonfingers_use`, `provenance_chain`. Optional: `reuse_state`. IDs in URN-Form `tisch_shared_core:<short_id>`.
- `capture.py` — nimmt rohe TiSCH-Run-Outputs / Chat-Excerpts entgegen, erzeugt `MemoryCandidate` mit `curation_state: raw` oder `candidate`. Persistierung: bevorzugt SQLite via aiosqlite, im Stil von `moonfingers_store.py`. Wenn das aus Survey-Sicht unpassend wirkt, JSONL-Fallback mit klarer Begründung in einem Inline-Kommentar.

Smoke-Test: `pytest` oder ein kleines `python -m tisch_shared_core.capture --demo` erzeugt einen Demo-Candidate und liest ihn zurück.

Commit: `feat(tisch-core): models for candidate/card/contextpack + capture pipeline`.

---

## Phase 2 — KiNTEGRiTY-Synthese und Curator

- `kintegrity_synthesis.py` — wenn KiNTEGRiTY im Repo existiert (siehe Phase 0), importiere es und wickle es. Wenn nicht, schreibe einen Stub mit der korrekten Signatur, der eine einfache TL;DR-Verdichtung macht und `source_role: kintegrity_synthesis`, `memory_layer: reusable_context` setzt.
- `curator.py` — Curator-Agent. Harte Regeln:
  - Eingang: `MemoryCandidate`s mit `curation_state` in {`raw`, `candidate`, `synthesized`}.
  - Erlaubte autonome Übergänge: bis maximal `reviewed` oder `curated_draft`. Darf `reuse_state: approved_for_reuse` setzen.
  - **Verboten autonom: `canonical`.** Übergang nur über expliziten API-Call (`POST /api/tisch-memory/curate` mit Body `{ target_state: "canonical", approved_by: "<user>" }`) ODER über Obsidian-Frontmatter-Sync, der `canonical: true` mit `canonical_approved_by`-Feld zurückspielt.
  - Dedupe: Hash + (optional, falls vorhanden) semantic similarity gegen vorhandene `MemoryCard`s.

Smoke-Test: ein Candidate durch die Pipeline → Curator stoppt bei `reviewed` → expliziter API-Call setzt `canonical`.

Commit: `feat(tisch-core): kintegrity synthesis + curator with canonical guard`.

---

## Phase 3 — Stable Answers, Context Packs, Obsidian-Export

- `stable_answers.py` — Index/Lookup über `MemoryCard`s mit `reuse_state: approved_for_reuse` ODER `curation_state: canonical`. Methode: `find_stable_answers(query, top_k)`.
- `context_packs.py` — `build_context_pack(task_or_question, top_k, max_tokens)` → kompakter Pack aus relevanten Karten, mit Provenance erhalten. Token-Budget grob respektieren.
- `obsidian_export.py` — schreibt eine `MemoryCard` als Markdown mit Frontmatter (Schema siehe Vertrag Punkt 6) in `OBSIDIAN_VAULT_PATH` (Env-Var). Wenn die Env-Var fehlt: no-op mit Warn-Log, **nicht** Fehler werfen. Das ist die Railway-Sicherheits-Regel: kein produktiver Pfad darf an einem lokalen Vault hängen.
- `obsidian_import.py` — liest Markdown-mit-Frontmatter aus dem Vault zurück, mappt auf `MemoryCard` oder updated bestehende Karten. Erkennt Frontmatter-Änderungen: insbesondere `canonical: true`/`canonical_approved_by` → schließt das Approval-Tor (siehe Phase 2).

Commit: `feat(tisch-core): stable-answer index, context packs, obsidian export/import`.

---

## Phase 4 — API

- `api.py` — FastAPI-Router mit:
  ```
  POST /api/tisch-memory/candidates
  GET  /api/tisch-memory/search
  POST /api/tisch-memory/context-pack
  GET  /api/tisch-memory/cards/{card_id}
  POST /api/tisch-memory/curate
  GET  /api/tisch-memory/obsidian/export
  ```
- Einbindung in `api_server.py` ausschließlich über **eine** neue Zeile:
  ```python
  app.include_router(tisch_shared_core_router, prefix="")
  ```
  Keine anderen Änderungen an `api_server.py`.
- Smoke-Test mit `curl`:
  ```
  curl -X POST http://localhost:8000/api/tisch-memory/candidates \
    -H "Content-Type: application/json" \
    -d '{"title":"Demo","content":"Hello","source_role":"chat_excerpt","family_line":["der-tisch"]}'
  ```
  Erwartet: 201 + URN-ID.

Commit: `feat(tisch-core): api router + integration into api_server`.

---

## Verboten

- Kein Anfassen von `/Users/ralfkirchner/moonfingers-recovery/`. Das ist Codex' Gebiet.
- Kein Anfassen von `der-tisch-backend/moonfingers_store.py`.
- Kein Anfassen von `der-tisch-backend/api_server.py` außer der **einen** `include_router`-Zeile.
- Kein Auto-Promote zu `canonical` ohne expliziten User-Identity-Flag oder Frontmatter-Bestätigung.
- Keine Railway-Runtime-Abhängigkeit auf lokale Pfade. `OBSIDIAN_VAULT_PATH` muss optional sein.
- Keine direkte DB- oder Filesystem-Kopplung zum Knowledge Core. Wenn du Knowledge-Daten brauchst, holst du sie später über HTTP — die Bridge ruft DICH auf, nicht umgekehrt.
- Kein neues Repo.
- Kein TypeScript-Backend.

---

## Koordination mit der Knowledge-Instanz (Codex)

Codex implementiert die Bridge und ruft deine Endpoints. Berührungspunkt ist die API-Form aus Vertrag-Sektion „API-Gruppen A".

Wenn dir auffällt:
- die Vertrags-API ist semantisch zu eng für was du in `MemoryCard` modellieren musst
- der Bridge-Side braucht ein Feld, das in den Endpoints fehlt
- die Provenance-Form passt nicht zum, was Codex erwartet

→ dokumentiere das in deinem Phasen-Report unter „Offene API-Fragen für die Bridge". Du änderst NICHT eigenmächtig den Vertrag. Der Nutzer entscheidet.

---

## Bericht am Ende

`der-tisch-backend/tisch_shared_core/PHASE1_REPORT_2026-05-18.md` mit:

- ausgeführte Phasen (0–4) mit Status (✓ / ⚠ / ✗)
- Liste aller neu angelegten Dateien
- Liste aller geänderten Dateien (sollte nur `api_server.py` + 1 Zeile sein)
- Liste aller Commits mit Hash und Message
- Beispiele: `MemoryCandidate`, `MemoryCard`, `ContextPack`, Obsidian-Frontmatter-Export
- Smoke-Test-Output (curl-Antworten) für alle 6 Endpoints
- Offene Risiken
- Offene API-Fragen für die Bridge / Knowledge-Seite

Pro Phase committen. Am Ende nicht alles in einem Mega-Commit.
