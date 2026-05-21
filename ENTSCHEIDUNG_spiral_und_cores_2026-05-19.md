# Entscheidungs-Dokument: SPiRAL MiND WRiTER ↔ die Shared Cores

**Stand:** 2026-05-19 · **Status:** Entscheidungsvorlage (von Ralf zu entscheiden)
**Erstellt mit:** Claude Code · **Grundlage:** read-only Review von `spiral_mind_writer`
(README, Sprint-0-Spec, Schema, alle 24 ADRs, Inspirations-Doku, CLAUDE.md)

> An `spiral_mind_writer` wurde **nichts verändert** — dieses Dokument ist reine Analyse.

---

## 1 · Worum es geht

Ralf baut mehrere Apps, die „fast alle auf die gleichen Dinge zurückgreifen" —
Leitprinzip: **modular, ein geteilter Kern, alle Apps docken an.**

Parallel existiert `spiral_mind_writer` (SPiRAL MiND WRiTER) — eine eigenständige
App, die aus LLM-Dialogen/Notizen ein Werk formt. Frage: **Wie kompatibel ist sie
mit den Shared Cores, und was müsste angepasst werden?**

Beteiligte Bausteine:
- **TiSCH-Core** (`der-tisch-backend/tisch_shared_core/`) — Antwort-/Wissensgedächtnis
- **Memory-Core** (`der-tisch-backend/memory_shared_core/`) — Notiz-/Chat-Gedächtnis, 1.122 Notizen
- **SPiRAL MiND WRiTER** (`/Users/ralfkirchner/Projects/spiral_mind_writer/`) — Sprint 0 fertig, 215 Tests grün

---

## 2 · Befund 1 — dieselbe Idee, dreimal unabhängig gebaut

| Konzept | TiSCH-Core | Memory-Core | SPiRAL MiND WRiTER |
|---|---|---|---|
| Wissens-Einheit | `MemoryCard` | `MemoryNote` | `IdeaCardRecord` |
| Roh-Eingang | `MemoryCandidate` | importierte Notiz | `SourceRecord` / `PassageRecord` |
| Herkunft / Provenance | `provenance_chain` | `source_path` | `ProvenanceLinkRecord` |
| Schutz vor Glättung | „nicht glätten" (Auftrag) | — | `do_not_smooth` (harter Status) |
| Reifegrad / Kuration | `curation_state`-Enum | — | `IdeaStatus` + `maturity` 0–5 |
| Synthese | KiNTEGRiTY / `ContextPack` | — | `SynthesisDraftRecord` |
| Speicher | JSONL → aiosqlite | sqlite | YAML pro Projekt |
| ID-Form | `tisch_shared_core:…` | `memory_shared_core:note_…` | `idea_…`, `src_…`, `prov_…` |

**Konzeptionell ~90 % deckungsgleich. Technisch 0 % verbunden.** Getrennte Repos,
Schemas, ID-Systeme, Speicher, Stacks — aktuell kann kein einziger Datensatz
zwischen ihnen wandern.

Nebenbefund: SPiRAL und der TiSCH-Core haben **denselben Ursprung** — die
KI-Tisch-/kiNtegrity-Mechanik (Multi-LLM-Cross-Review). Deshalb die große
Überlappung. Sie sind Geschwister mit gemeinsamem Elternteil.

---

## 3 · Befund 2 — der eigentliche Konflikt (kein Code-Problem)

Zwei **gegensätzliche Architektur-Philosophien**, in getrennten Sessions
entschieden, nie zusammengeführt:

- **SPiRAL sagt** (wörtlich in README + CLAUDE.md): *„Schwester-Apps. Kein
  Code-Lifting. Patterns spiegeln, nicht migrieren. Cross-Pollination, nicht
  Migration."* → jede App ist eine **eigene Insel**.
- **Ralfs Kernsatz:** *„meine Apps greifen fast alle auf die gleichen Dinge
  zurück — geteilter Kern."* → **ein Kernel, viele dünne Apps.**

Das ist der Punkt, der „angepasst" werden muss — **nicht Code, sondern die
Grundsatz-Entscheidung.** SPiRAL wurde bewusst als Insel gebaut; das widerspricht
dem Shared-Kernel-Prinzip.

---

## 4 · Die Optionen

### Option A — Insel (SPiRALs aktuelles Design)
SPiRAL bleibt komplett eigenständig. Cores bleiben getrennt. Austausch nur über
Konzepte/Patterns.
- ➕ kein Aufwand, SPiRAL ist schon so gebaut; jede App entwickelt sich frei
- ➖ Provenance/Cards/Anti-Glättung leben in 3 Codebasen, driften auseinander;
  kein gemeinsames Gedächtnis; SPiRAL ist *nicht* Teil des Shared-Kernel-Plans

### Option B — Kern-Client (empfohlen)
SPiRAL behält seine **gesamte interne Logik** (der redaktionelle Pipeline-Wert:
Passages → Cards → Cluster → Tensions → Synthese → Voice → Provenance — das ist
SPiRALs Eigenwert). Aber **Eingang und Ausgang** docken am Shared Core an.
- ➕ realisiert den Shared-Kernel; SPiRAL nutzt die 1.122 Memory-Core-Notizen;
  Synthesen werden für andere Apps sichtbar
- ➖ etwas Adapter-Arbeit; biegt SPiRALs „keine Integration"-Prinzip leicht

---

## 5 · Empfehlung

1. **SPiRALs Schema NICHT anfassen.** 215 Tests, harte Provenance-Gates, mitten
   in Phase E. Ein Umbau auf die Core-Schemas wäre zerstörerisch und gegen
   SPiRALs Design.
2. **SPiRAL Phase E normal fertig machen lassen** — in einer Sitzung, die *in*
   `spiral_mind_writer` läuft, nicht aus einer Core-Sitzung heraus.
3. **Danach bewusst entscheiden: Option A oder B** — als Entscheidung, nicht
   durch Drift.
4. **Empfehlung: Option B, aber nur an zwei sauberen Nähten** — additiv, kein
   Umbau, jederzeit rückbaubar.

---

## 6 · Die zwei Nähte (Option B konkret)

**Naht REIN — Memory-Core → SPiRAL `SourceRecord`:**
Die 1.122 Memory-Core-Notizen sind faktisch ein Stapel `llm_response` /
`chat_transcript`-Quellen. Ein Adapter `spiral source add --from memory-core`
lässt SPiRAL aus dem geteilten Kern trinken statt aus losen `.txt`-Dateien.
Klein, additiv, rührt SPiRALs Kern nicht an.

**Naht RAUS — SPiRAL-Synthese → Core `MemoryCard`:**
Ein fertiger `SynthesisDraftRecord` mit voller Provenance ist konzeptionell eine
`MemoryCard` mit `curation_state=curated`. SPiRALs kuratierte Ergebnisse können
in einen Core zurückfließen, damit andere Apps sie sehen.

**Gut:** SPiRALs eigenes Vokabular sieht das **schon vor** —
`InspirationCandidateRecord.targetModule` kennt bereits `"memory"`, `"table"`,
`"synthesis"`. Die Architektur hat die Andockpunkte bereits benannt.

---

## 7 · Was ausdrücklich NICHT zu tun ist

- ❌ SPiRALs Schema auf die Core-Schemas umbauen
- ❌ SPiRAL auf aiosqlite/JSONL zwingen (es ist bewusst YAML-first)
- ❌ IDs vereinheitlichen / eine gemeinsame DB erzwingen
- ❌ aus einer Core-Sitzung heraus in `spiral_mind_writer` schreiben
- ❌ `ki_tisch_mvp` oder das GEDANKENSPIRALE-Repo verändern (read-only)

---

## 8 · Offene Entscheidung für Ralf

> **Soll SPiRAL MiND WRiTER eine eigenständige Insel bleiben (A) — oder ein
> Client des geteilten Kerns werden (B, an den zwei Nähten)?**

Diese Entscheidung erst nach Phase E treffen. Vorher ändert sich nichts.

---

## 9 · Status

- Diese Analyse: **read-only abgeschlossen.** Keine Datei in `spiral_mind_writer`
  verändert.
- Phase E von SPiRAL: gehört in eine eigene, in `spiral_mind_writer` laufende
  Sitzung — unberührt von diesem Dokument.
- Nächster Bezug: ergänzt `FAHRPLAN_Wissenssystem_2026-05-19.md` (dort Phase 3 ff.).
