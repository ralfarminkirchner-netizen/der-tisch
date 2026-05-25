# MOONFiNGERS / Wissens-System — System-Sync-Briefing (alle Workspaces)

Stand: 2026-05-26 · **Löst das frühere `SYSTEM_SYNC_BRIEFING_2026-05-19.md` ab.**

> **Zweck.** Jede Instanz in jedem Workspace liest **zuerst dieses Briefing**,
> dann den kanonischen Vertrag `HANDOFF_dual_shared_cores_2026-05-18.md`, dann
> das eigene Workspace-`SURVEY`/`REPORT`. Ergänzend: der `FAHRPLAN_Wissenssystem_
> 2026-05-19.md` (Phasen-Plan) und `ENTSCHEIDUNG_spiral_und_cores_2026-05-19.md`
> (SPiRAL-Frage).

---

## 1. Das große Bild (Stand 2026-05-26)

Ralfs Leitprinzip: **modular · ein geteilter Kern, alle Apps docken an ·
Frontend und Backend immer getrennt.** Der FAHRPLAN benennt das Gesamtbild:

```
   WAS gespeichert wird   →  die 3 Cores (TiSCH · Memory · Wissen)
   WIE es verknüpft ist   →  Obsidian — der Wissensgraph
   WIE es aussieht        →  MiNDLAXY — die visuelle Galaxie
   WERKZEUG für Code      →  Graphify — Code-Graph
   DENK-MECHANISMUS       →  der TiSCH — mehrere LLMs, mehrere Runden
   WOFÜR das Ganze        →  gemeinsames Gedächtnis für Claude, Codex & Co.
```

Zwei Graphen, nicht verwechseln: **Wissens-Graph** (Notizen/Gedanken/Cores +
Obsidian) vs. **Code-Graph** (Programm-Projekte → Graphify).

---

## 2. Workspace-Landkarte

| Workspace | Pfad | Git-Remote | Rolle |
|---|---|---|---|
| **der-tisch** | `/Users/ralfkirchner/Documents/der-tisch/` | `ralfarminkirchner-netizen/der-tisch` | TiSCH-Backend; hostet `tisch_shared_core`, `memory_shared_core` (lokal, noch nicht deployed), Personen-Bibliothek, 11 Tische inkl. WiRTSCHAFT |
| **moonfingers-recovery** | `/Users/ralfkirchner/moonfingers-recovery/` | `ralfarminkirchner-netizen/moonfingers` | MOONFiNGERS-App; `knowledge_shared_core` (= Wissens-Core), Bridge, aktiv weiterentwickelter 3D-Globus/Atlas |
| **spiral_mind_writer** *(neu in der Landkarte)* | `/Users/ralfkirchner/Projects/spiral_mind_writer/` | (eigenes Repo) | LLM-Dialoge → Werk-Synthese, Sprint 0 fertig, 215 Tests grün, Phase E in Arbeit. **Entscheidung Option A/B nach Phase E.** Read-only aus Core-Sessions. |
| **shared-core** | `/Volumes/ThunderBolt4_2TB/MeineApps/shared-core/` | `kiNTEGRiTY/shared-core` | Sprachneutraler Wissens-Datenstamm (40 Personen, 7 Brücken, 23 Praktiken) — Quelle für `knowledge_shared_core`-Importer und für die Personen-Bibliothek |
| **knowledge-core** | `/Volumes/ThunderBolt4_2TB/MeineApps/knowledge-core/` | (kein Top-Git) | Zwei-Core-Konzept (Core A Obsidian-Vault + Core B Node-Engine) — soll laut FAHRPLAN-Logik unter den Wissens-Core (MOONFiNGERS) fallen, nicht mehr eigenständig wachsen |
| **shared-memory** | `/Volumes/ThunderBolt4_2TB/MeineApps/shared-memory/` | (kein Top-Git) | Legacy Node-API (Port 4788) — wird durch den Wissens-Core abgelöst |
| **mindlaxy** | `/Volumes/ThunderBolt4_2TB/MeineApps/mindlaxy/` | `kiNTEGRiTY/mindlaxy` | Familien-Umbrella; `core/shared-memory.ts` ≈ 98 Records → Knowledge-Core-Import (Vertrags-Phase 3b, offen) |
| **moonfingers (scratch)** | `/Volumes/ThunderBolt4_2TB/MeineApps/moonfingers/` | — | Claude-Code-Steuer-Checkout — treibt Implementierung, keine eigene App |

Ignorieren: `MeineApps/der-tisch-main/` (veraltete ZIP-Kopie).

---

## 3. Aktueller Stand (was läuft, was nicht)

### der-tisch — TiSCH-Core + Bibliothek live, Memory-Core noch Geist
- **TiSCH-Core** ✅ live auf Railway. `/api/health` 200, `/api/tisch-memory/*` antwortet
  mit eigenem Router (404 mit deutscher Fehlermeldung als Lebenszeichen). aiosqlite.
  Persistenz-Pfad via `TISCH_CORE_DB_PATH` überschreibbar.
- **Personen-Bibliothek** ✅ live. `/api/bibliothek` 200, 40 Personen / 11 Disziplinen.
  Page `/bibliothek` serviert.
- **11 Tische** ✅ alle live, inkl. **`/wirtschaftstisch.html`** (`774e0ee`),
  `tisch-hub.html` 200 mit Wirtschaft-Link 3× (Nav + Card + Footer) und korrekter
  Zählung „Elf"/„11".
- **CORS** ✅ verifiziert mit `Origin`-Header → `access-control-allow-origin: *`.
- **`memory_shared_core/`** ⚠️ **Realitäts-Gap.** Der FAHRPLAN sagt
  „ans Backend angebunden". Tatsächlich: Verzeichnis im Working-Tree **untracked**,
  api_server.py importiert es weder lokal noch auf origin/main, Railway antwortet
  auf `/api/memory/stats` mit HTTP 404. Die 1.122 Notizen liegen in einer
  uncommittetten lokalen SQLite. → **Entscheidung nötig** (s. §8).
- **anthropic** in `requirements.txt` ✅, `OPENAI_API_KEY` auf Railway gesetzt ✅,
  `ANTHROPIC_API_KEY` Status unbekannt (für echte KiNTEGRiTY-Synthese nötig).

### moonfingers-recovery — Wissens-Core + Bridge + aktiver Atlas
- **`knowledge_shared_core`** ✅ Phasen 2–3, ~9.6k Records (758 docs / 6856
  chunks / 1017 entities / 1010 relations). Provenance-Enums **wertgleich**
  zum TiSCH-Core (kein `curated_draft`, mit `context_pack`).
- **`moonfingers_dual_core_bridge`** ✅ Phase 4. `TISCH_CORE_BASE_URL` per
  Env-Var konfigurierbar (Default `http://localhost:8001`).
- **Atlas/Globe** 🟢 aktiv (letzte 8 Codex-Commits seit 2026-05-20):
  Esri-Imagery-Textur, Tag-/Nacht-Modi, vereintes Shading, progressive
  Tree-Connections, mobile Navigation. `https://ralfarminkirchner-netizen.github.io/moonfingers/`
  zeigt 1017 Entities / 1010 Relations / 414 Traditionen / 514 Thinker / 81
  Teachings / 8 Questions.
- Frontend ruft Backend nur unter `/api/kintegrity/moonfingers` — die neuen
  `/api/tisch-memory/*`, `/api/bibliothek*`, `/api/moonfingers/*` (Bridge) sind
  noch nicht im Atlas verdrahtet.

### spiral_mind_writer — Insel, Phase E in Arbeit
Sprint 0 fertig, 215 Tests grün, eigenes YAML-first-Schema (IdeaCardRecord,
SourceRecord, PassageRecord, SynthesisDraftRecord, ProvenanceLinkRecord),
konzeptionell ~90 % Überlappung mit TiSCH-Core, **technisch 0 % verbunden**.
ENTSCHEIDUNG-Dokument hält fest: **kein Code-Lifting jetzt**; nach Phase E
Option A (Insel) vs. Option B (Kern-Client an zwei Nähten: Memory-Core →
`SourceRecord` rein, `SynthesisDraftRecord` → Core-`MemoryCard` raus).

---

## 4. Geteilte Verträge (verbindlich für ALLE Workspaces)

1. **Kanonischer Vertrag:** `HANDOFF_dual_shared_cores_2026-05-18.md`. Liegt in
   der-tisch/ und moonfingers-recovery/. Bei Konflikten: Vertrag gewinnt.
2. **Provenance-Protokoll** — jeder persistierte Record jedes Cores trägt:
   `core_id`, `family_line`, `source_role`, `memory_layer`, `curation_state`,
   `visibility`, `moonfingers_use`, `provenance_chain`, optional `reuse_state`.
   Enum-**Werte** über alle Cores identisch.
   - **Geklärt (2026-05-20):** `curated_draft` ist **entfernt** (autonomer
     Curator-Endzustand = `reviewed`); `context_pack` ist als `source_role`
     **hinzugekommen** (für ContextPack-Aggregate). Beide Cores aktualisiert.
3. **IDs in URN-Form** `<core_id>:<short_id>` — Außenform an allen API-Grenzen.
4. **Föderation nur über HTTP.** Bridge ruft TiSCH-Core nur über die A-API,
   nicht über DB/FS-Kopplung. Cores importieren nicht den Code des anderen.
5. **Keine `/Volumes`-Laufzeitabhängigkeit.** Produktion (Railway) muss ohne
   externe Platte laufen. `/Volumes` ist nur Build-Zeit-Importquelle.
6. **`include_private=false`** ist Default für alle MOONFiNGERS-Endpunkte.
7. **Modellgenerierte Inhalte werden nie automatisch `canonical`.** Nur über
   Nutzer-Tor (explizit / Obsidian-Frontmatter).
8. **Scoped Commits** — `git add <datei>`, nie `-A`/`.`. Fremde uncommitted
   Änderungen bleiben unangetastet.
9. **Verträge nicht eigenmächtig ändern** — fehlende Felder/Endpunkte als
   „offene Frage" im eigenen Report melden, der Nutzer entscheidet.
10. **`spiral_mind_writer` ist read-only** aus Core-Sessions. Phase E dort
    gehört in eine eigene Session **in** `spiral_mind_writer`.

---

## 5. Bekannte Spannungen — Stand der offenen Punkte

| Punkt | Stand |
|---|---|
| **§5.A** — TiSCH-Core vs. Memory-Core: weiche Zuständigkeitsnaht für Chat-Ausschnitte | ⏳ noch informell. Solange Memory-Core nicht deployed ist, hat das Zeit. |
| **§5.B** — 4-fache Knowledge-Fragmentierung | ✅ **geklärt** im FAHRPLAN: 3-Core-Modell (TiSCH · Memory · Wissen); Wissens-Core lebt im MOONFiNGERS-Projekt; `knowledge-core` und `shared-memory` sind nicht mehr eigenständige Wachstumspfade. |
| **§5.C** — Obsidian-Vault-Wiring | 🟡 teilweise — TiSCH-Core hat den Export-/Import-Weg, `OBSIDIAN_VAULT_PATH` ist aber nirgends dauerhaft gesetzt. Memory-Core-Export ist **Phase 1** im FAHRPLAN, „startklar". |
| **§5.D** — Schema-Divergenz (visibility, content_hash, id-Form) zwischen TiSCH- und Memory-Core | ⏳ offen, akut erst wenn Memory-Core integriert ist. |
| **§5.E** — `memory_shared_core` Commit-Status | ⚠️ **Realitäts-Gap.** FAHRPLAN sagt „angebunden", Repo sagt „untracked". Höchste Priorität (§8.1). |
| **§5.F** — `curated_draft` vs. `curated` | ✅ **gelöst** (entfernt, beide Cores synced). |
| **§5.G** — Bridge ↔ TiSCH-Core-Port | 🟡 Default `localhost:8001`; für Deploy `TISCH_CORE_BASE_URL=https://der-tisch-production.up.railway.app` setzen. |
| **NEU** — SPiRAL MiND WRiTER vs. Cores | ⏳ Entscheidung Option A (Insel) / B (Kern-Client) nach SPiRALs Phase E. ENTSCHEIDUNG-Dok empfiehlt B an zwei Nähten. |

---

## 6. Bauregeln (modulares System)

- Bleib im Workspace. Andere Workspaces nur lesen oder per HTTP rufen.
- Provenance-Enums wertgleich halten (siehe §4.2).
- Föderation per HTTP, keine DB-/FS-Kopplung.
- Pro Phase / pro Konzern committen, scoped, mit Phasen-Report.
- Vor jedem Schreibvorgang: dieses Briefing + Vertrag + eigenes Survey gelesen?

---

## 7. Pro-Workspace-Auftrag (kurz, ausgerichtet auf FAHRPLAN)

- **der-tisch (FAHRPLAN-Phase 1 betrifft hier am meisten):** zuerst die
  Memory-Core-Realitäts-Lücke schließen (§8.1) — dann Memory-Core → Obsidian-
  Export als nächster Build-Schritt.
- **moonfingers-recovery:** Atlas-/Globe-Polish läuft. Offen aus dem
  Codex-PHASE1_REPORT: Mindlaxy-Export (3b), Media-Katalog, leere Manifeste,
  Tests, Legacy-Routen-Aufräumen. Bridge gegen den **echten** Railway-TiSCH-Core
  verifizieren (`TISCH_CORE_BASE_URL` setzen).
- **spiral_mind_writer:** Phase E in eigener Session abschließen. Cores
  unberührt lassen.
- **shared-core:** stabil halten — nur über das Export-Skript pflegen.
- **knowledge-core / shared-memory:** keine Neuentwicklung; sie sollen in den
  Wissens-Core (MOONFiNGERS) aufgehen.
- **mindlaxy:** `core/shared-memory.ts` → Export-Script → JSONL für den
  Knowledge-Core-Importer `mindlaxy_snapshot.py`.

---

## 8. Offene Entscheidungen — nur Ralf

1. **Memory-Core: deploy oder downgrade?** Aktuell: 1.122 Notizen lokal,
   nichts committet, nichts auf Railway. Entweder
   (a) `memory_shared_core/` committen, api_server.py-Wiring zurück, deployen
       (Bridge/Frontend können dann lesen),
   (b) FAHRPLAN-Wortwahl von „angebunden" auf „lokal vorbereitet" runterstufen
       und den Deploy bewusst aufschieben.
2. **TiSCH ↔ Memory-Zuständigkeit (§5.A):** klare Regel, was ein „TiSCH-Lauf"
   ist und in welchen Topf Chat/Notiz/KI-Antwort gehört.
3. **Obsidian-Vault (§5.C):** Welcher Vault? `OBSIDIAN_VAULT_PATH` wo gesetzt?
   `knowledge-core/core-a/entries/tisch-memory/` als Subordner, oder eigener
   Vault?
4. **SPiRAL MiND WRiTER Option A/B** (siehe ENTSCHEIDUNG-Dokument) — erst
   nach SPiRALs Phase E.
5. **`ANTHROPIC_API_KEY` auf Railway** setzen für echte KiNTEGRiTY-Synthese
   (sonst weiter lokaler TL;DR-Fallback).
6. **`TISCH_CORE_BASE_URL` für die Bridge** auf die Railway-URL setzen (sobald
   Bridge irgendwo deployed wird).

---

## 9. Confidence

Erstellt aus direkten Git-/Curl-/Filesystem-Checks und den Dokumenten
`ENTSCHEIDUNG_spiral_und_cores_2026-05-19.md` + `FAHRPLAN_Wissenssystem_
2026-05-19.md`. Sicher: TiSCH-/Bibliothek-/Tische-Live-Status, moonfingers-
recovery-Commits, Memory-Core-Realitäts-Gap. Übernommen aus den Dokumenten:
SPiRAL-Stand, die 1.122-Notizen-Zahl, MiNDLAXY/Graphify-Roadmap.
