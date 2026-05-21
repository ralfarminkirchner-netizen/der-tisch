# Fahrplan — Ralfs Wissenssystem

**Stand:** 2026-05-19 · erstellt mit Claude Code

Dieser Fahrplan hält die Gesamt-Vision fest — damit nichts verloren geht und
jeder Schritt für sich klar ist.

**Leitprinzip (von Ralf):** modular · Frontend und Backend immer getrennt ·
ein geteilter Kern, auf den (fast) alle Apps zugreifen. Kein Baustein ersetzt
einen anderen — jeder macht *eine* Sache.

---

## Das Gesamtbild

```
   WAS gespeichert wird   →  die 3 Cores (TiSCH · Memory · Wissen)
   WIE es verknüpft ist   →  Obsidian — der Wissensgraph
   WIE es aussieht        →  MiNDLAXY — die visuelle Galaxie
   WERKZEUG für Code      →  Graphify — Code-Graph (die Landkarte)
   DENK-MECHANISMUS       →  der TiSCH — mehrere LLMs, mehrere Runden
   WOFÜR das Ganze        →  gemeinsames Gedächtnis für Claude, Codex & Co.
```

Zwei getrennte Graphen, nicht verwechseln:
- **Wissens-Graph** = Notizen, Gedanken, KI-Gespräche → die Cores + Obsidian
- **Code-Graph** = die Programm-Projekte → Graphify

---

## Die Bausteine (Module)

| Modul | Zweck | Status |
|---|---|---|
| **TiSCH-Core** | Mehr-Perspektiven-Denkmaschine + Antwort-Gedächtnis | ✅ gebaut |
| **Memory-Core** | Notiz-/Chat-Gedächtnis — 1.122 iCloud-Notizen drin | ✅ gebaut & ans Backend angebunden |
| **Wissens-Core** | Weltwissen / Nachschlagewissen | ⬜ offen (Projekt MOONFiNGERS) |
| **Obsidian-Brücke** | Cores → Obsidian-Notizen (Wissensgraph, menschliche Sicht) | 🟡 TiSCH ja · Memory noch nicht |
| **Coding-TiSCH** ("Spiral Mind Writer") | TiSCH-Mechanismus auf Code: mehrere LLMs lösen gemeinsam | ⬜ offen |
| **Graphify-Anbindung** | Code → Graph (Landkarte), spart Tokens, füttert den Coding-TiSCH | ⬜ offen (erst prüfen) |
| **MiNDLAXY** | visuelle Galaxie über dem Wissensgraphen | ⬜ offen |
| **Agenten-Gedächtnis** | Cores als Gedächtnis für Claude / Codex (über die API) | 🟡 Grundlage da — APIs existieren |

---

## Die Phasen (Reihenfolge + Abhängigkeiten)

### Phase 0 — Fundament ✅ ERLEDIGT
- TiSCH-Core gebaut (Mehr-Perspektiven-Engine + Memory-API).
- Memory-Core gebaut, ans Backend angebunden, **1.122 iCloud-Notizen importiert**
  (1.063 sichtbar · 59 privat · 56 Passwort-Notizen bewusst ausgeschlossen).

### Phase 1 — Memory sichtbar machen  ← *nächster Schritt*
- Memory-Core → **Obsidian-Export**: die 1.122 Notizen werden verlinkte
  Obsidian-Notizen → der erste echte, durchblätterbare Wissensgraph.
- Hängt ab von: Phase 0 ✅. **Startklar.** Aufwand: klein.

### Phase 2 — Memory-Core erweitern (optional)
- Weitere Quellen einspeisen, falls gewünscht (z.B. `spiral-os/memory`).
- Schritt für Schritt, sortiert — nie als Müllhalde.

### Phase 3 — Wissens-Core
- Der 3. Topf: Weltwissen / Nachschlagewissen.
- Lebt laut Architektur im MOONFiNGERS-Projekt (`knowledge_shared_core`).

### Phase 4 — Coding-TiSCH ("Spiral Mind Writer")
- Den vorhandenen TiSCH-Mechanismus (Perspektiven → Reibung → Integration)
  **auf Code anwenden**: mehrere LLMs, mehrere Runden, gemeinsame Lösung.
- Hängt ab von: TiSCH-Core ✅. Wird durch Phase 5 deutlich stärker.

### Phase 5 — Graphify-Anbindung
- **Zuerst klären:** Ist Graphify ein echtes, installierbares Tool? Wie dockt
  es an (CLI? MCP?)? — bevor irgendetwas gebaut wird.
- Dann: Code-Graph als Kontext-Lieferant für den Coding-TiSCH (Token-Ersparnis).

### Phase 6 — MiNDLAXY
- Die visuelle Galaxie über dem Wissensgraphen.
- Hängt ab von: Phase 1 (der Graph muss erst gefüllt sein).

### Querschnitt — Agenten-Gedächtnis
- Läuft nebenher. Die Core-APIs existieren bereits — Claude/Codex *können den
  Memory-Core schon abfragen*. Wächst mit jeder Phase mit.

---

## Offen / ehrlich angemerkt

- **Graphify:** bisher nur aus einer Video-Beschreibung bekannt — vor Phase 5
  prüfen, ob/wie es real einsetzbar ist.
- **TiSCH-Core:** der „Automatik-Schalter" (jede `/api/ask`-Antwort wandert
  automatisch ins Gedächtnis) ist noch nicht eingebaut.
- **Festplatte:** war voll (257 MB frei); Cache geleert → 6 GB frei. Ein
  Mac-Neustart gibt weitere ~18 GB frei und stabilisiert iCloud.
- **Reihenfolge ist ein Vorschlag**, kein Gesetz. Phase 1 trägt aber am meisten
  und ist der natürliche nächste Schritt.

---

## Kurzfassung

> Fundament steht (TiSCH-Core + Memory-Core mit 1.122 Notizen). Nächster Schritt:
> Memory-Core nach Obsidian exportieren → sichtbarer Wissensgraph. Danach:
> Wissens-Core, dann der Coding-TiSCH, dann Graphify als dessen Landkarte, zum
> Schluss MiNDLAXY als visuelle Galaxie. Alles modular, ein geteilter Kern.
