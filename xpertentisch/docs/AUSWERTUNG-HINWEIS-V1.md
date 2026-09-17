# Auswertung Hinweis-v1 — Auftragsgrenzen

**Branch:** `honey/xpertentisch-auswertung-v1`  
**Basis:** PR #66 / `claude/hermes-linsen` @ `1b91d7c`  
**Kein Merge, kein Deployment, keine Produktabnahme.**

## Was geändert wurde

1. **Epistemik:** Begriffsüberschneidung = Themenbezug (Hinweis), kein Einigkeitsbeweis. Kein Gegensatz erkannt ≠ Übereinstimmung. Kein Treffer ≠ Einzigartigkeit. Labels in UI, Graph, Bericht und Marker-Notes angepasst; CSS-Klassen/Farben bleiben (Legende zieht mit).
2. **Folgen:** Ähnlichkeitsketten ohne gemeinsame Aussage → Themencluster mit Einzelaussagen und Zählung **Clusterbeteiligung**.
3. **Versionierte Läufe:** Tabelle `analysis_runs`; `assessments` zeigt den aktuellen Lauf; neue Läufe überschreiben keine alten.
4. **Zahlen:** Modul `zahlen.py` — Extraktion inkl. Wortzahlen, vorsichtige Vergleiche, keine stillen Trennzeichen-Annahmen; hängt an Summary.
5. **Eingabe-Fingerprints:** `request_fingerprints` beim Kontext-Freeze (shared vs. anbieterbezogen), ohne Zugangsdaten.
6. **Aussagen / Positionen:** Module `aussagen.py`, `positionen.py` — Belegprüfung und Protokollhilfen; KI-Zerlegung nur bei ausdrücklichem Aufruf und anderem Modell (sonst „nicht ausgeführt“).

## Bewusst offen / teilweise

- Eigene Zahlen-Linse in der UI (Daten in Summary, noch keine eigene Linsenfläche).
- Expliziter API-Trigger für Aussagenzerlegung an die Oberfläche.
- Gegenseitige-Lektüre-Durchgang als voller Sitzungsablauf.
- Ansichtsbelege (Screenshots) für die neuen Labels — nicht neu erzeugt.

## Prüfkommandos

```bash
cd xpertentisch/backend && .venv/bin/python -m pytest -q
cd xpertentisch/frontend && npm test -- --run
```
