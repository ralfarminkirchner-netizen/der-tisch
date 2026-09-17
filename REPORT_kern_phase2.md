# REPORT · Kern Phase 2

Zweig: `tisch/kern`  
Kein Merge, kein Deploy. `api_server.py` unverändert.

## Endpunkte

| Methode | Pfad |
|---|---|
| GET | `/api/tische` |
| GET | `/api/tische/{id}` |
| POST | `/api/laeufe` |
| GET | `/api/laeufe/{id}` |
| POST | `/api/laeufe/{id}/cancel/{perspective_id}` |
| GET | `/api/sessions/{id}/events` (SSE, bestehend) |

Lauf: Perspektiven parallel → Reibung → Integration. Kein Siegel, kein Auto-canonical.  
Modell je Phase über `model_id` / `reibung_model_id` / `integration_model_id` (sonst erstes konfiguriertes Modell). Kein hartes `gpt-4o-mini` in `tisch_lauf.py`.

Systemprompts der Kern-Agenten: wörtlich `AGENTS_DE` aus dem Katalog (`_agenten.json`, Herkunft `api_server.py`).

Memory: `POST {TISCH_CORE_BASE_URL}/api/tisch-memory/candidates` nur bei gesetzter Env; sonst Warn-Log, no-op.

## Fake-Tests (belegt)

`pytest tests/test_tisch_lauf.py`: **5 passed** (1.95s)

- FAMiLiENTiSCH modus `frage` → `table.perspectives` / `friction` / `integration` (TableResponse-Felder)
- Abbruch einer Perspektive
- Reibung startet erst nach Perspektiven-Phase
- Schnappschuss (`snapshots`) je Perspektive
- kein `gpt-4o-mini` im Laufmodul

## Gesamtgate

| Gate | Ergebnis |
|---|---|
| pytest | **91 passed**, 2 deselected |
| vitest | **66 passed** |

## Tor Phase 2

**Grün.** Phase 3 (Frontend/Parität, App, Capacitor, Moonfingers, Deploy) nicht angefasst.
