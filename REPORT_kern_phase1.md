# REPORT · Kern Phase 1

Zweig: `tisch/kern`  
Basis Phase 0: `421dc34`  
Kein Merge, kein Deploy. `api_server.py` unverändert (nur gelesen).

## Auftrag

`extract_catalog.py` aus 12 HTML + `api_server.py` → Katalog YAML.  
Rückprüfung: jedes extrahierte `position` byteidentisch in der Quelle.  
Pydantic-Schema. Lücken nur ausweisen.

## Zahlen (belegt)

| Größe | Zahl |
|---|---|
| HTML-Dateien (Liste) | 12/12 vorhanden |
| YAML `app/catalog/tische/<id>.yaml` | 12 |
| ArenaPair mit `position` | 23 (AST-`ArenaPair`-Aufrufe; weitere `position=`-Treffer in `api_server.py` sind andere Konstruktoren/Felder — Lücke) |
| AGENTS_DE / AGENTS_EN | 22 / 22 |
| `position` byteidentisch | **ok** (`verify_positions` leer) |
| pytest `tests/test_catalog.py` | **3 passed** |
| pytest gesamt | **86 passed**, 2 deselected |

## Lücken (nur ausgewiesen)

- `hub`, `der-tisch`, `expertentisch`, `literatentisch`: kein `const PERSPEKTIVEN`
- `teamtisch`, `integrationstisch`: Rollen aus `roleConfig` + Namensabgleich AGENTS_DE, nicht aus PERSPEKTIVEN-Array
- `familientisch`: 7 zusätzliche JS-`position:`-Strings ohne Zuordnung zu einer PERSPEKTIVEN-id
- AGENTS_DE-Prompts sind in der Quelle **konkatenierte** Literale — nicht als ein zusammenhängendes `position`-Bytefeld behandelt (sonst wäre Byteidentität falsch)

## Tor Phase 1

**Grün** für: 12 HTML, Schema valide, alle extrahierten `position`-Werte byteidentisch.  
Nicht 100 % aller HTML haben ein PERSPEKTIVEN-Array — das ist ausgewiesen, nicht erfunden.

## Artefakte

- `xpertentisch/backend/tools/extract_catalog.py`
- `xpertentisch/backend/app/catalog/` (Package; bisheriges `catalog.py` → `providers.py`, Re-Export in `__init__.py`)
- `xpertentisch/backend/app/catalog/tische/*.yaml`
