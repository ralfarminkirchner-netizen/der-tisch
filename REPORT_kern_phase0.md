# REPORT · Kern Phase 0

Zweig: `tisch/kern`  
Basis: `origin/main` `ea75cd414977879e17306b83b1931db05cd62188`  
Arbeitsbaum: `/Users/ralfkirchner/Projects/der-tisch-kern`  
Kein Merge, kein Deploy, nichts ratifiziert.

## Auftrag

Key-Fragment aus `CODEX_HANDOFF.md` entfernen und Datei als veraltet markieren. kiNTEGRiTY/`synthesize` nur prüfen. pytest + vitest + Build-Gate grün.

## Schlüssel

In `CODEX_HANDOFF.md` lag ein OpenAI-Key-ähnliches Fragment (1 Treffer, Länge 16). Es ist in der Arbeitskopie entfernt. **Git-History dieses Repos behält es.** Key in der OpenAI-Konsole rotieren; neuen Key nur in gitignorierter `.env` / Railway-Env.

Andere Treffer `sk-…` in Tests (`build.test.ts`, `test_einstellungen.py`) sind künstliche Platzhalter für die Build-Prüfung, keine Produktionskeys.

## kiNTEGRiTY / synthesize — nur geprüft, nicht gefixt

| Stelle | Befund |
|---|---|
| `der-tisch-backend/kintegrity.py` | `import anthropic` und `client = anthropic.Anthropic(...)` auf Modulebene. Import scheitert ohne Paket; Client entsteht beim Import. |
| `der-tisch-backend/tisch_shared_core/kintegrity_synthesis.py` | `async def synthesize_candidate`; Doku verlangt `ANTHROPIC_API_KEY`. |
| Auftrag | nichts fixen — eingehalten. |

## Prüfpunkte (belegt)

| Gate | Kommando | Ergebnis |
|---|---|---|
| pytest | `xpertentisch/backend/.venv/bin/pytest -q` | **83 passed**, 2 deselected (`not live`), 20.05s |
| Build | `npm run build` in `xpertentisch/frontend` | **bestanden** (`tsc --noEmit && vite build && check-build.mjs`); „keine Zugangsdaten im Auslieferungsstand“ |
| vitest | `npm test` **nach** dem Build (braucht `dist/`) | **66 passed** / 5 files. Vorher 2 Failures: `dist/` fehlte — Reihenfolge Build → Test. |

## Tor Phase 0

**Grün.** Weiter Phase 1.

## Nicht in diesem Auftrag

Phase 3 Frontend/Parität, App, Capacitor, Moonfingers, Deploy, Querleser, Neuro-Map. `api_server.py` nicht erweitert.
