# TiSCH Familie — Claude Cowork Onboarding
> Letzter Update: 2026-04-08 · Commit `3addab7` · 11 Apps live · **Shared Core v1.0 aktiv** · Betreuer: Claude Cowork

---

## Was ist das?
Die **TiSCH Familie** ist ein Portfolio aus 11 KI-gestützten Multiperspektiv-Analyse-Apps auf einem gemeinsamen Railway-Backend. Alle Apps sind in purem HTML/CSS/JS gebaut — kein Framework, kein Build-Step.

---

## GitHub Repo
**https://github.com/ralfarminkirchner-netizen/der-tisch**
- Root Directory im Repo: `der-tisch-backend/`
- Hauptdateien: alle `.html` Apps + `api_server.py` + `kintegrity.py` + `notizbuch.js` + `wiki-tooltip.js`

---

## Live URLs — alle 11 Apps

| App | URL | Akzentfarbe |
|-----|-----|-------------|
| **TiSCH HUB** | https://der-tisch-production.up.railway.app/tisch-hub.html | `#C9A84C` Gold |
| **TEAM TiSCH** | https://der-tisch-production.up.railway.app/ | `#C9A84C` Gold |
| **DER TiSCH** | https://der-tisch-production.up.railway.app/der-tisch.html | `#E8614A` Coral |
| **iNTEGRATiONS TiSCH** | https://der-tisch-production.up.railway.app/integrationstisch.html | `#3A6F73` Petrol |
| **LiTERATUR TiSCH** | https://der-tisch-production.up.railway.app/literatentisch.html | `#C87941` Espresso |
| **EXPERTiSEN TiSCH** | https://der-tisch-production.up.railway.app/expertentisch.html | `#2C5F8A` Stahlblau |
| **TRAiNiNGS TiSCH** | https://der-tisch-production.up.railway.app/trainingstisch.html | `#2ECC5A` Grasgrün |
| **COACHiNG TiSCH** | https://der-tisch-production.up.railway.app/coachingtisch.html | `#7C5CBF` Violett |
| **MEDiZiN TiSCH** | https://der-tisch-production.up.railway.app/medizintisch.html | `#4ECDC4` Mint |
| **JURiS TiSCH** | https://der-tisch-production.up.railway.app/juristisch.html | `#C9A84C` Gold |
| **FAMiLiEN TiSCH** | https://der-tisch-production.up.railway.app/familientisch.html | `#C4622D` Terrakotta |

---

## Railway Deployment

| Parameter | Wert |
|-----------|------|
| Project | "clever-gentleness" |
| Project ID | `f2c619f2-9c4d-4365-aeb5-856d01b13bb6` |
| Service ID | `f75d3760-41bb-4f09-bcef-89a17a6e40e5` |
| Environment ID | `902072f7-0ea1-41b3-9724-7e6d81f3dcfd` |
| Railway Token | `d7ef0453-6a1a-4940-9dc6-8fe977d46f13` |
| Port | `$PORT=8080` |
| Root Dir | `der-tisch-backend` |
| Anthropic API Key | In Railway Environment Variables gespeichert |

### Deploy-Befehl (nach jeder Änderung)
```bash
cd /tmp/der-tisch-deploy  # git clone des Repos
git config user.email "deploy@computer.ai"
git config user.name "Computer Deploy"
cp /pfad/zur/datei.html der-tisch-backend/
git add der-tisch-backend/ && git commit -m "Beschreibung"
git push origin main

# Railway Redeploy triggern:
curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Authorization: Bearer d7ef0453-6a1a-4940-9dc6-8fe977d46f13" \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { serviceInstanceRedeploy(environmentId:\"902072f7-0ea1-41b3-9724-7e6d81f3dcfd\", serviceId:\"f75d3760-41bb-4f09-bcef-89a17a6e40e5\") }"}'

# Health Check:
sleep 90 && curl https://der-tisch-production.up.railway.app/api/health
```

---

## Architektur

```
der-tisch-backend/
├── api_server.py          # FastAPI Backend — alle Routes, Endpoints, KI-Calls
├── kintegrity.py          # kiNTEGRiTY Modul — Integritätsprüfung, Profile je App
├── notizbuch.js           # Shared NOTiZBUCH — in alle 10 Apps eingebunden
├── wiki-tooltip.js        # Wikipedia Hover-Tooltips — in EXPERTiSEN eingebunden
├── tisch-hub.html         # Zentrale Landingpage aller Apps
├── index.html             # TEAM TiSCH (~7860 Zeilen)
├── der-tisch.html         # DER TiSCH
├── integrationstisch.html # iNTEGRATiONS TiSCH
├── literatentisch.html    # LiTERATUR TiSCH (Playfair Display, Sepia)
├── expertentisch.html     # EXPERTiSEN TiSCH (89 Disziplinen, Wiki-Tooltips)
├── trainingstisch.html    # TRAiNiNGS TiSCH (Rasen-BG, Sport-Silhouetten)
├── coachingtisch.html     # COACHiNG TiSCH (12 Perspektiven, 5 Modi)
├── medizintisch.html      # MEDiZiN TiSCH (Portrait-SVGs, Haftungshinweis)
├── juristisch.html        # JURiS TiSCH (Sockel-T Logo, 6 Rechtsbereiche)
├── familientisch.html     # FAMiLiEN TiSCH (Rundtisch-Sockel-T + 3D Aufstellung)
├── Procfile               # web: uvicorn api_server:app --host 0.0.0.0 --port $PORT
├── nixpacks.toml          # Python 3.12, pip install
├── requirements.txt       # fastapi, uvicorn, anthropic, pydantic
└── CLAUDE_COWORK_ONBOARDING.md  # diese Datei
```

---

## Backend Endpoints

| Endpoint | Methode | Zweck |
|----------|---------|-------|
| `/api/health` | GET | Health Check → `{"status":"ok","version":"7.0","shared_core":"active"}` |
| `/api/ask` | POST | Multiperspektiv-Analyse (alle Apps) + Auto-Save |
| `/api/ask-simple` | POST | DER TiSCH Einzelfrage + Auto-Save |
| `/api/ask-table` | POST | Custom Perspektiven + Auto-Save |
| `/api/ask-clarify` | POST | Klärungsgespräch |
| `/api/translate` | POST | Register/Ton-Übersetzung |
| `/api/kintegrity/synthesize` | POST | kiNTEGRiTY Synthese |
| `/wiki-tooltip.js` | GET | Shared Wikipedia-Tooltip JS |
| `/notizbuch.js` | GET | Shared NOTiZBUCH JS |
| `/<app>.html` | GET | App-Seiten |
| `/api/sessions` | GET | Sessions abrufen `?key=SHARED_CORE_KEY&app=DER-TiSCH` |
| `/api/sessions/export` | GET | Vollexport für Vault-Sync `?key=…&since=ISO` |
| `/api/sessions/patterns` | GET | myCEL-Muster abrufen `?key=…` |
| `/api/hooks/mycel/patterns` | GET/POST | myCEL Pattern-Store lesen/schreiben (aktiv) |

### `/api/ask` Payload
```json
{
  "question": "string (min 10 Zeichen)",
  "perspectives": ["Perspektive 1", "Perspektive 2"],
  "stil": "akademisch | therapeutisch | juristisch | alltag | ...",
  "register": "" | "fachsprache" | "einfach",
  "tone": "" | "achtsam" | "direkt",
  "lang": "de" | "en",
  "app": "teamtisch | expertentisch | ..."
}
```

---

## Design-System (NIEMALS ändern)

### Logo-Prinzip
`TiSCH` — das **T** ist immer grafisch als Tisch dargestellt (SVG). Das **i** ist immer in der App-Akzentfarbe. `SCH` normal weight.

**Logo-Typen je App:**
- **Doppel-T** (zwei Wörter mit T): EXPERTiSEN, LiTERATUR, TRAiNiNGS, COACHiNG, MEDiZiN — beide T-Querbalken berühren sich und bilden eine gemeinsame Tischplatte
- **Sockel-T** (ein T, Rundtisch von der Seite): JURiS TiSCH, FAMiLiEN TiSCH — einzelnes Mittelbein + breite Fußplatte
- **SVG-Tisch** (vollständig gezeichnet): TEAM TiSCH, DER TiSCH, iNTEGRATiONS TiSCH

### Schrift
- **Inter** (300/400/500/600/700) — alle Apps
- **Playfair Display** — nur LiTERATUR TiSCH

### Akzentfarben (fest, nicht tauschen)
```
TEAM TiSCH:          #C9A84C  Gold
DER TiSCH:           #E8614A  Coral
iNTEGRATiONS TiSCH: #3A6F73  Petrol
LiTERATUR TiSCH:     #C87941  Espresso
EXPERTiSEN TiSCH:    #2C5F8A  Stahlblau
TRAiNiNGS TiSCH:     #2ECC5A  Grasgrün
COACHiNG TiSCH:      #7C5CBF  Violett
MEDiZiN TiSCH:       #4ECDC4  Mint
JURiS TiSCH:         #C9A84C  Gold
FAMiLiEN TiSCH:      #C4622D  Terrakotta
```

### Icons
Ausschließlich **inline SVG** (stroke-only, `currentColor`, `stroke-width: 1.5`). Keine Emojis.

---

## Shared Module

### NOTiZBUCH (`notizbuch.js`)
- In allen 10 Apps via `<script src="/notizbuch.js"></script>` eingebunden
- Floating Button unten rechts, Tastenkürzel `N`
- `NB.addEntry(text, quelle)` — Eintrag hinzufügen
- `NB.makeButton(content, label)` — "Ins Notizbuch" Button erstellen
- localStorage Key: `tisch_notizbuch_v1`
- Export: TXT, JSON, Clipboard

### Wikipedia Tooltips (`wiki-tooltip.js`)
- In EXPERTiSEN TiSCH aktiv
- `TiSCHWiki.attach(element, 'Suchbegriff')` — Tooltip anhängen
- Zeigt Wikipedia-Snippet + Bild + Link on Hover
- Cache in Memory, 3s Timeout

### kiNTEGRiTY (`kintegrity.py`)
Profile je App:
- `familientisch_profile` — empathisch-systemisch
- `juristisch_profile` — juristisch-präzise
- `medizintisch_profile` — klinisch-verständlich
- `coachingtisch_profile` — empathisch-klar
- `trainingstisch_profile` — sportlich-präzise
- `expert_tisch_profile` — akademisch
- `literatentisch_profile` — literarisch-präzise
- `integrations_tisch_profile` — klar und ausgewogen
- `brainstorm_profile` — offen, kreativ

---

## App-Defaults (stil/tone/register)

| App | stil | tone | register |
|-----|------|------|----------|
| TEAM | oekonomisch | achtsam | fachsprache |
| DER TiSCH | alltag | achtsam | — |
| iNTEGRATiONS | alltag | achtsam | — |
| LiTERATUR | akademisch | achtsam | — |
| EXPERTiSEN | akademisch | achtsam | fachsprache |
| TRAiNiNGS | akademisch | achtsam | — |
| COACHiNG | therapeutisch | achtsam | — |
| MEDiZiN | akademisch | achtsam | fachsprache |
| JURiS | juristisch | achtsam | fachsprache |
| FAMiLiEN | therapeutisch | achtsam | — |

---

## Arbeitsregeln (KRITISCH)

1. **MINIMAL-INVASIV** — keine Architekturen neu schreiben, keine bestehenden Funktionen löschen
2. **Vor jeder Änderung**: Änderungsliste zeigen → auf GO warten → dann coden
3. **Execution Loop**: `READ → ANALYZE → PROPOSE → STOP (wait for GO) → CODE`
4. **Gleicher Backend** für alle Apps — kein separater Server
5. **Kein Session-Cache**, keine Varianten-History
6. **Keine Emojis** — ausschließlich inline SVG Icons
7. **Farben stabil** — Akzentfarben niemals zwischen Apps tauschen
8. **Timeout**: Alle `fetch()` Calls haben 120s `AbortController`

---

## History System (alle Apps)
- localStorage Key: `tisch_history_v1`
- Max 50 Einträge pro App
- Format: `{ id, ts, app, question, perspectives, result }`
- Geteilt über alle Apps (gemeinsamer Key)

---

## FAMiLiEN TiSCH — Sonderfeature
- **3D Familienaufstellung** via Three.js r128
- Orbit Controls: manuell implementiert (kein OrbitControls Import)
- Figuren-Typen: Erwachsen♂ (blau), Erwachsen♀ (terrakotta), Kind (grün), Ältere (grau)
- Beziehungslinien: nah (mint), distanziert (grau), konfliktreich (rot)
- Analyse-Endpoint: `/api/ask` mit Konstellations-Beschreibung als Frage

---

## EXPERTiSEN TiSCH — Besonderheiten
- **89 Fachbereiche** mit `wikiDE`/`wikiEN` Feldern
- Wikipedia Hover-Tooltips via `wiki-tooltip.js`
- KI-Auto-Modus wählt passende Experten automatisch

---

## Shared Core (NEU — v1.0)

Der Shared Core ist die gemeinsame SQLite-Datenbank aller TiSCH-Sessions.

**Was gespeichert wird:**
- Vollständige Perspectives, Friction, Integration jeder Session
- Frage, App-Herkunft, Zeitstempel, Sprache, Stil
- myCEL-Muster (erkannte Muster über alle Sessions hinweg)

**Dateien:**
- `shared_core_store.py` — SQLite-Engine (aiosqlite, analog zu moonfingers_store.py)
- `shared_core.db` — Datenbank (persistent auf Railway)

**Interner API-Key:** `SHARED_CORE_KEY` (Railway ENV, default: `tisch-shared-core-2026`)

**Vault-Sync:** täglich via Cowork Scheduled Task (tägl. 3:00 Uhr)
- Sync-Service: `app/services/tisch_session_sync.py`
- Pattern-Detektor: `app/services/tisch_pattern_detector.py`
- Sessions landen in: `tiSCH/sessions/`
- Muster landen in: `tiSCH/muster/`

**Session automatisch speichern:** aktiviert für `/api/ask`, `/api/ask-simple`, `/api/ask-table`
Optional: `source_app`-Feld in QueryRequest/TableRequest für App-Identifikation (z.B. `"LiTERATUR-TiSCH"`).

---

## Letzter Stand
```
Commit:  3addab7
Message: feat: Shared Core v1.0 — Session-Logging + myCEL Pattern-Store (v7.0)
Branch:  main
Repo:    https://github.com/ralfarminkirchner-netizen/der-tisch
Betreuer: Claude Cowork (ab 2026-04-08)
```
