# TiSCH Familie — Claude Cowork Onboarding

## Projekt-Überblick
Du arbeitest an der **TiSCH Familie** — einem Portfolio aus 11 KI-gestützten Multiperspektiv-Analyse-Apps, die alle auf einem gemeinsamen Railway-Backend laufen.

## GitHub Repo
**https://github.com/ralfarminkirchner-netizen/der-tisch**
- Root Directory: `der-tisch-backend/`
- Hauptdateien: `api_server.py`, `kintegrity.py`, alle `.html` Dateien

## Live URLs
| App | URL |
|-----|-----|
| **HUB** | https://der-tisch-production.up.railway.app/tisch-hub.html |
| TEAM TiSCH | https://der-tisch-production.up.railway.app/ |
| DER TiSCH | https://der-tisch-production.up.railway.app/der-tisch.html |
| iNTEGRATiONS TiSCH | https://der-tisch-production.up.railway.app/integrationstisch.html |
| LiTERATUR TiSCH | https://der-tisch-production.up.railway.app/literatentisch.html |
| EXPERTiSEN TiSCH | https://der-tisch-production.up.railway.app/expertentisch.html |
| TRAiNiNGS TiSCH | https://der-tisch-production.up.railway.app/trainingstisch.html |
| COACHiNG TiSCH | https://der-tisch-production.up.railway.app/coachingtisch.html |
| MEDiZiN TiSCH | https://der-tisch-production.up.railway.app/medizintisch.html |
| JURiS TiSCH | https://der-tisch-production.up.railway.app/juristisch.html |
| FAMiLiEN TiSCH | https://der-tisch-production.up.railway.app/familientisch.html |

## Railway Deployment
- Project: "clever-gentleness"
- Project ID: `f2c619f2-9c4d-4365-aeb5-856d01b13bb6`
- Service ID: `f75d3760-41bb-4f09-bcef-89a17a6e40e5`
- Environment ID: `902072f7-0ea1-41b3-9724-7e6d81f3dcfd`
- Railway Token: `d7ef0453-6a1a-4940-9dc6-8fe977d46f13`
- Port: `$PORT=8080`
- Root Directory: `der-tisch-backend`
- ANTHROPIC_API_KEY: `sk-ant-api03-[DEIN_KEY_AUS_RAILWAY_ENV_VARS]`

## Deploy-Befehl
```bash
# Nach Änderungen in der-tisch-backend/:
git add . && git commit -m "Beschreibung" && git push origin main
# Railway redeploy:
curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Authorization: Bearer d7ef0453-6a1a-4940-9dc6-8fe977d46f13" \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { serviceInstanceRedeploy(environmentId:\"902072f7-0ea1-41b3-9724-7e6d81f3dcfd\", serviceId:\"f75d3760-41bb-4f09-bcef-89a17a6e40e5\") }"}'
# Health check:
sleep 90 && curl https://der-tisch-production.up.railway.app/api/health
```

## Architektur
- **Backend**: FastAPI (`api_server.py`) auf Railway
- **KI**: Claude claude-sonnet-4-6 via Anthropic API
- **Frontend**: Reine HTML/CSS/JS Dateien (kein Framework, kein Build-Step)
- **kiNTEGRiTY**: Eigenes Modul (`kintegrity.py`) für Integritätsprüfung

## Backend Endpoints
| Endpoint | Zweck |
|----------|-------|
| `POST /api/ask` | Multiperspektiv-Analyse (alle Apps) |
| `POST /api/ask-simple` | DER TiSCH Einzelfrage |
| `POST /api/ask-table` | Custom Perspektiven |
| `POST /api/ask-clarify` | Klärungsgespräch |
| `POST /api/translate` | Register/Ton-Übersetzung |
| `POST /api/kintegrity/synthesize` | kiNTEGRiTY Synthese |
| `GET /api/health` | Health Check |

## Design-System (KRITISCH — niemals ändern)
- **Logo**: `TiSCH` — T ist Tischform (SVG: Querbalken + zwei Beine). `i` ist Akzentfarbe. `SCH` normal.
- **Schrift**: Inter durchgehend + Playfair Display (LiTERATUR TiSCH)
- **Akzentfarben** (fest, nicht tauschen):
  - TEAM TiSCH: `#C9A84C` Gold
  - DER TiSCH: `#E8614A` Coral
  - iNTEGRATiONS: `#3A6F73` Petrol
  - LiTERATUR: `#C87941` Espresso
  - EXPERTiSEN: `#2C5F8A` Stahlblau
  - TRAiNiNGS: `#2ECC5A` Grasgrün
  - COACHiNG: `#7C5CBF` Violett
  - MEDiZiN: `#4ECDC4` Mint
  - JURiS: `#C9A84C` Gold
  - FAMiLiEN: `#C4622D` Terrakotta

## Arbeitsregeln (WICHTIG)
1. **MINIMAL-INVASIV** — keine Architekturen neu schreiben, keine bestehenden Funktionen löschen
2. **Vor jeder Änderung**: Liste der Änderungen zeigen und auf GO warten
3. **Execution Loop**: READ → ANALYZE → PROPOSE → STOP (wait for GO) → CODE
4. **Kein Session-Cache**, keine Varianten-History — minimaler State
5. **Gleicher Backend** für alle Apps
6. **Icons**: Inline SVG (stroke-only, currentColor) — keine Emojis

## Stil/Ton System
- `stil`: philosophisch, therapeutisch, paedagogisch, juristisch, spirituell, achtsam, akademisch, oekonomisch, alltag, einfach, kindgerecht, jugend
- `tone`: "" | "achtsam" | "direkt"
- `register`: "" | "fachsprache" | "einfach"

## App-Defaults
- TEAM: `stil='oekonomisch'`, `tone='achtsam'`
- DER TiSCH: `stil='alltag'`, `tone='achtsam'`
- LiTERATUR: `stil='akademisch'`, `tone='achtsam'`
- EXPERTiSEN: `stil='akademisch'`, `tone='achtsam'`
- COACHiNG: `stil='therapeutisch'`, `tone='achtsam'`
- MEDiZiN: `stil='akademisch'`, `register='fachsprache'`, `tone='achtsam'`
- JURiS: `stil='juristisch'`, `register='fachsprache'`, `tone='achtsam'`
- TRAiNiNGS: `stil='akademisch'`, `tone='achtsam'`
- FAMiLiEN: `stil='therapeutisch'`, `tone='achtsam'`

## Wikipedia Integration
- `wiki-tooltip.js` liegt im Backend-Verzeichnis
- Route: `GET /wiki-tooltip.js`
- Nutzung: `<script src="/wiki-tooltip.js"></script>` + `TiSCHWiki.attach(element, 'Suchbegriff')`

## kiNTEGRiTY Profile
Jede App hat ein eigenes Profil in `kintegrity.py`:
`familientisch_profile`, `juristisch_profile`, `medizintisch_profile`, `coachingtisch_profile`, `trainingstisch_profile`, `expert_tisch_profile`, `literatentisch_profile`, `integrations_tisch_profile`, `brainstorm_profile`

## Letzter Git-Commit
`017094b` — Mega-Upgrade: TiSCH HUB + Wikipedia + 89 EXPERTiSEN Disziplinen + Portraits + FAMiLiEN 3D Fix
