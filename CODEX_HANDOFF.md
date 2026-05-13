# TiSCH — Übergabe an OpenAI Codex

**Projekt:** TiSCH Ecosystem — KI-gestützte Multiperspektiv-Analyse  
**GitHub:** `ralfarminkirchner-netizen/der-tisch`  
**Eigentümer:** Ralf Kirchner  
**Übergabe erstellt von:** Claude (Anthropic) — Mai 2026
**Zuletzt aktualisiert von:** OpenAI Codex — 14. Mai 2026

---

## Was ist TiSCH?

TiSCH ist ein Ökosystem aus 11 spezialisierten KI-Anwendungen, die alle dasselbe **FastAPI-Backend** teilen. Jede App hat ein eigenes Frontend (HTML/JS/CSS, Single File, kein Build-Step) mit unterschiedlichem Design, Funktionsumfang und Zielgruppe.

Das Backend orchestriert mehrere KI-Agenten (Multi-Perspektiv-Analyse): Eine Frage wird von bis zu 8 spezialisierten Agenten (philosophisch, systemisch, ethisch, empirisch, etc.) parallel analysiert, danach synthetisiert eine Reibungsanalyse die Widersprüche, und eine Integrationsanalyse destilliert ein Fazit.

---

## Repository-Struktur

```
der-tisch/
├── der-tisch-backend/          ← Alles hier drin
│   ├── api_server.py           ← FastAPI Backend (2100+ Zeilen)
│   ├── requirements.txt        ← fastapi, uvicorn, openai, pydantic
│   ├── nixpacks.toml           ← Railway Build-Config
│   ├── Procfile                ← uvicorn api_server:app --host 0.0.0.0 --port $PORT
│   ├── sw.js                   ← Service Worker (pass-through, kein aggressiver Cache)
│   ├── color-tokens.json       ← Design System Tokens
│   ├── icons/                  ← PWA Icons (192px, 512px)
│   ├── manifest-*.json         ← PWA Manifeste
│   │
│   ├── tisch-hub.html          ← TiSCH HUB (Startseite, Navigationszentrale)
│   ├── index.html              ← TEAM TiSCH (Hauptapp, ~400KB)
│   ├── integrationstisch.html  ← iNTEGRATiONS TiSCH
│   ├── der-tisch.html          ← DER TiSCH (Basisversion)
│   ├── coachingtisch.html      ← COACHiNG TiSCH
│   ├── expertentisch.html      ← EXPERTEN TiSCH
│   ├── familientisch.html      ← FAMiLiEN TiSCH
│   ├── juristisch.html         ← JURiSTiSCH TiSCH
│   ├── literatentisch.html     ← LiTERATEN TiSCH
│   ├── medizintisch.html       ← MEDiZiN TiSCH
│   └── trainingstisch.html     ← TRAiNiNGS TiSCH
└── vercel.json                 ← Vercel-Routing für Frontend-Deployment
```

---

## Aktueller Deployment-Stand

| Service | URL | Status |
|---|---|---|
| Backend (Railway) | `https://der-tisch-production.up.railway.app` | ✅ live |
| Backend-Routen | nur `/api/*` | ✅ HTML/Assets aus FastAPI entfernt |
| Frontend (Vercel) | noch nicht deployed | 🔲 TODO |
| Frontend-Assets im Repo | `der-tisch-backend/*.html` | ✅ Vercel-ready |

**Umgebungsvariablen auf Railway (Service `der-tisch`, Projekt `clever-gentleness`):**
- `OPENAI_API_KEY` = muss mit gültigem OpenAI-Key gesetzt/ersetzt werden
- `PORT` = von Railway automatisch gesetzt

---

## Backend: api_server.py

### KI-Stack (frisch migriert)
- **Vorher:** Anthropic Claude (Haiku + Sonnet)
- **Jetzt:** OpenAI GPT-4o-mini für alle Calls
- **SDK:** `openai>=1.50.0`
- **Client:** `client = OpenAI()` (liest `OPENAI_API_KEY` aus Environment)

### Kern-Funktion: `_call_api`
```python
def _call_api(model, max_tokens, system, tools, tool_name, messages) -> dict:
    # Erwartet native OpenAI Function-Calling-Tools
    # Gibt das strukturierte Output-Dict zurück
```
Die Tool-Definitionen sind im nativen OpenAI-Format:
`{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}`.
Die frühere `_anthropic_tool_to_openai()` Hilfsfunktion wurde entfernt.

### Frontend/Backend-Trennung
- `api_server.py` serviert keine HTML-, JS-, Icon- oder Manifest-Dateien mehr.
- Alle statischen Frontends liegen weiterhin in `der-tisch-backend/`.
- Die HTML-Apps nutzen `API_BASE = 'https://der-tisch-production.up.railway.app'`.
- `vercel.json` liegt im Repo-Root und routet Kurzpfade auf die jeweiligen HTML-Dateien.

### API-Endpunkte

| Endpunkt | Methode | Beschreibung |
|---|---|---|
| `/api/health` | GET | Health-Check, Version, Produktliste |
| `/api/ask` | POST | Alle 8 Agenten parallel (QueryRequest) |
| `/api/ask-simple` | POST | 4 intelligente Agenten (automatische Auswahl) |
| `/api/ask-table` | POST | Tisch-Modus: individuelle Agenten-Auswahl |
| `/api/ask-clarify` | POST | Klärungsgespräch-Modus |
| `/api/translate` | POST | Übersetzung + Re-Analyse |
| `/api/antagonisten/arena` | POST | Antagonisten-Modus (eskalierte Reibung) |
| `/api/antagonisten/fachgebiete` | GET | Verfügbare Fachgebiete |
| `/api/antagonisten/{fachgebiet}` | GET | Fachgebiet-Details |
| `/api/custom-perspectives/validate` | POST | Custom-Perspektiven validieren |
| `/api/custom-perspectives/save` | POST | Custom-Perspektiven speichern (Hook) |
| `/api/custom-perspectives/load` | GET | Custom-Perspektiven laden |
| `/api/hooks/black-hole-box` | POST | Ecosystem-Hook: Black Hole Box |
| `/api/hooks/ki-ntegrity` | POST | Ecosystem-Hook: KI-NTEGRITY |
| `/api/hooks/mycel/patterns` | GET/POST | Ecosystem-Hook: MYCEL |
| `/api/hooks/brainstormz` | POST | Ecosystem-Hook: BRAINSTORMZ |
| `/api/hooks/pandora-logic` | POST | Ecosystem-Hook: PANDORA LOGIC |

### Request-Format (Haupt-Endpunkte)
```json
{
  "question": "Soll ich meinen Job kündigen?",
  "lang": "de",
  "stil": "philosophisch",
  "register": "",
  "tone": "",
  "custom_perspectives": [],
  "methods": [],
  "reibungsintensitaet": "standard"
}
```

### Response-Format
```json
{
  "perspectives": [
    {
      "rolle": "Philosophisch",
      "anspruchstyp": "...",
      "kernanalyse": "...",
      "evidenz": "...",
      "blinder_fleck": "..."
    }
  ],
  "friction": {
    "uebersetzungsfehler": ["..."],
    "echte_widersprueche": ["..."],
    "uebersehenes": "..."
  },
  "integration": {
    "anspruchskarte": "...",
    "uebersetzbare_bruecken": ["..."],
    "echte_unvereinbarkeiten": ["..."],
    "praktische_optionen": ["..."],
    "offene_pruefpfade": ["..."],
    "vorlaeufiges_fazit": "...",
    "entscheidungshilfe": ["..."],
    "kurzfassung": ["..."],
    "einfach_gesagt": "...",
    "herzmensch": "...",
    "kopfmensch": "..."
  }
}
```

---

## Was noch zu tun ist (Priorität)

### 1. 🔴 KRITISCH: Gültigen OpenAI-Key auf Railway setzen
Die lokalen Testläufe gegen `/api/ask` und `/api/ask-simple` erreichten den OpenAI-Codepfad, wurden aber durch `AuthenticationError / invalid_api_key` blockiert. Zuerst einen gültigen Key setzen:

```bash
cd ~/Documents/der-tisch/der-tisch-backend
railway variables set OPENAI_API_KEY=sk-proj-...
railway up --detach
```

### 2. 🔴 KRITISCH: OpenAI-Migration mit gültigem Key testen
Sobald ein gültiger Key vorhanden ist:

```bash
# Lokal testen:
cd ~/Documents/der-tisch/der-tisch-backend
OPENAI_API_KEY=sk-... uvicorn api_server:app --port 8000

curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Was ist Wahrheit?", "lang": "de"}'
```

Bekannte potenzielle Probleme:
- `tool_choice` Format: OpenAI erwartet `{"type": "function", "function": {"name": "..."}}` — implementiert
- JSON-Parsing der Function Arguments: bereits mit `json.loads(tc.function.arguments)` implementiert
- `required` Felder in Tool-Schemas sind explizit gesetzt

### 3. 🟡 WICHTIG: Vercel deployment
- `vercel.json` liegt bereits im Root des Repos
- Vercel Projekt: Root Directory = `der-tisch-backend`
- Framework Preset: "Other" (static)
- Alle `.html` Dateien werden direkt als statische Assets serviert

### 4. 🟡 WICHTIG: CORS nach Vercel-Deployment einschränken
In `api_server.py` CORS-Origins nach Deployment auf tatsächliche Vercel-Domain einschränken. Aktuell ist für die Übergangsphase `allow_origins=["*"]` gesetzt.

### 5. 🟢 NICE TO HAVE: Streaming-Antworten
Für bessere UX: OpenAI Streaming API nutzen damit Antworten inkrementell erscheinen.

---

## Erledigt durch Codex

- OpenAI-Migration committed und gepusht.
- Tool-Definitionen auf natives OpenAI-Format migriert.
- `_anthropic_tool_to_openai()` entfernt.
- FastAPI serviert nur noch API-Endpunkte.
- Alle HTML-Dateien liegen im Repo und nutzen `API_BASE` für Railway.
- `vercel.json` im Repo-Root ergänzt.
- Frontend-Syntaxfehler in `index.html`, `literatentisch.html` und `trainingstisch.html` repariert.
- Verifiziert: `python3 -m py_compile api_server.py`, JSON-Check für `vercel.json`, Inline-JS-Syntaxcheck für alle 11 HTML-Dateien.

---

## Design-System

- **Primärfarbe:** `#C9A84C` (Gold/Amber)
- **Schrift:** System-UI Stack
- **Theme:** Dark-first, Light-Mode verfügbar
- **Design-Tokens:** `color-tokens.json` im Repo

---

## Wichtige Konventionen

1. **Sprache:** Code-Kommentare Englisch/Deutsch gemischt. User-facing Text Deutsch (mit EN-Support via `lang` Parameter).
2. **Agenten-Namen:** Groß, deutsch: `"Philosophisch"`, `"Systemisch"`, `"Ethisch"` etc.
3. **Fehlerbehandlung:** Backend gibt HTTP 500 mit Traceback zurück (für Debugging). In Produktion ggf. reduzieren.
4. **Persistenz:** Custom-Perspectives werden im Frontend (localStorage) gespeichert. Backend ist zustandslos.
5. **Parallelität:** Perspektiv-Agenten laufen mit `asyncio.gather()` parallel — das ist Performance-kritisch.

---

## Lokale Entwicklung

```bash
# Setup
cd ~/Documents/der-tisch/der-tisch-backend
pip install -r requirements.txt

# Backend starten
OPENAI_API_KEY=sk-... uvicorn api_server:app --reload --port 8000

# Frontend testen
cd ~/Documents/der-tisch/der-tisch-backend
python3 -m http.server 8082 --bind 127.0.0.1
# dann z.B. http://127.0.0.1:8082/tisch-hub.html öffnen
```

---

## Railway CLI (Deployment)

```bash
# Verlinken (einmalig)
railway link  # Projekt: clever-gentleness, Service: der-tisch

# Deployen
cd ~/Documents/der-tisch/der-tisch-backend
railway up --detach

# Logs
railway logs

# Variablen
railway variables set OPENAI_API_KEY=sk-proj-...
```

---

*Dokument erstellt: Mai 2026 — Claude für Ralf Kirchner*
