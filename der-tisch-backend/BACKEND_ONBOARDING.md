# der-tisch-backend — Claude Cowork Onboarding
> FastAPI Backend · Stand 2026-04-08 · Version 6.0

---

## Was ist das?
Das gemeinsame Backend für alle 11 TiSCH Web-Apps und die React Native Mobile App. FastAPI auf Railway. Orchestriert parallele Claude-Agenten via Anthropic Tool Use.

---

## Repository
```
https://github.com/ralfarminkirchner-netizen/der-tisch
└── der-tisch-backend/     ← dieses Verzeichnis
```

## Live-URL
```
https://der-tisch-production.up.railway.app
```

---

## Dateistruktur
```
der-tisch-backend/
├── api_server.py              # FastAPI App — Endpoints, Tool-Definitionen, Agent-Prompts
├── kintegrity.py              # kiNTEGRiTY Modul — Synthese-Engine
├── notizbuch.js               # Shared NOTiZBUCH (served als static file)
├── wiki-tooltip.js            # Wikipedia Hover-Tooltips (served als static file)
├── tisch-hub.html             # Zentrale Landingpage
├── index.html                 # TEAM TiSCH
├── der-tisch.html             # DER TiSCH
├── integrationstisch.html     # iNTEGRATiONS TiSCH
├── literatentisch.html        # LiTERATUR TiSCH
├── expertentisch.html         # EXPERTiSEN TiSCH (89 Disziplinen)
├── trainingstisch.html        # TRAiNiNGS TiSCH
├── coachingtisch.html         # COACHiNG TiSCH
├── medizintisch.html          # MEDiZiN TiSCH
├── juristisch.html            # JURiS TiSCH
├── familientisch.html         # FAMiLiEN TiSCH (inkl. 3D Aufstellung)
├── Procfile                   # web: uvicorn api_server:app --host 0.0.0.0 --port $PORT
├── nixpacks.toml              # Python 3.12
└── requirements.txt           # fastapi, uvicorn, anthropic, pydantic
```

---

## Alle Endpoints

| Method | Path | Pydantic Model | Zweck |
|--------|------|---------------|-------|
| GET | `/` | — | TEAM TiSCH HTML |
| GET | `/<app>.html` | — | App HTML-Dateien |
| GET | `/api/health` | — | `{"status":"ok","version":"6.0"}` |
| GET | `/notizbuch.js` | — | Shared NOTiZBUCH JS |
| GET | `/wiki-tooltip.js` | — | Wikipedia Tooltip JS |
| POST | `/api/ask` | `AskRequest` | Multiperspektiv-Analyse (Hauptendpoint) |
| POST | `/api/ask-simple` | `SimpleAskRequest` | DER TiSCH — vereinfachte Einzelfrage |
| POST | `/api/ask-table` | `TableRequest` | Custom Perspektiven (Eigener Tisch) |
| POST | `/api/ask-clarify` | `ClarifyRequest` | Klärungsgespräch-Modus |
| POST | `/api/translate` | `TranslateRequest` | Register/Ton-Übersetzung |
| POST | `/api/kintegrity/synthesize` | `KintegrityRequest` | kiNTEGRiTY Integritätsprüfung |

---

## Pydantic Models (api_server.py)

### Perspective
```python
class Perspective(BaseModel):
    rolle: str              # z.B. "Systemisch", "Tiefenpsychologisch"
    anspruchstyp: str       # Welchen Wahrheitsanspruch kann diese Methode bearbeiten?
    kernanalyse: str
    evidenz: str
    blinder_fleck: str
```

### Friction
```python
class Friction(BaseModel):
    uebersetzungsfehler: List[str]   # Wo reden sie aneinander vorbei (verschiedene Mone)?
    echte_widersprueche: List[str]   # Gleicher Anspruchstyp, verschiedene Antwort
    uebersehenes: str                # Was haben ALLE Methoden gemeinsam nicht gesehen?
```

### Integration (14 Felder)
```python
class Integration(BaseModel):
    anspruchskarte: str                    # Welche Wahrheitsansprüche liegen vor?
    uebersetzbare_bruecken: List[str]
    echte_unvereinbarkeiten: List[str]
    praktische_optionen: List[str]         # 3 Optionen mit explizitem Anspruchstyp
    offene_pruefpfade: List[str]
    vorlaeufiges_fazit: str                # Belastbare Arbeitsorientierung
    entscheidungshilfe: List[str]          # Welche Methode ist hier zuständig?
    kurzfassung: List[str]                 # 5-6 direkt verwendbare Bulletpoints
    einfach_gesagt: str                    # Plain-language für jedes Publikum
    herzmensch: str                        # Gefühl, Beziehung, Intuition
    kopfmensch: str                        # Logik, Fakten, Strategie
    maennlich: str                         # Archetypisch männliche Energie
    weiblich: str                          # Archetypisch weibliche Energie
```

### TableResponse
```python
class TableResponse(BaseModel):
    perspectives: List[Perspective]
    friction: Friction
    integration: Integration
```

---

## Agent-Architektur

### Prinzip
Jeder Agent ist **ein Finger, nicht der Mond**. Kein Agent hat ein Erkenntnismonopol. Jeder zeigt auf einen spezifischen Wahrheitsanspruch (Anspruchstyp), nicht auf Wahrheit selbst.

### Die 8 Agenten (AGENTS_DE)

| Agent | Finger zeigt auf | Methode |
|-------|-----------------|---------|
| **Systemisch** | Geltung in sozialen Systemen | Wechselwirkungen, Kontexte, Muster, Zirkularität |
| **Tiefenpsychologisch** | Das Wahrheitsbedürfnis selbst | Psychoanalyse, Bindungstheorie, Abwehrmechanismen |
| **Empirisch-Rational** | Intersubjektiv belastbare, revidierbare Aussagen | Falsifikation, Evidenz, Modellbildung |
| **Philosophisch** | Den Anspruchstyp selbst | Begriffsklärung, Kategorienfehler, Wahrheitsrelationen |
| **Ethisch** | Werte, Vertretbarkeit, moralische Konsequenzen | Wertkonflikte, Verteilungsfragen, Reversibilität |
| **Abwägung** | Verhältnis von Aufwand und Ertrag | Kosten/Nutzen, Ressourcen, Zeithorizont |
| **Strategisch** | Pfade und Pfadabhängigkeiten | Kurzfrist/Mittelfrist/Langfrist, Optionenerhalt |
| **Integrativ** | Meta-Synthese der anderen 7 | Kompatibilität, Prioritäten, Gesamtbewertung |

### Englische Entsprechungen (AGENTS_EN)
Identische Struktur, übersetzt.

---

## Anthropic Tool Use — Ablauf

### 3 Phasen, 3 Tools

```
Phase 1 — /api/ask → PERSPECTIVE_TOOL
┌─────────────────────────────────────────────────────────────┐
│ Parallel: 4-N asyncio.gather() Calls                        │
│ Jeder Agent: client.messages.create(tools=[PERSPECTIVE_TOOL])│
│ tool_choice = {"type": "tool", "name": "submit_perspective"} │
│ Result: Perspective Objekt mit 4 Feldern                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
Phase 2 — FRICTION_TOOL
┌─────────────────────────────────────────────────────────────┐
│ Friction-Agent liest alle Perspectives                       │
│ client.messages.create(tools=[FRICTION_TOOL])               │
│ tool_choice = {"type": "tool", "name": "submit_friction"}   │
│ Result: Friction Objekt mit 3 Feldern                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
Phase 3 — INTEGRATION_TOOL
┌─────────────────────────────────────────────────────────────┐
│ Integration-Agent liest Perspectives + Friction             │
│ client.messages.create(tools=[INTEGRATION_TOOL])            │
│ tool_choice = {"type": "tool", "name": "submit_integration"}│
│ Result: Integration Objekt mit 14 Feldern                    │
└─────────────────────────────────────────────────────────────┘
```

### Tool-Definitionen

**PERSPECTIVE_TOOL** — `input_schema`: `anspruchstyp`, `kernanalyse`, `evidenz`, `blinder_fleck`
- Kritische Regel: Kein Agent darf mit Meta-Phrasen beginnen ("Als philosophische Perspektive...")
- 3–6 Sätze pro Feld, keine Füller

**FRICTION_TOOL** — `input_schema`: `uebersetzungsfehler[]`, `echte_widersprueche[]`, `uebersehenes`
- Unterscheidet: scheinbare Widersprüche (verschiedene Mone) vs. echte Widersprüche

**INTEGRATION_TOOL** — `input_schema`: alle 14 Integration-Felder
- Herzmensch/Kopfmensch: archetypische Energie-Perspektiven (nicht Geschlecht)
- Entscheidungshilfe-Format: `[Methode] ist das richtige Werkzeug wenn [Bedingung]`

---

## Sprachstil-System (STIL_INSTRUCTIONS)

12 Stile mit DE/EN Varianten:

| stil | Charakter |
|------|-----------|
| `philosophisch` | Fachbegriffe, Begriffsklärung, Unterscheidungen |
| `akademisch` | Wissenschaftlich, hypothetisch, strukturiert |
| `alltag` | Kluger Freund beim Kaffee, keine Fremdwörter |
| `oekonomisch` | Kosten/Nutzen, handlungsorientiert |
| `kindgerecht` | Für aufgeweckte 12-Jährige, Analogien |
| `therapeutisch` | Warm, Ambivalenzen halten, keine Urteile |
| `paedagogisch` | Strukturiert, ermutigend, entwicklungsorientiert |
| `juristisch` | Tatbestand/Rechtsfolge, Normen, präzise |
| `einfach` | Eine Idee pro Satz, kein Jargon |
| `spirituell` | Kontemplativ, Metaphern, Raum für Unaussprechliches |
| `jugend` | Locker, direkt, authentisch — kein aufgesetzter Slang |
| `achtsam` | Langsam, präsenzorientiert, ohne Drängen |

### Tone-Modifikatoren
- `tone=""` — neutraler Standardton
- `tone="achtsam"` — zusätzliche Achtsamkeitsinstruktion im System-Prompt
- `tone="direkt"` — prägnant, ohne Umschweife

### Register
- `register=""` — Standard
- `register="fachsprache"` — explizit fachliche Terminologie
- `register="einfach"` — vereinfachte Ausgabe

---

## Sync/Async-Strategie

```python
# Parallele Perspective-Calls:
async def fetch_perspective(agent_name, prompt, question, lang, stil, tone):
    def sync_call():
        return client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=900,
            tools=[PERSPECTIVE_TOOL],
            tool_choice={"type": "tool", "name": "submit_perspective"},
            system=system_prompt,
            messages=[{"role": "user", "content": question}]
        )
    return await asyncio.to_thread(sync_call)  # ← Thread-Pool für sync Anthropic client

# Parallel starten:
results = await asyncio.gather(*[
    fetch_perspective(name, ...) for name in selected_agents
])
```

**Warum `asyncio.to_thread`?** Der Anthropic Python Client ist synchron. `to_thread` verhindert Blocking des Event Loops während paralleler API-Calls.

---

## kiNTEGRiTY (`kintegrity.py`)

### Was es ist
Integritätswahrende Synthese-Engine. Kein einfaches Zusammenfassen — strukturiertes Destillieren mit konfigurierbaren Parametern.

### Profile-Parameter

```python
{
  "compression_strength":        0.55,   # Wie stark komprimiert wird (0=minimal, 1=maximal)
  "reformulation_conservatism":  0.72,   # Wie viel Originalsprache erhalten bleibt
  "redundancy_reduction":        0.65,   # Wie aggressiv Duplikate entfernt werden
  "conflict_visibility":         0.85,   # Wie sichtbar Widersprüche bleiben
  "must_keep_sensitivity":       0.90,   # Wie stark MUST-KEEP Inhalte geschützt werden
  "tone":                        "...",  # Ton-Anweisung
  "special_rules": []                   # App-spezifische Sonderregeln
}
```

### Alle Profile

| Profil | Ton | Besonderheit |
|--------|-----|--------------|
| `familientisch_profile` | empathisch-systemisch | Emotionale Nuancen, Generationsmuster, keine Schuld |
| `juristisch_profile` | juristisch-präzise | Paragrafen erhalten, Rechtspositionen, Haftungshinweis |
| `medizintisch_profile` | klinisch-verständlich | Fachbegriffe + Erklärung, Haftungshinweis |
| `coachingtisch_profile` | empathisch-klar | Coaching-Spannungen, handlungsrelevant |
| `trainingstisch_profile` | sportlich-präzise | Umsetzbare Empfehlungen, Sportfachbegriffe |
| `expert_tisch_profile` | wissenschaftlich-sachlich | Terminologie, Fachbereichswidersprüche |
| `literatentisch_profile` | literarisch-präzise | Sprachrhythmus bewahren |
| `integrations_tisch_profile` | therapeutisch-integrierend | Perspektivenvielfalt, produktive Spannung |
| `brainstorm_profile` | explorativ-offen | Originalformulierungen maximal erhalten |
| `default_profile` | klar und ausgewogen | Fallback |

### KintegrityRequest
```python
class KintegrityRequest(BaseModel):
    inputs: List[InputBlock]         # Textblöcke mit id, role, content, is_user_authored
    integrity_field: str = ""        # MUST-KEEP Inhalt — höchste Priorität
    profile: str = "default_profile"
    lang: str = "de"
    question: str = ""               # Kontext/Originalfrage
```

### KintegrityResponse
```python
class KintegrityResponse(BaseModel):
    synthesis: str                    # Destillierter Haupttext
    aber_section: str                 # Echte Widersprüche / Unvereinbarkeiten
    questionable: str                 # Faktisch Fragwürdiges (separat, nicht gelöscht)
    redundancies_removed: List[str]   # Was entfernt wurde
    provenance: List[ProvenanceEntry] # Quellenrückverfolgung
    must_keep_honored: bool
    confidence: float                 # 0.0–1.0
```

### MUST-KEEP Mechanismus
Wenn `integrity_field` befüllt ist, wird es im Prompt als höchste Priorität markiert (`must_keep_sensitivity`). Inhalt, Formulierungen und Intentionen müssen im Syntheseresultat maximal erhalten bleiben.

---

## Wichtige Implementierungsdetails

### Modell
```python
model="claude-sonnet-4-6"
```
Überall einheitlich. Kein Modell-Wechsel ohne Rücksprache.

### max_tokens
- Perspective: `900`
- Friction: `600`  
- Integration: `2400`
- kiNTEGRiTY: `2400`
- Simple/Clarify: `1200`

### Timeout-Handling (Frontend)
Alle `fetch()` Calls in den HTML-Apps haben `AbortController` mit 120s Timeout — da parallele Perspective-Calls zusammen 40–80 Sekunden brauchen können.

### CORS
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

---

## Deployment

### Procfile
```
web: uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

### Environment Variables (Railway)
```
ANTHROPIC_API_KEY=sk-ant-api03-...
PORT=8080
```

### Deploy-Befehl
```bash
git add der-tisch-backend/api_server.py
git commit -m "Beschreibung"
git push origin main

# Railway Redeploy:
curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Authorization: Bearer d7ef0453-6a1a-4940-9dc6-8fe977d46f13" \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { serviceInstanceRedeploy(environmentId:\"902072f7-0ea1-41b3-9724-7e6d81f3dcfd\", serviceId:\"f75d3760-41bb-4f09-bcef-89a17a6e40e5\") }"}'

sleep 90 && curl https://der-tisch-production.up.railway.app/api/health
```

---

## Arbeitsregeln

1. **MINIMAL-INVASIV** — kein Umbau bestehender Architekturen
2. **Vor Änderungen**: Liste zeigen → auf GO warten → dann coden
3. **Modell nicht ändern** ohne Rücksprache — `claude-sonnet-4-6` überall
4. **Neue App hinzufügen**:
   - HTML-Datei erstellen
   - Route in `api_server.py` registrieren (`@app.get("/<name>.html")`)
   - kiNTEGRiTY Profil in `kintegrity.py` PROFILES dict hinzufügen
   - `CLAUDE_COWORK_ONBOARDING.md` aktualisieren
5. **Neue Sprach-Stil** → in `STIL_INSTRUCTIONS` mit DE + EN Variante hinzufügen
6. **Tool-Definitionen** (`PERSPECTIVE_TOOL` etc.) — sehr konservativ ändern, sie steuern die KI-Output-Qualität direkt
