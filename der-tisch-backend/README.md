# Der Tisch — Epistemischer Perspektivenraum

**Keine absolute Wahrheit. Vier Methoden, die gleichzeitig denken — gefolgt von schonungsloser Reibung und ehrlicher Integration.**

🌐 **Live:** [der-tisch-production.up.railway.app](https://der-tisch-production.up.railway.app)

---

## Was ist Der Tisch?

Der Tisch ist kein Chatbot. Er ist ein Raum der Klärung.

Du legst eine Frage auf den Tisch — eine echte, schwierige, persönliche Frage. Dann denken vier spezialisierte KI-Agenten **gleichzeitig und unabhängig** aus vier völlig verschiedenen methodischen Perspektiven:

| Agent | Methode |
|-------|---------|
| **Systemisch** | Wechselwirkungen, Kontexte, Muster, Zirkularität |
| **Tiefenpsychologisch** | Unbewusstes, Schutzstrategien, Bindungsmuster |
| **Empirisch-Rational** | Kausalität, kognitive Verzerrungen, Evidenz |
| **Philosophisch** | Begriffsklärung, Kategorienfehler, implizite Annahmen |

Danach folgt:
- **Reibungs-Agent:** Findet echte Widersprüche zwischen den Perspektiven — kein Harmonisieren
- **Integrations-Agent:** Meta-Synthese ohne Kuschelkonsens

---

## Architektur

```
Frontend (HTML/CSS/JS)
└── 3 Phasen: Perspektiven → Reibung → Integration

Backend (Python / FastAPI)
└── 4 parallele Agenten (asyncio.gather)
└── Structured Outputs via Anthropic Tool Use (Pydantic)
└── Reibungs-Agent (Phase 2)
└── Integrations-Agent (Phase 3)
```

Das Geheimnis sind **Structured Outputs**: Die KIs werden auf API-Ebene in harte methodische Vorgaben gezwungen. Kein freier Text — jeder Agent muss exakt liefern:

- `kernanalyse` — Kernaussage aus der eigenen Methode
- `evidenz` — Womit wird die Analyse begründet?
- `blinder_fleck` — Was kann diese Methode prinzipiell NICHT sehen?
- `harte_widersprueche` — Echte Widersprüche zwischen den Agenten
- `scheinkonsens` — Wo scheinen sie sich einig, meinen aber Verschiedenes?
- `praktische_optionen` — Konkrete Handlungsoptionen
- `offene_pruefpfade` — Was bleibt offen und muss weiter untersucht werden?

---

## Selbst hosten

### Voraussetzungen
- Python 3.12+
- Anthropic API Key ([console.anthropic.com](https://console.anthropic.com))

### Installation

```bash
git clone https://github.com/ralfarminkirchner-netizen/der-tisch.git
cd der-tisch/der-tisch-backend
pip install -r requirements.txt
```

### Starten

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python api_server.py
```

Dann öffne `http://localhost:8000` im Browser.

### Auf Railway deployen

1. Fork dieses Repo
2. Neues Projekt auf [railway.app](https://railway.app) → "Deploy from GitHub"
3. Root Directory: `der-tisch-backend`
4. Variable setzen: `ANTHROPIC_API_KEY=sk-ant-...`
5. Domain generieren — fertig

---

## Kosten

Pro vollständiger Analyse (alle 3 Phasen): ca. **$0.003** (claude-sonnet-4-6).

---

## Lizenz

MIT — frei verwendbar, veränderbar, weiterzugeben.

---

*Gebaut mit [Perplexity Computer](https://www.perplexity.ai/computer)*
