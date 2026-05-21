# TiSCH — Handoff
**Projekt:** `ralfarminkirchner-netizen/der-tisch`
**Repo lokal:** `/Users/ralfkirchner/Documents/der-tisch`
**Branch:** `main`
**Zuletzt aktualisiert:** 2026-05-17 22:15 CEST

---

## ⚡ Erster Schritt im neuen Chat

```bash
cd ~/Documents/der-tisch/der-tisch-backend
python3 -c "
import re
with open('trainingstisch.html', 'r') as f:
    c = f.read()
c = re.sub(r'\n<style id=\"trainingstisch-warm-redesign\">.*?</style>', '', c, flags=re.DOTALL)
with open('trainingstisch.html', 'w') as f:
    f.write(c)
print('Done')
"
```

Dann prüfen: `http://127.0.0.1:8082/trainingstisch.html` — muss dunkles Sportfeld-Design zeigen.

---

## Kurzstand

- Backend läuft auf Railway: `https://der-tisch-production.up.railway.app`
- OPENAI_API_KEY ist korrekt gesetzt (`sk-proj-iVw6z88h...`)
- Alle 11 TiSCH-Seiten sind live und erreichbar
- OpenAI GPT-4o-mini liefert Antworten — `/api/ask` getestet ✅
- Lokaler Preview: `http://127.0.0.1:8082/` (python3 -m http.server 8082 in der-tisch-backend)
- Git worktree ist dirty — nichts committed

---

## trainingstisch.html — aktuelles Problem

Die Datei hat zwei `<style>`-Blöcke:
1. **Zeilen 22–502:** Originales dunkles Sportfeld-Design ← **GUT, das wollen wir**
2. **Zeilen 504–2414:** `<style id="trainingstisch-warm-redesign">` ← **WEG DAMIT** (Codex-Fehler, überschreibt alles mit Fraunces/Cream/!important)

Fix: Python-Befehl oben (erster Schritt).

---

## Design — was locked ist

### Das TT = Tisch Prinzip (Kern der Marke)
Zwei gestapelte T-Buchstaben bilden die Silhouette eines Tisches:
- Kleines T aus "TRAiNiNGS" (oben, kleiner)
- Großes T aus "TiSCH" (unten, dominant)
→ Zusammen: Tisch-Silhouette. Das gilt für ALLE TiSCH-Apps.

### Was NICHT geht (explizit abgelehnt)
- ❌ Fraunces oder andere Serif-Fonts im Logo — Logo braucht Inter Black/900
- ❌ Split-Layout (grün links / cream rechts) — der Nutzer hat das klar abgelehnt
- ❌ Warm Cream als Full-Page-Hintergrund — zu generisch
- ❌ Codex für Design-Arbeit — er versteht visuelle Konzepte nicht aus Textbeschreibungen

### Was richtig ist
- ✅ Dunkles Sportfeld als Hintergrund für Trainingstisch
- ✅ TT=Tisch-Logo in Inter Black, weiß/hell auf dunklem Grund
- ✅ Orange italic `i`-Buchstaben in allen Markennamen (TRAiN**i**NGS T**i**SCH)
- ✅ Design direkt in HTML/CSS von Claude umsetzen, nicht delegieren

---

## api_server.py — Sync nötig

Verbesserte Version liegt in:
`/Volumes/ThunderBolt4_2TB/MeineApps/der-tisch-main/der-tisch-backend/api_server.py`

Verbesserungen:
- max_tokens pro Perspektive: 500 → 1200
- Agenten-Prompts: kein "Halte dich kurz", kein Situations-Wiederholen
- kernanalyse: 4-6 Sätze statt 2-3

Sync-Befehl:
```bash
cp /Volumes/ThunderBolt4_2TB/MeineApps/der-tisch-main/der-tisch-backend/api_server.py \
   ~/Documents/der-tisch/der-tisch-backend/api_server.py
cd ~/Documents/der-tisch
git add der-tisch-backend/api_server.py
git commit -m "fix: deeper perspectives, no meta-commentary, max_tokens 1200"
git push
```

---

## Aufgaben-Reihenfolge

### 🔴 Sofort
1. trainingstisch.html zweiten Style-Block entfernen (Python-Befehl oben)
2. Preview prüfen, Logo analysieren
3. Logo direkt in HTML fixen (Claude, kein Codex)
4. Nutzer gibt Design frei

### 🟡 Danach
5. api_server.py synchen und deployen (Befehle oben)
6. Design-System auf alle anderen TiSCH-Apps übertragen
7. sw.js 404 fixen

### 🟢 Nice to have
8. Frontend/Backend trennen: `const API_BASE = 'https://der-tisch-production.up.railway.app'` in alle HTMLs
9. Pydantic `register`-Warnungen fixen
10. Vercel-Deployment für Frontends

---

## Deployment

| Service | URL | Status |
|---|---|---|
| Backend + Frontend | `der-tisch-production.up.railway.app` | ✅ live |
| OPENAI_API_KEY | Railway env | ✅ korrekt |
| GitHub | `ralfarminkirchner-netizen/der-tisch` | main branch |

**Railway:** Projekt `clever-gentleness`, Service `der-tisch`
```bash
cd ~/Documents/der-tisch/der-tisch-backend
railway logs        # Logs ansehen
railway up --detach # Manuell deployen
railway variables   # Env vars prüfen
```

---

## Datei-Landkarte

```
/Users/ralfkirchner/Documents/der-tisch/
├── der-tisch-backend/
│   ├── api_server.py          ← FastAPI Backend (OpenAI GPT-4o-mini)
│   ├── requirements.txt       ← openai>=1.50.0
│   ├── trainingstisch.html    ← ⚠️ ERSTER FIX NÖTIG (zweiter Style-Block weg)
│   ├── tisch-hub.html
│   ├── index.html             ← TEAM TiSCH
│   ├── der-tisch.html
│   ├── integrationstisch.html
│   ├── coachingtisch.html
│   ├── expertentisch.html
│   ├── familientisch.html
│   ├── juristisch.html
│   ├── literatentisch.html
│   ├── medizintisch.html
│   ├── tisch-responsive.css   ← gemeinsame Responsive-Schicht
│   ├── sw.js                  ← ⚠️ 404 auf Railway
│   └── notizbuch.js
└── vercel.json

Handoff-Dateien:
/Users/ralfkirchner/spiral-os/memory/session-compress/2026-05-17-2215.md
/Users/ralfkirchner/Documents/der-tisch/CODEX_HANDOFF.md
```

---

## Kontexte für neuen Chat

```
Lies zuerst:
/Users/ralfkirchner/spiral-os/memory/session-compress/2026-05-17-2215.md
/Users/ralfkirchner/Documents/der-tisch/CODEX_HANDOFF.md
Repo: cd /Users/ralfkirchner/Documents/der-tisch
```
