# der-tisch-app — Claude Cowork Onboarding
> React Native / Expo Mobile App · Stand 2026-04-08

---

## Was ist das?
Die native Mobile-App der **TiSCH Familie**. Expo-basiertes React Native Projekt für iOS & Android. Verbindet sich mit demselben Railway-Backend wie alle Web-Apps.

---

## Repository
```
https://github.com/ralfarminkirchner-netizen/der-tisch
└── der-tisch-app/        ← dieses Verzeichnis
```

---

## Backend — Railway Live-URL
```
https://der-tisch-production.up.railway.app
```

**Wichtig:** In `src/services/api.js` steht noch ein Platzhalter:
```js
export const API_BASE_URL = "https://YOUR-RAILWAY-URL.up.railway.app";
```
→ Ersetzen mit: `"https://der-tisch-production.up.railway.app"`

### Verwendete Endpoints
| Endpoint | Methode | Payload |
|----------|---------|---------|
| `/api/ask` | POST | `{ question: string }` |
| `/api/health` | GET | — |

---

## Projektstruktur
```
der-tisch-app/
├── App.js                        # Root: NavigationContainer + Stack.Navigator
├── app.json                      # Expo config (name, bundleIdentifier, splash)
├── package.json                  # Dependencies
├── babel.config.js               # Babel (expo preset)
├── assets/
│   └── icon.png
└── src/
    ├── theme.js                  # Design Tokens (colors, typography, spacing, radius)
    ├── screens/
    │   ├── HomeScreen.js         # Eingabe-Screen
    │   └── ResultScreen.js       # Ergebnis-Screen (3 Phasen)
    ├── components/               # (leer — Komponenten noch in screens)
    ├── services/
    │   └── api.js                # askTheTable(), checkHealth()
    └── hooks/                    # (leer — für zukünftige Hooks)
```

---

## Navigation
```
Stack Navigator
├── Home  → HomeScreen.js
└── Result → ResultScreen.js
     props: route.params = { question: string, result: ApiResponse }
```

---

## API Response Struktur (`/api/ask`)
```typescript
{
  perspectives: Array<{
    rolle: string,         // z.B. "Systemisch", "Philosophisch"
    kernanalyse: string,
    evidenz: string,
    blinder_fleck: string,
    anspruchstyp: string,
  }>,
  friction: {
    harte_widersprueche: string[],
    scheinkonsens: string[],
    uebersehenes: string,
  },
  integration: {
    vorlaeufiger_konsens: string,
    fruchtbare_differenzen: string[],
    uebersetzbarkeit: string[],
    echte_unvereinbarkeiten: string[],
    praktische_optionen: string[],
    offene_pruefpfade: string[],
  }
}
```

---

## Design Tokens (`src/theme.js`)

### Farben
```js
colors = {
  bg:            "#0e0d0b",   // Haupt-Hintergrund
  bgCard:        "#161410",   // Karten
  bgInput:       "#1a1815",   // Eingabefelder
  border:        "#2a2720",
  textPrimary:   "#f0ede6",
  textSecondary: "#9a9080",
  textMuted:     "#5a5448",
  amber:         "#c8993a",   // Primär-Akzent (Gold)
  red:           "#c0574a",   // Reibungs-Phase
  blue:          "#5a8fb8",   // Integrations-Phase
}
```

### Agenten-Farben (pro Perspektive)
```js
agentColors = {
  "Systemisch":          { accent: "#6b9e6e", bg: "#0a120a" },
  "Tiefenpsychologisch": { accent: "#9e7b6e", bg: "#120e0a" },
  "Empirisch-Rational":  { accent: "#6e8e9e", bg: "#0a0f12" },
  "Philosophisch":       { accent: "#9e9a6e", bg: "#12110a" },
}
```

### Typography
```js
typography = {
  hero: 36, h1: 28, h2: 20, h3: 16,
  body: 15, small: 13, xs: 11,
  display: "Georgia",  // serif für Überschriften
  body: "System",
}
```

### Spacing / Radius
```js
spacing = { xs:4, sm:8, md:16, lg:24, xl:32, xxl:48 }
radius  = { sm:6, md:12, lg:18, full:999 }
```

---

## HomeScreen (`src/screens/HomeScreen.js`)

**Was es tut:**
- TextInput für die Frage (max 600 Zeichen)
- "Untersuchen" Button → ruft `askTheTable(question)` → navigiert zu Result
- 4 Beispielfragen als Chips
- Animated fade-in beim Mount
- Loading-State zeigt 4 Agenten-Labels mit ActivityIndicator
- Error-Handling mit roter Fehlerkarte
- KeyboardAvoidingView für iOS/Android

**Logo-Komponente (`TableLogo`):**
- React Native View-basiertes T-als-Tisch Symbol
- Querbalken + zwei Beine + 4 farbige Dots (amber, grün, braun, blau)

---

## ResultScreen (`src/screens/ResultScreen.js`)

**Empfängt via `route.params`:** `{ question, result }`

**3-Phasen Layout:**

**Phase I — Perspektiven**
- `PerspectiveCard` pro Perspektive (staggered fade-in, delay 120ms × index)
- Accent-Bar links, agentColors per `rolle`-Feld
- Felder: Kernanalyse, Evidenz, Blinder Fleck

**Phase II — Reibung**
- `FrictionCard` — roter Background (`#160d0b`)
- Harte Widersprüche (Bullet-List, rot)
- Scheinkonsens (Bullet-List, redDim)
- Kollektiv Übersehen (Text)

**Phase III — Integration**
- `IntegrationCard` — blauer Background (`#0a1118`)
- 6 Sektionen: Konsens, Differenzen, Übersetzbarkeit, Unvereinbarkeiten, Optionen, Prüfpfade

---

## Dependencies
```json
{
  "expo": "~51.0.0",
  "react-native": "0.74.5",
  "@react-navigation/native": "^6.1.18",
  "@react-navigation/native-stack": "^6.11.0",
  "react-native-safe-area-context": "4.10.5",
  "react-native-screens": "~3.31.1",
  "expo-linear-gradient": "~13.0.2"
}
```

---

## Lokales Setup
```bash
cd der-tisch-app
npm install
npx expo start

# iOS Simulator:
npx expo start --ios

# Android Emulator:
npx expo start --android

# Expo Go (physisches Gerät):
# QR-Code scannen mit Expo Go App
```

**Node Version:** 18+ empfohlen  
**Expo CLI:** `npm install -g expo-cli` falls nötig

---

## App Bundle IDs
```
iOS:     com.dertisch.app
Android: com.dertisch.app
```

---

## Capacitor Wrapper (im Repo unter `/capacitor/`)
Neben der React Native App gibt es 3 Capacitor-basierte Wrapper für native App-Builds:
```
capacitor/
├── der-tisch/           # DER TiSCH Web → iOS/Android
├── team-tisch/          # TEAM TiSCH Web → iOS/Android
└── integrations-tisch/  # iNTEGRATiONS TiSCH → iOS/Android
```
Setup-Anleitung: `capacitor/SETUP_ANLEITUNG.md`

---

## Erweiterungsideen (noch nicht gebaut)
- [ ] Alle 11 TiSCH Apps als auswählbare Modi in der Mobile App
- [ ] NOTiZBUCH-Funktion (`AsyncStorage` statt `localStorage`)
- [ ] Verlauf (History) mit `AsyncStorage`, Key: `tisch_history_v1`
- [ ] KI-Auto-Modus (wie EXPERTiSEN TiSCH)
- [ ] Dark/Light Mode Toggle
- [ ] Push Notifications für gespeicherte Fragen
- [ ] Share-Button für Ergebnisse

---

## Verbindung zum Web-Backend
Das Backend (`der-tisch-backend/api_server.py`) läuft auf Railway und bedient sowohl Web-Apps als auch die Mobile App über dieselben Endpoints. Kein separater Mobile-Backend nötig.

**Health Check:**
```bash
curl https://der-tisch-production.up.railway.app/api/health
# → {"status":"ok","version":"6.0"}
```
