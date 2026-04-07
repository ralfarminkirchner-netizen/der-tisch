# TiSCH Suite — App Store Setup

## Struktur
```
capacitor/
├── der-tisch/           → DER TiSCH (app.tisch.der)
├── team-tisch/          → TEAM TiSCH (app.tisch.team)
└── integrations-tisch/  → iNTEGRATiONS TiSCH (app.tisch.integrations)
```

Alle 3 Apps nutzen **Capacitor** als Native Shell — die Web-App läuft auf Railway
und wird in die App als Live-URL geladen. Kein Code-Rewrite, kein Build-Step.

---

## Schritt 1 — Developer Accounts anlegen

| Store | Kosten | Link |
|---|---|---|
| Apple App Store | $99/Jahr | https://developer.apple.com/programs/enroll/ |
| Google Play Store | $25 einmalig | https://play.google.com/console/signup |

---

## Schritt 2 — Voraussetzungen lokal installieren

```bash
# Node.js (bereits installiert wenn du npm hast)
node --version  # sollte v18+ sein

# Android Studio (für Android builds)
# https://developer.android.com/studio

# Xcode (für iOS — nur auf Mac)
# https://developer.apple.com/xcode/
```

---

## Schritt 3 — Für jede App: Setup

```bash
# Beispiel für DER TiSCH (gleich für die anderen)
cd capacitor/der-tisch

# Dependencies installieren
npm install

# Android-Projekt hinzufügen
npx cap add android

# iOS-Projekt hinzufügen (nur Mac)
npx cap add ios

# Sync (nach jeder Änderung an capacitor.config.json)
npx cap sync
```

---

## Schritt 4 — Android APK/AAB bauen

```bash
cd capacitor/der-tisch

# Android Studio öffnen
npx cap open android
```

In Android Studio:
1. `Build → Generate Signed Bundle/APK`
2. Wähle `Android App Bundle (.aab)` für Play Store
3. Keystore erstellen (einmalig pro App — sicher aufbewahren!)
4. Release-Build erstellen

---

## Schritt 5 — iOS IPA bauen (Mac erforderlich)

```bash
cd capacitor/der-tisch
npx cap open ios
```

In Xcode:
1. `Product → Archive`
2. `Distribute App → App Store Connect`
3. Apple Developer Zertifikat verwenden

---

## App-IDs (Bundle Identifiers)

| App | Bundle ID | App Name |
|---|---|---|
| DER TiSCH | `app.tisch.der` | DER TiSCH |
| TEAM TiSCH | `app.tisch.team` | TEAM TiSCH |
| iNTEGRATiONS TiSCH | `app.tisch.integrations` | iNTEGRATiONS TiSCH |

---

## App Icons & Splash Screens

Benötigte Größen für Android:
- `ic_launcher.png`: 48×48, 72×72, 96×96, 144×144, 192×192
- `splash.png`: 1080×1920

Benötigte Größen für iOS:
- 1024×1024 (App Store)
- 60×60, 76×76, 83.5×83.5, 120×120, 152×152, 167×167, 180×180

Platzierung nach `npx cap add android`:
- Android: `android/app/src/main/res/mipmap-*/`
- iOS: `ios/App/App/Assets.xcassets/AppIcon.appiconset/`

---

## Hinweis: Live-URL Architektur

Die Apps laden die Web-App direkt von Railway:
- DER TiSCH → `https://der-tisch-production.up.railway.app/der-tisch.html`
- TEAM TiSCH → `https://der-tisch-production.up.railway.app/`
- iNTEGRATiONS TiSCH → `https://der-tisch-production.up.railway.app/integrationstisch.html`

**Vorteil:** Updates an der Web-App sind sofort in der nativen App sichtbar — kein Re-Submit zum Store nötig.

**Achtung:** Für App Store Compliance muss sichergestellt sein:
- HTTPS überall (✓ Railway hat SSL)
- Keine irreführende Beschreibung
- Datenschutzerklärung (Privacy Policy) URL erforderlich
