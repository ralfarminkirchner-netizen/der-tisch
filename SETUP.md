# Der Tisch — App Setup

## Schritt 1: Node.js installieren

1. Geh auf https://nodejs.org
2. Lade **"LTS"** (Long Term Support) herunter und installiere es
3. Überprüfe die Installation:
   ```
   node --version   # sollte v20.x.x oder höher zeigen
   npm --version    # sollte 10.x.x oder höher zeigen
   ```

## Schritt 2: Expo CLI installieren

```bash
npm install -g expo-cli
```

## Schritt 3: Expo Go auf dein iPhone installieren

App Store → "Expo Go" suchen → installieren

## Schritt 4: Projekt einrichten

1. Entpacke den Ordner `der-tisch-app` (den du heruntergeladen hast)
2. Öffne Terminal (Spotlight → "Terminal") und navigiere in den Ordner:
   ```bash
   cd ~/Downloads/der-tisch-app
   npm install
   ```

## Schritt 5: Railway-URL eintragen

Öffne `src/services/api.js` in einem Texteditor und ersetze:
```
"https://YOUR-RAILWAY-URL.up.railway.app"
```
mit deiner echten Railway-URL, zum Beispiel:
```
"https://der-tisch-production.up.railway.app"
```

## Schritt 6: App starten

```bash
npm start
```

Ein QR-Code erscheint im Terminal. Öffne die **Kamera-App** auf deinem iPhone,
scanne den QR-Code → die App öffnet sich automatisch in Expo Go.

---

## App Store veröffentlichen (optional, später)

1. `npm install -g eas-cli`
2. `eas login` → mit Expo-Account anmelden
3. `eas build --platform ios` → baut die App für den App Store
4. `eas submit --platform ios` → lädt sie hoch

Für Android identisch mit `--platform android`.
