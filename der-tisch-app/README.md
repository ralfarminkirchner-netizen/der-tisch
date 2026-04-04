# DER TiSCH — React Native App

React Native / Expo source files for the DER TiSCH mobile app.

## Features
- KI-Auto expert preview: shows which perspectives will be chosen *before* submitting
- Live debounced preview (1.2s after typing stops)
- Connects to: https://der-tisch-production.up.railway.app

## Setup
```bash
npm install
npx expo start
```

## Required packages
- react-native
- react-native-safe-area-context
- expo (or bare RN)

## Files
- `src/screens/HomeScreen.js` — Main screen with KI-Auto preview
- `src/screens/ResultScreen.js` — Results display
- `src/services/api.js` — API calls (askTheTable, askExperts, kintegritySynthesize)
- `src/theme.js` — Colors, spacing, typography
