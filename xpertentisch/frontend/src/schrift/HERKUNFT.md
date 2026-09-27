# Schriften

Beide Schriften liegen **hier im Repository** und werden mitgebaut. Zur Laufzeit
wird nichts von einem fremden Server geladen — weder von Google Fonts noch von
einem CDN. Das ist Absicht: die Anwendung hinterlässt keine Datenspur bei
Dritten, nur weil jemand sie öffnet.

| Datei | Schrift | Achsen | Herkunft | Lizenz |
| --- | --- | --- | --- | --- |
| `inter-latin-variabel.woff2` | Inter Variable | `opsz` 14–32, `wght` 100–900 | `@fontsource-variable/inter@5`, Datei `inter-latin-opsz-normal.woff2` | SIL Open Font License 1.1 (`LIZENZ-Inter.txt`) |
| `newsreader-latin-variabel.woff2` | Newsreader Variable | `wght` 200–800 | `@fontsource-variable/newsreader`, Datei `newsreader-latin-wght-normal.woff2` | SIL Open Font License 1.1 (`LIZENZ-Newsreader.txt`) |

Nur der Zeichenvorrat `latin` ist enthalten. Deutsch braucht daraus Umlaute und
ß; beides ist darin. Fehlt einem Zeichen die Deckung, greift die im
Schriftstapel dahinterstehende Systemschrift.

## Warum zwei Schriften

Die Frage des Menschen und die Antwort der Maschine haben in dieser Anwendung
nicht denselben Rang. Der **Funke** ist eine Setzung und steht darum in der
Serife (Newsreader). Alles, was die Maschine beisteuert — Antworttext,
Zustände, Zahlen, Bedienung — steht in der Grotesk (Inter). Die Unterscheidung
ist damit keine Verzierung, sondern trägt eine Aussage.

## Nicht im HTML-Bericht

Der exportierte Bericht bindet diese Dateien **nicht** ein. Er soll klein,
offline lesbar und in jedem Programm zu öffnen sein; eine eingebettete Schrift
als Base64 würde ihn um ein Vielfaches aufblähen. Der Bericht setzt darum auf
einen sorgfältig gewählten Systemschrift-Stapel, der dieselbe Gliederung trägt.

## Aktualisieren

```
npm i --no-save @fontsource-variable/inter@5 @fontsource-variable/newsreader
cp node_modules/@fontsource-variable/inter/files/inter-latin-opsz-normal.woff2 \
   src/schrift/inter-latin-variabel.woff2
cp node_modules/@fontsource-variable/newsreader/files/newsreader-latin-wght-normal.woff2 \
   src/schrift/newsreader-latin-variabel.woff2
```

Die Pakete bleiben bewusst **keine** Abhängigkeit des Projekts: was gebraucht
wird, liegt als Datei hier.
