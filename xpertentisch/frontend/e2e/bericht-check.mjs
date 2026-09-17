#!/usr/bin/env node
/**
 * Prüft den exportierten HTML-Bericht als eigenständiges Artefakt.
 *
 *   node e2e/bericht-check.mjs http://127.0.0.1:8077 [zielbild.png]
 *
 * Der Bericht wird vom Server geholt, auf die Platte gelegt und dann über
 * file:// geöffnet — ohne laufenden Anwendungsserver. Jede Anfrage, die den
 * Browser trotzdem nach außen schicken würde, wird abgewiesen und gemeldet.
 */
import { chromium } from 'playwright';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const BASE = process.argv[2] ?? 'http://127.0.0.1:8077';
const BILD = process.argv[3] ?? '';
const ergebnisse = [];
function pruefe(name, bedingung, detail = '') {
  ergebnisse.push({ name, ok: Boolean(bedingung), detail });
  console.log(`${bedingung ? 'OK  ' : 'FEHL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

/* Der Bericht muss an einer Sitzung geprüft werden, in der auch etwas steht.
   Über einer Sitzung ohne Auswertung bestünde jede Zusicherung hier von
   selbst: kein Marker, keine Themen, kein Baum — und damit auch kein Beleg,
   dass das eingebettete SVG den Bericht nicht doch nach außen öffnet. Fehlt
   eine solche Sitzung, wird eine angelegt, statt über Leerem zu bestehen. */
async function sitzungMitInhalt() {
  const liste = (await (await fetch(`${BASE}/api/sessions`)).json()).sessions ?? [];
  for (const kandidat of liste) {
    const buendel = await (await fetch(`${BASE}/api/sessions/${kandidat.id}`)).json();
    if ((buendel.sparks ?? []).some((e) => e.summary && e.markers.length > 0)) {
      return { sitzung: kandidat, buendel };
    }
  }

  const neue = await (await fetch(`${BASE}/api/sessions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ title: 'Berichtsprüfung' }),
  })).json();
  await fetch(`${BASE}/api/sessions/${neue.id}/sparks`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      prompt: 'Ist eine tägliche Sicherung der Datenbank notwendig?',
      client_request_id: `bericht-${Date.now()}`,
    }),
  });
  const frist = Date.now() + 180000;
  while (Date.now() < frist) {
    const buendel = await (await fetch(`${BASE}/api/sessions/${neue.id}`)).json();
    const jobs = (buendel.sparks ?? []).flatMap((e) => e.jobs);
    const ruht = jobs.length > 0
      && jobs.every((j) => !['queued', 'running', 'streaming'].includes(j.status));
    if (ruht && (buendel.sparks ?? []).every((e) => e.summary)) return { sitzung: neue, buendel };
    await new Promise((weiter) => setTimeout(weiter, 500));
  }
  return null;
}

const gefunden = await sitzungMitInhalt();
if (!gefunden) {
  console.log('FEHL  Keine Sitzung mit ausgewerteten Antworten — nichts zu prüfen.');
  process.exit(1);
}
const { sitzung, buendel } = gefunden;

const html = await (await fetch(`${BASE}/api/sessions/${sitzung.id}/report.html`)).text();
const ordner = mkdtempSync(join(tmpdir(), 'xt-bericht-'));
const datei = join(ordner, 'bericht.html');
writeFileSync(datei, html, 'utf8');
console.log(`Bericht abgelegt: ${datei} (${html.length} Zeichen)`);

// Statische Prüfungen am Artefakt selbst.
pruefe('Keine Skripte im Bericht', !/<script/i.test(html));
pruefe('Keine eingebundenen Fremdressourcen',
  !/<(link|iframe|object|embed)\b/i.test(html));
const externe = html.replace(/http-equiv/g, '').match(/https?:\/\/[^\s"'<)]+/g) ?? [];
pruefe('Keine externe Adresse im Quelltext', externe.length === 0, externe.join(', '));
pruefe('Eigenes Stylesheet eingebettet', html.includes('<style>'));
pruefe('Eigene Druckregeln vorhanden', html.includes('@media print'));

// Die Linsen im Bericht: der Verlaufsbaum ist das einzige eingebettete Bild.
// Er darf die Offline-Zusicherung nicht aufweichen — auch nicht über einen
// Namensraum, der als Adresse im Quelltext stünde.
pruefe('Der Verlauf steht als eingebettetes SVG im Bericht',
  /<svg class="verlauf"/.test(html));
pruefe('Das eingebettete SVG bringt keine Adresse mit',
  !/<svg[^>]*xmlns/i.test(html));
const hatMarker = (buendel.sparks ?? []).some((e) => (e.markers ?? []).length > 0);
pruefe('Die Themen der Auswertung stehen im Bericht',
  !hatMarker || html.includes('<h3>Themen</h3>'),
  hatMarker ? '' : 'keine Fundstellen in dieser Sitzung — Themen wären hier leer');

const browser = await chromium.launch({ executablePath: process.env.PW_CHROMIUM ?? undefined });
try {
  const context = await browser.newContext({ viewport: { width: 1000, height: 1400 } });
  const nachAussen = [];
  // Alles, was nicht aus der Datei selbst kommt, wird geblockt und vermerkt.
  await context.route('**/*', (route) => {
    const url = route.request().url();
    if (url.startsWith('file://') || url.startsWith('data:')) return route.continue();
    nachAussen.push(url);
    return route.abort();
  });
  const page = await context.newPage();
  const fehler = [];
  page.on('pageerror', (e) => fehler.push(String(e)));

  await page.goto(`file://${datei}`, { waitUntil: 'load' });
  pruefe('Bericht öffnet ohne laufenden Server',
    (await page.locator('h1').count()) >= 1);
  pruefe('Kein Zugriff nach außen beim Öffnen',
    nachAussen.length === 0, nachAussen.join(', '));
  pruefe('Keine JavaScript-Fehler', fehler.length === 0, fehler.join(' | '));

  // Lange Wörter dürfen den Satzspiegel nicht sprengen.
  const mass = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    client: document.documentElement.clientWidth,
  }));
  pruefe('Keine Überbreite im Bericht', mass.scroll <= mass.client + 1,
    `${mass.scroll} / ${mass.client}`);

  await page.setViewportSize({ width: 390, height: 1400 });
  await page.waitForTimeout(200);
  const schmal = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    client: document.documentElement.clientWidth,
  }));
  pruefe('Keine Überbreite auf dem Telefon', schmal.scroll <= schmal.client + 1,
    `${schmal.scroll} / ${schmal.client}`);

  // Druck: der Bericht muss auch im Druckbild lesbar bleiben.
  await page.setViewportSize({ width: 1000, height: 1400 });
  await page.emulateMedia({ media: 'print' });
  const imDruck = await page.locator('.answer').first().isVisible();
  pruefe('Antworttext ist auch im Druckbild sichtbar', imDruck);
  const baumImDruck = await page.locator('svg.verlauf').first().isVisible();
  pruefe('Der Verlaufsbaum ist auch im Druckbild sichtbar', baumImDruck);
  await page.emulateMedia({ media: 'screen' });

  if (BILD) {
    await page.screenshot({ path: BILD, fullPage: true });
    console.log(`Bild: ${BILD}`);
  }
  await context.close();
} finally {
  await browser.close();
}

const fehlgeschlagen = ergebnisse.filter((r) => !r.ok);
console.log(`\n${ergebnisse.length - fehlgeschlagen.length}/${ergebnisse.length} Prüfungen bestanden.`);
process.exit(fehlgeschlagen.length === 0 ? 0 : 1);
