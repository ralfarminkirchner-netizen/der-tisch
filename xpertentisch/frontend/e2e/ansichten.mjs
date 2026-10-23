#!/usr/bin/env node
/**
 * Erzeugt den Ansichtsnachweis: sechs Ansichten, je hell und dunkel, je bei
 * 390 px und 1440 px.
 *
 *   node e2e/ansichten.mjs http://127.0.0.1:8077 ./ansichten
 *
 * Jede Theme-/Breiten-Kombination bekommt einen eigenen Browserkontext und
 * damit eine eigene, frische Sitzung. Fotografiert werden tatsächliche
 * Anwendungszustände; nichts wird nachträglich zusammengesetzt.
 */
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, mkdtempSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const BASE = process.argv[2] ?? 'http://127.0.0.1:8077';
const ZIEL = process.argv[3] ?? './ansichten';
const ZUGANGSWORT = process.env.XT_ADMIN_TOKEN ?? 'demo-zugangswort';
const FRAGE = 'Braucht ein kleines Team eine tägliche Sicherung der Datenbank?';

mkdirSync(ZIEL, { recursive: true });
const gemacht = [];
const fehlerGesamt = [];

const browser = await chromium.launch({ executablePath: process.env.PW_CHROMIUM ?? undefined });

async function schuss(page, ansicht, thema, breite, ziel) {
  const name = `${ansicht}--${thema}--${breite}.png`;
  const pfad = join(ZIEL, name);
  if (ziel) await ziel.screenshot({ path: pfad });
  else await page.screenshot({ path: pfad, fullPage: true });
  gemacht.push(name);
  console.log(`  ${name}`);
}

try {
  for (const thema of ['hell', 'dunkel']) {
    for (const breite of [390, 1440]) {
      console.log(`\n${thema} / ${breite} px`);
      const context = await browser.newContext({
        viewport: { width: breite, height: breite === 390 ? 844 : 960 },
        deviceScaleFactor: 2,
        colorScheme: thema === 'dunkel' ? 'dark' : 'light',
      });
      const page = await context.newPage();
      page.on('pageerror', (e) => fehlerGesamt.push(`${thema}/${breite}: ${e}`));
      await page.goto(BASE, { waitUntil: 'domcontentloaded' });
      await page.waitForSelector('form.spark');

      // 1) Der leere Tisch — vor dem ersten Funken.
      await page.waitForSelector('#leerer-tisch');
      await schuss(page, 'leerer-tisch', thema, breite);

      // 2) Der Moment des Einlaufens.
      await page.fill('#prompt', FRAGE);
      await page.click('form.spark button.primary');
      await page.waitForSelector('.card.zustand-streaming', { timeout: 30000 });
      await page.waitForTimeout(2300);
      await schuss(page, 'einlaufen', thema, breite);

      // 3) Fertiger Funke mit Auswertung.
      await page.waitForSelector('.panel-grid table', { timeout: 120000 });
      await page.waitForTimeout(800);
      await schuss(page, 'auswertung', thema, breite);

      // 4) Das Beziehungsnetz für sich.
      const netz = page.locator('.panel', { has: page.locator('svg.graph') }).first();
      await netz.scrollIntoViewIfNeeded();
      await schuss(page, 'beziehungsnetz', thema, breite, netz);

      // 5) Die Einstellungen.
      await page.click('#settings-open');
      await page.fill('#admin-token', ZUGANGSWORT);
      await page.click('#einstellungen button.primary');
      await page.waitForSelector('#kurator', { timeout: 20000 });
      await page.locator('#einstellungen').scrollIntoViewIfNeeded();
      await schuss(page, 'einstellungen', thema, breite, page.locator('#einstellungen'));

      await context.close();
    }
  }

  // 6) Die Zustandsschau, sofern angelegt: alle acht Auftragszustände
  // nebeneinander. Ohne sie fehlt dieser Beleg — und das wird gesagt.
  const listeVorab = await (await fetch(`${BASE}/api/sessions`)).json();
  const schau = (listeVorab.sessions ?? []).find((x) => x.title.startsWith('Zustandsschau'));
  if (!schau) {
    console.log('\nZustandsschau nicht angelegt — dieser Beleg FEHLT ' +
      '(tools/zustaende_seed.py ausführen).');
  } else {
    console.log('\nZustandsschau');
    for (const thema of ['hell', 'dunkel']) {
      for (const breite of [390, 1440]) {
        const context = await browser.newContext({
          viewport: { width: breite, height: 1000 },
          deviceScaleFactor: 2,
          colorScheme: thema === 'dunkel' ? 'dark' : 'light',
        });
        const page = await context.newPage();
        await page.goto(`${BASE}/#/s/${schau.id}`, { waitUntil: 'domcontentloaded' });
        await page.waitForSelector('.card');
        await page.waitForTimeout(500);
        await schuss(page, 'zustaende', thema, breite, page.locator('#sparks'));
        await context.close();
      }
    }
  }

  // 6) Der Bericht als eigenständiges Artefakt, über file:// geöffnet.
  const sitzungen = await (await fetch(`${BASE}/api/sessions`)).json();
  const sitzung = sitzungen.sessions[0];
  const html = await (await fetch(`${BASE}/api/sessions/${sitzung.id}/report.html`)).text();
  const datei = join(mkdtempSync(join(tmpdir(), 'xt-bericht-')), 'bericht.html');
  writeFileSync(datei, html, 'utf8');
  console.log(`\nBericht (offline, file://): ${datei}`);
  for (const thema of ['hell', 'dunkel']) {
    for (const breite of [390, 1440]) {
      const context = await browser.newContext({
        viewport: { width: breite, height: 1200 },
        deviceScaleFactor: 2,
        colorScheme: thema === 'dunkel' ? 'dark' : 'light',
      });
      const page = await context.newPage();
      await page.goto(`file://${datei}`, { waitUntil: 'load' });
      await schuss(page, 'bericht', thema, breite);
      await context.close();
    }
  }
} finally {
  await browser.close();
}

console.log(`\n${gemacht.length} Ansichtsbelege in ${ZIEL}`);
if (fehlerGesamt.length) {
  console.log(`JavaScript-Fehler: ${fehlerGesamt.join(' | ')}`);
  process.exit(1);
}
