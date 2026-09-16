#!/usr/bin/env node
/**
 * Browserprüfung gegen einen laufenden XPERTENTiSCH-Server.
 *
 *   npm i -D playwright            # nicht Teil der Abhängigkeiten
 *   node e2e/browser-check.mjs http://127.0.0.1:8077
 *
 * Geprüft wird, was sich nur im echten Layout zeigt:
 * keine horizontale Seitenüberbreite auf Mobilgeräten, unabhängige
 * Modellkarten und der Graphklick, der die richtigen Antworten öffnet.
 */
import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://127.0.0.1:8077';
const VIEWPORTS = [
  { name: 'iPhone SE', width: 320, height: 568 },
  { name: 'iPhone 12', width: 390, height: 844 },
  { name: 'Tablet', width: 768, height: 1024 },
  { name: 'Desktop', width: 1440, height: 900 },
];

const ergebnisse = [];
function pruefe(name, bedingung, detail = '') {
  ergebnisse.push({ name, ok: Boolean(bedingung), detail });
  console.log(`${bedingung ? 'OK  ' : 'FEHL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

const browser = await chromium.launch({
  executablePath: process.env.PW_CHROMIUM ?? undefined,
});

try {
  const context = await browser.newContext({ viewport: VIEWPORTS[1] });
  const page = await context.newPage();
  const fehler = [];
  page.on('pageerror', (e) => fehler.push(String(e)));

  // Der Ereignisstrom bleibt offen; 'networkidle' tritt darum nie ein.
  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('form.spark', { timeout: 15000 });
  pruefe('Oberfläche lädt', await page.locator('h1').first().isVisible());

  // Funke setzen
  await page.fill('#prompt', 'Ist eine tägliche Sicherung der Datenbank notwendig?');
  await page.click('button.primary');
  await page.waitForSelector('.card .answer', { timeout: 60000 });
  await page.waitForSelector('.panel-grid table', { timeout: 60000 });

  const karten = await page.locator('.card').count();
  pruefe('Modellkarten erscheinen', karten >= 2, `${karten} Karten`);

  const antworten = await page.locator('.card .answer').count();
  pruefe('Jede Karte trägt ihre eigene Antwort', antworten >= 1, `${antworten} Antwortblöcke`);

  // Graphklick öffnet die richtigen Antworten
  const knoten = page.locator('svg.graph .node').first();
  pruefe('Diagramm gezeichnet', (await page.locator('svg.graph .node').count()) >= 1);
  const jobId = await knoten.getAttribute('data-job-id');
  await knoten.click();
  const markiert = await page.locator('.card.highlight').evaluateAll((nodes) =>
    nodes.map((n) => n.getAttribute('data-job-id')),
  );
  pruefe('Graphklick öffnet genau die richtige Antwort',
    markiert.length === 1 && markiert[0] === jobId,
    `markiert: ${markiert.join(', ')} / erwartet: ${jobId}`);

  // Geklickt wird die Gruppe: die Trefferfläche darin gehört zu ihr,
  // ein Klick darauf landet über das Ereignis ohnehin bei der Gruppe.
  const kante = page.locator('svg.graph g.edge').first();
  if (await kante.count()) {
    const a = await kante.getAttribute('data-job-a');
    const b = await kante.getAttribute('data-job-b');
    await kante.click();
    const paar = await page.locator('.card.highlight').evaluateAll((nodes) =>
      nodes.map((n) => n.getAttribute('data-job-id')).sort(),
    );
    pruefe('Kantenklick öffnet beide beteiligten Antworten',
      JSON.stringify(paar) === JSON.stringify([a, b].sort()),
      `markiert: ${paar.join(', ')}`);
  } else {
    pruefe('Kantenklick', true, 'keine Kante vorhanden (keine Bezüge gefunden)');
  }

  // Keine horizontale Überbreite in allen Viewports
  for (const vp of VIEWPORTS) {
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await page.waitForTimeout(250);
    const mass = await page.evaluate(() => ({
      scroll: document.documentElement.scrollWidth,
      client: document.documentElement.clientWidth,
      breiteste: (() => {
        let max = 0;
        let schuldiger = '';
        for (const el of document.querySelectorAll('body *')) {
          const r = el.getBoundingClientRect();
          if (r.right > max) { max = r.right; schuldiger = el.tagName + '.' + el.className; }
        }
        return { max: Math.round(max), schuldiger };
      })(),
    }));
    pruefe(`Keine Seitenüberbreite (${vp.name}, ${vp.width}px)`,
      mass.scroll <= mass.client + 1,
      `scrollWidth ${mass.scroll} / clientWidth ${mass.client}; breitestes Element bis ${mass.breiteste.max}px (${mass.breiteste.schuldiger})`);
  }

  pruefe('Keine JavaScript-Fehler', fehler.length === 0, fehler.join(' | '));
  await context.close();
} finally {
  await browser.close();
}

const fehlgeschlagen = ergebnisse.filter((r) => !r.ok);
console.log(`\n${ergebnisse.length - fehlgeschlagen.length}/${ergebnisse.length} Prüfungen bestanden.`);
process.exit(fehlgeschlagen.length === 0 ? 0 : 1);
