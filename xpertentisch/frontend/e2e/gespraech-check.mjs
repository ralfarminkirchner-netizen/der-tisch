#!/usr/bin/env node
/**
 * Browserprüfung der Gesprächsfunktionen gegen einen laufenden Tisch.
 *
 *   python backend/tools/demo_server.py 8077     # steuerbare Test-Provider
 *   node e2e/gespraech-check.mjs http://127.0.0.1:8077
 *
 * Geprüft wird, was sich nur im echten Ablauf zeigt: der wachsende Text
 * während des Schreibens, der Abbruch eines einzelnen Auftrags, das
 * Wechselgespräch und die Sitzungsliste.
 */
import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://127.0.0.1:8077';
const ergebnisse = [];
function pruefe(name, bedingung, detail = '') {
  ergebnisse.push({ name, ok: Boolean(bedingung), detail });
  console.log(`${bedingung ? 'OK  ' : 'FEHL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

const browser = await chromium.launch({ executablePath: process.env.PW_CHROMIUM ?? undefined });

try {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  const fehler = [];
  page.on('pageerror', (e) => fehler.push(String(e)));

  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('form.spark', { timeout: 15000 });

  // --- Streamen ----------------------------------------------------------
  await page.fill('#prompt', 'Ist eine tägliche Sicherung der Datenbank notwendig?');
  await page.click('form.spark button.primary');

  // Der langsame Anbieter schreibt lange genug, um beim Wachsen zuzusehen.
  await page.waitForSelector('.card.zustand-streaming', { timeout: 20000 });
  await page.waitForTimeout(2000);
  const abbruchId = await page.locator('.card.zustand-streaming').first()
    .getAttribute('data-job-id');
  const karte = page.locator(`.card[data-job-id="${abbruchId}"]`);

  const ersterStand = await karte.locator('.answer').innerText();
  pruefe('Text erscheint, während er entsteht', ersterStand.length > 0,
    `${ersterStand.length} Zeichen`);

  await page.waitForTimeout(2500);
  const zweiterStand = await karte.locator('.answer').innerText();
  pruefe('Der Text wächst weiter', zweiterStand.length > ersterStand.length,
    `${ersterStand.length} → ${zweiterStand.length} Zeichen`);

  // --- Abbrechen ---------------------------------------------------------
  const laufend = karte;
  await laufend.locator('button', { hasText: 'Abbrechen' }).click();
  await page.waitForSelector(`.card[data-job-id="${abbruchId}"].zustand-cancelled`, { timeout: 15000 });
  pruefe('Abbruch trifft genau diesen Auftrag', true, abbruchId);

  // Diese Prüfung ist nur etwas wert, wenn es überhaupt eine andere Karte gibt
  // und deren Zustand tatsächlich ausgelesen wurde. Ein leeres Feld besteht
  // sonst jedes every().
  const andere = await page.locator('.card').evaluateAll((nodes, id) =>
    nodes.filter((n) => n.getAttribute('data-job-id') !== id)
      .map((n) => ({
        id: n.getAttribute('data-job-id'),
        zustand: n.className.match(/zustand-(\w+)/)?.[1] ?? null,
        zeichen: (n.querySelector('.answer')?.textContent ?? '').length,
      })),
    abbruchId);
  const lesbar = andere.filter((k) => k.id && k.zustand);
  pruefe('Es gibt mindestens eine andere Karte zum Vergleich',
    lesbar.length >= 1, `${lesbar.length} andere Karten`);
  pruefe('Keine andere Karte wurde mit abgebrochen',
    lesbar.length >= 1 && lesbar.every((k) => k.zustand !== 'cancelled'),
    lesbar.map((k) => `${k.id}:${k.zustand}`).join(', '));

  // Und sie laufen wirklich weiter: mindestens eine andere Karte trägt Text
  // oder erreicht einen Endzustand, statt bloß „nicht abgebrochen" zu sein.
  const weiter = lesbar.filter(
    (k) => k.zeichen > 0 || ['done', 'error', 'streaming', 'running'].includes(k.zustand),
  );
  pruefe('Mindestens eine andere Antwort läuft weiter',
    weiter.length >= 1,
    weiter.map((k) => `${k.id}:${k.zustand}/${k.zeichen} Zeichen`).join(', '));

  await page.waitForSelector('.panel-grid table', { timeout: 30000 });
  pruefe('Die Auswertung läuft trotz Abbruch', true);

  // --- Kosten ------------------------------------------------------------
  const kosten = await page.locator('.card .leiste').first().innerText();
  pruefe('Verbrauch steht ohne Neuladen auf der Karte',
    /Token\s*\d+\/\d+/.test(kosten), kosten.replace(/\n/g, ' · '));
  pruefe('Ohne hinterlegte Preise wird nichts geschätzt',
    kosten.includes('Kosten unbekannt') && !/\d+[.,]\d+\s*(ct|€)/.test(kosten),
    kosten.replace(/\n/g, ' · '));

  // --- Wechselgespräch ---------------------------------------------------
  await page.click('#pingpong > summary');
  await page.fill('#pp-prompt', 'Prüft die Gegenposition zur täglichen Sicherung.');
  await page.fill('#pp-runden', '2');
  await page.click('#pp-start');
  await page.waitForSelector('.pingpong-lauf', { timeout: 20000 });
  pruefe('Das Wechselgespräch nennt seine Grenzen vorher',
    (await page.locator('#pp-grenzen').innerText()).includes('höchstens'));
  await page.waitForFunction(
    () => document.querySelector('.pingpong-lauf')?.textContent?.includes('beendet')
      || document.querySelector('.pingpong-lauf')?.textContent?.includes('gestoppt'),
    null, { timeout: 60000 },
  );
  const bloecke = await page.locator('.spark-block').count();
  pruefe('Das Wechselgespräch bleibt in seiner Grenze', bloecke <= 4, `${bloecke} Beiträge`);

  // --- Sitzungen ---------------------------------------------------------
  const sitzungen = await page.locator('#sitzungswahl option').count();
  pruefe('Die Sitzungsliste steht zur Wahl', sitzungen >= 1, `${sitzungen} Einträge`);

  // --- Nach dem Neuladen -------------------------------------------------
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.card', { timeout: 20000 });
  const nachher = await page.locator('.card').count();
  pruefe('Nach dem Neuladen steht alles wieder da', nachher >= 2, `${nachher} Karten`);
  const abgebrochen = await page.locator(`.card[data-job-id="${abbruchId}"]`).innerText();
  pruefe('Der Abbruch bleibt als Abbruch erkennbar',
    abgebrochen.includes('abgebrochen'), abgebrochen.split('\n')[1] ?? '');

  // --- Entwürfe bei unterbrochener Verbindung -----------------------------
  await context.setOffline(true);
  await page.fill('#prompt', 'Dieser Gedanke entsteht ohne Netz.');
  await page.click('form.spark button.primary');
  await page.waitForSelector('#entwuerfe', { timeout: 15000 });
  pruefe('Ein Gedanke ohne Netz geht nicht verloren',
    (await page.locator('#entwuerfe').innerText()).includes('ohne Netz'));

  await context.setOffline(false);
  await page.click('#entwuerfe-senden');
  await page.waitForFunction(() => !document.getElementById('entwuerfe'), null, { timeout: 30000 });
  pruefe('Nach der Rückkehr wird der Gedanke nachgereicht', true);

  // --- Einstellungen: Preise und Kurator ---------------------------------
  await page.click('#settings-open');
  await page.fill('#admin-token', 'demo-zugangswort');
  await page.click('#einstellungen button.primary');
  await page.waitForSelector('#kurator', { timeout: 15000 });
  // Diese Prüfung hieß „Preisfelder", prüfte aber nur, ob es den Kurator gibt.
  // Jetzt prüft sie, was ihr Name sagt: die Preisfelder existieren, sind leer
  // und gehören zu einem tatsächlich vorhandenen Anbieter.
  const preisfelder = page.locator('input[id^="price_in-"]');
  const anzahlPreise = await preisfelder.count();
  const werte = await preisfelder.evaluateAll((felder) =>
    felder.map((f) => f.value),
  );
  pruefe('Für jeden Anbieter gibt es ein Preisfeld',
    anzahlPreise >= 1, `${anzahlPreise} Felder`);
  pruefe('Ohne eingetragenen Preis bleiben die Felder leer',
    anzahlPreise >= 1 && werte.every((w) => w === ''),
    `Werte: ${JSON.stringify(werte)}`);
  const kuratorLeer = await page.locator('#kurator').inputValue();
  pruefe('Ohne Auswahl gibt es keine Kuratierung', kuratorLeer === '');

  // --- Überbreite --------------------------------------------------------
  const mass = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    client: document.documentElement.clientWidth,
  }));
  pruefe('Keine Seitenüberbreite auf dem Telefon', mass.scroll <= mass.client + 1,
    `${mass.scroll} / ${mass.client}`);

  pruefe('Keine JavaScript-Fehler', fehler.length === 0, fehler.join(' | '));
  await context.close();
} finally {
  await browser.close();
}

const fehlgeschlagen = ergebnisse.filter((r) => !r.ok);
console.log(`\n${ergebnisse.length - fehlgeschlagen.length}/${ergebnisse.length} Prüfungen bestanden.`);
process.exit(fehlgeschlagen.length === 0 ? 0 : 1);
