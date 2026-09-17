#!/usr/bin/env node
/**
 * Prüft, was sich aus einem Screenshot NICHT ablesen lässt:
 * Kontrast, Tastaturbedienung, Fokus, Themenvorrang, reduzierte Bewegung und
 * die schmalen Breiten auch in geöffneten Panels.
 *
 *   node e2e/zugang-check.mjs http://127.0.0.1:8077
 */
import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://127.0.0.1:8077';
const ZUGANGSWORT = process.env.XT_ADMIN_TOKEN ?? 'demo-zugangswort';
const ergebnisse = [];
function pruefe(name, bedingung, detail = '') {
  ergebnisse.push({ name, ok: Boolean(bedingung), detail });
  console.log(`${bedingung ? 'OK  ' : 'FEHL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

/* Farben kommen aus dem Browser heute als oklch(...) heraus. Statt sie zu
   parsen, lässt sich der echte sRGB-Wert über ein Canvas-Pixel bestimmen —
   und zwar bereits über dem tatsächlichen Untergrund zusammengerechnet. */
async function messeKontrast(page, waehler) {
  return page.evaluate((w) => {
    const el = document.querySelector(w);
    if (!el) return null;

    const leinwand = document.createElement('canvas');
    leinwand.width = 1;
    leinwand.height = 1;
    const ctx = leinwand.getContext('2d', { willReadFrequently: true });

    const alsRGB = (farbe, unterlage) => {
      ctx.clearRect(0, 0, 1, 1);
      if (unterlage) {
        ctx.fillStyle = unterlage;
        ctx.fillRect(0, 0, 1, 1);
      }
      ctx.fillStyle = farbe;
      ctx.fillRect(0, 0, 1, 1);
      const d = ctx.getImageData(0, 0, 1, 1).data;
      return [d[0], d[1], d[2]];
    };

    // Untergrund: die erste Fläche, die tatsächlich deckt — über Weiß bzw.
    // Schwarz zusammengerechnet, damit halbdurchsichtige Lagen mitzählen.
    let knoten = el;
    let hinten = null;
    const lagen = [];
    while (knoten) {
      const bg = getComputedStyle(knoten).backgroundColor;
      const teil = alsRGB(bg);
      ctx.clearRect(0, 0, 1, 1);
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, 1, 1);
      const alpha = ctx.getImageData(0, 0, 1, 1).data[3];
      if (alpha > 0) lagen.push(bg);
      if (alpha === 255) { hinten = bg; break; }
      knoten = knoten.parentElement;
    }
    if (!hinten && lagen.length) hinten = lagen[lagen.length - 1];
    if (!hinten) hinten = getComputedStyle(document.body).backgroundColor;

    // Die gefundenen Lagen von hinten nach vorn übereinanderlegen.
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillStyle = hinten;
    ctx.fillRect(0, 0, 1, 1);
    for (const lage of lagen.reverse()) {
      ctx.fillStyle = lage;
      ctx.fillRect(0, 0, 1, 1);
    }
    const u = ctx.getImageData(0, 0, 1, 1).data;
    const unterlage = `rgb(${u[0]}, ${u[1]}, ${u[2]})`;

    // SVG-Text färbt über 'fill', nicht über 'color'. Wer hier 'color' misst,
    // misst den vererbten Wert und damit etwas anderes als das, was dasteht.
    const stil = getComputedStyle(el);
    const istSVG = el.namespaceURI === 'http://www.w3.org/2000/svg';
    const fuellung = stil.fill;
    const vorneRoh = istSVG && fuellung && fuellung !== 'none' ? fuellung : stil.color;
    const vorne = alsRGB(vorneRoh, unterlage);

    const lum = (rgb) => {
      const [r, g, b] = rgb.map((v) => {
        const c = v / 255;
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const a = lum(vorne);
    const b = lum([u[0], u[1], u[2]]);
    const [hell, dunkel] = a > b ? [a, b] : [b, a];
    return {
      wert: (hell + 0.05) / (dunkel + 0.05),
      vorne: `rgb(${vorne.join(', ')})`,
      hinten: unterlage,
      groesse: parseFloat(stil.fontSize),
      gewicht: stil.fontWeight,
    };
  }, waehler);
}

/* Bedienbarkeit über die echte Tabulatortaste, nicht über focus():
   programmatisches focus() löst :focus-visible nicht aus und prüfte etwas
   anderes, als ein Mensch erlebt. */
async function erreichbarPerTabulator(page, waehler, schluesselAttribut) {
  const ziel = page.locator(waehler).first();
  if ((await ziel.count()) === 0) return false;
  const kennung = await ziel.getAttribute(schluesselAttribut);
  if (kennung === null) return false;
  await ziel.scrollIntoViewIfNeeded();
  await page.locator('#prompt').focus();
  for (let i = 0; i < 400; i += 1) {
    await page.keyboard.press('Tab');
    const getroffen = await page.evaluate(
      ([attr, wert]) => document.activeElement?.getAttribute?.(attr) === wert,
      [schluesselAttribut, kennung],
    );
    if (getroffen) return true;
  }
  return false;
}

const browser = await chromium.launch({ executablePath: process.env.PW_CHROMIUM ?? undefined });
try {
  for (const thema of ['light', 'dark']) {
    const context = await browser.newContext({
      viewport: { width: 1280, height: 900 }, colorScheme: thema,
    });
    const page = await context.newPage();
    await page.goto(BASE, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('form.spark');
    await page.fill('#prompt', 'Braucht ein kleines Team eine tägliche Sicherung?');
    await page.click('form.spark button.primary');
    await page.waitForSelector('.panel-grid table', { timeout: 120000 });

    // --- Kontrast: Fließtext und Nebenangaben gegen ihren echten Untergrund.
    const messe = async (waehler, mindest, name) => {
      const wert = await messeKontrast(page, waehler);
      if (!wert) {
        pruefe(`${name} (${thema})`, false, 'Element nicht gefunden');
        return;
      }
      pruefe(`${name} (${thema}) erreicht ${mindest}:1`, wert.wert >= mindest,
        `${wert.wert.toFixed(2)}:1 — ${wert.vorne} auf ${wert.hinten}, ` +
        `${wert.groesse}px/${wert.gewicht}`);
    };

    await messe('.answer', 4.5, 'Antworttext');
    await messe('.question', 4.5, 'Funke');
    await messe('.leiste', 4.5, 'Nebenangaben');
    await messe('.hint', 4.5, 'Hinweistext');
    await messe('.tag.zustand', 4.5, 'Zustandsschild');
    await messe('td', 4.5, 'Tabellenzelle');
    await messe('footer.foot', 4.5, 'Fußzeile');

    // --- Ein Szenario durchspielen, damit die Folgen-Linse etwas zu zeigen hat.
    // Ohne diesen Schritt liefe die Prüfung über die leere Fläche und bestünde
    // aus dem falschen Grund.
    const folgenKnopf = page
      .locator('.card .card-actions button', { hasText: 'Folgen durchspielen' })
      .first();
    if (await folgenKnopf.count()) {
      await folgenKnopf.click();
      await page.click('form.spark button.primary');
      await page.waitForFunction(
        () => document.querySelectorAll('.spark-block').length >= 2, null, { timeout: 120000 },
      );
      await page.waitForFunction(
        () => document.querySelectorAll('.spark-block .panel-grid table').length >= 2,
        null, { timeout: 180000 },
      );
    }
    pruefe(`Szenario für die Folgen-Linse angelegt (${thema})`,
      (await page.locator('.spark-block').count()) >= 2,
      'ohne Szenario wäre die Folgen-Linse UNGEPRÜFT');

    // --- Der Verlauf steht jetzt in einer eigenen Spalte der Linsenfläche.
    // Er ist keine Linse unter mehreren mehr, aber dieselben Zusicherungen
    // gelten weiter: lesbar, mit der echten Tabulatortaste erreichbar.
    const verlaufZweige = await page.locator('.verlaufwrap .zweig').count();
    pruefe(`Der Verlauf zeichnet etwas (${thema})`, verlaufZweige >= 1,
      `${verlaufZweige} Zweige`);
    if (verlaufZweige >= 1) {
      const wert = await messeKontrast(page, '.verlaufwrap .zweigtext');
      pruefe(`Beschriftung des Verlaufs (${thema}) erreicht 4.5:1`,
        wert !== null && wert.wert >= 4.5,
        wert ? `${wert.wert.toFixed(2)}:1 — ${wert.vorne} auf ${wert.hinten}` : 'nicht gefunden');
      pruefe(`Der Verlauf ist per Tabulator erreichbar (${thema})`,
        await erreichbarPerTabulator(page, '.verlaufwrap .zweig', 'data-spark-id'),
        'mit der echten Tabulatortaste');
    }

    // --- Die Linsen: jede muss lesbar und ohne Maus bedienbar sein, nicht nur
    // die, die zufällig zuerst angezeigt wird.
    const LINSEN_PRUEFUNG = [
      { id: 'themen', element: 'svg.graph .thema', schrift: '.themaname',
        schluessel: 'data-thema', nebensache: null },
      // Die Nebenangaben tragen die leisere Farbe und die kleinere Schrift —
      // sie sind der Teil, an dem der Kontrast zuerst reißt. Sie mitzuprüfen
      // ist der eigentliche Nachweis, nicht die kräftige Hauptzeile.
      { id: 'szenario', element: 'svg.graph .folge', schrift: '.folgentext',
        schluessel: 'data-folge', nebensache: '.haeufigkeit' },
      { id: 'herkunft', element: 'svg.graph .bezug', schrift: '.bezugname',
        schluessel: 'data-relation', nebensache: '.bezugart' },
      { id: 'zeit', element: 'svg.graph .zeitzeile', schrift: '.zeitname',
        schluessel: 'data-job-id', nebensache: '.zeitmarke' },
    ];

    for (const linse of LINSEN_PRUEFUNG) {
      const schalter = page.locator(`.linsenwahl button[data-linse="${linse.id}"]`).first();
      if ((await schalter.count()) === 0) {
        pruefe(`Linse „${linse.id}" vorhanden (${thema})`, false, 'Schalter nicht gefunden');
        continue;
      }
      await schalter.click();
      await page.waitForTimeout(250);

      const gezeichnet = await page.locator(`.graphwrap ${linse.element}`).count();
      pruefe(`Linse „${linse.id}" zeichnet etwas (${thema})`, gezeichnet >= 1,
        gezeichnet === 0
          ? 'nichts gezeichnet — diese Linse ist damit UNGEPRÜFT'
          : `${gezeichnet} Elemente`);
      if (gezeichnet === 0) continue;

      const wert = await messeKontrast(page, `.graphwrap ${linse.schrift}`);
      pruefe(`Beschriftung der Linse „${linse.id}" (${thema}) erreicht 4.5:1`,
        wert !== null && wert.wert >= 4.5,
        wert ? `${wert.wert.toFixed(2)}:1 — ${wert.vorne} auf ${wert.hinten}` : 'nicht gefunden');

      if (linse.nebensache) {
        const neben = await messeKontrast(page, `.graphwrap ${linse.nebensache}`);
        pruefe(`Nebenangabe der Linse „${linse.id}" (${thema}) erreicht 4.5:1`,
          neben !== null && neben.wert >= 4.5,
          neben
            ? `${neben.wert.toFixed(2)}:1 — ${neben.vorne} auf ${neben.hinten}, `
              + `${neben.groesse}px/${neben.gewicht}`
            : 'nicht gefunden');
      }

      pruefe(`Linse „${linse.id}" ist per Tabulator erreichbar (${thema})`,
        await erreichbarPerTabulator(page, `.graphwrap ${linse.element}`, linse.schluessel),
        'mit der echten Tabulatortaste');

      // Bedienen: die Tastatur muss dieselbe Handlung auslösen wie die Maus.
      await page.locator(`.graphwrap ${linse.element}`).first().focus();
      await page.keyboard.press('Enter');
      await page.waitForTimeout(150);
      const geoeffnet = await page.locator('.card.highlight').count();
      pruefe(`Linse „${linse.id}" öffnet per Tastatur eine Antwort (${thema})`,
        geoeffnet >= 1, `${geoeffnet} Karten geöffnet`);
    }

    await page.locator('.linsenwahl button[data-linse="stimmen"]').first().click();
    await page.waitForTimeout(200);

    // --- Tastatur: das Netz ist ohne Maus bedienbar, und der Fokus ist sichtbar.
    // Programmatisches focus() löst :focus-visible nicht aus — es muss die
    // echte Tabulatortaste sein, sonst prüft der Test etwas anderes als das,
    // was ein Mensch erlebt.
    const knoten = page.locator('svg.graph .node').first();
    const knotenId = await knoten.getAttribute('data-job-id');
    await page.locator('.graphwrap').scrollIntoViewIfNeeded();
    await page.locator('#prompt').focus();
    let getroffen = false;
    for (let i = 0; i < 400 && !getroffen; i += 1) {
      await page.keyboard.press('Tab');
      getroffen = await page.evaluate(
        (id) => document.activeElement?.getAttribute?.('data-job-id') === id,
        knotenId,
      );
    }
    pruefe(`Knoten ist per Tabulator erreichbar (${thema})`, getroffen, knotenId ?? '');

    if (getroffen) {
      const sichtbar = await knoten.evaluate((el) => {
        const kreis = el.querySelector('circle');
        return kreis ? getComputedStyle(kreis).strokeWidth : '';
      });
      pruefe(`Fokus am Knoten ist sichtbar (${thema})`,
        parseFloat(sichtbar) >= 2.5, `Strichbreite ${sichtbar}`);

      await page.keyboard.press('Enter');
      const markiert = await page.locator('.card.highlight').count();
      pruefe(`Knoten per Tastatur bedienbar (${thema})`, markiert === 1,
        `${markiert} Karten geöffnet`);
    }

    const kante = page.locator('svg.graph .edge').first();
    if (await kante.count()) {
      await kante.focus();
      await page.keyboard.press('Enter');
      const paar = await page.locator('.card.highlight').count();
      pruefe(`Kante per Tastatur bedienbar (${thema})`, paar === 2, `${paar} Karten`);
    }

    await context.close();
  }

  // --- Jeder Zustand einzeln: die Schilder müssen alle lesbar sein, nicht nur
  // das eine, das im Durchlauf zufällig entsteht. Dafür braucht es die
  // Zustandsschau aus tools/zustaende_seed.py.
  const alleSitzungen = await (await fetch(`${BASE}/api/sessions`)).json();
  const schau = (alleSitzungen.sessions ?? []).find((x) => x.title.startsWith('Zustandsschau'));
  if (!schau) {
    pruefe('Zustandsschau vorhanden', false,
      'nicht angelegt — die Kontraste der einzelnen Zustandsschilder sind UNGEPRÜFT ' +
      '(tools/zustaende_seed.py ausführen)');
  } else {
    for (const thema of ['light', 'dark']) {
      const ctx = await browser.newContext({
        viewport: { width: 1280, height: 900 }, colorScheme: thema,
      });
      const seite = await ctx.newPage();
      await seite.goto(`${BASE}/#/s/${schau.id}`, { waitUntil: 'domcontentloaded' });
      await seite.waitForSelector('.card .tag.zustand');
      const zustaende = await seite.locator('.card').evaluateAll((karten) =>
        karten.map((k) => k.className.match(/zustand-(\w+)/)?.[1]).filter(Boolean));
      const gesehen = new Set();
      for (const zustand of zustaende) {
        if (gesehen.has(zustand)) continue;
        gesehen.add(zustand);
        const wert = await messeKontrast(seite, `.card.zustand-${zustand} .tag.zustand`);
        pruefe(`Schild „${zustand}" (${thema}) erreicht 4.5:1`,
          wert !== null && wert.wert >= 4.5,
          wert ? `${wert.wert.toFixed(2)}:1 — ${wert.vorne} auf ${wert.hinten}` : 'nicht gefunden');
      }
      pruefe(`Alle acht Zustände in der Schau vorhanden (${thema})`,
        gesehen.size >= 8, `${gesehen.size}: ${[...gesehen].join(', ')}`);
      await ctx.close();
    }
  }

  // --- Themenvorrang: ausdrückliche Wahl schlägt die Vorgabe des Geräts.
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 }, colorScheme: 'dark',
  });
  const page = await context.newPage();
  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#thema-schalter');
  const grund = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  await page.click('#thema-schalter'); // system -> hell
  const hell = await page.evaluate(() => ({
    attr: document.documentElement.getAttribute('data-theme'),
    grund: getComputedStyle(document.body).backgroundColor,
  }));
  pruefe('Ausdrücklich hell schlägt ein dunkles Gerät',
    hell.attr === 'light' && hell.grund !== grund, `${grund} → ${hell.grund}`);

  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#thema-schalter');
  const nachNeuladen = await page.evaluate(() =>
    document.documentElement.getAttribute('data-theme'));
  pruefe('Die Wahl überdauert das Neuladen', nachNeuladen === 'light', String(nachNeuladen));

  await page.click('#thema-schalter'); // hell -> dunkel
  const dunkel = await page.evaluate(() =>
    document.documentElement.getAttribute('data-theme'));
  pruefe('Ausdrücklich dunkel ist erreichbar', dunkel === 'dark', String(dunkel));

  await page.click('#thema-schalter'); // dunkel -> system
  const zurueck = await page.evaluate(() => ({
    attr: document.documentElement.getAttribute('data-theme'),
    grund: getComputedStyle(document.body).backgroundColor,
  }));
  pruefe('Zurück zur Vorgabe des Geräts',
    zurueck.attr === null && zurueck.grund === grund, `${zurueck.grund}`);

  // --- Schmale Breiten, auch in geöffneten Panels.
  await page.fill('#prompt', 'Kurze Frage für den Breitentest.');
  await page.click('form.spark button.primary');
  await page.waitForSelector('.panel-grid table', { timeout: 120000 });
  await page.click('#settings-open');
  await page.fill('#admin-token', ZUGANGSWORT);
  await page.click('#einstellungen button.primary');
  await page.waitForSelector('#kurator', { timeout: 20000 });
  await page.locator('.pingpong > summary').click();
  for (const gruppe of await page.locator('.weitergabe').all()) {
    await gruppe.evaluate((el) => el.setAttribute('open', ''));
  }
  for (const breite of [320, 390, 768, 1440]) {
    await page.setViewportSize({ width: breite, height: 900 });
    await page.waitForTimeout(250);
    const mass = await page.evaluate(() => {
      let schuldig = '';
      let max = 0;
      for (const el of document.querySelectorAll('body *')) {
        const r = el.getBoundingClientRect();
        if (r.right > max) { max = r.right; schuldig = el.tagName + '.' + el.className; }
      }
      return {
        scroll: document.documentElement.scrollWidth,
        client: document.documentElement.clientWidth,
        max: Math.round(max), schuldig,
      };
    });
    pruefe(`Keine Überbreite bei ${breite}px mit offenen Panels`,
      mass.scroll <= mass.client + 1,
      `scroll ${mass.scroll} / client ${mass.client}; breitestes bis ${mass.max}px (${mass.schuldig})`);
  }

  // --- Reduzierte Bewegung: dieselbe Information, ohne Animation.
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.card');
  const ruhig = await page.evaluate(() => {
    const karte = document.querySelector('.card');
    const text = document.querySelector('.answer');
    return {
      sichtbar: !!text && text.textContent.trim().length > 0,
      dauer: karte ? getComputedStyle(karte).transitionDuration : '',
    };
  });
  pruefe('Bei reduzierter Bewegung bleibt der Inhalt vollständig da', ruhig.sichtbar);
  pruefe('Übergänge sind bei reduzierter Bewegung praktisch aus',
    parseFloat(ruhig.dauer) < 0.01, `transition-duration ${ruhig.dauer}`);

  await context.close();
} finally {
  await browser.close();
}

const fehlgeschlagen = ergebnisse.filter((r) => !r.ok);
console.log(`\n${ergebnisse.length - fehlgeschlagen.length}/${ergebnisse.length} Prüfungen bestanden.`);
process.exit(fehlgeschlagen.length === 0 ? 0 : 1);
