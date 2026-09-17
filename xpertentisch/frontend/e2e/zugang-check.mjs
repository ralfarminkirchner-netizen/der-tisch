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

    // --- Die Linsen: jede muss lesbar und ohne Maus bedienbar sein, nicht nur
    // die, die zufällig zuerst angezeigt wird.
    for (const linse of ['themen', 'verlauf']) {
      const schalter = page.locator(`.linsenwahl button[data-linse="${linse}"]`).first();
      if ((await schalter.count()) === 0) {
        pruefe(`Linse „${linse}" vorhanden (${thema})`, false, 'Schalter nicht gefunden');
        continue;
      }
      await schalter.click();
      await page.waitForTimeout(250);

      const gezeichnet = await page.locator(
        linse === 'themen' ? 'svg.graph .thema' : 'svg.graph .zweig',
      ).count();
      pruefe(`Linse „${linse}" zeichnet etwas (${thema})`, gezeichnet >= 1,
        `${gezeichnet} Elemente`);

      if (gezeichnet >= 1) {
        const beschriftung = linse === 'themen' ? '.themaname' : '.zweigtext';
        const wert = await messeKontrast(page, `svg.graph ${beschriftung}`);
        pruefe(`Beschriftung der Linse „${linse}" (${thema}) erreicht 4.5:1`,
          wert !== null && wert.wert >= 4.5,
          wert ? `${wert.wert.toFixed(2)}:1 — ${wert.vorne} auf ${wert.hinten}` : 'nicht gefunden');

        // Bedienbarkeit über die echte Tabulatortaste, nicht über focus().
        const ziel = page.locator(
          linse === 'themen' ? 'svg.graph .thema' : 'svg.graph .zweig',
        ).first();
        const kennung = await ziel.evaluate((el) =>
          el.getAttribute('data-thema') ?? el.getAttribute('data-spark-id'));
        await page.locator('#prompt').focus();
        let erreicht = false;
        for (let i = 0; i < 400 && !erreicht; i += 1) {
          await page.keyboard.press('Tab');
          erreicht = await page.evaluate((k) => {
            const a = document.activeElement;
            return !!a && (a.getAttribute?.('data-thema') === k
              || a.getAttribute?.('data-spark-id') === k);
          }, kennung);
        }
        pruefe(`Linse „${linse}" ist per Tabulator erreichbar (${thema})`,
          erreicht, String(kennung));
      }
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
