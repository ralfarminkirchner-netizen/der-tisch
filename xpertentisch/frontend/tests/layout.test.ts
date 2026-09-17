import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { createCard, renderTable } from '../src/render';
import type { Job, JobStatus, Summary } from '../src/types';

/** Die acht Sachverhalte, die nicht zusammenfallen dürfen. */
const ZUSTAENDE: JobStatus[] = [
  'not_requested', 'queued', 'running', 'streaming',
  'done', 'error', 'interrupted', 'cancelled',
];

const css = readFileSync(resolve(__dirname, '../src/styles.css'), 'utf8');

describe('Mobile Ansicht', () => {
  it('nutzt keine festen Breiten, die über den Viewport hinausragen', () => {
    expect(css).not.toMatch(/width:\s*100vw/);
    // Breite min-width-Angaben nur in Media Queries ("(min-width: 760px)"),
    // nicht als Elementbreite — sonst entsteht auf schmalen Geräten Überbreite.
    expect(css).not.toMatch(/[^(]min-width:\s*[5-9]\d{2,}px/);
  });

  it('lässt Raster auf schmalen Geräten umbrechen', () => {
    // minmax(min(...)) verhindert, dass Spalten breiter als der Viewport werden.
    const grids = css.match(/repeat\(auto-fit,\s*minmax\([^)]*\)[^)]*\)/g) ?? [];
    expect(grids.length).toBeGreaterThan(0);
    for (const grid of grids) expect(grid).toContain('min(');
  });

  it('bricht lange Wörter und URLs um', () => {
    expect(css).toMatch(/overflow-wrap:\s*anywhere/);
  });

  it('gibt breiten Elementen einen eigenen Scrollbereich', () => {
    expect(css).toMatch(/\.tablewrap\s*\{[^}]*overflow-x:\s*auto/);
    expect(css).toMatch(/\.graphwrap\s*\{[^}]*max-width:\s*100%/);
  });

  it('skaliert das Diagramm mit der Breite', () => {
    expect(css).toMatch(/svg\.graph\s*\{[^}]*width:\s*100%/);
  });

  it('setzt eine Breitenbegrenzung auf Wurzelelementen', () => {
    expect(css).toMatch(/html,\s*body\s*\{[^}]*max-width:\s*100%/);
  });
});

const job: Job = {
  id: 'j1', session_id: 's', spark_id: 'f', model_id: 'a', label: 'Modell A',
  provider: 'fake', model: 'm', status: 'done',
  text: 'Ein sehr langes Wort: ' + 'a'.repeat(300),
  error: null, partial: false, latency_ms: 12,
  tokens_in: null, tokens_out: null, cost_micro: null, cost_source: 'unbekannt',
  created_at: 1000, started_at: 1000.2, finished_at: 1001,
};

describe('Modellkarte', () => {
  it('hält den Text in einem umbruchfähigen Block', () => {
    const card = createCard(job, []);
    expect(card.querySelector('.answer')).not.toBeNull();
    expect(card.getAttribute('data-job-id')).toBe('j1');
  });

  it('zeigt Fehler, ohne den Text anderer Karten zu berühren', () => {
    const fehler = { ...job, id: 'j2', status: 'error' as const, text: '', error: 'Ausfall' };
    const a = createCard(job, []);
    const b = createCard(fehler, []);
    expect(b.querySelector('.errorbox')?.textContent).toBe('Ausfall');
    expect(a.querySelector('.answer')?.textContent).toContain('Ein sehr langes Wort');
  });

  it('kennzeichnet Teilantworten', () => {
    const card = createCard({ ...job, partial: true }, []);
    expect(card.textContent).toContain('Teilantwort');
  });
});

describe('Vergleichstabelle', () => {
  it('steckt in einem eigenen Scrollbereich', () => {
    const summary: Summary = {
      spark_id: 'f', computed_at: 0,
      models: [{ job_id: 'j1', model_id: 'a', label: 'Modell A', provider: 'fake', status: 'done', chars: 10, sentences: 1, latency_ms: 5, partial: false, error: null, unique: 0, agreements: 0, contradictions: 0 }],
      pairs: [], counts: { uebereinstimmung: 0, widerspruch: 0, einzigartig: 0 },
      analysed_jobs: ['j1'], method: 'regelbasiert',
    };
    const panel = renderTable(summary);
    expect(panel.querySelector('.tablewrap table')).not.toBeNull();
    expect(panel.textContent).toContain('keine Rangfolge');
  });
});

describe('Verstecken von Bausteinen', () => {
  it('überstimmt die Anzeigeart, damit [hidden] wirklich versteckt', () => {
    // Ohne diese Regel bleibt ein .row mit display:flex trotz hidden sichtbar.
    expect(css).toMatch(/\[hidden\]\s*\{[^}]*display:\s*none\s*!important/);
  });
});

describe('Werkbank-Gestaltung', () => {
  it('bringt Farbe nur mit Bedeutung ins Spiel', () => {
    for (const marke of ['--einig', '--gegen', '--einzeln', '--marke']) {
      expect(css).toContain(`${marke}:`);
    }
  });

  it('beschreibt alle drei Theme-Zustände', () => {
    expect(css).toMatch(/^:root \{/m);
    expect(css).toMatch(/@media \(prefers-color-scheme: dark\)/);
    expect(css).toMatch(/:root:not\(\[data-theme='light'\]\)/);
    expect(css).toMatch(/:root\[data-theme='dark'\]/);
  });

  it('malt den Untergrund selbst, statt ihn vom Wirt zu erben', () => {
    expect(css).toMatch(/body \{[^}]*background: var\(--flaeche\)/);
  });

  it('trägt den Zustand einer Karte an der oberen Lichtkante', () => {
    // Geändert gegenüber der Werkbank: die Kante ist keine border-top-color
    // mehr, sondern eine eigene Fläche, die ihre Farbe aus --zustandsfarbe
    // bezieht. Geprüft wird weiterhin dasselbe: der Zustand steht an der
    // oberen Kante, und Fehler bekommt die Widerspruchsfarbe.
    expect(css).toMatch(/\.card::before \{[^}]*background: var\(--zustandsfarbe/);
    expect(css).toMatch(/\.card\.zustand-error \{ --zustandsfarbe: var\(--gegen\)/);
  });
});

describe('Laufende und abgebrochene Karten', () => {
  it('zeigt „schreibt“, solange der Text noch wächst', () => {
    const card = createCard({ ...job, status: 'streaming', text: 'Anfang …' }, []);
    expect(card.className).toContain('zustand-streaming');
    expect(card.textContent).toContain('schreibt');
    // Der bisher eingetroffene Text steht schon da, nicht erst am Ende.
    expect(card.querySelector('.answer')?.textContent).toContain('Anfang');
  });

  it('nennt einen Abbruch beim Namen, ohne ihn zum Fehler zu erklären', () => {
    const card = createCard(
      { ...job, status: 'cancelled', text: '', error: 'Von dir abgebrochen.' },
      [],
    );
    expect(card.textContent).toContain('abgebrochen');
    expect(card.textContent).toContain('Abgebrochen, bevor eine Antwort kam.');
  });

  it('verschweigt bei laufenden Karten die Dauer, weil sie noch nicht feststeht', () => {
    const card = createCard({ ...job, status: 'streaming', latency_ms: 900 }, []);
    // Die Dauer steht jetzt in der Nebenangabenzeile — dort darf sie ebenso
    // wenig auftauchen, solange sie nicht feststeht.
    expect(card.querySelector('.leiste')?.textContent).not.toContain('900');
    expect(card.querySelector('.leiste')?.textContent).not.toContain('0,9');
  });

  it('gibt jedem der acht Zustände eine eigene Darstellung', () => {
    // Dieselbe Formänderung wie oben; die Aussage bleibt: acht Sachverhalte,
    // acht unterscheidbare Darstellungen.
    for (const zustand of ZUSTAENDE) {
      expect(css).toMatch(new RegExp(`\\.card\\.zustand-${zustand}\\b`));
    }
    expect(css).toMatch(/\.tag\.streaming/);
    expect(css).toMatch(/\.tag\.cancelled/);
  });

  it('macht den Zustand nie allein an der Farbe fest', () => {
    // Jeder Zustand trägt zusätzlich eine Beschriftung, und das Zustandsschild
    // bekommt einen eigenen Punkt — wer Farben nicht unterscheidet, liest ihn
    // trotzdem.
    const beschriftungen = new Set<string>();
    for (const zustand of ZUSTAENDE) {
      const card = createCard({ ...job, status: zustand }, []);
      const schild = card.querySelector('.tag.zustand');
      expect(schild, `Zustand ${zustand} ohne Schild`).not.toBeNull();
      const text = (schild!.textContent ?? '').trim();
      expect(text.length, `Zustand ${zustand} ohne Beschriftung`).toBeGreaterThan(0);
      beschriftungen.add(text);
    }
    expect(beschriftungen.size).toBe(ZUSTAENDE.length);
    expect(css).toMatch(/\.tag\.zustand::before \{[^}]*content: ''/);
  });

  it('hält die Schreibanimation für Menschen zurück, die keine Bewegung wollen', () => {
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\)/);
  });
});

describe('Kosten auf der Karte', () => {
  it('schätzt nichts, wenn keine Preise hinterlegt sind', () => {
    const card = createCard(
      { ...job, tokens_in: 100, tokens_out: 200, cost_micro: null, cost_source: 'unbekannt' },
      [],
    );
    // Geändert: Verbrauch und Kosten stehen nicht mehr als gleichrangige
    // Schilder, sondern in der Nebenangabenzeile. Geprüft bleibt, DASS der
    // gemeldete Verbrauch dasteht und dass ohne Preise nichts geschätzt wird.
    const leiste = card.querySelector('.leiste')!;
    expect(leiste.textContent).toContain('Token');
    expect(leiste.textContent).toContain('100/200');
    expect(leiste.textContent).toContain('Kosten unbekannt');
  });

  it('zeigt einen Betrag nur, wenn er wirklich gerechnet wurde', () => {
    const card = createCard(
      { ...job, tokens_in: 100, tokens_out: 200, cost_micro: 4200, cost_source: 'berechnet' },
      [],
    );
    expect(card.textContent).not.toContain('Kosten unbekannt');
  });
});

describe('Nebenangaben auf der Karte', () => {
  it('trennt ohne Trennzeichen, damit beim Umbruch kein Rest stehenbleibt', () => {
    // Ein '·' als ::before am Folgeelement landet beim Zeilenumbruch am
    // Zeilenanfang. Getrennt wird darum über Abstand und Benennung.
    expect(css).not.toMatch(/\.leiste > span \+ span::before/);
    expect(css).toMatch(/\.leiste \{[^}]*gap:/);
  });

  it('benennt jede Angabe, statt bloße Zahlen aneinanderzureihen', () => {
    const card = createCard(
      { ...job, latency_ms: 2300, tokens_in: 180, tokens_out: 140 },
      [],
    );
    const leiste = card.querySelector('.leiste')!;
    const namen = [...leiste.querySelectorAll('.leiste-name')].map((n) => n.textContent);
    expect(namen).toContain('Dauer');
    expect(namen).toContain('Token');
    // Die Herkunft steht für sich und braucht keine Benennung.
    expect(leiste.querySelector('.leiste-herkunft')?.textContent).toContain('fake');
  });
});

describe('Hinweis auf neue Beiträge', () => {
  it('liegt über dem Text und drängt sich nicht in den Lesefluss', () => {
    expect(css).toMatch(/\.neue-beitraege \{[^}]*position:\s*fixed/);
  });
});

describe('Die Linsenfläche', () => {
  it('stapelt auf schmalen Geräten und stellt erst mit Platz nebeneinander', () => {
    expect(css).toMatch(/\.linsenraster \{[^}]*grid-template-columns: minmax\(0, 1fr\)/);
    expect(css).toMatch(
      /@media \(min-width: 860px\) \{\s*\.linsenraster \{[^}]*grid-template-columns:/,
    );
  });

  it('lässt fünf Linsenschalter umbrechen, statt die Seite breiter zu machen', () => {
    expect(css).toMatch(/\.linsenwahl \{[^}]*flex-wrap: wrap/);
    expect(css).toMatch(/\.linsenwahl \{[^}]*max-width: 100%/);
  });

  it('gibt dem Verlaufsbaum einen eigenen Scrollbereich statt ihn zu schrumpfen', () => {
    expect(css).toMatch(/\.verlaufwrap \{[^}]*overflow-x: auto/);
    expect(css).toMatch(/\.verlaufwrap svg\.graph \{[^}]*width: auto/);
  });

  it('begrenzt die zeilenförmigen Linsen, damit ihre Schrift nicht mitwächst', () => {
    expect(css).toMatch(
      /svg\.graph\.szenario,\s*svg\.graph\.herkunft,\s*svg\.graph\.zeit \{[^}]*max-width:/,
    );
  });
});

describe('Die neuen Linsen in der Gestaltung', () => {
  it('gibt jeder bedienbaren Linsenzeile einen sichtbaren Fokus', () => {
    for (const teil of ['.folge', '.bezug', '.zeitzeile']) {
      const regel = new RegExp(
        `svg\\.graph\\.\\w+ \\${teil}:focus-visible [^{]*\\{[^}]*stroke: var\\(--marke\\)`,
      );
      expect(css, `${teil} braucht einen sichtbaren Fokus`).toMatch(regel);
    }
  });

  it('kodiert den Stand eines Bezugs über die Strichart, nicht über eine Farbe', () => {
    expect(css).toMatch(/\.bezug\.bestaetigt \.bezugstrich \{ stroke-dasharray: none/);
    expect(css).toMatch(/\.bezug\.vorschlag \.bezugstrich \{ stroke-dasharray: \d/);
    expect(css).toMatch(/\.bezug\.abgelehnt \.bezugstrich \{[^}]*stroke-dasharray: /);
  });

  it('unterscheidet die Herkunft über die Form, nicht über eine eigene Farbe', () => {
    // Beide Zeichen tragen dieselben Farbtöne; verschieden ist die Form
    // (gefüllte Raute gegenüber offenem Ring) und die Füllung.
    expect(css).toMatch(/\.herkunftszeichen \{[^}]*fill: var\(--tinte-matt\)/);
    expect(css).toMatch(/\.herkunftszeichen\.maschine \{[^}]*fill: var\(--flaeche-tief\)/);
  });

  it('gibt der Häufigkeit einer Folge ihre Bedeutungsfarbe — dieselben drei wie überall', () => {
    expect(css).toMatch(/\.stimmpunkt\.voll\.agree \{ fill: var\(--einig\)/);
    expect(css).toMatch(/\.stimmpunkt\.voll\.contra \{ fill: var\(--gegen\)/);
    expect(css).toMatch(/\.stimmpunkt\.voll\.unique \{ fill: var\(--einzeln\)/);
  });

  it('richtet den Begriff der Themen-Linse hinter seinem Punkt aus', () => {
    // Die Sammelregel für Knotentexte setzt `middle` und überstimmt das
    // Attribut am Element. Ohne diese Regel lag die Scheibe im Wort.
    expect(css).toMatch(/\.thema \.themaname \{ text-anchor: start/);
    expect(css).toMatch(/\.stimme \.zahl \{ text-anchor: end/);
  });
});
