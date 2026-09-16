import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { createCard, renderTable } from '../src/render';
import type { Job, Summary } from '../src/types';

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

  it('trägt den Zustand einer Karte an der oberen Kante', () => {
    expect(css).toMatch(/\.card\.zustand-error \{ border-top-color: var\(--gegen\)/);
  });
});
