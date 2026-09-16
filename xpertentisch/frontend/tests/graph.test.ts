import { describe, expect, it } from 'vitest';

import { renderGraph } from '../src/graph';
import type { Summary } from '../src/types';

const summary: Summary = {
  spark_id: 'f1',
  computed_at: 0,
  models: [
    { job_id: 'j1', model_id: 'a', label: 'Modell A', provider: 'fake', status: 'done', chars: 100, sentences: 4, latency_ms: 20, partial: false, error: null, unique: 1, agreements: 2, contradictions: 1 },
    { job_id: 'j2', model_id: 'b', label: 'Modell B', provider: 'fake', status: 'done', chars: 90, sentences: 3, latency_ms: 30, partial: false, error: null, unique: 0, agreements: 2, contradictions: 1 },
  ],
  pairs: [
    { a_job_id: 'j1', b_job_id: 'j2', a_model_id: 'a', b_model_id: 'b', a_label: 'Modell A', b_label: 'Modell B', agreements: 2, contradictions: 1, topics: ['sicherung'] },
  ],
  counts: { uebereinstimmung: 2, widerspruch: 1, einzigartig: 1 },
  analysed_jobs: ['j1', 'j2'],
  method: 'regelbasiert',
};

describe('renderGraph', () => {
  it('zeichnet Knoten und Kanten', () => {
    const svg = renderGraph(summary, () => {});
    expect(svg.querySelectorAll('.node')).toHaveLength(2);
    expect(svg.querySelectorAll('.edge.agree')).toHaveLength(1);
    expect(svg.querySelectorAll('.edge.contra')).toHaveLength(1);
  });

  it('öffnet beim Knotenklick genau die eine Antwort', () => {
    const selections: string[][] = [];
    const svg = renderGraph(summary, (s) => selections.push(s.jobIds));
    const node = svg.querySelector<SVGGElement>('.node[data-job-id="j2"]')!;
    node.dispatchEvent(new MouseEvent('click'));
    expect(selections).toEqual([['j2']]);
  });

  it('öffnet beim Kantenklick beide beteiligten Antworten', () => {
    const selections: string[][] = [];
    const svg = renderGraph(summary, (s) => selections.push(s.jobIds));
    svg.querySelector<SVGLineElement>('.edge.contra')!.dispatchEvent(new MouseEvent('click'));
    expect(selections).toEqual([['j1', 'j2']]);
  });

  it('ist über die Tastatur bedienbar', () => {
    const selections: string[][] = [];
    const svg = renderGraph(summary, (s) => selections.push(s.jobIds));
    const node = svg.querySelector<SVGGElement>('.node[data-job-id="j1"]')!;
    node.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(selections).toEqual([['j1']]);
  });

  it('zeigt einen Hinweis, wenn nichts auswertbar ist', () => {
    const leer: Summary = { ...summary, models: [], pairs: [], analysed_jobs: [] };
    const svg = renderGraph(leer, () => {});
    expect(svg.querySelectorAll('.node')).toHaveLength(0);
    expect(svg.textContent).toContain('Noch keine auswertbaren Antworten');
  });
});
