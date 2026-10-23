import { describe, expect, it } from 'vitest';

import { baueZweige, themenAusMarkern, renderThemen, renderVerlauf } from '../src/linsen';
import type { Job, Marker, SessionBundle, SparkEntry, Summary } from '../src/types';

function marker(over: Partial<Marker> & Pick<Marker, 'job_id' | 'kind' | 'topics'>): Marker {
  return {
    id: `m-${Math.random()}`, session_id: 's', spark_id: 'f1', related_job_id: null,
    start_offset: 0, end_offset: 10, quote: 'Ein Satz.', note: '',
    ...over,
  } as Marker;
}

describe('Themen aus Fundstellen', () => {
  it('sammelt jeden Begriff mit den Stimmen, die daran hängen', () => {
    const themen = themenAusMarkern([
      marker({ job_id: 'j1', kind: 'uebereinstimmung', topics: ['sicherung', 'probe'] }),
      marker({ job_id: 'j2', kind: 'uebereinstimmung', topics: ['sicherung'] }),
      marker({ job_id: 'j1', kind: 'widerspruch', topics: ['rhythmus'] }),
    ]);
    const nach = new Map(themen.map((t) => [t.begriff, t]));
    expect([...nach.get('sicherung')!.einig].sort()).toEqual(['j1', 'j2']);
    expect([...nach.get('rhythmus')!.gegen]).toEqual(['j1']);
    expect(nach.get('probe')!.einig.size).toBe(1);
  });

  it('stellt die schwersten Begriffe nach vorn und bleibt dabei stabil', () => {
    const themen = themenAusMarkern([
      marker({ job_id: 'j1', kind: 'uebereinstimmung', topics: ['viele'] }),
      marker({ job_id: 'j2', kind: 'uebereinstimmung', topics: ['viele'] }),
      marker({ job_id: 'j3', kind: 'uebereinstimmung', topics: ['viele'] }),
      marker({ job_id: 'j1', kind: 'einzigartig', topics: ['wenige'] }),
    ]);
    expect(themen[0].begriff).toBe('viele');
    // Zweimal dieselbe Eingabe ergibt dieselbe Reihenfolge.
    const nochmal = themenAusMarkern([
      marker({ job_id: 'j1', kind: 'einzigartig', topics: ['b'] }),
      marker({ job_id: 'j2', kind: 'einzigartig', topics: ['a'] }),
    ]);
    expect(nochmal.map((t) => t.begriff)).toEqual(['a', 'b']);
  });

  it('erfindet nichts: ohne Fundstellen gibt es keine Begriffe', () => {
    expect(themenAusMarkern([])).toEqual([]);
    expect(themenAusMarkern([marker({ job_id: 'j1', kind: 'einzigartig', topics: [] })]))
      .toEqual([]);
  });
});

function job(id: string, spark: string): Job {
  return {
    id, session_id: 's', spark_id: spark, model_id: 'a', label: 'A',
    provider: 'fake', model: 'm', status: 'done', text: 'x', error: null,
    partial: false, latency_ms: 1, tokens_in: null, tokens_out: null,
    cost_micro: null, cost_source: 'unbekannt',
  } as Job;
}

function eintrag(id: string, seq: number, kind: string, refs: string[]): SparkEntry {
  return {
    spark: {
      id, session_id: 's', seq, prompt: `Frage ${seq}`, client_request_id: `r${seq}`,
      created_at: seq, kind, refs,
    },
    jobs: [job(`${id}-j`, id)],
    markers: [],
    summary: null,
  } as unknown as SparkEntry;
}

describe('Verlaufsbaum', () => {
  it('hängt einen Beitrag an den, aus dessen Antwort er hervorging', () => {
    const zweige = baueZweige([
      eintrag('f1', 1, 'funke', []),
      eintrag('f2', 2, 'gegenposition', ['f1-j']),
      eintrag('f3', 3, 'szenario', ['f2-j']),
    ]);
    expect(zweige.map((z) => z.tiefe)).toEqual([0, 1, 2]);
    expect(zweige[1].elternId).toBe('f1');
    expect(zweige[2].elternId).toBe('f2');
  });

  it('lässt einen Beitrag ohne Bezug als eigenen Ursprung stehen', () => {
    const zweige = baueZweige([
      eintrag('f1', 1, 'funke', []),
      eintrag('f2', 2, 'funke', []),
    ]);
    expect(zweige.every((z) => z.tiefe === 0 && z.elternId === null)).toBe(true);
  });

  it('kommt mit einem Ring in den Bezügen zurecht, statt sich aufzuhängen', () => {
    const zweige = baueZweige([
      eintrag('f1', 1, 'funke', ['f2-j']),
      eintrag('f2', 2, 'antwort', ['f1-j']),
    ]);
    expect(zweige).toHaveLength(2);
    expect(zweige.every((z) => Number.isFinite(z.tiefe))).toBe(true);
  });
});

const summary: Summary = {
  spark_id: 'f1', computed_at: 0,
  models: [
    { job_id: 'j1', model_id: 'a', label: 'Alto', provider: 'fake', status: 'done', chars: 10, sentences: 2, latency_ms: 5, partial: false, error: null, unique: 0, agreements: 1, contradictions: 1 },
    { job_id: 'j2', model_id: 'b', label: 'Basso', provider: 'fake', status: 'done', chars: 10, sentences: 2, latency_ms: 5, partial: false, error: null, unique: 0, agreements: 1, contradictions: 1 },
  ],
  pairs: [], counts: { uebereinstimmung: 1, widerspruch: 1, einzigartig: 0 },
  analysed_jobs: ['j1', 'j2'], method: 'regelbasiert',
};

describe('Themen-Linse im Bild', () => {
  it('öffnet beim Klick auf einen Begriff alle Stimmen, die daran hängen', () => {
    const auswahl: string[][] = [];
    const svg = renderThemen(
      summary,
      [
        marker({ job_id: 'j1', kind: 'widerspruch', topics: ['sicherung'] }),
        marker({ job_id: 'j2', kind: 'widerspruch', topics: ['sicherung'] }),
      ],
      (s) => auswahl.push(s.jobIds.sort()),
    );
    const thema = svg.querySelector<SVGGElement>('.thema[data-thema="sicherung"]')!;
    expect(thema).not.toBeNull();
    thema.dispatchEvent(new MouseEvent('click'));
    expect(auswahl).toEqual([['j1', 'j2']]);
  });

  it('ist über die Tastatur bedienbar', () => {
    const auswahl: string[][] = [];
    const svg = renderThemen(
      summary,
      [marker({ job_id: 'j1', kind: 'uebereinstimmung', topics: ['probe'] })],
      (s) => auswahl.push(s.jobIds),
    );
    const thema = svg.querySelector<SVGGElement>('.thema')!;
    expect(thema.getAttribute('tabindex')).toBe('0');
    thema.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(auswahl).toEqual([['j1']]);
  });

  it('sagt es, wenn es nichts zu zeigen gibt', () => {
    const svg = renderThemen(summary, [], () => {});
    expect(svg.textContent).toContain('Noch keine Begriffe gefunden');
  });
});

describe('Verlauf im Bild', () => {
  const bundle = {
    sparks: [
      eintrag('f1', 1, 'funke', []),
      eintrag('f2', 2, 'szenario', ['f1-j']),
    ],
  } as unknown as SessionBundle;

  it('zeichnet einen Kasten je Beitrag und hebt den aktuellen hervor', () => {
    const svg = renderVerlauf(bundle, 'f2', () => {});
    expect(svg.querySelectorAll('.zweig')).toHaveLength(2);
    expect(svg.querySelector('.zweig.aktuell')?.getAttribute('data-spark-id')).toBe('f2');
    expect(svg.querySelectorAll('.ast')).toHaveLength(1);
  });

  it('führt beim Klick zu genau diesem Beitrag', () => {
    const ziele: string[] = [];
    const svg = renderVerlauf(bundle, null, (id) => ziele.push(id));
    svg.querySelector<SVGGElement>('.zweig[data-spark-id="f2"]')!
      .dispatchEvent(new MouseEvent('click'));
    expect(ziele).toEqual(['f2']);
  });
});
