/**
 * Die neuen Linsen: Folgen, Herkunft, Zeit.
 *
 * Jede wird mit leeren Daten, mit einer Stimme, mit fünfen und mit
 * widersprüchlichen Daten geprüft. Eine Linse, die nur mit den Demo-Daten
 * schön aussieht, ist nicht fertig — und eine Zusicherung, die auch über
 * leeren Daten besteht, sichert nichts zu.
 */
import { describe, expect, it } from 'vitest';

import {
  LINSEN,
  dauerText,
  folgenArt,
  renderHerkunft,
  renderSzenario,
  renderZeit,
  umbruch,
} from '../src/linsen';
import type { Folge, Job, Relation, Szenario } from '../src/types';

function folge(over: Partial<Folge> & Pick<Folge, 'id' | 'text'>): Folge {
  return {
    themen: [],
    nennungen: [],
    anzahl: 1,
    von: 3,
    gegensatz: [],
    ...over,
  };
}

function nennung(job_id: string, label: string, quote: string) {
  return { job_id, label, quote, start_offset: 0, end_offset: quote.length };
}

function szenario(over: Partial<Szenario> = {}): Szenario {
  return {
    spark_id: 'f2',
    seq: 2,
    prompt: 'Welche Folgen hätte das?',
    ausgang: { job_id: 'j0', label: 'Alto', auszug: 'Täglich sichern.', gekuerzt: false },
    stimmen: [
      { job_id: 'j1', label: 'Alto' },
      { job_id: 'j2', label: 'Basso' },
      { job_id: 'j3', label: 'Cantus' },
    ],
    folgen: [],
    uebergangen: 0,
    methode: 'regelbasiert',
    ...over,
  };
}

function job(over: Partial<Job> & Pick<Job, 'id' | 'label'>): Job {
  return {
    session_id: 's', spark_id: 'f1', model_id: over.id, provider: 'fake', model: 'm',
    status: 'done', text: 'x', error: null, partial: false, latency_ms: 100,
    tokens_in: null, tokens_out: null, cost_micro: null, cost_source: 'unbekannt',
    created_at: 1000, started_at: 1000.1, finished_at: 1001,
    ...over,
  } as Job;
}

// ----------------------------------------------------------------- Umbruch

describe('Umbruch im SVG', () => {
  it('bricht an Wortgrenzen und zerschneidet kein Wort', () => {
    const zeilen = umbruch('eins zwei drei vier fünf sechs', 10, 9);
    expect(zeilen.every((z) => z.length <= 10 || !z.includes(' '))).toBe(true);
    expect(zeilen.join(' ')).toBe('eins zwei drei vier fünf sechs');
  });

  it('lässt ein einzelnes zu langes Wort lieber überstehen als es zu zerreißen', () => {
    expect(umbruch('Donaudampfschifffahrtsgesellschaft', 10, 4))
      .toEqual(['Donaudampfschifffahrtsgesellschaft']);
  });

  it('kappt mit Auslassungszeichen, statt stillschweigend abzuschneiden', () => {
    const zeilen = umbruch('a '.repeat(80), 10, 3);
    expect(zeilen).toHaveLength(3);
    expect(zeilen[2].endsWith('…')).toBe(true);
  });

  it('macht aus leerem Text keine leere Zeile', () => {
    expect(umbruch('   ', 10, 3)).toEqual([]);
    expect(umbruch('', 10, 3)).toEqual([]);
  });
});

// ------------------------------------------------------------ Szenario-Linse

describe('Szenario-Linse', () => {
  it('sagt es, wenn noch kein Szenario durchgespielt wurde', () => {
    const svg = renderSzenario(null, () => {});
    expect(svg.textContent).toContain('Noch kein Szenario durchgespielt');
    expect(svg.querySelectorAll('.folge')).toHaveLength(0);
  });

  it('sagt es, wenn ein Szenario keine auswertbare Folge ergab', () => {
    const svg = renderSzenario(szenario({ folgen: [] }), () => {});
    expect(svg.textContent).toContain('Keine auswertbare Folge genannt');
  });

  it('beschriftet die Häufigkeit als Häufigkeit — nie als Wahrscheinlichkeit', () => {
    const svg = renderSzenario(
      szenario({
        folgen: [folge({
          id: 'flg1', text: 'Der Aufwand steigt.', anzahl: 2, von: 5,
          nennungen: [nennung('j1', 'Alto', 'Der Aufwand steigt.'),
                      nennung('j2', 'Basso', 'Der Aufwand steigt.')],
        })],
      }),
      () => {},
    );
    expect(svg.textContent).toContain('von 2 von 5 Stimmen genannt');
    for (const wort of ['wahrscheinlich', 'Wahrscheinlich', 'Prognose', 'Risiko', 'Punkte']) {
      expect(svg.textContent).not.toContain(wort);
    }
  });

  it('zählt Stimmen, nicht Nennungen: dieselbe Stimme zweimal bleibt eine', () => {
    const eine = folge({
      id: 'flg1', text: 'Der Aufwand steigt.', anzahl: 1, von: 3,
      nennungen: [nennung('j1', 'Alto', 'a'), nennung('j1', 'Alto', 'b')],
    });
    const auswahl: string[][] = [];
    const svg = renderSzenario(szenario({ folgen: [eine] }), (s) => auswahl.push(s.jobIds));
    svg.querySelector<SVGGElement>('.folge')!.dispatchEvent(new MouseEvent('click'));
    expect(auswahl).toEqual([['j1']]);
  });

  it('gibt jeder Folge so viele Punkte wie Stimmen — gefüllt, wie viele sie nannten', () => {
    const svg = renderSzenario(
      szenario({
        folgen: [folge({ id: 'flg1', text: 'Folge.', anzahl: 2, von: 5 })],
      }),
      () => {},
    );
    expect(svg.querySelectorAll('.stimmpunkt')).toHaveLength(5);
    expect(svg.querySelectorAll('.stimmpunkt.voll')).toHaveLength(2);
    expect(svg.querySelectorAll('.stimmpunkt.leer')).toHaveLength(3);
  });

  it('gibt widersprechenden Folgen die Widerspruchsfarbe, sonst die geteilte oder einzelne', () => {
    expect(folgenArt(folge({ id: 'a', text: 't', anzahl: 3, gegensatz: ['b'] }))).toBe('contra');
    expect(folgenArt(folge({ id: 'a', text: 't', anzahl: 3 }))).toBe('agree');
    expect(folgenArt(folge({ id: 'a', text: 't', anzahl: 1 }))).toBe('unique');
  });

  it('ist über die Tastatur bedienbar und sagt jeder Folge ihren Sinn an', () => {
    const auswahl: string[][] = [];
    const svg = renderSzenario(
      szenario({
        folgen: [folge({
          id: 'flg1', text: 'Der Aufwand steigt.', anzahl: 1, von: 3,
          nennungen: [nennung('j1', 'Alto', 'Der Aufwand steigt.')],
          gegensatz: ['flg2'],
        })],
      }),
      (s) => auswahl.push(s.jobIds),
    );
    const eintrag = svg.querySelector<SVGGElement>('.folge')!;
    expect(eintrag.getAttribute('tabindex')).toBe('0');
    expect(eintrag.getAttribute('aria-label')).toContain('von 1 von 3 Stimmen genannt');
    expect(eintrag.getAttribute('aria-label')).toContain('entgegen');
    eintrag.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(auswahl).toEqual([['j1']]);
  });

  it('verschweigt übergangene Folgen nicht, sondern zählt sie', () => {
    const svg = renderSzenario(
      szenario({ folgen: [folge({ id: 'flg1', text: 'Folge.' })], uebergangen: 7 }),
      () => {},
    );
    expect(svg.textContent).toContain('+ 7 weitere genannte Folgen');
  });

  it('kommt mit fünf Stimmen und vielen Folgen zurecht, ohne dass sich Zeilen überlagern', () => {
    const viele = Array.from({ length: 8 }, (_, i) =>
      folge({ id: `flg${i}`, text: `Eine genannte Folge Nummer ${i}.`, anzahl: 1, von: 5 }));
    const svg = renderSzenario(szenario({ folgen: viele }), () => {});
    const kaesten = [...svg.querySelectorAll<SVGRectElement>('.folgenflaeche')].map((r) => ({
      y: Number(r.getAttribute('y')),
      h: Number(r.getAttribute('height')),
    }));
    expect(kaesten).toHaveLength(8);
    for (let i = 1; i < kaesten.length; i += 1) {
      expect(kaesten[i].y).toBeGreaterThanOrEqual(kaesten[i - 1].y + kaesten[i - 1].h);
    }
  });
});

// ----------------------------------------------------------- Herkunfts-Linse

function bezug(over: Partial<Relation> & Pick<Relation, 'id' | 'status' | 'origin'>): Relation {
  return {
    session_id: 's', from_id: 'j1', to_id: 'j2', type: 'uebereinstimmung',
    note: '', created_at: 1,
    ...over,
  } as Relation;
}

const NAMEN = new Map([['j1', 'Alto'], ['j2', 'Basso']]);

describe('Herkunfts-Linse', () => {
  it('sagt bei leeren Daten, dass es nichts gibt, worauf man sich berufen könnte', () => {
    const svg = renderHerkunft([], NAMEN, () => {});
    expect(svg.textContent).toContain('Noch kein Bezug gesetzt');
    expect(svg.querySelectorAll('.bezug')).toHaveLength(0);
  });

  it('trennt bestätigt, Vorschlag und verworfen und zählt jedes Band', () => {
    const svg = renderHerkunft(
      [
        bezug({ id: 'b1', status: 'bestaetigt', origin: 'mensch' }),
        bezug({ id: 'b2', status: 'vorschlag', origin: 'maschine' }),
        bezug({ id: 'b3', status: 'vorschlag', origin: 'maschine' }),
      ],
      NAMEN,
      () => {},
    );
    expect(svg.textContent).toContain('Von dir bestätigt — 1');
    expect(svg.textContent).toContain('Vorschlag der Auswertung — 2');
    // Ein leeres Band verschwindet nicht: „null bestätigt" ist selbst ein Befund.
    expect(svg.textContent).toContain('Von dir verworfen — 0');
    expect(svg.textContent).toContain('— keiner —');
  });

  it('trägt den Stand in der Strichart, nicht in einer eigenen Farbe', () => {
    const svg = renderHerkunft(
      [
        bezug({ id: 'b1', status: 'bestaetigt', origin: 'mensch' }),
        bezug({ id: 'b2', status: 'vorschlag', origin: 'maschine' }),
        bezug({ id: 'b3', status: 'abgelehnt', origin: 'maschine' }),
      ],
      NAMEN,
      () => {},
    );
    for (const stand of ['bestaetigt', 'vorschlag', 'abgelehnt']) {
      expect(svg.querySelector(`.bezug.${stand} .bezugstrich`)).not.toBeNull();
    }
    // Keine Kodierung über ein Farbattribut am Element selbst.
    for (const strich of svg.querySelectorAll('.bezugstrich')) {
      expect(strich.getAttribute('stroke')).toBeNull();
      expect(strich.getAttribute('fill')).toBeNull();
    }
  });

  it('unterscheidet Herkunft über die Form und benennt sie in Worten', () => {
    const svg = renderHerkunft(
      [
        bezug({ id: 'b1', status: 'bestaetigt', origin: 'mensch' }),
        bezug({ id: 'b2', status: 'bestaetigt', origin: 'maschine' }),
      ],
      NAMEN,
      () => {},
    );
    expect(svg.querySelectorAll('.herkunftszeichen.mensch')).toHaveLength(1);
    expect(svg.querySelectorAll('.herkunftszeichen.maschine')).toHaveLength(1);
    const beschriftungen = [...svg.querySelectorAll('.bezug')]
      .map((g) => g.getAttribute('aria-label') ?? '');
    expect(beschriftungen.some((b) => b.includes('von dir gesetzt'))).toBe(true);
    expect(beschriftungen.some((b) => b.includes('maschineller Vorschlag'))).toBe(true);
  });

  it('öffnet beim Klick beide beteiligten Beiträge', () => {
    const auswahl: string[][] = [];
    const svg = renderHerkunft(
      [bezug({ id: 'b1', status: 'bestaetigt', origin: 'mensch' })],
      NAMEN,
      (s) => auswahl.push(s.jobIds),
    );
    const eintrag = svg.querySelector<SVGGElement>('.bezug')!;
    expect(eintrag.getAttribute('tabindex')).toBe('0');
    eintrag.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(auswahl).toEqual([['j1', 'j2']]);
  });

  it('zeigt einen unbekannten Beitrag mit seiner Kennung, statt ihn zu verschlucken', () => {
    const svg = renderHerkunft(
      [bezug({ id: 'b1', status: 'bestaetigt', origin: 'mensch', to_id: 'fremd' })],
      NAMEN,
      () => {},
    );
    expect(svg.textContent).toContain('fremd');
  });
});

// ---------------------------------------------------------------- Zeitlinse

describe('Zeitlinse', () => {
  it('sagt es, wenn nichts gemessen wurde, statt eine leere Skala zu zeichnen', () => {
    expect(renderZeit([], () => {}).textContent).toContain('Noch keine Zeiten erfasst');
    const ungestartet = [job({ id: 'j1', label: 'Alto', status: 'not_requested',
      started_at: null, finished_at: null })];
    expect(renderZeit(ungestartet, () => {}).textContent)
      .toContain('Noch keine Zeiten erfasst');
  });

  it('zeigt Warten und Schreiben getrennt — sonst sähe Warten aus wie Langsamkeit', () => {
    const svg = renderZeit(
      [job({ id: 'j1', label: 'Alto', created_at: 100, started_at: 105, finished_at: 106 })],
      () => {},
    );
    expect(svg.querySelectorAll('.warteband')).toHaveLength(1);
    expect(svg.querySelectorAll('.schreibband')).toHaveLength(1);
    expect(svg.querySelector('.zeitzeile')?.getAttribute('aria-label'))
      .toContain('5.0 s gewartet');
  });

  it('zeichnet keinen Wartebalken, wo nicht gewartet wurde', () => {
    const svg = renderZeit(
      [job({ id: 'j1', label: 'Alto', created_at: 100, started_at: 100, finished_at: 101 })],
      () => {},
    );
    expect(svg.querySelectorAll('.warteband')).toHaveLength(0);
  });

  it('erfindet kein Ende für einen laufenden Auftrag', () => {
    const svg = renderZeit(
      [job({ id: 'j1', label: 'Alto', status: 'streaming',
        created_at: 100, started_at: 101, finished_at: null })],
      () => {},
    );
    expect(svg.querySelectorAll('.schreibband')).toHaveLength(0);
    expect(svg.querySelectorAll('.offenesband')).toHaveLength(1);
    expect(svg.querySelector('.zeitzeile')?.getAttribute('aria-label')).toContain('läuft noch');
  });

  it('behält die Reihenfolge des Tisches bei und sortiert nicht nach Dauer', () => {
    const svg = renderZeit(
      [
        job({ id: 'j1', label: 'Langsam', created_at: 100, started_at: 100, finished_at: 130 }),
        job({ id: 'j2', label: 'Schnell', created_at: 100, started_at: 100, finished_at: 101 }),
      ],
      () => {},
    );
    const namen = [...svg.querySelectorAll('.zeitname')].map((t) => t.textContent);
    expect(namen).toEqual(['Langsam', 'Schnell']);
  });

  it('kommt mit fünf Stimmen und mit einer einzigen gleichermaßen zurecht', () => {
    const fuenf = ['Alto', 'Basso', 'Cantus', 'Discant', 'Echo'].map((label, i) =>
      job({ id: `j${i}`, label, created_at: 100, started_at: 100 + i, finished_at: 110 + i }));
    expect(renderZeit(fuenf, () => {}).querySelectorAll('.zeitzeile')).toHaveLength(5);
    expect(renderZeit([fuenf[0]], () => {}).querySelectorAll('.zeitzeile')).toHaveLength(1);
  });

  it('teilt nicht durch null, wenn alles im selben Augenblick geschah', () => {
    const svg = renderZeit(
      [job({ id: 'j1', label: 'Alto', created_at: 100, started_at: 100, finished_at: 100 })],
      () => {},
    );
    for (const rechteck of svg.querySelectorAll('rect')) {
      expect(Number.isFinite(Number(rechteck.getAttribute('x')))).toBe(true);
      expect(Number.isFinite(Number(rechteck.getAttribute('width')))).toBe(true);
    }
  });

  it('ist über die Tastatur bedienbar', () => {
    const auswahl: string[][] = [];
    const svg = renderZeit(
      [job({ id: 'j1', label: 'Alto' })],
      (s) => auswahl.push(s.jobIds),
    );
    const zeile = svg.querySelector<SVGGElement>('.zeitzeile')!;
    expect(zeile.getAttribute('tabindex')).toBe('0');
    zeile.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(auswahl).toEqual([['j1']]);
  });

  it('schreibt Dauern lesbar, ohne eine Genauigkeit zu behaupten', () => {
    expect(dauerText(0.002)).toBe('2 ms');
    expect(dauerText(2.34)).toBe('2.3 s');
  });
});

// ------------------------------------------------------------ Die Linsenwahl

describe('Die Linsenwahl', () => {
  it('führt fünf Linsen, jede mit Namen und Erklärung', () => {
    expect(LINSEN.map((l) => l.id))
      .toEqual(['stimmen', 'themen', 'szenario', 'herkunft', 'zeit']);
    for (const linse of LINSEN) {
      expect(linse.name.length).toBeGreaterThan(0);
      expect(linse.erklaerung.length).toBeGreaterThan(20);
    }
  });

  it('verspricht in keiner Erklärung eine Vorhersage', () => {
    for (const linse of LINSEN) {
      expect(linse.erklaerung).not.toMatch(/Prognose|Risikowert|Punktzahl|Konfidenz/);
      // Das Wort darf vorkommen — aber nur, um es auszuschließen.
      for (const treffer of linse.erklaerung.matchAll(/(\S+)\s+Wahrscheinlichkeit/g)) {
        expect(treffer[1].toLowerCase()).toBe('keine');
      }
    }
  });
});
