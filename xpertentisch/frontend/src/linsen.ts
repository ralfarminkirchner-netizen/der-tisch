/**
 * Linsen auf dieselben Daten.
 *
 * Das Beziehungsnetz zeigte bisher genau eine Sache: welche Stimme welcher
 * widerspricht. Dieselben Daten tragen mehr — worüber gestritten wird, was nur
 * einer sagt, und welche Wege im Gespräch tatsächlich verfolgt wurden.
 *
 * Jede Linse ist eine andere Projektion **vorhandener** Daten. Keine Linse
 * rechnet etwas hinzu, schätzt oder sagt etwas voraus: was man sieht, steht so
 * in den Antworten, den Fundstellen oder den gesetzten Bezügen.
 */

import { SVG_NS, bedienbar, initialen, shorten } from './graph';
import type { GraphSelection } from './graph';
import type { Marker, SessionBundle, SparkEntry, Summary } from './types';

export type LinsenArt = 'stimmen' | 'themen' | 'verlauf';

export const LINSEN: { id: LinsenArt; name: string; erklaerung: string }[] = [
  {
    id: 'stimmen',
    name: 'Stimmen',
    erklaerung:
      'Wer trifft sich mit wem, und wo widersprechen sie einander? Jede Kante ' +
      'steht für ausgewiesene Fundstellen in beiden Antworten.',
  },
  {
    id: 'themen',
    name: 'Themen',
    erklaerung:
      'Woran hängen die Befunde? Links die Stimmen, rechts die Begriffe, an ' +
      'denen sie sich treffen, streiten oder alleine stehen.',
  },
  {
    id: 'verlauf',
    name: 'Verlauf',
    erklaerung:
      'Welche Wege wurden verfolgt? Jeder Zweig ist ein Beitrag, der aus einem ' +
      'anderen hervorgegangen ist — keine Vorhersage, sondern das, was geschah.',
  },
];

function text(
  x: number, y: number, inhalt: string, klasse = '', anker = 'middle',
): SVGTextElement {
  const t = document.createElementNS(SVG_NS, 'text');
  t.setAttribute('x', String(x));
  t.setAttribute('y', String(y));
  t.setAttribute('text-anchor', anker);
  if (klasse) t.setAttribute('class', klasse);
  t.textContent = inhalt;
  return t;
}

function leereFlaeche(svg: SVGSVGElement, breite: number, hoehe: number, satz: string): void {
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);
  svg.appendChild(text(breite / 2, hoehe / 2, satz, 'leer'));
}

// ------------------------------------------------------------- Themen-Linse

interface ThemenKnoten {
  begriff: string;
  /** Job-Kennungen, die an diesem Begriff hängen — je Art getrennt. */
  einig: Set<string>;
  gegen: Set<string>;
  einzeln: Set<string>;
}

/** Sammelt die Begriffe aus den Fundstellen. Nichts wird dazuerfunden. */
export function themenAusMarkern(markers: Marker[]): ThemenKnoten[] {
  const nach = new Map<string, ThemenKnoten>();
  for (const marker of markers) {
    for (const begriff of marker.topics ?? []) {
      if (!begriff) continue;
      let knoten = nach.get(begriff);
      if (!knoten) {
        knoten = { begriff, einig: new Set(), gegen: new Set(), einzeln: new Set() };
        nach.set(begriff, knoten);
      }
      if (marker.kind === 'uebereinstimmung') knoten.einig.add(marker.job_id);
      else if (marker.kind === 'widerspruch') knoten.gegen.add(marker.job_id);
      else knoten.einzeln.add(marker.job_id);
    }
  }
  // Schwere zuerst: woran die meisten Stimmen hängen, steht oben. Bei
  // Gleichstand entscheidet das Alphabet, damit die Reihenfolge stabil bleibt.
  return [...nach.values()].sort((a, b) => {
    const ga = a.einig.size + a.gegen.size + a.einzeln.size;
    const gb = b.einig.size + b.gegen.size + b.einzeln.size;
    return gb - ga || a.begriff.localeCompare(b.begriff, 'de');
  });
}

const THEMEN_HOECHSTENS = 9;

export function renderThemen(
  summary: Summary,
  markers: Marker[],
  onSelect: (selection: GraphSelection) => void,
): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'graph themen');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Themen und Stimmen');

  const stimmen = summary.models.filter((m) => summary.analysed_jobs.includes(m.job_id));
  const alle = themenAusMarkern(markers);
  const themen = alle.slice(0, THEMEN_HOECHSTENS);

  if (stimmen.length === 0 || themen.length === 0) {
    leereFlaeche(svg, 380, 160, 'Noch keine Begriffe gefunden.');
    return svg;
  }

  const zeile = 34;
  const oben = 26;
  const hoehe = Math.max(stimmen.length, themen.length) * zeile + oben * 2;
  const breite = 380;
  const xStimme = 54;
  const xThema = 250;
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);

  const yStimme = new Map<string, number>();
  stimmen.forEach((m, i) => {
    const mitte = (hoehe - stimmen.length * zeile) / 2;
    yStimme.set(m.job_id, mitte + i * zeile + zeile / 2);
  });
  const yThema = new Map<string, number>();
  themen.forEach((t, i) => {
    const mitte = (hoehe - themen.length * zeile) / 2;
    yThema.set(t.begriff, mitte + i * zeile + zeile / 2);
  });

  // Kanten zuerst, damit die Knoten darüber lesbar bleiben.
  for (const thema of themen) {
    const zy = yThema.get(thema.begriff)!;
    const arten: [Set<string>, string][] = [
      [thema.einig, 'agree'],
      [thema.gegen, 'contra'],
      [thema.einzeln, 'unique'],
    ];
    for (const [menge, art] of arten) {
      for (const jobId of menge) {
        const sy = yStimme.get(jobId);
        if (sy === undefined) continue;
        const pfad = document.createElementNS(SVG_NS, 'path');
        const mx = (xStimme + xThema) / 2;
        pfad.setAttribute('d', `M ${xStimme + 16} ${sy} C ${mx} ${sy}, ${mx} ${zy}, ${xThema - 8} ${zy}`);
        pfad.setAttribute('class', `faden ${art}`);
        svg.appendChild(pfad);
      }
    }
  }

  // Welche Stimmen überhaupt an einem der gezeigten Begriffe hängen.
  const angebunden = new Set<string>();
  for (const thema of themen) {
    for (const id of [...thema.einig, ...thema.gegen, ...thema.einzeln]) angebunden.add(id);
  }

  for (const stimme of stimmen) {
    const y = yStimme.get(stimme.job_id)!;
    const ohne = !angebunden.has(stimme.job_id);
    const gruppe = document.createElementNS(SVG_NS, 'g');
    gruppe.setAttribute('class', `node stimme${ohne ? ' ohne-faden' : ''}`);
    gruppe.setAttribute('tabindex', '0');
    gruppe.setAttribute('role', 'button');
    gruppe.setAttribute('aria-label', `Antwort von ${stimme.label} öffnen`);
    gruppe.dataset.jobId = stimme.job_id;

    const kreis = document.createElementNS(SVG_NS, 'circle');
    kreis.setAttribute('cx', String(xStimme));
    kreis.setAttribute('cy', String(y));
    kreis.setAttribute('r', '14');
    gruppe.appendChild(kreis);
    gruppe.appendChild(text(xStimme, y + 4, initialen(stimme.label)));
    gruppe.appendChild(text(xStimme - 22, y + 4, shorten(stimme.label), 'zahl', 'end'));

    const titel = document.createElementNS(SVG_NS, 'title');
    titel.textContent = ohne
      ? `${stimme.label}: berührt keinen der gezeigten Begriffe.`
      : `${stimme.label}: ${stimme.agreements} Übereinstimmungen, ` +
        `${stimme.contradictions} Widersprüche, ${stimme.unique} einzigartig`;
    gruppe.appendChild(titel);

    bedienbar(gruppe, () => onSelect({ jobIds: [stimme.job_id], label: stimme.label }));
    svg.appendChild(gruppe);
  }

  for (const thema of themen) {
    const y = yThema.get(thema.begriff)!;
    const beteiligt = new Set([...thema.einig, ...thema.gegen, ...thema.einzeln]);
    const gruppe = document.createElementNS(SVG_NS, 'g');
    gruppe.setAttribute('class', 'node thema');
    gruppe.setAttribute('tabindex', '0');
    gruppe.setAttribute('role', 'button');
    const wie = thema.gegen.size > 0
      ? 'umstritten'
      : thema.einig.size > 0 ? 'geteilt' : 'einzeln genannt';
    gruppe.setAttribute(
      'aria-label',
      `Begriff „${thema.begriff}", ${wie}, ${beteiligt.size} Stimmen — Antworten öffnen`,
    );
    gruppe.dataset.thema = thema.begriff;

    // Ein Begriff, der umstritten ist, bekommt die Widerspruchsfarbe — das ist
    // dieselbe Bedeutung wie überall sonst, keine zweite Bedeutung der Farbe.
    const r = 4 + Math.min(beteiligt.size, 5);
    const punkt = document.createElementNS(SVG_NS, 'circle');
    punkt.setAttribute('cx', String(xThema));
    punkt.setAttribute('cy', String(y));
    punkt.setAttribute('r', String(r));
    punkt.setAttribute(
      'class',
      `themapunkt ${thema.gegen.size > 0 ? 'contra' : thema.einig.size > 0 ? 'agree' : 'unique'}`,
    );
    gruppe.appendChild(punkt);
    // Der Name beginnt hinter dem Punkt, nicht auf ihm — sonst überdeckte die
    // Größe des Punktes die ersten Buchstaben.
    gruppe.appendChild(text(xThema + r + 7, y + 4, thema.begriff, 'themaname', 'start'));

    const titel = document.createElementNS(SVG_NS, 'title');
    titel.textContent =
      `„${thema.begriff}" — ${thema.einig.size} einig, ${thema.gegen.size} im Widerspruch, ` +
      `${thema.einzeln.size} allein`;
    gruppe.appendChild(titel);

    bedienbar(gruppe, () =>
      onSelect({ jobIds: [...beteiligt], label: `Begriff „${thema.begriff}"` }),
    );
    svg.appendChild(gruppe);
  }

  if (alle.length > themen.length) {
    svg.appendChild(
      text(breite / 2, hoehe - 8, `+ ${alle.length - themen.length} weitere Begriffe`, 'leer'),
    );
  }

  return svg;
}

// ------------------------------------------------------------ Verlaufsbaum

interface Zweig {
  entry: SparkEntry;
  tiefe: number;
  elternId: string | null;
}

/**
 * Ordnet die Beiträge einer Sitzung zu einem Baum.
 *
 * Ein Beitrag hängt an dem Beitrag, aus dessen Antwort er hervorgegangen ist —
 * das steht in seinen ausdrücklich gesetzten Bezügen. Ohne Bezug steht er als
 * eigener Ursprung da.
 */
export function baueZweige(sparks: SparkEntry[]): Zweig[] {
  const sparkVonJob = new Map<string, string>();
  for (const eintrag of sparks) {
    for (const job of eintrag.jobs) sparkVonJob.set(job.id, eintrag.spark.id);
  }

  const elternVon = new Map<string, string | null>();
  for (const eintrag of sparks) {
    let eltern: string | null = null;
    for (const ref of eintrag.spark.refs ?? []) {
      const quelle = sparkVonJob.get(ref);
      if (quelle && quelle !== eintrag.spark.id) {
        eltern = quelle;
        break;
      }
    }
    elternVon.set(eintrag.spark.id, eltern);
  }

  const tiefeVon = new Map<string, number>();
  const tiefe = (id: string, gesehen = new Set<string>()): number => {
    if (tiefeVon.has(id)) return tiefeVon.get(id)!;
    if (gesehen.has(id)) return 0; // Schutz gegen Ringe in den Bezügen
    gesehen.add(id);
    const eltern = elternVon.get(id) ?? null;
    const wert = eltern ? tiefe(eltern, gesehen) + 1 : 0;
    tiefeVon.set(id, wert);
    return wert;
  };

  return sparks.map((eintrag) => ({
    entry: eintrag,
    tiefe: tiefe(eintrag.spark.id),
    elternId: elternVon.get(eintrag.spark.id) ?? null,
  }));
}

const ART_KURZ: Record<string, string> = {
  funke: 'Funke',
  antwort: 'Antwort',
  weitergabe: 'Weitergabe',
  gegenposition: 'Gegenposition',
  vertiefung: 'Vertiefung',
  szenario: 'Szenario',
  kuratierung: 'Kuratierung',
  pingpong: 'Wechselrede',
};

export function renderVerlauf(
  bundle: SessionBundle,
  aktuellerSpark: string | null,
  onSelect: (sparkId: string) => void,
): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'graph verlauf');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Verlauf der Sitzung');

  const zweige = baueZweige(bundle.sparks);
  if (zweige.length === 0) {
    leereFlaeche(svg, 380, 120, 'Noch kein Beitrag gesetzt.');
    return svg;
  }

  const kastenBreite = 128;
  const kastenHoehe = 38;
  const spaltenAbstand = 46;
  const zeilenAbstand = 14;
  const rand = 12;

  const maxTiefe = Math.max(...zweige.map((z) => z.tiefe));
  const breite = rand * 2 + (maxTiefe + 1) * kastenBreite + maxTiefe * spaltenAbstand;
  const hoehe = rand * 2 + zweige.length * (kastenHoehe + zeilenAbstand) - zeilenAbstand;
  svg.setAttribute('viewBox', `0 0 ${breite} ${hoehe}`);

  const pos = new Map<string, { x: number; y: number }>();
  zweige.forEach((zweig, i) => {
    pos.set(zweig.entry.spark.id, {
      x: rand + zweig.tiefe * (kastenBreite + spaltenAbstand),
      y: rand + i * (kastenHoehe + zeilenAbstand),
    });
  });

  // Verbindungen: rechtwinklig, damit die Abzweigung als Abzweigung lesbar ist.
  for (const zweig of zweige) {
    if (!zweig.elternId) continue;
    const von = pos.get(zweig.elternId);
    const nach = pos.get(zweig.entry.spark.id);
    if (!von || !nach) continue;
    const x1 = von.x + kastenBreite;
    const y1 = von.y + kastenHoehe / 2;
    const x2 = nach.x;
    const y2 = nach.y + kastenHoehe / 2;
    const mx = x1 + spaltenAbstand / 2;
    const pfad = document.createElementNS(SVG_NS, 'path');
    pfad.setAttribute('d', `M ${x1} ${y1} H ${mx} V ${y2} H ${x2}`);
    pfad.setAttribute('class', 'ast');
    svg.appendChild(pfad);
  }

  for (const zweig of zweige) {
    const { spark, jobs } = zweig.entry;
    const p = pos.get(spark.id)!;
    const gruppe = document.createElementNS(SVG_NS, 'g');
    const aktuell = spark.id === aktuellerSpark;
    gruppe.setAttribute('class', `zweig${aktuell ? ' aktuell' : ''}`);
    gruppe.setAttribute('tabindex', '0');
    gruppe.setAttribute('role', 'button');
    gruppe.dataset.sparkId = spark.id;

    const kasten = document.createElementNS(SVG_NS, 'rect');
    kasten.setAttribute('x', String(p.x));
    kasten.setAttribute('y', String(p.y));
    kasten.setAttribute('width', String(kastenBreite));
    kasten.setAttribute('height', String(kastenHoehe));
    kasten.setAttribute('rx', '8');
    gruppe.appendChild(kasten);

    gruppe.appendChild(
      text(p.x + 10, p.y + 15, ART_KURZ[spark.kind] ?? spark.kind, 'zweigart', 'start'),
    );
    gruppe.appendChild(
      text(p.x + 10, p.y + 29, gekuerzt(spark.prompt, 20), 'zweigtext', 'start'),
    );

    // Ein Strich je Antwort — der Zustand der Runde auf einen Blick, ohne Zahl.
    jobs.slice(0, 8).forEach((job, i) => {
      const strich = document.createElementNS(SVG_NS, 'rect');
      strich.setAttribute('x', String(p.x + kastenBreite - 9 - i * 5));
      strich.setAttribute('y', String(p.y + 8));
      strich.setAttribute('width', '3');
      strich.setAttribute('height', '8');
      strich.setAttribute('rx', '1.5');
      strich.setAttribute('class', `zustandsstrich zustand-${job.status}`);
      gruppe.appendChild(strich);
    });

    const titel = document.createElementNS(SVG_NS, 'title');
    titel.textContent =
      `${ART_KURZ[spark.kind] ?? spark.kind} ${spark.seq}: ${spark.prompt}` +
      `\n${jobs.length} Antwort(en)`;
    gruppe.appendChild(titel);
    gruppe.setAttribute(
      'aria-label',
      `${ART_KURZ[spark.kind] ?? spark.kind} ${spark.seq}, ${jobs.length} Antworten — hingehen`,
    );

    bedienbar(gruppe, () => onSelect(spark.id));
    svg.appendChild(gruppe);
  }

  return svg;
}

function gekuerzt(satz: string, zeichen: number): string {
  const sauber = satz.replace(/\s+/g, ' ').trim();
  return sauber.length > zeichen ? `${sauber.slice(0, zeichen - 1)}…` : sauber;
}
